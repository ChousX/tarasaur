//! GPU storage for resident chunks.
//!
//! An arena holds one LOD's chunks in fixed-size slots (padded SDF,
//! materials, visibility, metadata, draw args) plus shared vertex/index
//! buffers carved up by `RangeHeap`. Each arena is sized to the device's
//! storage-binding limit; `VoxelChunkArenaSet` adds arenas per LOD up to
//! `ArenaConfig` and moves chunks between them as their LOD changes.

use std::collections::{BTreeMap, HashMap, HashSet};

use bevy::{
    prelude::*,
    render::{
        render_resource::{BindGroup, Buffer, BufferInitDescriptor, BufferUsages},
        renderer::{RenderDevice, RenderQueue},
        sync_world::MainEntity,
    },
};
use bytemuck::{Pod, Zeroable};

use crate::{
    LOD,
    num::{f32_from_u32, u32_from_f32, u64_from_usize, usize_from},
    voxel::{
        types::{CreateStorage, DrawIndexedIndirectArgs, entries},
        util::{hash_chunk_pos, make_batch_uniform_buffer},
    },
};

// Average fraction of cells per chunk provisioned for vertex/index output.
// Each chunk gets an exact-size range from a shared heap (see `RangeHeap`),
// so this only has to cover the arena-wide average. Sandbox terrain
// peaked at ~1.3% average in sandbox runs. A full heap never drops terrain: uploads wait and the
// chunk keeps its current mesh.
const ACTIVE_FRACTION_ESTIMATE: f32 = 0.03;

/// Cells of per-dispatch scratch (flags + compacted offsets, 8 bytes/cell)
/// per arena: ~64 MB, i.e. ~29 High chunks, ~223 Medium, every Low chunk.
/// Uploads past it in one frame wait for the next (keeping their mesh).
const SCRATCH_CELLS: u32 = 8 * 1024 * 1024;

/// Indices emitted per active cell, worst case (3 quads x 6).
pub const INDICES_PER_CELL: u32 = 18;

/// Bytes per vertex in `final_vertex_buffer` (two `vec4<f32>`).
const VERTEX_BYTES: u64 = 32;

const STORAGE_DST: BufferUsages = BufferUsages::STORAGE.union(BufferUsages::COPY_DST);
const STORAGE_SRC: BufferUsages = BufferUsages::STORAGE.union(BufferUsages::COPY_SRC);

/// Largest chunk count that keeps every per-slot buffer (the SDF dominates)
/// under the device's actual `max_storage_buffer_binding_size`, with a 10%
/// safety margin.
fn max_chunks(
    render_device: &RenderDevice,
    total_cells: u32,
    sdf_elems_per_chunk: u32,
    budget_cells_per_chunk: u32,
) -> u32 {
    let limit = render_device
        .limits()
        .max_storage_buffer_binding_size
        .saturating_mul(9)
        / 10;

    let sdf_bytes = u64::from(sdf_elems_per_chunk).saturating_mul(2); // snorm16
    let material_bytes = u64::from(sdf_elems_per_chunk);
    let vertex_bytes = u64::from(budget_cells_per_chunk).saturating_mul(VERTEX_BYTES);
    let index_bytes = u64::from(budget_cells_per_chunk)
        .saturating_mul(u64::from(INDICES_PER_CELL))
        .saturating_mul(4);

    let worst_bytes_per_chunk = sdf_bytes
        .max(material_bytes)
        .max(vertex_bytes)
        .max(index_bytes);

    let fit = limit.checked_div(worst_bytes_per_chunk).unwrap_or(0).max(1);
    let capped = u32::try_from(fit).unwrap_or(u32::MAX);
    info!(
        "[VoxelChunkArena] limit={} bytes, worst-case/chunk={} bytes, capping to {} chunks (budget={} cells/chunk, {:.0}% of {})",
        limit,
        worst_bytes_per_chunk,
        capped,
        budget_cells_per_chunk,
        ACTIVE_FRACTION_ESTIMATE * 100.0,
        total_cells
    );
    capped
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct ChunkMeta {
    pub chunk_world_origin: [f32; 3],
    /// Neighbour LODs for seam stitching; see `systems::seam_lod_ranks`.
    pub lod_ranks: u32,
}

/// First-fit allocator over a range of vertex cells, with coalescing frees.
/// Index ranges mirror it at `INDICES_PER_CELL` x the vertex offset.
#[derive(Default)]
pub struct RangeHeap {
    free: BTreeMap<u32, u32>, // start -> len
}

impl RangeHeap {
    fn new(len: u32) -> Self {
        let mut free = BTreeMap::new();
        if len > 0 {
            free.insert(0, len);
        }
        Self { free }
    }

    fn alloc(&mut self, len: u32) -> Option<u32> {
        if len == 0 {
            return Some(0);
        }
        let (&start, &free_len) = self.free.iter().find(|&(_, &l)| l >= len)?;
        self.free.remove(&start);
        if let Some(rest) = free_len.checked_sub(len).filter(|&r| r > 0) {
            self.free.insert(start.saturating_add(len), rest);
        }
        Some(start)
    }

    fn free(&mut self, mut start: u32, mut len: u32) {
        if len == 0 {
            return;
        }
        if let Some((&prev, &prev_len)) = self.free.range(..start).next_back()
            && prev.checked_add(prev_len) == Some(start)
        {
            self.free.remove(&prev);
            start = prev;
            len = len.saturating_add(prev_len);
        }
        if let Some(next_len) = self.free.remove(&start.saturating_add(len)) {
            len = len.saturating_add(next_len);
        }
        self.free.insert(start, len);
    }

    /// (total free cells, largest free run), for logging.
    fn stats(&self) -> (u32, u32) {
        let total = self
            .free
            .values()
            .fold(0u32, |sum, &l| sum.saturating_add(l));
        (total, self.free.values().max().copied().unwrap_or(0))
    }
}

/// Capacity to reserve for `count` vertices: a little slack so small edits
/// re-mesh in place, rounded to 64 cells to limit fragmentation.
const fn reserve_for(count: u32) -> u32 {
    if count == 0 {
        return 0;
    }
    match count.saturating_add(count / 8).checked_next_multiple_of(64) {
        Some(cap) => cap,
        None => u32::MAX,
    }
}

#[derive(Resource)]
pub struct VoxelChunkArena {
    pub max_chunks: u32,
    /// Chunks one dispatch can mesh; sizes the per-dispatch scratch.
    pub scratch_chunks: u32,
    pub cell_count: u32,          // chunk_voxels + 1, fixed for this LOD bucket
    pub texture_size: u32,        // padded SDF size, fixed for this LOD bucket
    pub total_cells: u32,         // cell_count^3
    pub sdf_elems_per_chunk: u32, // texture_size^3
    pub budget_cells_per_chunk: u32, // average vertex/index budget per chunk (ACTIVE_FRACTION_ESTIMATE * total_cells)

    pub sdf_buffer: Buffer,
    pub material_buffer: Buffer,
    pub visibility_mask_buffer: Buffer,
    pub chunk_has_mask_buffer: Buffer,
    /// Per-dispatch scratch from pass1: `[vertex] -> cell` per chunk.
    pub active_cells_buffer: Buffer,
    /// Per-dispatch scratch from pass1: `[cell] -> vertex` per chunk.
    pub cell_vertex_buffer: Buffer,
    /// Per-dispatch: active cells found per chunk; zeroed before pass1.
    pub chunk_counts_buffer: Buffer,
    pub final_vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub indirect_args_buffer: Buffer, // indexed by slot
    pub chunk_meta_buffer: Buffer,
    pub batch_uniform_buffer: Buffer,
    pub batch_uniform_buffer_pass3: Buffer, // pass3's uniforms, also read by the raster pass

    pub pass1_bind_group: BindGroup,
    pub pass3_bind_group: BindGroup,
    pub raster_chunk_bind_group: BindGroup,

    pub mask_slots: Vec<bool>,
    pub free_slots: Vec<u32>,
    pub slot_of_main_entity: HashMap<MainEntity, u32>,
    pub active_slots: Vec<u32>, // every slot holding a chunk; drawn each frame
    /// Slots to re-mesh this frame. Drained by the compute dispatch, so only
    /// chunks whose inputs changed pay for surface nets.
    pub dirty_slots: Vec<u32>,

    /// Vertex range per slot: (base, capacity) in vertex cells.
    pub slot_range: Vec<(u32, u32)>,
    heap: RangeHeap,

    pub slot_vertex_base_buffer: Buffer, // [slot] -> vertex base, CPU-written
    pub slot_vertex_count_buffer: Buffer, // [slot] -> exact active cell count, CPU-written
    pub active_slot_map_buffer: Buffer,  // [dispatch idx] -> slot, CPU-written per dispatch

    // pass3 records [trips, slot, gpu_count, cpu_count] when its range guard
    // fires; read back by `map_guard_readbacks` when `GuardTelemetry` is on.
    pub guard_debug_buffer: Buffer,
    pub guard_readback_buffer: Buffer,
    pub guard_state: std::sync::Arc<std::sync::atomic::AtomicU8>,

    /// `chunk_pos` <-> slot, for the collision query's merged lookup table
    /// (see `VoxelChunkArenaSet::rebuild_merged_lookup`).
    pub slot_of_chunk_pos: HashMap<IVec3, u32>,
    pub chunk_pos_of_slot: Vec<Option<IVec3>>, // len == max_chunks, O(1) release
}

/// Per-chunk sizes an arena's buffers are allocated from.
struct ArenaSizes {
    cell_count: u32,
    texture_size: u32,
    total_cells: u32,
    sdf_elems_per_chunk: u32,
    budget_cells_per_chunk: u32,
    max_chunks: u32,
    scratch_chunks: u32,
}

impl ArenaSizes {
    fn new(render_device: &RenderDevice, cell_count: u32, texture_size: u32) -> Self {
        let total_cells = cell_count.saturating_pow(3);
        let sdf_elems_per_chunk = texture_size.saturating_pow(3);
        // snorm16 packs 2 voxels per word: an even count keeps every slot word-aligned.
        debug_assert!(sdf_elems_per_chunk.is_multiple_of(2));
        let budget_cells_per_chunk =
            u32_from_f32((f32_from_u32(total_cells) * ACTIVE_FRACTION_ESTIMATE).ceil());
        let max_chunks = max_chunks(
            render_device,
            total_cells,
            sdf_elems_per_chunk,
            budget_cells_per_chunk,
        );
        // Per-dispatch scratch, indexed by position in the dispatch, not slot:
        // only chunks re-meshed this frame need flags and offsets.
        let scratch_chunks = SCRATCH_CELLS
            .checked_div(total_cells)
            .unwrap_or(1)
            .min(max_chunks)
            .max(1);
        Self {
            cell_count,
            texture_size,
            total_cells,
            sdf_elems_per_chunk,
            budget_cells_per_chunk,
            max_chunks,
            scratch_chunks,
        }
    }
}

/// Every GPU buffer an arena owns; moved into `VoxelChunkArena` by `new`.
struct ArenaBuffers {
    sdf: Buffer,
    material: Buffer,
    visibility_mask: Buffer,
    chunk_has_mask: Buffer,
    active_cells: Buffer,
    cell_vertex: Buffer,
    chunk_counts: Buffer,
    final_vertex: Buffer,
    index: Buffer,
    indirect_args: Buffer,
    chunk_meta: Buffer,
    batch_uniform: Buffer,
    batch_uniform_pass3: Buffer,
    slot_vertex_base: Buffer,
    slot_vertex_count: Buffer,
    active_slot_map: Buffer,
    guard_debug: Buffer,
    guard_readback: Buffer,
}

impl ArenaBuffers {
    fn new(render_device: &RenderDevice, sizes: &ArenaSizes) -> Self {
        let dev = render_device;
        let max = u64::from(sizes.max_chunks);
        let scratch = u64::from(sizes.scratch_chunks);
        let per_chunk = |bytes: u64| bytes.saturating_mul(max);
        let sdf_elems = u64::from(sizes.sdf_elems_per_chunk);
        let total_cells = u64::from(sizes.total_cells);
        let budget = u64::from(sizes.budget_cells_per_chunk);

        // 1 bit/voxel, same element count as sdf_buffer/material_buffer, but each
        // slot gets its OWN word-aligned region (mask_words_per_chunk words) —
        // deliberately not a flat globally-packed bit array, since that would let
        // adjacent slots' write_buffer calls corrupt each other's boundary words
        // whenever texture_size^3 isn't a multiple of 32 (it never is, for the
        // LOD sizes in use).
        let mask_words_per_chunk = u64::from(sizes.sdf_elems_per_chunk.div_ceil(32));
        // Tiny: 1 bit per SLOT (not per voxel), packed 32 slots/word.
        let has_mask_words = u64::from(sizes.max_chunks.div_ceil(32));
        let index_bytes = budget
            .saturating_mul(u64::from(INDICES_PER_CELL))
            .saturating_mul(4);
        let meta_bytes = u64_from_usize(std::mem::size_of::<ChunkMeta>());

        Self {
            sdf: dev.storage(
                "arena_sdf_buffer",
                per_chunk(sdf_elems.saturating_mul(2)),
                STORAGE_DST,
            ),
            material: dev.storage("arena_material_buffer", per_chunk(sdf_elems), STORAGE_DST),
            visibility_mask: dev.storage(
                "arena_visibility_mask_buffer",
                per_chunk(mask_words_per_chunk.saturating_mul(4)),
                STORAGE_DST,
            ),
            chunk_has_mask: dev.storage(
                "arena_chunk_has_mask_buffer",
                has_mask_words.saturating_mul(4),
                STORAGE_DST,
            ),
            active_cells: dev.storage(
                "arena_active_cells_buffer",
                total_cells.saturating_mul(4).saturating_mul(scratch),
                BufferUsages::STORAGE,
            ),
            cell_vertex: dev.storage(
                "arena_cell_vertex_buffer",
                total_cells.saturating_mul(4).saturating_mul(scratch),
                BufferUsages::STORAGE,
            ),
            chunk_counts: dev.storage(
                "arena_chunk_counts_buffer",
                scratch.saturating_mul(4),
                STORAGE_DST,
            ),
            final_vertex: dev.storage(
                "arena_final_vertex_buffer",
                per_chunk(budget.saturating_mul(VERTEX_BYTES)),
                BufferUsages::STORAGE | BufferUsages::VERTEX,
            ),
            index: dev.storage(
                "arena_index_buffer",
                per_chunk(index_bytes),
                BufferUsages::STORAGE | BufferUsages::INDEX,
            ),
            indirect_args: indirect_args_buffer(dev, sizes.max_chunks),
            chunk_meta: dev.storage(
                "arena_chunk_meta_buffer",
                per_chunk(meta_bytes),
                STORAGE_DST,
            ),
            batch_uniform: batch_uniform_buffer(dev, sizes, "arena_batch_uniform_buffer", 4),
            batch_uniform_pass3: batch_uniform_buffer(
                dev,
                sizes,
                "arena_batch_uniform_buffer_pass3",
                8,
            ),
            slot_vertex_base: dev.storage(
                "arena_slot_vertex_base_buffer",
                per_chunk(4),
                STORAGE_DST,
            ),
            slot_vertex_count: dev.storage(
                "arena_slot_vertex_count_buffer",
                per_chunk(4),
                STORAGE_DST,
            ),
            active_slot_map: dev.storage("arena_active_slot_map_buffer", per_chunk(4), STORAGE_DST),
            guard_debug: dev.storage(
                "arena_guard_debug_buffer",
                16,
                STORAGE_SRC | BufferUsages::COPY_DST,
            ),
            guard_readback: dev.storage(
                "arena_guard_readback_buffer",
                16,
                BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            ),
        }
    }
}

/// One draw per slot, all empty until the slot gets a vertex range.
fn indirect_args_buffer(render_device: &RenderDevice, max_chunks: u32) -> Buffer {
    let zeroed_args: Vec<DrawIndexedIndirectArgs> = (0..max_chunks)
        .map(|_| DrawIndexedIndirectArgs {
            index_count: 0,
            instance_count: 1,
            first_index: 0, // written when the slot gets a vertex range
            base_vertex: 0,
            first_instance: 0,
        })
        .collect();
    render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("arena_indirect_args_buffer"),
        contents: bytemuck::cast_slice(&zeroed_args),
        usage: BufferUsages::STORAGE
            | BufferUsages::INDIRECT
            | BufferUsages::COPY_DST
            | BufferUsages::COPY_SRC,
    })
}

/// Pass 1 uses `@workgroup_size(4, 4, 4)` and pass 3 `@workgroup_size(8, 8, 8)`,
/// so each gets its own z-stride.
fn batch_uniform_buffer(
    render_device: &RenderDevice,
    sizes: &ArenaSizes,
    label: &'static str,
    wg_size: u32,
) -> Buffer {
    let chunk_voxels = sizes.texture_size.saturating_sub(2);
    let voxel_size = crate::CHUNK_SIZE / f32_from_u32(chunk_voxels);
    make_batch_uniform_buffer(
        render_device,
        label,
        sizes.cell_count,
        sizes.texture_size,
        wg_size,
        voxel_size,
    )
}

/// The arena's three bind groups, in field order.
fn arena_bind_groups(
    render_device: &RenderDevice,
    layouts: &super::pipeline::VoxelPipelineLayouts,
    b: &ArenaBuffers,
) -> [BindGroup; 3] {
    let pass1 = render_device.create_bind_group(
        Some("arena_pass1_bind_group"),
        &layouts.pass1_surface_layout,
        &entries(&[
            (0, &b.sdf),
            (1, &b.active_cells),
            (2, &b.cell_vertex),
            (3, &b.chunk_counts),
            (4, &b.final_vertex),
            (5, &b.indirect_args),
            (6, &b.batch_uniform),
            (7, &b.chunk_meta),
            (8, &b.active_slot_map),
            (9, &b.visibility_mask),
            (10, &b.chunk_has_mask),
        ]),
    );
    let pass3 = render_device.create_bind_group(
        Some("arena_pass3_bind_group"),
        &layouts.pass3_surface_layout,
        &entries(&[
            (0, &b.sdf),
            (1, &b.active_cells),
            (2, &b.cell_vertex),
            (3, &b.final_vertex),
            (4, &b.index),
            (5, &b.indirect_args),
            (6, &b.batch_uniform_pass3),
            (7, &b.chunk_meta),
            (8, &b.slot_vertex_base),
            (9, &b.slot_vertex_count),
            (10, &b.active_slot_map),
            (11, &b.material),
            (12, &b.guard_debug),
            (13, &b.chunk_counts),
        ]),
    );
    let raster = render_device.create_bind_group(
        Some("arena_raster_chunk_bind_group"),
        &layouts.raster_chunk_visibility_layout,
        &entries(&[
            (0, &b.chunk_meta),
            (1, &b.visibility_mask),
            (2, &b.chunk_has_mask),
            (3, &b.batch_uniform_pass3),
            (4, &b.material),
        ]),
    );
    [pass1, pass3, raster]
}

impl VoxelChunkArena {
    pub fn new(
        render_device: &RenderDevice,
        layouts: &super::pipeline::VoxelPipelineLayouts,
        cell_count: u32,
        texture_size: u32,
    ) -> Self {
        let sizes = ArenaSizes::new(render_device, cell_count, texture_size);
        let buffers = ArenaBuffers::new(render_device, &sizes);
        let [pass1_bind_group, pass3_bind_group, raster_chunk_bind_group] =
            arena_bind_groups(render_device, layouts, &buffers);
        let ArenaBuffers {
            sdf,
            material,
            visibility_mask,
            chunk_has_mask,
            active_cells,
            cell_vertex,
            chunk_counts,
            final_vertex,
            index,
            indirect_args,
            chunk_meta,
            batch_uniform,
            batch_uniform_pass3,
            slot_vertex_base,
            slot_vertex_count,
            active_slot_map,
            guard_debug,
            guard_readback,
        } = buffers;
        let slots = usize_from(sizes.max_chunks);

        Self {
            max_chunks: sizes.max_chunks,
            scratch_chunks: sizes.scratch_chunks,
            cell_count,
            texture_size,
            total_cells: sizes.total_cells,
            sdf_elems_per_chunk: sizes.sdf_elems_per_chunk,
            budget_cells_per_chunk: sizes.budget_cells_per_chunk,
            sdf_buffer: sdf,
            active_cells_buffer: active_cells,
            cell_vertex_buffer: cell_vertex,
            chunk_counts_buffer: chunk_counts,
            final_vertex_buffer: final_vertex,
            index_buffer: index,
            indirect_args_buffer: indirect_args,
            chunk_meta_buffer: chunk_meta,
            batch_uniform_buffer: batch_uniform,
            batch_uniform_buffer_pass3: batch_uniform_pass3,
            pass1_bind_group,
            pass3_bind_group,
            free_slots: (0..sizes.max_chunks).rev().collect(),
            slot_of_main_entity: HashMap::new(),
            active_slots: Vec::new(),
            dirty_slots: Vec::new(),
            slot_range: vec![(0, 0); slots],
            heap: RangeHeap::new(
                sizes
                    .budget_cells_per_chunk
                    .saturating_mul(sizes.max_chunks),
            ),
            slot_vertex_base_buffer: slot_vertex_base,
            slot_vertex_count_buffer: slot_vertex_count,
            active_slot_map_buffer: active_slot_map,
            guard_debug_buffer: guard_debug,
            guard_readback_buffer: guard_readback,
            guard_state: std::sync::Arc::default(),
            material_buffer: material,
            visibility_mask_buffer: visibility_mask,
            chunk_has_mask_buffer: chunk_has_mask,
            mask_slots: vec![false; slots],
            raster_chunk_bind_group,
            slot_of_chunk_pos: HashMap::new(),
            chunk_pos_of_slot: vec![None; slots],
        }
    }

    /// Marks whether `slot` currently has valid visibility-mask data
    /// uploaded. The has-mask bits are uploaded each frame by
    /// `prepare_visibility_for_arena`.
    pub fn set_slot_has_mask(&mut self, slot: u32, has_mask: bool) {
        if let Some(m) = self.mask_slots.get_mut(usize_from(slot)) {
            *m = has_mask;
        }
    }

    /// Records/updates which chunk coordinate owns `slot`, keeping
    /// `slot_of_chunk_pos` and `chunk_pos_of_slot` in sync.
    ///
    /// Called from `prepare_voxel_arena` right after slot allocation.
    pub fn register_chunk_pos(&mut self, slot: u32, chunk_pos: IVec3) {
        let Some(owner) = self.chunk_pos_of_slot.get_mut(usize_from(slot)) else {
            return;
        };
        if *owner == Some(chunk_pos) {
            return;
        }
        if let Some(old) = owner.replace(chunk_pos) {
            self.slot_of_chunk_pos.remove(&old);
        }
        self.slot_of_chunk_pos.insert(chunk_pos, slot);
    }

    /// Gives this chunk a slot (reusing its current one) and a vertex range
    /// of at least `count` cells.
    ///
    /// `None` = no free slot or no room in the heap; nothing is changed, so
    /// the chunk keeps whatever mesh it has.
    pub fn place(&mut self, main_entity: MainEntity, count: u32) -> Option<u32> {
        let existing = self.slot_of_main_entity.get(&main_entity).copied();
        if !existing.is_some_and(|s| self.can_mark_dirty(s)) && self.dirty_full() {
            return None; // no scratch left to mesh it this frame
        }
        if let Some(slot) = existing {
            let range = self.slot_range.get_mut(usize_from(slot))?;
            let (base, cap) = *range;
            // Keep the range unless it is too small or far too big.
            if count <= cap && count >= cap / 4 {
                return Some(slot);
            }
            let new_cap = reserve_for(count);
            let new_base = self.heap.alloc(new_cap)?;
            self.heap.free(base, cap);
            *range = (new_base, new_cap);
            return Some(slot);
        }
        let slot = *self.free_slots.last()?;
        let range = self.slot_range.get_mut(usize_from(slot))?;
        let cap = reserve_for(count);
        let base = self.heap.alloc(cap)?;
        self.free_slots.pop();
        self.slot_of_main_entity.insert(main_entity, slot);
        *range = (base, cap);
        self.active_slots.push(slot);
        Some(slot)
    }

    pub fn release(&mut self, main_entity: MainEntity) {
        let Some(slot) = self.slot_of_main_entity.remove(&main_entity) else {
            return;
        };
        self.free_slots.push(slot);
        let i = usize_from(slot);
        if let Some(range) = self.slot_range.get_mut(i) {
            let (base, cap) = std::mem::take(range);
            self.heap.free(base, cap);
        }
        self.active_slots.retain(|&s| s != slot);
        // Its range may be handed out again this frame; never re-mesh it
        // with the old owner's data.
        self.dirty_slots.retain(|&s| s != slot);

        if let Some(m) = self.mask_slots.get_mut(i) {
            *m = false;
        }
        if let Some(pos) = self.chunk_pos_of_slot.get_mut(i).and_then(Option::take) {
            self.slot_of_chunk_pos.remove(&pos);
        }
    }

    /// This frame's dispatch already fills the per-dispatch scratch.
    pub const fn dirty_full(&self) -> bool {
        self.dirty_slots.len() >= usize_from(self.scratch_chunks)
    }

    /// Whether `slot` can be re-meshed this frame (already queued, or room left).
    pub fn can_mark_dirty(&self, slot: u32) -> bool {
        self.dirty_slots.contains(&slot) || !self.dirty_full()
    }

    /// Queues `slot` for this frame's re-mesh. Callers check `can_mark_dirty`
    /// first: an upload that can't be meshed this frame must wait instead.
    pub fn mark_dirty(&mut self, slot: u32) {
        if !self.dirty_slots.contains(&slot) {
            self.dirty_slots.push(slot);
        }
    }

    pub fn existing_slot(&self, main_entity: MainEntity) -> Option<u32> {
        self.slot_of_main_entity.get(&main_entity).copied()
    }
}

/// Max arenas per LOD, indexed by `LOD::rank()` (Lowest, Low, Medium, High).
/// Each arena is ~one device storage-binding-limit in size, so the real
/// capacity is arenas * `max_chunks`. Give the LODs that hold many chunks more.
#[derive(Clone, Debug)]
pub struct ArenaConfig {
    pub max_arenas: [usize; LOD::COUNT],
}

impl Default for ArenaConfig {
    fn default() -> Self {
        Self {
            max_arenas: [1, 1, 2, 4],
        }
    }
}

/// How many arenas per LOD the cursor-query shader can currently address.
/// `cursor_query.wgsl` / `query_entries()` bind exactly one `sdf_buffer` +
/// `chunk_meta` per LOD, so only arena 0 is visible to queries for now.
pub const QUERY_ARENAS_PER_LOD: usize = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaSlot {
    pub arena_idx: usize,
    pub slot: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuChunkLookupEntry {
    pub pos_x: u32,
    pub pos_y: u32,
    pub pos_z: u32,
    pub packed_slot_and_rank: u32,
}

#[derive(Resource, Default)]
pub struct VoxelChunkArenaSet {
    pub arenas: [Vec<VoxelChunkArena>; LOD::COUNT],
    pub config: ArenaConfig,
    /// Chunks that should be resident, written each frame by
    /// `release_despawned_chunks`. Pending uploads for anything else are dropped.
    pub live: HashSet<MainEntity>,
    /// Chunks whose SDF upload is waiting this frame (budget or no room).
    /// Their material/visibility uploads wait too, so the three stay in step.
    pub deferred: HashSet<MainEntity>,
    /// Live chunks whose latest SDF meshes to nothing (exact CPU count of 0).
    /// They hold no slot; their material/visibility uploads are dropped.
    pub empty: HashSet<MainEntity>,
}

impl VoxelChunkArenaSet {
    /// (`lod_rank`, arena) for every LOD's query-visible arena.
    pub fn query_arenas(&self) -> impl Iterator<Item = (usize, &VoxelChunkArena)> {
        self.arenas
            .iter()
            .enumerate()
            .filter_map(|(rank, v)| v.first().map(|a| (rank, a)))
    }

    #[inline]
    pub const fn lod_index(lod: LOD) -> usize {
        usize_from(lod.rank())
    }

    pub fn iter(&self) -> impl Iterator<Item = &VoxelChunkArena> {
        self.arenas.iter().flatten()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut VoxelChunkArena> {
        self.arenas.iter_mut().flatten()
    }

    /// True if no arena exists for any LOD.
    pub fn is_empty(&self) -> bool {
        self.arenas.iter().all(std::vec::Vec::is_empty)
    }

    #[inline]
    pub fn arena_mut(&mut self, lod: LOD, arena_idx: usize) -> Option<&mut VoxelChunkArena> {
        self.arenas
            .get_mut(Self::lod_index(lod))?
            .get_mut(arena_idx)
    }

    /// Finds which arena (within `lod`) already holds this entity, if any.
    pub fn find_slot(&self, lod: LOD, main_entity: MainEntity) -> Option<ArenaSlot> {
        self.arenas
            .get(Self::lod_index(lod))?
            .iter()
            .enumerate()
            .find_map(|(arena_idx, a)| {
                a.existing_slot(main_entity)
                    .map(|slot| ArenaSlot { arena_idx, slot })
            })
    }

    /// Places this chunk in a `lod` arena with room for `count` vertices.
    ///
    /// In order of preference:
    /// 1. the arena already holding it, if its range fits or can grow
    /// 2. any other arena of this LOD with a free slot and room
    /// 3. a new arena, up to `config.max_arenas[lod]`
    ///
    /// Only after it is placed is it evicted from any other arena (an old
    /// LOD, or the arena it outgrew), so it never goes undrawn. `None` = no
    /// room anywhere; nothing changed, the caller retries next frame.
    pub fn place(
        &mut self,
        lod: LOD,
        main_entity: MainEntity,
        count: u32,
        make_arena: impl FnOnce() -> VoxelChunkArena,
    ) -> Option<ArenaSlot> {
        let lod_idx = Self::lod_index(lod);
        let found = self.find_slot(lod, main_entity);
        let max_arenas = self.config.max_arenas.get(lod_idx).copied().unwrap_or(0);
        let empty_chunks = self.empty.len();
        let arenas = self.arenas.get_mut(lod_idx)?;
        // Out of scratch is a "this frame" limit, not a lack of room: wait
        // rather than moving the chunk or growing a new arena over it.
        if let Some(found) = found
            && !arenas
                .get(found.arena_idx)
                .is_some_and(|a| a.can_mark_dirty(found.slot))
        {
            return None;
        }

        let current = found.map(|s| s.arena_idx);
        let others = (0..arenas.len()).filter(move |i| Some(*i) != current);
        let mut placed = current.into_iter().chain(others).find_map(|arena_idx| {
            let slot = arenas.get_mut(arena_idx)?.place(main_entity, count)?;
            Some(ArenaSlot { arena_idx, slot })
        });
        if placed.is_none() && arenas.iter().any(VoxelChunkArena::dirty_full) {
            return None;
        }
        if placed.is_none() && arenas.len() < max_arenas {
            for (i, a) in arenas.iter().enumerate() {
                let (free_cells, largest_run) = a.heap.stats();
                info!(
                    "[VoxelChunkArenaSet] LOD {:?} arena {} full for a {}-cell chunk: slots {}/{}, free vertex cells {} (largest run {}); {} empty chunks hold no slot",
                    lod,
                    i,
                    count,
                    usize_from(a.max_chunks).saturating_sub(a.free_slots.len()),
                    a.max_chunks,
                    free_cells,
                    largest_run,
                    empty_chunks,
                );
            }
            let mut arena = make_arena();
            if let Some(slot) = arena.place(main_entity, count) {
                placed = Some(ArenaSlot {
                    arena_idx: arenas.len(),
                    slot,
                });
                arenas.push(arena);
                info!(
                    "[VoxelChunkArenaSet] LOD {:?} now has {} arena(s)",
                    lod,
                    arenas.len()
                );
            }
        }
        let placed = placed?;

        for (i, arenas) in self.arenas.iter_mut().enumerate() {
            for (j, a) in arenas.iter_mut().enumerate() {
                if (i, j) != (lod_idx, placed.arena_idx) {
                    a.release(main_entity);
                }
            }
        }
        Some(placed)
    }

    /// Releases the entity from whichever arena holds it, across all LODs.
    pub fn release(&mut self, main_entity: MainEntity) {
        for a in self.iter_mut() {
            a.release(main_entity);
        }
    }

    pub fn total_max_chunks(&self) -> u32 {
        self.iter()
            .fold(0u32, |sum, a| sum.saturating_add(a.max_chunks))
    }

    pub fn rebuild_merged_lookup(
        &self,
        render_queue: &RenderQueue,
        buffer: &Buffer,
        capacity: u32,
    ) {
        const EMPTY: u32 = u32::MAX;
        let cap = usize_from(capacity);
        if cap == 0 {
            return;
        }

        let mut table = vec![
            GpuChunkLookupEntry {
                pos_x: 0,
                pos_y: 0,
                pos_z: 0,
                packed_slot_and_rank: EMPTY,
            };
            cap
        ];
        let mut total_entries: usize = 0;
        // Linear probing; `cap` is non-zero, so the remainders can't fail.
        let wrap = |i: usize| i.checked_rem(cap).unwrap_or(0);

        'outer: for (rank, arenas) in self.arenas.iter().enumerate() {
            let rank = u32::try_from(rank).unwrap_or(0);
            for arena in arenas.iter().take(QUERY_ARENAS_PER_LOD) {
                for (&pos, &slot) in &arena.slot_of_chunk_pos {
                    if total_entries >= cap {
                        bevy::log::warn!("Merged lookup table exceeded capacity ({cap})!");
                        break 'outer;
                    }
                    let packed = (rank << 28) | (slot & 0x0FFF_FFFF);
                    let initial_idx = wrap(usize_from(hash_chunk_pos(pos)));
                    let mut idx = initial_idx;
                    loop {
                        let Some(entry) = table.get_mut(idx) else {
                            break 'outer;
                        };
                        if entry.packed_slot_and_rank == EMPTY {
                            *entry = GpuChunkLookupEntry {
                                pos_x: pos.x.cast_unsigned(),
                                pos_y: pos.y.cast_unsigned(),
                                pos_z: pos.z.cast_unsigned(),
                                packed_slot_and_rank: packed,
                            };
                            total_entries = total_entries.saturating_add(1);
                            break;
                        }
                        idx = wrap(idx.wrapping_add(1));
                        if idx == initial_idx {
                            bevy::log::error!("Merged lookup table completely full!");
                            break 'outer;
                        }
                    }
                }
            }
        }

        render_queue.write_buffer(buffer, 0, bytemuck::cast_slice(&table));
    }
}

#[cfg(test)]
mod tests {
    use super::RangeHeap;

    #[test]
    fn heap_reuses_and_coalesces() {
        let mut h = RangeHeap::new(256);
        let a = h.alloc(64).unwrap();
        let b = h.alloc(64).unwrap();
        let c = h.alloc(128).unwrap();
        assert!(h.alloc(1).is_none());
        h.free(a, 64);
        h.free(c, 128);
        h.free(b, 64); // joins both neighbours
        assert_eq!(h.alloc(256), Some(0));
    }
}
