use std::collections::{BTreeMap, HashMap, HashSet};

use bevy::{
    prelude::*,
    render::{
        render_resource::*,
        renderer::{RenderDevice, RenderQueue},
        sync_world::MainEntity,
    },
};
use bytemuck::{Pod, Zeroable};

use crate::{
    LOD,
    voxel::{
        types::{CreateStorage, DrawIndexedIndirectArgs, entries},
        util::{hash_chunk_pos, make_batch_uniform_buffer, next_pow2},
    },
};

// Average fraction of cells per chunk provisioned for vertex/index output.
// Each chunk gets an exact-size range from a shared heap (see `RangeHeap`),
// so this only has to cover the arena-wide average. Sandbox terrain
// measured 1.2-3.4%. A full heap never drops terrain: uploads wait and the
// chunk keeps its current mesh.
const ACTIVE_FRACTION_ESTIMATE: f32 = 0.06;

/// Indices emitted per active cell, worst case (3 quads x 6).
pub const INDICES_PER_CELL: u32 = 18;

/// Largest chunk count that keeps every per-chunk-strided buffer (vertex
/// buffers dominate, at 32 bytes/cell) under the device's actual
/// max_storage_buffer_binding_size, with a 10% safety margin.
fn max_chunks(
    render_device: &RenderDevice,
    total_cells: u32,
    sdf_elems_per_chunk: u32,
    budget_cells_per_chunk: u32,
) -> u32 {
    let limit = (render_device.limits().max_storage_buffer_binding_size as u64 * 9) / 10;

    let sdf_bytes = sdf_elems_per_chunk as u64 * 4;
    let material_bytes = sdf_elems_per_chunk as u64;
    let flags_bytes = total_cells as u64 * 4;
    let offsets_bytes = total_cells as u64 * 4;
    let scattered_bytes = budget_cells_per_chunk as u64 * 32;
    let vertex_bytes = budget_cells_per_chunk as u64 * 32;
    let index_bytes = budget_cells_per_chunk as u64 * INDICES_PER_CELL as u64 * 4;

    let worst_bytes_per_chunk = [
        sdf_bytes,
        material_bytes,
        flags_bytes,
        offsets_bytes,
        scattered_bytes,
        vertex_bytes,
        index_bytes,
    ]
    .into_iter()
    .max()
    .unwrap();

    let capped = (limit / worst_bytes_per_chunk).max(1) as u32;
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
    pub active_list_pos: u32, // unused; kept so every shader's ChunkMeta stays 16 bytes
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
        if free_len > len {
            self.free.insert(start + len, free_len - len);
        }
        Some(start)
    }

    fn free(&mut self, mut start: u32, mut len: u32) {
        if len == 0 {
            return;
        }
        if let Some((&prev, &prev_len)) = self.free.range(..start).next_back() {
            if prev + prev_len == start {
                self.free.remove(&prev);
                start = prev;
                len += prev_len;
            }
        }
        if let Some(next_len) = self.free.remove(&(start + len)) {
            len += next_len;
        }
        self.free.insert(start, len);
    }
}

/// Capacity to reserve for `count` vertices: a little slack so small edits
/// re-mesh in place, rounded to 64 cells to limit fragmentation.
fn reserve_for(count: u32) -> u32 {
    if count == 0 {
        0
    } else {
        (count + count / 8).next_multiple_of(64)
    }
}

#[derive(Resource)]
pub struct VoxelChunkArena {
    pub max_chunks: u32,
    pub cell_count: u32,   // chunk_voxels + 1, fixed for this LOD bucket
    pub texture_size: u32, // padded SDF size, fixed for this LOD bucket
    pub total_cells: u32,  // cell_count^3
    pub blocks_per_chunk: u32,
    pub sdf_elems_per_chunk: u32,    // texture_size^3
    pub budget_cells_per_chunk: u32, // average vertex/index budget per chunk (ACTIVE_FRACTION_ESTIMATE * total_cells)

    pub sdf_buffer: Buffer,
    pub material_buffer: Buffer,
    pub visibility_mask_buffer: Buffer,
    pub chunk_has_mask_buffer: Buffer,
    pub flags_buffer: Buffer,
    pub compacted_offsets_buffer: Buffer,
    pub scattered_vertex_buffer: Buffer,
    pub final_vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub indirect_args_buffer: Buffer, // indexed by slot
    pub block_sums_buffer: Buffer,
    pub chunk_meta_buffer: Buffer,
    pub compaction_uniform_buffer: Buffer,
    pub batch_uniform_buffer: Buffer,
    pub batch_uniform_buffer_pass3: Buffer, // pass3 uses workgroup_size(8,8,8), needs its own wg_per_chunk_z stride

    pub pass1_bind_group: BindGroup,
    pub pass3_bind_group: BindGroup,
    pub compaction_bind_group: BindGroup,
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

    pub slot_vertex_base_buffer: Buffer,  // [slot] -> vertex base, CPU-written
    pub slot_vertex_count_buffer: Buffer, // [slot] -> exact active cell count, CPU-written
    pub active_slot_map_buffer: Buffer,   // [dispatch idx] -> slot, CPU-written per dispatch

    // Temporary diagnostic: pass3 records [trips, slot, gpu_count, cpu_count]
    // when its range guard fires; read back by `map_guard_readbacks`.
    pub guard_debug_buffer: Buffer,
    pub guard_readback_buffer: Buffer,
    pub guard_state: std::sync::Arc<std::sync::atomic::AtomicU8>,

    // --- GPU-resident chunk_pos -> slot lookup, for the collision query
    // pass's DDA (see voxel/query.rs). Rebuilt CPU-side every frame from
    // slot_of_chunk_pos and re-uploaded — cheap at these table sizes.
    pub slot_of_chunk_pos: HashMap<IVec3, u32>,
    pub chunk_pos_of_slot: Vec<Option<IVec3>>, // len == max_chunks, O(1) release
    pub chunk_lookup_buffer: Buffer,
    pub lookup_capacity: u32,
}

impl VoxelChunkArena {
    pub fn new(
        render_device: &RenderDevice,
        layouts: &super::pipeline::VoxelPipelineLayouts,
        cell_count: u32,
        texture_size: u32,
    ) -> Self {
        let total_cells = cell_count * cell_count * cell_count;
        let sdf_elems_per_chunk = texture_size * texture_size * texture_size;
        let budget_cells_per_chunk =
            ((total_cells as f32) * ACTIVE_FRACTION_ESTIMATE).ceil() as u32;
        let max = max_chunks(
            render_device,
            total_cells,
            sdf_elems_per_chunk,
            budget_cells_per_chunk,
        ) as u64;

        let lookup_capacity = next_pow2((max as u32).saturating_mul(2)).max(4);
        let sdf_buffer = render_device.storage(
            "arena_sdf_buffer",
            sdf_elems_per_chunk as u64 * 4 * max,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        let material_buffer = render_device.storage(
            "arena_material_buffer",
            sdf_elems_per_chunk as u64 * max,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );

        // 1 bit/voxel, same element count as sdf_buffer/material_buffer, but each
        // slot gets its OWN word-aligned region (mask_words_per_chunk words) —
        // deliberately not a flat globally-packed bit array, since that would let
        // adjacent slots' write_buffer calls corrupt each other's boundary words
        // whenever texture_size^3 isn't a multiple of 32 (it never is, for the
        // LOD sizes in use).
        let mask_words_per_chunk = sdf_elems_per_chunk.div_ceil(32);
        let visibility_mask_buffer = render_device.storage(
            "arena_visibility_mask_buffer",
            mask_words_per_chunk as u64 * 4 * max,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );

        // Tiny: 1 bit per SLOT (not per voxel), packed 32 slots/word.
        let has_mask_words = (max as u32).div_ceil(32);
        let chunk_has_mask_buffer = render_device.storage(
            "arena_chunk_has_mask_buffer",
            has_mask_words as u64 * 4,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );

        let flags_buffer = render_device.storage(
            "arena_flags_buffer",
            total_cells as u64 * 4 * max,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );

        let compacted_offsets_buffer = render_device.storage(
            "arena_compacted_offsets_buffer",
            total_cells as u64 * 4 * max,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );

        // Not read by any shader seen so far, but still bound at pass1
        // binding 3; remove it together with that layout entry if dropping it.
        let scattered_vertex_buffer = render_device.storage(
            "arena_scattered_vertex_buffer",
            budget_cells_per_chunk as u64 * 32 * max,
            BufferUsages::STORAGE,
        );
        let final_vertex_buffer = render_device.storage(
            "arena_final_vertex_buffer",
            budget_cells_per_chunk as u64 * 32 * max,
            BufferUsages::STORAGE | BufferUsages::VERTEX,
        );

        let index_buffer = render_device.storage(
            "arena_index_buffer",
            budget_cells_per_chunk as u64 * INDICES_PER_CELL as u64 * 4 * max,
            BufferUsages::STORAGE | BufferUsages::INDEX,
        );

        let zeroed_args: Vec<DrawIndexedIndirectArgs> = (0..max as u32)
            .map(|_| DrawIndexedIndirectArgs {
                index_count: 0,
                instance_count: 1,
                first_index: 0, // written when the slot gets a vertex range
                base_vertex: 0,
                first_instance: 0,
            })
            .collect();
        let indirect_args_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("arena_indirect_args_buffer"),
            contents: bytemuck::cast_slice(&zeroed_args),
            usage: BufferUsages::STORAGE
                | BufferUsages::INDIRECT
                | BufferUsages::COPY_DST
                | BufferUsages::COPY_SRC,
        });

        let workgroup_capacity = 512u32; // WORKGROUP_SIZE * 2
        let blocks_per_chunk = total_cells.div_ceil(workgroup_capacity);
        let block_sums_buffer = render_device.storage(
            "arena_block_sums_buffer",
            (blocks_per_chunk as u64 * max) * 4,
            BufferUsages::STORAGE,
        );

        let chunk_meta_buffer = render_device.storage(
            "arena_chunk_meta_buffer",
            std::mem::size_of::<ChunkMeta>() as u64 * max,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );

        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct BatchCompactionUniforms {
            chunk_size: u32,
            total_cells: u32,
            blocks_per_chunk: u32,
            _pad0: u32,
        }
        let compaction_uniform_buffer =
            render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("arena_compaction_uniform_buffer"),
                contents: bytemuck::bytes_of(&BatchCompactionUniforms {
                    chunk_size: cell_count,
                    total_cells,
                    blocks_per_chunk,
                    _pad0: 0,
                }),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            });

        let chunk_voxels = texture_size - 2;
        let voxel_size = crate::CHUNK_SIZE / chunk_voxels as f32;

        // Pass 1 uses @workgroup_size(4, 4, 4) -> its own z-stride.
        let batch_uniform_buffer = make_batch_uniform_buffer(
            render_device,
            "arena_batch_uniform_buffer",
            cell_count,
            texture_size,
            4,
            voxel_size,
        );

        // Pass 3 uses @workgroup_size(8, 8, 8) -> a different z-stride.
        let batch_uniform_buffer_pass3 = make_batch_uniform_buffer(
            render_device,
            "arena_batch_uniform_buffer_pass3",
            cell_count,
            texture_size,
            8,
            voxel_size,
        );

        let slot_vertex_base_buffer = render_device.storage(
            "arena_slot_vertex_base_buffer",
            4 * max,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        let slot_vertex_count_buffer = render_device.storage(
            "arena_slot_vertex_count_buffer",
            4 * max,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        let active_slot_map_buffer = render_device.storage(
            "arena_active_slot_map_buffer",
            4 * max,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );

        let guard_debug_buffer = render_device.storage(
            "arena_guard_debug_buffer",
            16,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        );
        let guard_readback_buffer = render_device.storage(
            "arena_guard_readback_buffer",
            16,
            BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        );

        let pass1_bind_group = render_device.create_bind_group(
            Some("arena_pass1_bind_group"),
            &layouts.pass1_surface_layout,
            &entries(&[
                (0, &sdf_buffer),
                (1, &flags_buffer),
                (2, &compacted_offsets_buffer),
                (3, &scattered_vertex_buffer),
                (4, &final_vertex_buffer),
                (5, &indirect_args_buffer),
                (6, &batch_uniform_buffer),
                (7, &chunk_meta_buffer),
                (8, &active_slot_map_buffer),
                (9, &visibility_mask_buffer),
                (10, &chunk_has_mask_buffer),
            ]),
        );

        let pass3_bind_group = render_device.create_bind_group(
            Some("arena_pass3_bind_group"),
            &layouts.pass3_surface_layout,
            &entries(&[
                (0, &sdf_buffer),
                (1, &flags_buffer),
                (2, &compacted_offsets_buffer),
                (3, &final_vertex_buffer),
                (4, &index_buffer),
                (5, &indirect_args_buffer),
                (6, &batch_uniform_buffer_pass3),
                (7, &chunk_meta_buffer),
                (8, &slot_vertex_base_buffer),
                (9, &slot_vertex_count_buffer),
                (10, &active_slot_map_buffer),
                (11, &material_buffer),
                (12, &guard_debug_buffer),
            ]),
        );
        let compaction_bind_group = render_device.create_bind_group(
            Some("arena_compaction_bind_group"),
            &layouts.compaction_bind_group_layout,
            &entries(&[
                (0, &compaction_uniform_buffer),
                (1, &flags_buffer),
                (2, &compacted_offsets_buffer),
                (3, &block_sums_buffer),
                (5, &active_slot_map_buffer), // compaction addresses cells by real slot
            ]),
        );

        let raster_chunk_bind_group = render_device.create_bind_group(
            Some("arena_raster_chunk_bind_group"),
            &layouts.raster_chunk_visibility_layout,
            &entries(&[
                (0, &chunk_meta_buffer),
                (1, &visibility_mask_buffer),
                (2, &chunk_has_mask_buffer),
                (3, &batch_uniform_buffer_pass3),
                (4, &material_buffer),
            ]),
        );

        let empty_lookup: Vec<[u32; 4]> = vec![[u32::MAX; 4]; lookup_capacity as usize];
        let chunk_lookup_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("arena_chunk_lookup_buffer"),
            contents: bytemuck::cast_slice(&empty_lookup),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        Self {
            max_chunks: max as u32,
            cell_count,
            texture_size,
            total_cells,
            sdf_elems_per_chunk,
            budget_cells_per_chunk,
            sdf_buffer,
            flags_buffer,
            compacted_offsets_buffer,
            scattered_vertex_buffer,
            final_vertex_buffer,
            index_buffer,
            indirect_args_buffer,
            block_sums_buffer,
            chunk_meta_buffer,
            compaction_uniform_buffer,
            batch_uniform_buffer,
            batch_uniform_buffer_pass3,
            pass1_bind_group,
            pass3_bind_group,
            compaction_bind_group,
            free_slots: (0..(max as u32)).rev().collect(),
            slot_of_main_entity: HashMap::new(),
            active_slots: Vec::new(),
            dirty_slots: Vec::new(),
            slot_range: vec![(0, 0); max as usize],
            heap: RangeHeap::new(budget_cells_per_chunk * max as u32),
            blocks_per_chunk,
            slot_vertex_base_buffer,
            slot_vertex_count_buffer,
            active_slot_map_buffer,
            guard_debug_buffer,
            guard_readback_buffer,
            guard_state: Default::default(),
            material_buffer,
            visibility_mask_buffer,
            chunk_has_mask_buffer,
            mask_slots: vec![false; max as usize],
            raster_chunk_bind_group,
            slot_of_chunk_pos: HashMap::new(),
            chunk_pos_of_slot: vec![None; max as usize],
            chunk_lookup_buffer,
            lookup_capacity,
        }
    }

    /// Marks whether `slot` currently has valid visibility-mask data
    /// uploaded. The has-mask bits are uploaded each frame by
    /// `prepare_visibility_for_arena`.
    pub fn set_slot_has_mask(&mut self, slot: u32, has_mask: bool) {
        if (slot as usize) < self.mask_slots.len() {
            self.mask_slots[slot as usize] = has_mask;
        }
    }

    /// Records/updates which chunk coordinate owns `slot`, keeping
    /// `slot_of_chunk_pos` and `chunk_pos_of_slot` in sync. Called from
    /// `prepare_voxel_arena` right after slot allocation.
    pub fn register_chunk_pos(&mut self, slot: u32, chunk_pos: IVec3) {
        if self.chunk_pos_of_slot[slot as usize] == Some(chunk_pos) {
            return;
        }
        if let Some(old) = self.chunk_pos_of_slot[slot as usize].take() {
            self.slot_of_chunk_pos.remove(&old);
        }
        self.slot_of_chunk_pos.insert(chunk_pos, slot);
        self.chunk_pos_of_slot[slot as usize] = Some(chunk_pos);
    }

    /// Rebuilds and re-uploads the GPU-resident open-addressing
    /// `chunk_pos -> slot` table from `slot_of_chunk_pos`.
    pub fn rebuild_chunk_lookup(&self, render_queue: &bevy::render::renderer::RenderQueue) {
        const EMPTY: u32 = u32::MAX;
        let cap = self.lookup_capacity as usize;
        let mut table = vec![[EMPTY; 4]; cap];
        for (&pos, &slot) in self.slot_of_chunk_pos.iter() {
            let mut idx = (hash_chunk_pos(pos) as usize) % cap;
            while table[idx][3] != EMPTY {
                idx = (idx + 1) % cap;
            }
            table[idx] = [pos.x as u32, pos.y as u32, pos.z as u32, slot];
        }
        render_queue.write_buffer(&self.chunk_lookup_buffer, 0, bytemuck::cast_slice(&table));
    }

    /// Gives this chunk a slot (reusing its current one) and a vertex range
    /// of at least `count` cells. `None` = no free slot or no room in the
    /// heap; nothing is changed, so the chunk keeps whatever mesh it has.
    pub fn place(&mut self, main_entity: MainEntity, count: u32) -> Option<u32> {
        let existing = self.slot_of_main_entity.get(&main_entity).copied();
        if let Some(slot) = existing {
            let (base, cap) = self.slot_range[slot as usize];
            // Keep the range unless it is too small or far too big.
            if count <= cap && count >= cap / 4 {
                return Some(slot);
            }
            let new_cap = reserve_for(count);
            let new_base = self.heap.alloc(new_cap)?;
            self.heap.free(base, cap);
            self.slot_range[slot as usize] = (new_base, new_cap);
            return Some(slot);
        }
        let slot = *self.free_slots.last()?;
        let cap = reserve_for(count);
        let base = self.heap.alloc(cap)?;
        self.free_slots.pop();
        self.slot_of_main_entity.insert(main_entity, slot);
        self.slot_range[slot as usize] = (base, cap);
        self.active_slots.push(slot);
        Some(slot)
    }

    pub fn release(&mut self, main_entity: MainEntity) {
        if let Some(slot) = self.slot_of_main_entity.remove(&main_entity) {
            self.free_slots.push(slot);
            let (base, cap) = std::mem::take(&mut self.slot_range[slot as usize]);
            self.heap.free(base, cap);
            self.active_slots.retain(|&s| s != slot);
            // Its range may be handed out again this frame; never re-mesh it
            // with the old owner's data.
            self.dirty_slots.retain(|&s| s != slot);

            self.mask_slots[slot as usize] = false;
            if let Some(pos) = self.chunk_pos_of_slot[slot as usize].take() {
                self.slot_of_chunk_pos.remove(&pos);
            }
        }
    }

    pub fn active_chunk_count(&self) -> u32 {
        self.active_slots.len() as u32
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
/// cursor_query.wgsl / query_entries() bind exactly one sdf_buffer +
/// chunk_meta per LOD, so only arena 0 is visible to queries for now.
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
    pub arenas: [Vec<VoxelChunkArena>; LOD::get_number_of_lods()],
    pub config: ArenaConfig,
    /// Chunks that should be resident, written each frame by
    /// `release_despawned_chunks`. Pending uploads for anything else are dropped.
    pub live: HashSet<MainEntity>,
    /// Chunks whose SDF upload is waiting this frame (budget or no room).
    /// Their material/visibility uploads wait too, so the three stay in step.
    pub deferred: HashSet<MainEntity>,
}

impl VoxelChunkArenaSet {
    pub fn with_config(config: ArenaConfig) -> Self {
        Self {
            config,
            ..Default::default()
        }
    }

    /// The query-visible arena for `lod` (the first one), if it exists.
    pub fn first_arena(&self, lod: LOD) -> Option<&VoxelChunkArena> {
        self.arenas[Self::lod_index(lod)].first()
    }

    /// (lod_rank, arena) for every LOD's query-visible arena.
    pub fn query_arenas(&self) -> impl Iterator<Item = (usize, &VoxelChunkArena)> {
        self.arenas
            .iter()
            .enumerate()
            .filter_map(|(rank, v)| v.first().map(|a| (rank, a)))
    }

    #[inline]
    pub fn lod_index(lod: LOD) -> usize {
        lod.rank() as usize
    }

    pub fn iter(&self) -> impl Iterator<Item = &VoxelChunkArena> {
        self.arenas.iter().flatten()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut VoxelChunkArena> {
        self.arenas.iter_mut().flatten()
    }

    /// True if no arena exists for any LOD.
    pub fn is_empty(&self) -> bool {
        self.arenas.iter().all(|v| v.is_empty())
    }

    #[inline]
    pub fn arena_mut(&mut self, lod: LOD, arena_idx: usize) -> &mut VoxelChunkArena {
        &mut self.arenas[Self::lod_index(lod)][arena_idx]
    }

    /// Finds which arena (within `lod`) already holds this entity, if any.
    pub fn find_slot(&self, lod: LOD, main_entity: MainEntity) -> Option<ArenaSlot> {
        self.arenas[Self::lod_index(lod)]
            .iter()
            .enumerate()
            .find_map(|(arena_idx, a)| {
                a.existing_slot(main_entity)
                    .map(|slot| ArenaSlot { arena_idx, slot })
            })
    }

    /// Places this chunk in a `lod` arena with room for `count` vertices:
    ///   1. the arena already holding it, if its range fits or can grow
    ///   2. any other arena of this LOD with a free slot and room
    ///   3. a new arena, up to `config.max_arenas[lod]`
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
        let current = self.find_slot(lod, main_entity).map(|s| s.arena_idx);

        let order = current
            .into_iter()
            .chain((0..self.arenas[lod_idx].len()).filter(|i| Some(*i) != current));
        let mut placed = None;
        for arena_idx in order.collect::<Vec<_>>() {
            if let Some(slot) = self.arenas[lod_idx][arena_idx].place(main_entity, count) {
                placed = Some(ArenaSlot { arena_idx, slot });
                break;
            }
        }
        if placed.is_none() && self.arenas[lod_idx].len() < self.config.max_arenas[lod_idx] {
            for (i, a) in self.arenas[lod_idx].iter().enumerate() {
                info!(
                    "[VoxelChunkArenaSet] LOD {:?} arena {} full for a {}-cell chunk: slots {}/{}, free vertex cells {} (largest run {})",
                    lod,
                    i,
                    count,
                    a.max_chunks as usize - a.free_slots.len(),
                    a.max_chunks,
                    a.heap.free.values().sum::<u32>(),
                    a.heap.free.values().max().copied().unwrap_or(0),
                );
            }
            let mut arena = make_arena();
            if let Some(slot) = arena.place(main_entity, count) {
                let list = &mut self.arenas[lod_idx];
                list.push(arena);
                info!(
                    "[VoxelChunkArenaSet] LOD {:?} now has {} arena(s)",
                    lod,
                    list.len()
                );
                placed = Some(ArenaSlot {
                    arena_idx: list.len() - 1,
                    slot,
                });
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
        self.iter().map(|a| a.max_chunks).sum()
    }

    /// Max entries `rebuild_merged_lookup` can emit (only query-visible arenas).
    pub fn query_visible_max_chunks(&self) -> u32 {
        self.arenas
            .iter()
            .flat_map(|v| v.iter().take(QUERY_ARENAS_PER_LOD))
            .map(|a| a.max_chunks)
            .sum()
    }

    pub fn rebuild_merged_lookup(
        &self,
        render_queue: &RenderQueue,
        buffer: &Buffer,
        capacity: u32,
    ) {
        const EMPTY: u32 = u32::MAX;
        let cap = capacity as usize;
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
        let mut total_entries = 0;

        'outer: for (rank, arenas) in self.arenas.iter().enumerate() {
            for arena in arenas.iter().take(QUERY_ARENAS_PER_LOD) {
                for (&pos, &slot) in arena.slot_of_chunk_pos.iter() {
                    if total_entries >= cap {
                        bevy::log::warn!("Merged lookup table exceeded capacity ({cap})!");
                        break 'outer;
                    }
                    let packed = ((rank as u32) << 28) | (slot & 0x0FFF_FFFF);
                    let initial_idx = (hash_chunk_pos(pos) as usize) % cap;
                    let mut idx = initial_idx;
                    loop {
                        if table[idx].packed_slot_and_rank == EMPTY {
                            table[idx] = GpuChunkLookupEntry {
                                pos_x: pos.x as u32,
                                pos_y: pos.y as u32,
                                pos_z: pos.z as u32,
                                packed_slot_and_rank: packed,
                            };
                            total_entries += 1;
                            break;
                        }
                        idx = (idx + 1) % cap;
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

unsafe impl Send for VoxelChunkArena {}
unsafe impl Sync for VoxelChunkArena {}

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
