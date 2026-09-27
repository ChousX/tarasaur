use std::collections::HashMap;

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

const ACTIVE_FRACTION_ESTIMATE: f32 = 0.20; // tune from telemetry

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
    let index_bytes = budget_cells_per_chunk as u64 * 18 * 4;

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
    pub active_list_pos: u32,
}

#[derive(Resource)]
pub struct VoxelChunkArena {
    pub max_chunks: u32,
    pub cell_count: u32,   // chunk_voxels + 1, fixed for this LOD bucket
    pub texture_size: u32, // padded SDF size, fixed for this LOD bucket
    pub total_cells: u32,  // cell_count^3
    pub blocks_per_chunk: u32,
    pub sdf_elems_per_chunk: u32,    // texture_size^3
    pub budget_cells_per_chunk: u32, // provisioned vertex/index budget per chunk (ACTIVE_FRACTION_ESTIMATE * total_cells)

    pub sdf_buffer: Buffer,
    pub material_buffer: Buffer,
    pub visibility_mask_buffer: Buffer,
    pub chunk_has_mask_buffer: Buffer,
    pub flags_buffer: Buffer,
    pub compacted_offsets_buffer: Buffer,
    pub scattered_vertex_buffer: Buffer,
    pub final_vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub indirect_args_buffer: Buffer,
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
    pub active_slots: Vec<u32>, // stable order used for this frame's batched dispatch
    pub dirty_slots: Vec<u32>,  // slots whose ChunkMeta/SDF changed since last upload

    pub chunk_active_counts_buffer: Buffer, // [active_list_pos] -> active cell count, GPU-written
    pub chunk_vertex_base_buffer: Buffer,   // [active_list_pos] -> exclusive prefix sum
    pub chunk_index_base_buffer: Buffer,    // [active_list_pos] -> vertex_base * 18
    pub active_slot_map_buffer: Buffer, // [active_list_pos] -> real arena slot, CPU-written per frame
    pub overflow_flag_buffer: Buffer,   // single atomic<u32>, cleared each frame
    pub overflow_readback_buffer: Buffer, // CPU-mapped copy for telemetry/next-frame drop
    pub chunk_bases_uniform_buffer: Buffer, // active_count + budget constants
    pub chunk_bases_bind_group: BindGroup,

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

        // Tiny: 1 bit per SLOT (not per voxel), packed 32 slots/word. Kept
        // deliberately separate from ChunkMeta rather than a bit stolen from
        // active_list_pos — pass1 doesn't read ChunkMeta at all since the
        // earlier shrink, and pass3 uses active_list_pos as a raw array index in
        // several places that would all need defensive masking if the flag lived
        // there. This buffer is orders of magnitude smaller than ChunkMeta per
        // slot and keeps pass3 completely untouched by visibility work.
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

        let scattered_vertex_buffer = render_device.storage(
            "arena_scattered_vertex_buffer",
            budget_cells_per_chunk as u64 * 32 * max, // was total_cells — shrunk to match the dynamic budget
            BufferUsages::STORAGE,
        );
        let final_vertex_buffer = render_device.storage(
            "arena_final_vertex_buffer",
            budget_cells_per_chunk as u64 * 32 * max,
            BufferUsages::STORAGE | BufferUsages::VERTEX,
        );

        let index_buffer = render_device.storage(
            "arena_index_buffer",
            budget_cells_per_chunk as u64 * 18 * 4 * max,
            BufferUsages::STORAGE | BufferUsages::INDEX,
        );

        let zeroed_args: Vec<DrawIndexedIndirectArgs> = (0..max as u32)
            .map(|_| DrawIndexedIndirectArgs {
                index_count: 0,
                instance_count: 1,
                first_index: 0, // overwritten every frame by compute_chunk_bases
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
        let blocks_per_chunk = (total_cells + workgroup_capacity - 1) / workgroup_capacity;
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
        // --- Dynamic bump-allocation buffers (created before the bind groups
        // that reference them, since compaction_bind_group and
        // chunk_bases_bind_group both need chunk_active_counts_buffer et al.) ---
        let chunk_active_counts_buffer = render_device.storage(
            "arena_chunk_active_counts_buffer",
            4 * max,
            BufferUsages::STORAGE,
        );
        let chunk_vertex_base_buffer = render_device.storage(
            "arena_chunk_vertex_base_buffer",
            4 * max,
            BufferUsages::STORAGE,
        );
        let chunk_index_base_buffer = render_device.storage(
            "arena_chunk_index_base_buffer",
            4 * max,
            BufferUsages::STORAGE,
        );
        let active_slot_map_buffer = render_device.storage(
            "arena_active_slot_map_buffer",
            4 * max,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        let overflow_flag_buffer = render_device.storage(
            "arena_overflow_flag_buffer",
            4,
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        );
        let overflow_readback_buffer = render_device.storage(
            "arena_overflow_readback_buffer",
            4,
            BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        );

        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct ChunkBasesUniforms {
            active_count: u32,
            budget_cells_per_chunk: u32,
            total_budget_cells: u32, // budget_cells_per_chunk * max
            _pad0: u32,
        }
        let chunk_bases_uniform_buffer =
            render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("arena_chunk_bases_uniform_buffer"),
                contents: bytemuck::bytes_of(&ChunkBasesUniforms {
                    active_count: 0, // updated per-frame via write_buffer
                    budget_cells_per_chunk,
                    total_budget_cells: budget_cells_per_chunk * max as u32,
                    _pad0: 0,
                }),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            });

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
                (8, &chunk_vertex_base_buffer),
                (9, &chunk_index_base_buffer),
                (10, &active_slot_map_buffer),
                (11, &material_buffer),
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
                (4, &chunk_active_counts_buffer),
            ]),
        );

        let raster_chunk_bind_group = render_device.create_bind_group(
            Some("arena_raster_chunk_bind_group"),
            &layouts.raster_chunk_visibility_layout,
            &entries(&[
                (0, &chunk_meta_buffer),
                (1, &chunk_meta_buffer),
                (2, &chunk_has_mask_buffer),
                (3, &batch_uniform_buffer_pass3),
            ]),
        );

        let chunk_bases_bind_group = render_device.create_bind_group(
            Some("arena_chunk_bases_bind_group"),
            &layouts.chunk_bases_layout,
            &entries(&[
                (0, &chunk_bases_uniform_buffer),
                (1, &chunk_active_counts_buffer),
                (2, &active_slot_map_buffer),
                (3, &chunk_vertex_base_buffer),
                (4, &chunk_index_base_buffer),
                (5, &indirect_args_buffer),
                (6, &overflow_flag_buffer),
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
            blocks_per_chunk,
            chunk_active_counts_buffer,
            chunk_vertex_base_buffer,
            chunk_index_base_buffer,
            active_slot_map_buffer,
            overflow_flag_buffer,
            overflow_readback_buffer,
            chunk_bases_uniform_buffer,
            chunk_bases_bind_group,
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
    /// uploaded. Doesn't itself write chunk_has_mask_buffer — that upload
    /// still needs to happen somewhere each frame; see the note in
    /// prepare_visibility_for_arena's caller about wiring that write.
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
    /// `chunk_pos -> slot` table from `slot_of_chunk_pos`. Called once per
    /// frame from `prepare_voxel_queries`, not from the main arena prepare
    /// path — only the collision LOD's arena needs this.
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

    /// Returns the slot for this chunk, allocating a fresh one if it's new.
    pub fn slot_for(&mut self, main_entity: MainEntity) -> Option<u32> {
        if let Some(&slot) = self.slot_of_main_entity.get(&main_entity) {
            return Some(slot);
        }
        let slot = self.free_slots.pop()?; // None => arena full, caller should warn/drop
        self.slot_of_main_entity.insert(main_entity, slot);
        Some(slot)
    }

    pub fn release(&mut self, main_entity: MainEntity) {
        if let Some(slot) = self.slot_of_main_entity.remove(&main_entity) {
            self.free_slots.push(slot);
            self.active_slots.retain(|&s| s != slot);
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

#[derive(Resource, Default)]
pub struct VoxelChunkArenaSet {
    pub arenas: HashMap<LOD, VoxelChunkArena>,
}
impl VoxelChunkArenaSet {
    pub fn total_max_chunks(&self) -> u32 {
        self.arenas.values().map(|a| a.max_chunks).sum()
    }

    /// Rebuilds a combined chunk_pos -> (lod_rank, slot) table across every
    /// live LOD arena. Unlike each arena's own lookup (which only knows
    /// about its own slots), this is what the query shader now walks so a
    /// ray resolves whatever LOD is actually resident at a position.
    pub fn rebuild_merged_lookup(
        &self,
        render_queue: &RenderQueue,
        buffer: &Buffer,
        capacity: u32,
    ) {
        const EMPTY: u32 = u32::MAX;
        let cap = capacity as usize;
        let mut table = vec![[EMPTY; 4]; cap];
        for (&lod, arena) in self.arenas.iter() {
            let rank = lod.rank();
            for (&pos, &slot) in arena.slot_of_chunk_pos.iter() {
                let packed = (rank << 28) | slot;
                let mut idx = (hash_chunk_pos(pos) as usize) % cap;
                while table[idx][3] != EMPTY {
                    idx = (idx + 1) % cap;
                }
                table[idx] = [pos.x as u32, pos.y as u32, pos.z as u32, packed];
            }
        }
        render_queue.write_buffer(buffer, 0, bytemuck::cast_slice(&table));
    }
}
unsafe impl Send for VoxelChunkArena {}
unsafe impl Sync for VoxelChunkArena {}
