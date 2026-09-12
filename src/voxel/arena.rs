use std::collections::HashMap;

use bevy::{
    prelude::*,
    render::{render_resource::*, renderer::RenderDevice, sync_world::MainEntity},
};
use bytemuck::{Pod, Zeroable};

use crate::voxel::types::DrawIndexedIndirectArgs;

use std::sync::OnceLock;

pub static MAX_CHUNKS: OnceLock<u32> = OnceLock::new();

/// Largest chunk count that keeps every per-chunk-strided buffer (vertex
/// buffers dominate, at 32 bytes/cell) under the device's actual
/// max_storage_buffer_binding_size, with a 10% safety margin.
/// Computed once from the real device limit and cached for the app's lifetime.
fn max_chunks(
    render_device: &RenderDevice,
    total_cells: u32,
    sdf_elems_per_chunk: u32,
    budget_cells_per_chunk: u32,
) -> u32 {
    *MAX_CHUNKS.get_or_init(|| {
        let limit = (render_device.limits().max_storage_buffer_binding_size as u64 * 9) / 10;

        let sdf_bytes     = sdf_elems_per_chunk as u64 * 4;
        let flags_bytes   = total_cells as u64 * 4;
        let offsets_bytes = total_cells as u64 * 4;
        // vertex/index strides now sized to the active-cell budget, not total_cells.
        let vertex_bytes  = budget_cells_per_chunk as u64 * 32;
        let index_bytes   = budget_cells_per_chunk as u64 * 18 * 4;

        let worst_bytes_per_chunk = [sdf_bytes, flags_bytes, offsets_bytes, vertex_bytes, index_bytes]
            .into_iter().max().unwrap();

        let capped = (limit / worst_bytes_per_chunk).max(1) as u32;
        info!(
            "[VoxelChunkArena] limit={} bytes, worst-case/chunk={} bytes, capping to {} chunks (budget={} cells/chunk, {:.0}% of {})",
            limit, worst_bytes_per_chunk, capped, budget_cells_per_chunk,
            ACTIVE_FRACTION_ESTIMATE * 100.0, total_cells
        );
        capped
    })
}

const ACTIVE_FRACTION_ESTIMATE: f32 = 0.20; // tune from telemetry

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct ChunkMeta {
    pub chunk_world_origin: [f32; 3],
    pub voxel_size: f32,
    pub sdf_offset: u32,
    pub cell_offset: u32,
    pub active_list_pos: u32, // was `vertex_offset` — now dead, replaced by dynamic bases
    pub _pad: u32,
}

#[derive(Resource)]
pub struct VoxelChunkArena {
    pub max_chunks: u32,
    pub cell_count: u32,   // chunk_voxels + 1, fixed for this LOD bucket
    pub texture_size: u32, // padded SDF size, fixed for this LOD bucket
    pub total_cells: u32,  // cell_count^3
    pub blocks_per_chunk: u32,
    pub sdf_elems_per_chunk: u32, // texture_size^3

    pub sdf_buffer: Buffer,
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
    pub batch_uniform_buffer_pass3: Buffer, // NEW: pass3 uses workgroup_size(8,8,8), needs its own wg_per_chunk_z stride

    pub readback_vertex_buffer: Buffer,
    pub readback_index_buffer: Buffer,
    pub readback_indirect_buffer: Buffer,

    pub pass1_bind_group: BindGroup,
    pub pass3_bind_group: BindGroup,
    pub compaction_bind_group: BindGroup,

    free_slots: Vec<u32>,
    slot_of_main_entity: HashMap<MainEntity, u32>,
    pub active_slots: Vec<u32>, // stable order used for this frame's batched dispatch
    pub dirty_slots: Vec<u32>,  // slots whose ChunkMeta/SDF changed since last upload
    pub chunk_active_counts_buffer: Buffer, // [active_list_pos] -> active cell count, GPU-written
    pub chunk_vertex_base_buffer: Buffer, // [active_list_pos] -> exclusive prefix sum
    pub chunk_index_base_buffer: Buffer, // [active_list_pos] -> vertex_base * 18
    pub active_slot_map_buffer: Buffer, // [active_list_pos] -> real arena slot, CPU-written per frame
    pub overflow_flag_buffer: Buffer,   // single atomic<u32>, cleared each frame
    pub overflow_readback_buffer: Buffer, // CPU-mapped copy for telemetry/next-frame drop
    pub chunk_bases_uniform_buffer: Buffer, // active_count + budget constants
    pub chunk_bases_bind_group: BindGroup,
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
        let sdf_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("arena_sdf_buffer"),
            size: sdf_elems_per_chunk as u64 * 4 * max,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let flags_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("arena_flags_buffer"),
            size: total_cells as u64 * 4 * max,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let compacted_offsets_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("arena_compacted_offsets_buffer"),
            size: total_cells as u64 * 4 * max,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let scattered_vertex_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("arena_scattered_vertex_buffer"),
            size: total_cells as u64 * 32 * max,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let final_vertex_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("arena_final_vertex_buffer"),
            size: budget_cells_per_chunk as u64 * 32 * max,
            usage: BufferUsages::STORAGE | BufferUsages::VERTEX | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let index_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("arena_index_buffer"),
            size: budget_cells_per_chunk as u64 * 18 * 4 * max,
            usage: BufferUsages::STORAGE | BufferUsages::INDEX | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

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
        let block_sums_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("arena_block_sums_buffer"),
            size: (blocks_per_chunk as u64 * max) * 4,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let chunk_meta_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("arena_chunk_meta_buffer"),
            size: std::mem::size_of::<ChunkMeta>() as u64 * max,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

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

        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct BatchUniforms {
            cell_count: u32,
            texture_size: u32,
            wg_per_chunk_z: u32,
            _pad0: u32,
        }

        // Pass 1 uses @workgroup_size(4, 4, 4) -> its own z-stride.
        let wg_per_chunk_z_pass1 = cell_count.div_ceil(4);
        let batch_uniform_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("arena_batch_uniform_buffer"),
            contents: bytemuck::bytes_of(&BatchUniforms {
                cell_count,
                texture_size,
                wg_per_chunk_z: wg_per_chunk_z_pass1,
                _pad0: 0,
            }),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        // Pass 3 uses @workgroup_size(8, 8, 8) -> a different z-stride.
        // Must NOT share batch_uniform_buffer with pass 1, or chunk_idx
        // resolution in surface_nets_pass3.wgsl divides by the wrong stride
        // and every chunk after the first aliases chunk 0's data.
        let wg_per_chunk_z_pass3 = cell_count.div_ceil(8);
        let batch_uniform_buffer_pass3 =
            render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("arena_batch_uniform_buffer_pass3"),
                contents: bytemuck::bytes_of(&BatchUniforms {
                    cell_count,
                    texture_size,
                    wg_per_chunk_z: wg_per_chunk_z_pass3,
                    _pad0: 0,
                }),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            });

        let readback_vertex_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("arena_readback_vertex_buffer"),
            size: total_cells as u64 * 32 * max,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let readback_index_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("arena_readback_index_buffer"),
            size: total_cells as u64 * 18 * 4 * max,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let readback_indirect_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("arena_readback_indirect_buffer"),
            size: 4 * max, // index_count only, per slot
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let pass1_bind_group = render_device.create_bind_group(
            Some("arena_pass1_bind_group"),
            &layouts.pass1_surface_layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: sdf_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: flags_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: compacted_offsets_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: scattered_vertex_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: final_vertex_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: indirect_args_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: batch_uniform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: chunk_meta_buffer.as_entire_binding(),
                },
            ],
        );

        let pass3_bind_group = render_device.create_bind_group(
            Some("arena_pass3_bind_group"),
            &layouts.pass3_surface_layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: sdf_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: flags_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: compacted_offsets_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: final_vertex_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: index_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: indirect_args_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    // NOTE: pass3's own uniform buffer, not batch_uniform_buffer.
                    resource: batch_uniform_buffer_pass3.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: chunk_meta_buffer.as_entire_binding(),
                },
            ],
        );

        let compaction_bind_group = render_device.create_bind_group(
            Some("arena_compaction_bind_group"),
            &layouts.compaction_bind_group_layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: compaction_uniform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: flags_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: compacted_offsets_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: block_sums_buffer.as_entire_binding(),
                },
            ],
        );
        let chunk_active_counts_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("arena_chunk_active_counts_buffer"),
            size: 4 * max,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let chunk_vertex_base_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("arena_chunk_vertex_base_buffer"),
            size: 4 * max,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let chunk_index_base_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("arena_chunk_index_base_buffer"),
            size: 4 * max,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let active_slot_map_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("arena_active_slot_map_buffer"),
            size: 4 * max,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let overflow_flag_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("arena_overflow_flag_buffer"),
            size: 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let overflow_readback_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("arena_overflow_readback_buffer"),
            size: 4,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

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

        Self {
            max_chunks: max as u32,
            cell_count,
            texture_size,
            total_cells,
            sdf_elems_per_chunk,
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
            readback_vertex_buffer,
            readback_index_buffer,
            readback_indirect_buffer,
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
        }
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
        }
    }

    pub fn active_chunk_count(&self) -> u32 {
        self.active_slots.len() as u32
    }
}
