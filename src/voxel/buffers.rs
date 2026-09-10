use std::marker::PhantomData;

use bevy::{
    prelude::*,
    render::{render_resource::*, sync_world::MainEntity},
};

#[derive(Component)]
pub struct GpuVoxelChunkBuffers {
    pub chunk_coord: IVec3,
    pub lod: u32,
    pub sdf_texture: Texture,
    pub chunk_voxels: u32,
    pub sdf_view: TextureView,

    pub flags_buffer: Buffer,
    pub compacted_offsets_buffer: Buffer,
    pub scattered_vertex_buffer: Buffer,
    pub final_vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub indirect_args_buffer: Buffer,
    pub compaction_uniform_buffer: Buffer,
    pub block_sums_buffer: Buffer,
    pub pass_uniform_buffer: Buffer,

    pub pass1_surface_bind_group: BindGroup,
    pub pass3_surface_bind_group: BindGroup,
    pub compaction_bind_group: BindGroup,

    pub readback_vertex_buffer: Buffer,
    pub readback_index_buffer: Buffer,
    pub readback_indirect_buffer: Buffer,

    pub mesh_generation: u64,
}

impl GpuVoxelChunkBuffers {
    /// Number of dual cells along one axis of this chunk (`chunk_voxels + 1`).
    /// Matches `Pass1Uniforms::cell_count` / `Pass3Uniforms::cell_count`.
    pub fn cell_count(&self) -> u32 {
        self.chunk_voxels + 1
    }

    /// Total number of dual cells in the chunk (`cell_count^3`).
    /// Matches `CompactionUniforms::total_cells`.
    pub fn total_cells(&self) -> u32 {
        let c = self.cell_count();
        c * c * c
    }

    /// Workgroup dispatch dimensions for a compute pass that covers the
    /// chunk's cell grid on all three axes with a cubic workgroup of size
    /// `wg_size` (e.g. 4 for pass1's `@workgroup_size(4,4,4)`, 8 for pass3's
    /// `@workgroup_size(8,8,8)`).
    pub fn dispatch_grid_3d(&self, wg_size: u32) -> UVec3 {
        let cells = self.cell_count();
        let groups = (cells + wg_size - 1) / wg_size;
        UVec3::splat(groups)
    }

    /// Workgroup count for a 1D compute pass over all cells, flattened, with
    /// workgroup size `wg_size` (e.g. 512 for the stream-compaction passes).
    /// Each workgroup in `scan_workgroup` covers `wg_size * 2` cells, matching
    /// the up-sweep/down-sweep sizing in `stream_compaction.wgsl`.
    pub fn dispatch_blocks_1d(&self, wg_size: u32) -> u32 {
        let total = self.total_cells();
        (total + wg_size - 1) / wg_size
    }
}
