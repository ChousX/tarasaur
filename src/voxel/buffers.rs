use bevy::{
    prelude::*,
    render::{render_resource::*, sync_world::MainEntity},
};

#[derive(Component)]
pub struct GpuVoxelChunkBuffers {
    pub chunk_coord: IVec3,
    pub lod: u32,
    pub sdf_texture: Texture,
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
}

#[derive(Component)]
pub struct ExtractedChunkSdf {
    pub main_entity: MainEntity,
    pub chunk_pos: IVec3,
    pub padded_sdf_data: Vec<u8>,
    pub size: u32,
}
