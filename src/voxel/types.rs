#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawIndexedIndirectArgs {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub first_instance: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CompactionUniforms {
    pub chunk_size: u32,
    pub total_cells: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Pass3Uniforms {
    pub chunk_size: u32,
    pub voxel_size: f32,
    pub _pad0: [u32; 2], // pad chunk_world_origin to 16-byte offset
    pub chunk_world_origin: [f32; 3],
    pub _pad1: u32, // pad struct to 32 bytes (multiple of 16)
}
