use bevy::{
    prelude::*,
    render::{
        render_resource::{Buffer, BufferUsages},
        renderer::RenderDevice,
    },
};
use bytemuck::{Pod, Zeroable};
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct DrawIndexedIndirectArgs {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub first_instance: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct Pass1Uniforms {
    pub cell_count: u32,
    pub texture_size: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BatchCompactionUniforms {
    chunk_size: u32,
    total_cells: u32,
    blocks_per_chunk: u32,
    _pad0: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct Pass3Uniforms {
    pub cell_count: u32,
    pub texture_size: u32,
    pub voxel_size: f32,
    pub _pad0: [u32; 1], // pad chunk_world_origin to 16-byte offset
    pub chunk_world_origin: [f32; 3],
    pub _pad1: u32, // pad struct to 32 bytes (multiple of 16)
}

pub struct CollisionMeshData {
    pub chunk_pos: IVec3, // or whatever crate::chunk::ChunkPosition wraps
    pub generation: u64,
    pub vertices: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct RayQuery {
    pub origin: [f32; 3],
    pub max_distance: f32,
    pub direction: [f32; 3],
    pub user_id: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct HitResult {
    pub hit_pos_world: [f32; 3],
    pub hit_distance: f32,
    pub hit_normal: [f32; 3],
    pub did_hit: u32,
    pub voxel_coord: [u32; 3],
    pub chunk_slot: u32,
}
use bevy::render::render_resource::*;

pub trait CreateStorage {
    fn storage(&self, label: &'static str, size: u64, usage: BufferUsages) -> Buffer;
}

impl CreateStorage for RenderDevice {
    fn storage(&self, label: &'static str, size: u64, usage: BufferUsages) -> Buffer {
        self.create_buffer(&BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    }
}
