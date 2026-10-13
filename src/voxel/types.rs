use bevy::{
    prelude::*,
    render::{
        render_resource::{BindGroupEntry, Buffer, BufferDescriptor, BufferUsages},
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

#[must_use]
pub fn entries<'a>(bufs: &[(u32, &'a Buffer)]) -> Vec<BindGroupEntry<'a>> {
    bufs.iter()
        .map(|&(binding, buf)| BindGroupEntry {
            binding,
            resource: buf.as_entire_binding(),
        })
        .collect()
}
