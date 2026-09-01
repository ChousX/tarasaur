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
pub struct Pass1Uniforms {
    pub cell_count: u32,
    pub texture_size: u32,
    pub _pad0: u32,
    pub _pad1: u32,
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
    pub cell_count: u32,
    pub texture_size: u32,
    pub voxel_size: f32,
    pub _pad0: [u32; 1], // pad chunk_world_origin to 16-byte offset
    pub chunk_world_origin: [f32; 3],
    pub _pad1: u32, // pad struct to 32 bytes (multiple of 16)
}

#[derive(bevy::ecs::component::Component)]
pub struct PendingMeshReadback(u64);

impl PendingMeshReadback {
    pub fn new(start_at: u64) -> Self {
        Self(start_at)
    }

    pub fn incroment(&mut self) {
        self.0 += 1;
    }

    pub fn get_val(&self) -> u64 {
        self.0
    }
}

use bevy::prelude::*;
use crossbeam_channel::{Receiver, Sender};

pub struct CollisionMeshData {
    pub chunk_pos: IVec3, // or whatever crate::chunk::ChunkPosition wraps
    pub generation: u64,
    pub vertices: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
}

// Lives in the render world — systems.rs reads this to send results.
#[derive(Resource, Clone)]
pub struct MeshReadbackChannel {
    pub sender: Sender<CollisionMeshData>,
}

// Lives in the main world — a Bevy system drains this every Update.
#[derive(Resource)]
pub struct MeshReadbackChannelReceiver {
    pub receiver: Receiver<CollisionMeshData>,
}
