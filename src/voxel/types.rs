//! GPU-shared structs (layouts must match the WGSL) and buffer helpers.

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
