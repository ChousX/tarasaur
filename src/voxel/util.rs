//! Small GPU helpers shared by the arenas and the meshing dispatch.

use bevy::{
    prelude::*,
    render::{
        render_resource::{
            BindGroup, Buffer, BufferInitDescriptor, BufferUsages, CommandEncoder,
            ComputePassDescriptor, ComputePipeline,
        },
        renderer::RenderDevice,
    },
};

/// Must match `hash_chunk` in `cursor_query.wgsl` exactly.
#[must_use]
pub const fn hash_chunk_pos(pos: IVec3) -> u32 {
    // Bit reinterpretation, as WGSL's `bitcast<u32>` does.
    let x = pos.x.cast_unsigned();
    let y = pos.y.cast_unsigned();
    let z = pos.z.cast_unsigned();
    x.wrapping_mul(0x9E37_79B1) ^ y.wrapping_mul(0x85EB_CA77) ^ z.wrapping_mul(0xC2B2_AE3D)
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BatchUniforms {
    cell_count: u32,
    texture_size: u32,
    wg_per_chunk_z: u32,
    voxel_size: f32,
}

#[must_use]
pub fn make_batch_uniform_buffer(
    render_device: &RenderDevice,
    label: &'static str,
    cell_count: u32,
    texture_size: u32,
    wg_size: u32,
    voxel_size: f32,
) -> Buffer {
    render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::bytes_of(&BatchUniforms {
            cell_count,
            texture_size,
            wg_per_chunk_z: cell_count.div_ceil(wg_size),
            voxel_size,
        }),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    })
}

pub fn dispatch(
    encoder: &mut CommandEncoder,
    label: &'static str,
    pipeline: &ComputePipeline,
    bind_group: &BindGroup,
    (x, y, z): (u32, u32, u32),
) {
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(x, y, z);
}
