use bevy::{
    prelude::*,
    render::{
        render_resource::{Buffer, BufferInitDescriptor, BufferUsages},
        renderer::RenderDevice,
    },
};
// voxel/util.rs
pub fn next_pow2(x: u32) -> u32 {
    if x <= 1 {
        1
    } else {
        1u32 << (32 - (x - 1).leading_zeros())
    }
}

/// Must match `hash_chunk` in cursor_query.wgsl exactly.
pub fn hash_chunk_pos(pos: IVec3) -> u32 {
    let x = pos.x as u32;
    let y = pos.y as u32;
    let z = pos.z as u32;
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

fn pack_bits(bits: impl ExactSizeIterator<Item = bool>) -> Vec<u32> {
    let mut packed = vec![0u32; bits.len().div_ceil(32)];
    for (i, b) in bits.enumerate() {
        if b {
            packed[i / 32] |= 1 << (i % 32);
        }
    }
    packed
}
