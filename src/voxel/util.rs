use bevy::prelude::*;
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
