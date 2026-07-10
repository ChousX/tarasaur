use bevy::math::UVec3;

pub const MAX_SIZE: u32 = 64;
pub const MAX_VOLUME: usize = (MAX_SIZE * MAX_SIZE * MAX_SIZE) as usize; // 262,144

// 262,144 / 64 = 4,096 total u64 words needed
pub const MAX_VISIBILITY_WORDS: usize = MAX_VOLUME / 64;

#[inline]
pub fn flatten_with_size(x: u32, y: u32, z: u32, size: UVec3) -> u32 {
    // Index = z * (width * height) + y * width + x
    z * (size.x * size.y) + y * size.x + x
}
