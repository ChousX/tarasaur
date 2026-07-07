// fields/consts.rs
pub const CHUNK_SIZE: u32 = 32;
pub const CHUNK_VOLUME: usize = (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) as usize; // 32,768
pub const VISIBILITY_WORDS: usize = CHUNK_VOLUME.div_ceil(64); // 512

#[inline]
pub const fn flatten(x: u32, y: u32, z: u32) -> u32 {
    z * CHUNK_SIZE * CHUNK_SIZE + y * CHUNK_SIZE + x
}

#[inline]
pub const fn unflatten(flat: u32) -> (u32, u32, u32) {
    let x = flat % CHUNK_SIZE;
    let y = (flat / CHUNK_SIZE) % CHUNK_SIZE;
    let z = flat / (CHUNK_SIZE * CHUNK_SIZE);
    (x, y, z)
}
