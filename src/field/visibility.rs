// fields/visibility.rs
use super::{Field, consts::*};
use bevy::prelude::*;

#[derive(Component, Clone)]
pub struct VisibilityField {
    words: Box<[u64; VISIBILITY_WORDS]>,
}

impl Default for VisibilityField {
    fn default() -> Self {
        Self {
            words: Box::new([u64::MAX; VISIBILITY_WORDS]),
        } // default visible
    }
}

impl Field<bool> for VisibilityField {
    fn size(&self) -> UVec3 {
        UVec3::splat(CHUNK_SIZE)
    }

    fn get(&self, x: u32, y: u32, z: u32) -> bool {
        let bit = flatten(x, y, z) as usize;
        (self.words[bit / 64] >> (bit % 64)) & 1 != 0
    }

    fn set(&mut self, x: u32, y: u32, z: u32, value: bool) {
        let bit = flatten(x, y, z) as usize;
        let (word, shift) = (bit / 64, bit % 64);
        if value {
            self.words[word] |= 1 << shift;
        } else {
            self.words[word] &= !(1 << shift);
        }
    }
}
