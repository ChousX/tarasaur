// fields/visibility.rs
use super::{Field, consts::*};
use bevy::prelude::*;

#[derive(Component, Clone)]
pub struct VisibilityField {
    pub words: Box<[u64; MAX_VISIBILITY_WORDS]>,
}

impl VisibilityField {
    #[inline]
    pub fn all_false(&self) -> bool {
        todo!()
    }
    #[inline]
    pub fn all_true(&self) -> bool {
        todo!()
    }
}

impl Default for VisibilityField {
    fn default() -> Self {
        Self {
            words: Box::new([0; MAX_VISIBILITY_WORDS]),
        }
    }
}

impl Field<bool> for VisibilityField {
    fn size(&self) -> UVec3 {
        // Keeps trait matching system-wide MAX footprint
        UVec3::splat(MAX_SIZE)
    }

    fn get(&self, x: u32, y: u32, z: u32) -> bool {
        let bit = flatten_with_size(x, y, z, MAX_SIZE);
        let word_idx = (bit / 64) as usize;
        let shift = bit % 64;
        ((self.words[word_idx] >> shift) & 1) == 1
    }

    fn set(&mut self, x: u32, y: u32, z: u32, value: bool) {
        let bit = flatten_with_size(x, y, z, MAX_SIZE);
        let word_idx = (bit / 64) as usize;
        let shift = bit % 64;

        if value {
            self.words[word_idx] |= 1 << shift;
        } else {
            self.words[word_idx] &= !(1 << shift);
        }
    }
}
