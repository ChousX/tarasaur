use crate::LOD;

// fields/visibility.rs
use super::{Field, consts::*};
use bevy::prelude::*;

#[derive(Component, Clone)]
pub struct VisibilityField {
    pub lod: LOD,
    pub words: Box<[u64; MAX_VISIBILITY_WORDS]>,
}

impl VisibilityField {
    pub fn new(lod: LOD) -> Self {
        Self {
            lod,
            words: Box::new([0; MAX_VISIBILITY_WORDS]),
        }
    }

    /// Total number of active voxels/bits for the current LOD size
    #[inline]
    fn total_bits(&self) -> usize {
        let size = self.size();
        (size.x * size.y * size.z) as usize
    }

    #[inline]
    pub fn all_false(&self) -> bool {
        let total_bits = self.total_bits();
        let full_words = total_bits / 64;
        let remaining_bits = total_bits % 64;

        // 1. Check all fully-utilized words are completely 0
        if !self.words[..full_words].iter().all(|&w| w == 0) {
            return false;
        }

        // 2. Check the active portion of the final partial word (if it exists)
        if remaining_bits > 0 {
            let mask = (1 << remaining_bits) - 1;
            if (self.words[full_words] & mask) != 0 {
                return false;
            }
        }

        true
    }

    #[inline]
    pub fn all_true(&self) -> bool {
        let total_bits = self.total_bits();
        let full_words = total_bits / 64;
        let remaining_bits = total_bits % 64;

        // 1. Check all fully-utilized words are completely filled with 1s
        if !self.words[..full_words].iter().all(|&w| w == u64::MAX) {
            return false;
        }

        // 2. Check the active portion of the final partial word (if it exists)
        if remaining_bits > 0 {
            let mask = (1 << remaining_bits) - 1;
            // The active bits must match the mask exactly
            if (self.words[full_words] & mask) != mask {
                return false;
            }
        }

        true
    }
}

impl Default for VisibilityField {
    fn default() -> Self {
        Self {
            lod: LOD::default(),
            words: Box::new([0; MAX_VISIBILITY_WORDS]),
        }
    }
}

impl Field<bool> for VisibilityField {
    fn size(&self) -> UVec3 {
        UVec3::splat(self.lod as u32)
    }

    fn get(&self, x: u32, y: u32, z: u32) -> bool {
        let size = self.size();
        let bit = flatten_with_size(x, y, z, size);
        let word_idx = (bit / 64) as usize;
        let shift = bit % 64;
        ((self.words[word_idx] >> shift) & 1) == 1
    }

    fn set(&mut self, x: u32, y: u32, z: u32, value: bool) {
        let size = self.size();
        let bit = flatten_with_size(x, y, z, size);
        let word_idx = (bit / 64) as usize;
        let shift = bit % 64;

        if value {
            self.words[word_idx] |= 1 << shift;
        } else {
            self.words[word_idx] &= !(1 << shift);
        }
    }
}
