use crate::{
    ApronSample, ExtractGate, FieldLOD, LOD, Versionable, VoxelDataSlice, flatten_with_size,
};
// fields/visibility.rs
use super::Field;
use bevy::prelude::*;

#[derive(Component, Clone)]
pub struct VisibilityField {
    pub lod: LOD,
    words: Box<[u64]>,
    byte_mirror: Box<[u8]>, // authoritative, GPU-upload-sized representation
    version: u64,
}

impl VisibilityField {
    pub fn new(lod: LOD) -> Self {
        let volume = lod.volume();
        Self {
            lod,
            words: vec![0u64; words_for_lod(lod)].into_boxed_slice(),
            byte_mirror: vec![0u8; volume].into_boxed_slice(),
            version: 0,
        }
    }

    /// Total number of active voxels/bits for the current LOD size
    #[inline]
    fn total_bits(&self) -> usize {
        self.lod.volume()
    }

    #[inline]
    pub fn is_uniform(&self) -> Option<bool> {
        let total_bits = self.total_bits();
        let full_words = total_bits / 64;
        let remaining_bits = total_bits % 64;

        let mut all_zero = true;
        let mut all_ones = true;

        for &w in &self.words[..full_words] {
            all_zero &= w == 0;
            all_ones &= w == u64::MAX;
            if !all_zero && !all_ones {
                return None;
            }
        }

        if remaining_bits > 0 {
            let mask = (1u64 << remaining_bits) - 1;
            let tail = self.words[full_words] & mask;
            all_zero &= tail == 0;
            all_ones &= tail == mask;
        }

        match (all_zero, all_ones) {
            (false, false) => None,
            (_, true) => Some(true),
            _ => Some(false),
        }
    }

    #[inline]
    pub fn all_false(&self) -> bool {
        let total_bits = self.total_bits();
        let full_words = total_bits / 64;
        let remaining_bits = total_bits % 64;
        if !self.words[..full_words].iter().all(|&w| w == 0) {
            return false;
        }
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
        if !self.words[..full_words].iter().all(|&w| w == u64::MAX) {
            return false;
        }
        if remaining_bits > 0 {
            let mask = (1 << remaining_bits) - 1;
            if (self.words[full_words] & mask) != mask {
                return false;
            }
        }
        true
    }
}

impl Default for VisibilityField {
    fn default() -> Self {
        let lod = LOD::default();
        Self::new(lod)
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
        let old = ((self.words[word_idx] >> shift) & 1) == 1;
        if old != value {
            if value {
                self.words[word_idx] |= 1 << shift;
            } else {
                self.words[word_idx] &= !(1 << shift);
            }
            self.byte_mirror[bit as usize] = value as u8;
            self.version += 1;
        }
    }
}

impl VoxelDataSlice for VisibilityField {
    type Elem = u8;
    fn data_slice(&self) -> &[u8] {
        &self.byte_mirror
    }
}

impl ApronSample for VisibilityField {
    fn sample_apron(data: &[u8], size: u32, x: f32, y: f32, z: f32) -> u8 {
        crate::voxel::systems::nearest_neighbor_sample(data, size, x, y, z)
    }
}

impl Versionable for VisibilityField {
    fn version(&self) -> u64 {
        self.version
    }
    fn incorment_version(&mut self) {
        self.version += 1;
    }
}

impl ExtractGate for VisibilityField {
    fn should_extract(&self) -> bool {
        self.is_uniform().is_none()
    }
}

/// Number of u64 words needed to store one bit per voxel at the given LOD.
fn words_for_lod(lod: LOD) -> usize {
    lod.volume().div_ceil(64)
}

impl FieldLOD for VisibilityField {
    fn lod(&self) -> LOD {
        self.lod
    }
}
