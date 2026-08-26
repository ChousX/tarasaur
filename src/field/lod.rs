use crate::CHUNK_SIZE;
use bevy::prelude::*;

#[allow(clippy::upper_case_acronyms)]
#[derive(Component, Default, Clone, Copy, Debug, PartialEq, Eq)]
pub enum LOD {
    Lowest = 4,
    Low = 16,
    #[default]
    Medium = 32,
    High = 64,
}

impl LOD {
    /// Returns the CHUNK_SIZE for this specific level of detail
    #[inline]
    pub fn size(self) -> u32 {
        self as u32
    }

    /// Returns the total number of voxels (Volume) for this LOD
    #[inline]
    pub fn volume(self) -> usize {
        let s = self.size() as usize;
        s * s * s
    }

    /// Dynamically calculates voxel spatial size based on CHUNK_SIZE.x and grid resolution
    /// Example: LOD::Medium (32) -> 10.0 / 32.0 = 0.3125 world units per voxel
    #[inline]
    pub fn voxel_size(self) -> f32 {
        CHUNK_SIZE / self.size() as f32
    }
}
