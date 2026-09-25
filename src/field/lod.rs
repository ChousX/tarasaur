use crate::CHUNK_SIZE;
use bevy::prelude::*;

#[allow(clippy::upper_case_acronyms)]
#[derive(
    serde::Serialize,
    serde::Deserialize,
    Component,
    Default,
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Hash,
)]
pub enum LOD {
    Lowest = 4,
    Low = 16,
    #[default]
    Medium = 32,
    High = 64,
}

impl LOD {
    pub const COUNT: usize = 4;
    /// Returns the CHUNK_SIZE for this specific level of detail
    #[inline]
    pub fn size(self) -> u32 {
        self as u32
    }

    #[inline]
    pub fn rank(self) -> u32 {
        match self {
            LOD::Lowest => 0,
            LOD::Low => 1,
            LOD::Medium => 2,
            LOD::High => 3,
        }
    }

    #[inline]
    pub fn from_rank(rank: u32) -> Option<Self> {
        match rank {
            0 => Some(LOD::Lowest),
            1 => Some(LOD::Low),
            2 => Some(LOD::Medium),
            3 => Some(LOD::High),
            _ => None,
        }
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
