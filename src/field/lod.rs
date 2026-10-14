//! Levels of detail. A chunk is always `CHUNK_SIZE` wide; LOD sets how
//! many voxels span it.

use crate::CHUNK_SIZE;
use crate::num::{f32_from_u32, usize_from};
use bevy::prelude::*;

/// Voxels per chunk edge (the discriminant). Ordered coarse < fine.
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
    PartialOrd,
    Ord,
)]
pub enum LOD {
    Lowest = 4,
    Low = 16,
    #[default]
    Medium = 32,
    High = 64,
}

/// What the loader wants this chunk to be. `LOD` (below) is what its fields
/// currently hold; the load pipeline closes the gap between the two.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetLOD(pub LOD);

/// The only LOD at which fields may be edited or saved.
#[derive(Resource, Clone, Copy, Debug)]
pub struct MaxEditLod(pub LOD);
impl Default for MaxEditLod {
    fn default() -> Self {
        Self(LOD::High)
    }
}

impl LOD {
    pub const COUNT: usize = 4;
    #[inline]
    #[must_use]
    pub const fn size(self) -> u32 {
        match self {
            Self::Lowest => 4,
            Self::Low => 16,
            Self::Medium => 32,
            Self::High => 64,
        }
    }
    #[inline]
    #[must_use]
    pub const fn rank(self) -> u32 {
        match self {
            Self::Lowest => 0,
            Self::Low => 1,
            Self::Medium => 2,
            Self::High => 3,
        }
    }
    #[inline]
    #[must_use]
    pub const fn volume(self) -> usize {
        usize_from(self.size()).saturating_pow(3)
    }
    #[inline]
    #[must_use]
    pub fn voxel_size(self) -> f32 {
        CHUNK_SIZE / f32_from_u32(self.size())
    }
}
