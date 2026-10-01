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
    #[inline]
    pub fn volume(self) -> usize {
        let s = self.size() as usize;
        s * s * s
    }
    #[inline]
    pub fn voxel_size(self) -> f32 {
        CHUNK_SIZE / self.size() as f32
    }
    pub const fn get_number_of_lods() -> usize {
        Self::COUNT
    }
}
