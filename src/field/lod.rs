use bevy::prelude::*;

#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lod {
    Lowest = 4,
    Low = 16,
    Medium = 32,
    High = 64,
}

impl Lod {
    /// Returns the CHUNK_SIZE for this specific level of detail
    pub fn size(self) -> u32 {
        self as u32
    }

    /// Returns the total number of voxels (Volume) for this LOD
    pub fn volume(self) -> usize {
        let s = self.size() as usize;
        s * s * s
    }
}
