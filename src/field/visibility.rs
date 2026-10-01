use super::Field;
use crate::{
    ApronSample, ExtractGate, FieldFromRaw, FieldLOD, FieldNew, LOD, Versionable, VoxelDataSlice,
    flatten_with_size,
};
use bevy::prelude::*;

/// Optional per-chunk mask. Absent or all-true = fully visible;
/// all-false = chunk hidden; mixed = rendered and masked per fragment.
#[derive(Component, Clone)]
pub struct VisibilityField {
    pub lod: LOD,
    words: Box<[u64]>,
    byte_mirror: Box<[u8]>, // GPU-upload representation
    set_count: u32,         // number of true voxels, maintained by `set`
    version: u64,
}

impl VisibilityField {
    pub const DEFAULT_VALUE: bool = true;

    pub fn new(lod: LOD) -> Self {
        Self::filled(lod, Self::DEFAULT_VALUE)
    }

    /// `filled(lod, true)` = fully visible mask you can carve into.
    pub fn filled(lod: LOD, value: bool) -> Self {
        Self {
            lod,
            words: vec![if value { u64::MAX } else { 0 }; words_for_lod(lod)].into_boxed_slice(),
            byte_mirror: vec![value as u8; lod.volume()].into_boxed_slice(),
            set_count: if value { lod.volume() as u32 } else { 0 },
            version: 0,
        }
    }

    #[inline]
    pub fn is_all_empty(&self) -> bool {
        self.set_count == 0
    }
    #[inline]
    pub fn is_all_full(&self) -> bool {
        self.set_count as usize == self.lod.volume()
    }
    #[inline]
    pub fn is_mixed(&self) -> bool {
        !self.is_all_empty() && !self.is_all_full()
    }
}

impl Default for VisibilityField {
    fn default() -> Self {
        Self::new(LOD::default())
    }
}

impl Field<bool> for VisibilityField {
    fn size(&self) -> UVec3 {
        UVec3::splat(self.lod as u32)
    }

    fn get(&self, x: u32, y: u32, z: u32) -> bool {
        let bit = flatten_with_size(x, y, z, self.size());
        ((self.words[(bit / 64) as usize] >> (bit % 64)) & 1) == 1
    }

    fn set(&mut self, x: u32, y: u32, z: u32, value: bool) {
        let bit = flatten_with_size(x, y, z, self.size());
        let (w, shift) = ((bit / 64) as usize, bit % 64);
        let old = ((self.words[w] >> shift) & 1) == 1;
        if old != value {
            if value {
                self.words[w] |= 1 << shift;
                self.set_count += 1;
            } else {
                self.words[w] &= !(1 << shift);
                self.set_count -= 1;
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

impl FieldFromRaw for VisibilityField {
    const OPTIONAL: bool = true;
    fn from_raw(lod: LOD, bytes: Vec<u8>) -> Self {
        let mut words = vec![0u64; words_for_lod(lod)];
        for (i, &b) in bytes.iter().enumerate() {
            if b != 0 {
                words[i / 64] |= 1u64 << (i % 64);
            }
        }
        let set_count = words.iter().map(|w| w.count_ones()).sum();
        let mirror: Vec<u8> = bytes.into_iter().map(|b| (b != 0) as u8).collect();
        Self {
            lod,
            words: words.into_boxed_slice(),
            byte_mirror: mirror.into_boxed_slice(),
            set_count,
            version: 0,
        }
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

/// Only a MIXED mask needs GPU data. All-true uploads nothing, which clears
/// the slot's has-mask flag and leaves the chunk fully visible. All-false is
/// filtered out before this is consulted.
impl ExtractGate for VisibilityField {
    fn should_extract(&self) -> bool {
        self.is_mixed()
    }
}

fn words_for_lod(lod: LOD) -> usize {
    lod.volume().div_ceil(64)
}

impl FieldLOD for VisibilityField {
    fn lod(&self) -> LOD {
        self.lod
    }
}
