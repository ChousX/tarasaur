//! Per-voxel visibility mask. Hidden voxels still mesh; the fragment shader
//! discards them.

use super::Field;
use crate::{
    field::{ApronSample, ExtractGate, FieldFromRaw, FieldLOD, LOD, Versionable, VoxelDataSlice},
    grid::{flatten_with_size, nearest_neighbor_sample},
    num::usize_from,
};
use bevy::prelude::*;

/// Optional per-chunk mask. Absent or all-true = fully visible;
/// all-false = chunk hidden; mixed = rendered and masked per fragment.
#[derive(Component, Clone)]
pub struct VisibilityField {
    pub lod: LOD,
    words: Box<[u64]>,
    byte_mirror: Box<[u8]>, // GPU-upload representation
    set_count: usize,       // number of true voxels, maintained by `set`
    version: u64,
}

impl VisibilityField {
    pub const DEFAULT_VALUE: bool = true;

    #[must_use]
    pub fn new(lod: LOD) -> Self {
        Self::filled(lod, Self::DEFAULT_VALUE)
    }

    /// `filled(lod, true)` = fully visible mask you can carve into.
    #[must_use]
    pub fn filled(lod: LOD, value: bool) -> Self {
        Self {
            lod,
            words: vec![if value { u64::MAX } else { 0 }; words_for_lod(lod)].into_boxed_slice(),
            byte_mirror: vec![u8::from(value); lod.volume()].into_boxed_slice(),
            set_count: if value { lod.volume() } else { 0 },
            version: 0,
        }
    }

    #[inline]
    #[must_use]
    pub const fn is_all_empty(&self) -> bool {
        self.set_count == 0
    }
    #[inline]
    #[must_use]
    pub const fn is_all_full(&self) -> bool {
        self.set_count == self.lod.volume()
    }
    #[inline]
    #[must_use]
    pub const fn is_mixed(&self) -> bool {
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
        UVec3::splat(self.lod.size())
    }

    fn get(&self, x: u32, y: u32, z: u32) -> bool {
        let bit = flatten_with_size(x, y, z, self.size());
        self.words
            .get(bit / 64)
            .is_some_and(|w| ((w >> (bit % 64)) & 1) == 1)
    }

    fn set(&mut self, x: u32, y: u32, z: u32, value: bool) {
        let bit = flatten_with_size(x, y, z, self.size());
        let shift = bit % 64;
        let (Some(word), Some(byte)) =
            (self.words.get_mut(bit / 64), self.byte_mirror.get_mut(bit))
        else {
            return;
        };
        let old = ((*word >> shift) & 1) == 1;
        if old != value {
            if value {
                *word |= 1 << shift;
                self.set_count = self.set_count.saturating_add(1);
            } else {
                *word &= !(1 << shift);
                self.set_count = self.set_count.saturating_sub(1);
            }
            *byte = u8::from(value);
            self.version = self.version.wrapping_add(1);
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
            if b != 0
                && let Some(word) = words.get_mut(i / 64)
            {
                *word |= 1u64 << (i % 64);
            }
        }
        let set_count = words.iter().map(|w| usize_from(w.count_ones())).sum();
        let mirror: Vec<u8> = bytes.into_iter().map(|b| u8::from(b != 0)).collect();
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
        nearest_neighbor_sample(data, size, x, y, z)
    }
}

impl Versionable for VisibilityField {
    fn version(&self) -> u64 {
        self.version
    }
}

impl ExtractGate for VisibilityField {
    fn should_extract(&self) -> bool {
        !self.is_all_full() // mixed OR all-empty; all-full needs no mask
    }
}

const fn words_for_lod(lod: LOD) -> usize {
    lod.volume().div_ceil(64)
}

impl FieldLOD for VisibilityField {
    fn lod(&self) -> LOD {
        self.lod
    }
}
