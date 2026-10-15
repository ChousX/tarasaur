//! Per-voxel material ids for a user-defined `VoxelMaterial` enum. Blended
//! between voxels at mesh and fragment time, never in the stored data.

use crate::{
    field::{
        GpuChannel, VoxelField,
        ops::{AccumulateExt, BlendExt},
    },
    grid::flatten_with_size,
};

use super::{Field, LOD};
use bevy::prelude::*;
use std::marker::PhantomData;

/// One `u8` material id per voxel; the id is the palette layer.
#[derive(Component, Clone)]
pub struct MaterialField<M: VoxelMaterial> {
    lod: LOD,
    data: Box<[u8]>, // raw material ids, GPU-ready
    version: u64,
    _marker: PhantomData<M>,
}

impl<M: VoxelMaterial> Default for MaterialField<M> {
    fn default() -> Self {
        Self::new(LOD::default())
    }
}

impl<M: VoxelMaterial> MaterialField<M> {
    #[must_use]
    pub fn new(lod: LOD) -> Self {
        Self::from_raw(lod, vec![M::default().to_id(); lod.volume()])
    }
}

impl<M: VoxelMaterial> Field<M> for MaterialField<M> {
    fn size(&self) -> UVec3 {
        UVec3::splat(self.lod.size())
    }

    fn get(&self, x: u32, y: u32, z: u32) -> M {
        self.data
            .get(flatten_with_size(x, y, z, self.size()))
            .map_or_else(M::default, |&id| M::from_id(id))
    }

    fn set(&mut self, x: u32, y: u32, z: u32, value: M) {
        let idx = flatten_with_size(x, y, z, self.size());
        let id = value.to_id();
        if let Some(slot) = self.data.get_mut(idx)
            && *slot != id
        {
            *slot = id;
            self.version = self.version.wrapping_add(1);
        }
    }
}
/// A user-defined enum of voxel material variants.
///
/// Implement this on your own enum to use it with `MaterialField<Self>` and
/// `WorldEditor`. `AccumulateExt`/`BlendExt` are required supertraits, not a
/// blanket impl the engine provides — that's deliberate. A blanket impl
/// would mean every material set gets the same edit-brush semantics with no
/// way to override them (Rust's coherence rules won't let you specialize a
/// blanket impl for your own type). Writing these two impls yourself is a
/// few lines, and it's the hook you'll actually want — e.g. weighted-random
/// selection instead of a hard threshold, or majority-vote blending.
pub trait VoxelMaterial:
    Copy + Eq + Default + Send + Sync + AccumulateExt + BlendExt + 'static
{
    /// Number of distinct variants. Must fit in a u8 (<= 255) since that's
    /// the palette layer index every voxel stores.
    const COUNT: u8;

    fn to_id(self) -> u8;
    fn from_id(id: u8) -> Self;
}
impl<M: VoxelMaterial> VoxelField for MaterialField<M> {
    type Value = M;
    type Elem = u8;
    const GPU: GpuChannel = GpuChannel::Material;

    fn new_for_lod(lod: LOD) -> Self {
        Self::new(lod)
    }

    fn from_raw(lod: LOD, data: Vec<u8>) -> Self {
        debug_assert_eq!(data.len(), lod.volume());
        Self {
            lod,
            data: data.into_boxed_slice(),
            _marker: PhantomData,
            version: 0,
        }
    }

    fn lod(&self) -> LOD {
        self.lod
    }

    fn data_slice(&self) -> &[u8] {
        &self.data
    }

    fn version(&self) -> u64 {
        self.version
    }
}
