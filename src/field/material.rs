use crate::ops::{AccumulateExt, BlendExt};

use super::{Field, LOD};
use bevy::prelude::*;
use std::marker::PhantomData;

#[derive(Component, Clone)]
pub struct MaterialField<M: VoxelMaterial> {
    lod: LOD,
    data: Box<[u8]>, // storage stays raw u8 — same GPU-ready layout as before
    version: u64,    // needed later for extraction dirty-checking; added now
    // while we'rMaterialFielde already rewriting this struct
    _marker: PhantomData<M>,
}

impl<M: VoxelMaterial> Default for MaterialField<M> {
    fn default() -> Self {
        Self::new(LOD::default())
    }
}

impl<M: VoxelMaterial> MaterialField<M> {
    pub fn new(lod: LOD) -> Self {
        let volume = lod.volume();
        Self {
            lod,
            data: vec![M::default().to_id(); volume].into_boxed_slice(),
            version: 0,
            _marker: PhantomData,
        }
    }
}

impl<M: VoxelMaterial> Field<M> for MaterialField<M> {
    fn size(&self) -> UVec3 {
        UVec3::splat(self.lod as u32)
    }

    fn get(&self, x: u32, y: u32, z: u32) -> M {
        let size = self.lod.size();
        let idx = (z * size * size + y * size + x) as usize;
        M::from_id(self.data[idx])
    }

    fn set(&mut self, x: u32, y: u32, z: u32, value: M) {
        let size = self.lod.size();
        let idx = (z * size * size + y * size + x) as usize;
        let id = value.to_id();
        if self.data[idx] != id {
            self.data[idx] = id;
            self.version += 1;
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
