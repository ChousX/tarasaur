//! Per-voxel material ids for a user-defined `VoxelMaterial` enum, each
//! with the distance to the nearest material boundary. Blended between
//! voxels at mesh and fragment time, never in the stored data.
//!
//! # Why a distance rides along with the id
//!
//! An id alone quantises a material boundary onto the voxel grid, and the
//! error that costs grows with the voxel: measured against the generator
//! on the bench terrain, 0.78% of surface points carry the wrong material
//! at High and 17.1% at Lowest, and sampling the id better recovers
//! essentially none of it (a majority filter per voxel: 0.00 percentage
//! points above Lowest). The boundary's *position* is simply not in the
//! id field once it has been sampled.
//!
//! A distance to that boundary is a smooth quantity, so it survives point
//! sampling and interpolates. Keeping one byte of it per voxel lets the
//! mesher and the raster put the boundary back sub-voxel instead of on the
//! grid. The distance is stored unsigned: which side a voxel is on is
//! already its id, and with several materials about there is no single
//! sign to agree on. A blend signs it against the pair it is blending
//! (`+d` for one id, `-d` for the other) and interpolates that.

use crate::{
    field::{
        GpuChannel, VoxelField,
        ops::{AccumulateExt, BlendExt},
    },
    grid::flatten_with_size,
    num::u32_from_f32,
};

use super::{Field, LOD};
use bevy::prelude::*;
use std::marker::PhantomData;

/// Material boundary distances saturate here, in **world** units.
///
/// One `LOD::Lowest` voxel. World units rather than voxels of the chunk's
/// own LOD for three reasons: a coarser LOD's stride downsample then keeps
/// the value unchanged, an apron texel read across an LOD seam already
/// means the same thing (the SDF needs `apron_unit_shift` for exactly this
/// and gets it wrong while a seam rank is stale), and the raster's
/// fragment stage has no seam ranks at all, so it could not rescale even
/// if it wanted to. One byte over this band is 0.0098 world units, well
/// inside what a blend needs.
pub const MATERIAL_BAND: f32 = 2.5;

/// Whether an arena at `lod` carries material boundary distances.
///
/// They buy the least where the voxels are already small. Measured
/// against the generator, the material error without them is 0.78% of
/// surface points at `High` but 3.0% at `Low` and 17.1% at `Lowest`, and
/// `High` is also the arena with the most vertices and the most screen
/// coverage, so it pays the most for them and gains the least. Dropping
/// them there returns that arena to exactly its old layout and cost.
#[must_use]
pub const fn keeps_material_distance(lod: LOD) -> bool {
    !matches!(lod, LOD::High)
}

/// Bytes an arena stores per material voxel: the id, and the boundary
/// distance where [`keeps_material_distance`] holds.
#[must_use]
pub const fn material_elem_bytes(lod: LOD) -> u32 {
    if keeps_material_distance(lod) { 2 } else { 1 }
}

/// One voxel: the palette layer in the low byte, and in the high byte the
/// distance to the nearest material boundary (see [`quantize_material_distance`]).
pub type MaterialVoxel = u16;

/// The palette layer of a stored voxel.
#[must_use]
pub const fn material_id(voxel: MaterialVoxel) -> u8 {
    voxel.to_le_bytes()[0]
}

/// The stored distance byte of a voxel.
#[must_use]
pub const fn material_distance_byte(voxel: MaterialVoxel) -> u8 {
    voxel.to_le_bytes()[1]
}

/// A voxel from an id and a distance byte.
#[must_use]
pub const fn pack_material(id: u8, distance: u8) -> MaterialVoxel {
    MaterialVoxel::from_le_bytes([id, distance])
}

/// A distance to the nearest material boundary, in world units, as the
/// byte stored with the id. Saturates at [`MATERIAL_BAND`].
#[must_use]
pub fn quantize_material_distance(world: f32) -> u8 {
    let t = (world / MATERIAL_BAND).clamp(0.0, 1.0);
    // `as` is denied in this crate; the product is in 0..=255 by the clamp.
    u8::try_from(u32_from_f32((t * 255.0).round())).unwrap_or(u8::MAX)
}

/// The inverse of [`quantize_material_distance`], in world units.
#[must_use]
pub fn material_distance(byte: u8) -> f32 {
    f32::from(byte) / 255.0 * MATERIAL_BAND
}

/// One `u16` per voxel: the palette layer, and the distance to the nearest
/// material boundary (see the module docs).
#[derive(Component, Clone)]
pub struct MaterialField<M: VoxelMaterial> {
    lod: LOD,
    data: Box<[MaterialVoxel]>, // GPU-ready: id byte, then distance byte
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
        // No boundary anywhere near: a uniform field blends nothing.
        let voxel = pack_material(M::default().to_id(), u8::MAX);
        Self::from_raw(lod, vec![voxel; lod.volume()])
    }
}

impl<M: VoxelMaterial> Field<M> for MaterialField<M> {
    fn size(&self) -> UVec3 {
        UVec3::splat(self.lod.size())
    }

    fn get(&self, x: u32, y: u32, z: u32) -> M {
        self.data
            .get(flatten_with_size(x, y, z, self.size()))
            .map_or_else(M::default, |&v| M::from_id(material_id(v)))
    }

    /// An edit knows the id it is painting but not where the boundary it
    /// just moved now lies, so the voxel is marked as far from one as the
    /// byte goes. The blend then falls back to the grid there, which is
    /// what it did before the distance existed, rather than steering the
    /// boundary with a stale number.
    fn set(&mut self, x: u32, y: u32, z: u32, value: M) {
        let idx = flatten_with_size(x, y, z, self.size());
        let voxel = pack_material(value.to_id(), u8::MAX);
        if let Some(slot) = self.data.get_mut(idx)
            && *slot != voxel
        {
            *slot = voxel;
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
    type Elem = MaterialVoxel;
    const GPU: GpuChannel = GpuChannel::Material;

    /// Planar within a chunk: every id, then every distance. Interleaved,
    /// one word would hold two voxels; planar it holds four ids, which is
    /// the density the ids had before they carried a distance at all, and
    /// the distances get their own four-per-word runs. Almost every cell
    /// is one material and reads only ids, so that is the fetch to keep
    /// dense. An arena that carries no distances uploads the first half
    /// alone (see [`keeps_material_distance`]).
    fn gpu_bytes(vol: &[Self::Elem]) -> Vec<u8> {
        let mut out = Vec::with_capacity(vol.len().saturating_mul(2));
        out.extend(vol.iter().copied().map(material_id));
        out.extend(vol.iter().copied().map(material_distance_byte));
        out
    }

    /// An arena that carries no distances is sent the ids alone, so the
    /// distance half is never built or transferred for it either.
    fn gpu_bytes_at(vol: &[Self::Elem], lod: LOD) -> Vec<u8> {
        if keeps_material_distance(lod) {
            return Self::gpu_bytes(vol);
        }
        vol.iter().copied().map(material_id).collect()
    }


    fn new_for_lod(lod: LOD) -> Self {
        Self::new(lod)
    }

    fn from_raw(lod: LOD, data: Vec<MaterialVoxel>) -> Self {
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

    fn data_slice(&self) -> &[MaterialVoxel] {
        &self.data
    }

    fn version(&self) -> u64 {
        self.version
    }
}
