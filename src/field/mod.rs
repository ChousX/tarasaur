//! Per-chunk voxel grids ("fields") and the traits that make them generic.
//!
//! A field is a component holding `LOD::size()`^3 values. [`VoxelField`]
//! describes how to build, version, save and upload one, and [`Field`] how
//! to edit it; `app.add_voxel_field::<F>()` (see `AppVoxelFieldExt`) wires
//! up everything else. `EditableChunks` is the gate every mutation goes
//! through.

use bevy::ecs::component::Mutable;
use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use std::marker::PhantomData;

use crate::grid::nearest_neighbor_sample;
use ops::{AccumulateExt, BlendExt};

pub(crate) mod apron;
pub mod editor;
mod lod;
mod material;
pub mod ops;
mod plugin;
pub(crate) mod sdf;
pub(crate) mod surface;
pub(crate) mod systems;
mod visibility;
pub use lod::{LOD, MaxEditLod, TargetLOD};
pub use material::{MaterialField, VoxelMaterial};
pub(crate) use plugin::register_edits;
pub use plugin::{FieldSet, FieldsPlugin};
pub use sdf::{SDF, SDF_BAND, quantize_sdf};
pub use visibility::VisibilityField;

/// Marks a chunk whose `F` was edited this cycle. For `SDF` it means the
/// values are signs, not distances, until the async JFA reinit finishes.
#[derive(Component, Clone, Copy)]
pub struct DirtyField<F, V>(PhantomData<(F, V)>)
where
    F: Field<V>,
    V: Copy + Default;

impl<F, V> Default for DirtyField<F, V>
where
    F: Field<V>,
    V: Copy + Default,
{
    fn default() -> Self {
        Self(PhantomData)
    }
}
/// Core trait representing a 3D grid of data.
pub trait Field<T: Copy + Default>: Component {
    /// Returns the dimensions of this specific field.
    fn size(&self) -> UVec3;
    /// Gets the value at the given grid coordinates.
    fn get(&self, x: u32, y: u32, z: u32) -> T;
    /// Sets the value at the given grid coordinates.
    fn set(&mut self, x: u32, y: u32, z: u32, value: T);
}

/// Which GPU buffer a field's padded volume feeds. At most one field per
/// channel: two would overwrite each other's slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuChannel {
    /// The shape the mesher reads: `gpu_bytes` must be snorm16 signed
    /// distances (see `quantize_sdf`). `SDF`.
    Shape,
    /// One `u8` palette id per voxel, blended into vertices and read per
    /// fragment. `MaterialField<M>`.
    Material,
    /// One `u8` per voxel, 0 = hidden, discarded per fragment. Empty data
    /// (no field, or `should_extract() == false`) means fully visible.
    /// `VisibilityField`.
    Mask,
    /// Edited and saved on the CPU, never uploaded.
    None,
}

/// A per-chunk voxel grid the engine can build, version, save and upload.
///
/// Register one with `app.add_voxel_field::<F>()`. Editing it also needs
/// `Field<Self::Value>`.
pub trait VoxelField: Component<Mutability = Mutable> + Sized {
    /// The value edits read and write (`Field<Self::Value>`).
    type Value: Copy + Default + Send + Sync + 'static + AccumulateExt + BlendExt;
    /// The stored element, as saved and uploaded: f32 distances for `SDF`,
    /// u8 ids for materials.
    type Elem: bytemuck::Pod + Default + Send + Sync;

    /// The GPU buffer this field feeds.
    const GPU: GpuChannel;
    /// Removed (not defaulted) when a load has no data for it.
    const OPTIONAL: bool = false;
    /// Whether `DirtyField` is cleared at the end of every edit cycle.
    /// `SDF` keeps it until its jump flood lands.
    const CLEAR_DIRTY_EACH_CYCLE: bool = true;

    /// A default field at `lod`, for new chunks and edits on absent fields.
    fn new_for_lod(lod: LOD) -> Self;
    /// A field at `lod` from saved or generated elements (version 0).
    fn from_raw(lod: LOD, data: Vec<Self::Elem>) -> Self;
    /// The LOD the data is stored at (may lag the chunk's `TargetLOD`).
    fn lod(&self) -> LOD;
    /// The stored elements, `lod().size()`^3 of them, x fastest.
    fn data_slice(&self) -> &[Self::Elem];
    /// Bumped on every change; drives re-extraction and unsaved-edit detection.
    fn version(&self) -> u64;
    /// Version of only the voxels that `NEIGHBORS_MASK[neighbor]`'s chunk reads
    /// into its apron (the low slab(s) facing it), at equal LOD. Lets an edit
    /// that misses a face skip re-meshing the chunk across it. Default: any change.
    fn border_version(&self, _neighbor: usize) -> u64 {
        self.version()
    }
    /// Whether to build and upload a padded volume at all. `false` sends an
    /// empty one (for the mask: "no mask needed").
    fn should_extract(&self) -> bool {
        true
    }
    /// The bytes uploaded for a padded volume. Default: the raw elements.
    fn gpu_bytes(vol: &[Self::Elem]) -> Vec<u8> {
        bytemuck::cast_slice(vol).to_vec()
    }
    /// Converts a value sampled from a grid `ratio`x finer (or, if <1,
    /// coarser) than the destination. Only distances care: SDF values are
    /// in voxels of their own LOD, so they divide by `ratio`. Categorical
    /// data is unchanged. Used by downsampling AND by the neighbour apron.
    fn rescale(v: Self::Elem, _ratio: f32) -> Self::Elem {
        v
    }
    /// Samples a neighbour's data at a non-integer coordinate for the apron.
    /// Default: nearest voxel, as categorical data (ids, masks) must never
    /// blend; `SDF` interpolates.
    fn sample_apron(data: &[Self::Elem], size: u32, x: f32, y: f32, z: f32) -> Self::Elem {
        nearest_neighbor_sample(data, size, x, y, z)
    }
}

use crate::chunk::{Chunk, EditPin};

/// Per-chunk state `EditableChunks` checks before handing out a field.
type EditableChunkState = (
    Option<&'static LOD>,
    Has<crate::storage::loading::ChunkDataTask>,
    Option<&'static mut EditPin>,
);

/// The only sanctioned way to mutate a chunk field.
///
/// `get_mut` returns `Some` only when the chunk is at `MaxEditLod` with no
/// load in flight. Otherwise it pins the chunk at max LOD (upgrading it) and
/// returns `None`, so the caller retries next frame. If the field is optional
/// and absent, it is created at its "editable default" (for visibility:
/// fully visible).
#[derive(SystemParam)]
pub struct EditableChunks<'w, 's, F: VoxelField> {
    max: Res<'w, MaxEditLod>,
    commands: Commands<'w, 's>,
    chunks: Query<'w, 's, EditableChunkState, With<Chunk>>,
    fields: Query<'w, 's, &'static mut F, With<Chunk>>,
}

impl<F: VoxelField> EditableChunks<'_, '_, F> {
    /// A loaded chunk's field to read, at whatever LOD, without pinning it.
    #[must_use]
    pub fn get(&self, e: Entity) -> Option<&F> {
        self.fields.get(e).ok()
    }

    pub fn get_mut(&mut self, e: Entity) -> Option<Mut<'_, F>> {
        let max = self.max.0;
        let (lod, loading, pin) = self.chunks.get_mut(e).ok()?;

        let ready_lod = lod.copied().filter(|l| *l == max && !loading);
        let Some(lod) = ready_lod else {
            match pin {
                Some(mut p) => p.idle = 0.0, // waiting on the load: keep the pin alive
                None => {
                    self.commands
                        .entity(e)
                        .try_insert((EditPin::default(), TargetLOD(max)));
                }
            }
            return None;
        };
        if let Some(mut p) = pin {
            p.idle = 0.0;
        }
        if !self.fields.contains(e) {
            self.commands.entity(e).try_insert(F::new_for_lod(lod));
            return None; // present next frame
        }
        self.fields.get_mut(e).ok()
    }
}
