use bevy::ecs::component::Mutable;
use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use std::marker::PhantomData;

pub mod editor;
pub mod generator;
pub mod loading;
pub mod lod;
pub mod material;
pub mod ops;
pub mod persistence;
mod plugin;
pub mod sdf;
pub mod surface;
pub mod systems;
pub mod visibility;
pub use lod::{LOD, MaxEditLod, TargetLOD};
pub use material::MaterialField;
pub use material::VoxelMaterial;
pub use plugin::*;
pub use sdf::SDF;
pub use visibility::VisibilityField;

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

pub trait FieldGen<T: Copy + Default>: Field<T> {
    /// `chunk_pos` is in chunk-grid coordinates; `local` is 0..size() per axis.
    fn build(&mut self, chunk_pos: IVec3, local: UVec3) -> T;
}

#[inline]
pub fn flatten_with_size(x: u32, y: u32, z: u32, size: UVec3) -> u32 {
    // Index = z * (width * height) + y * width + x
    z * (size.x * size.y) + y * size.x + x
}

pub trait Versionable: Component {
    /// Gets the value at the given grid coordinates.
    fn version(&self) -> u64;
    /// Sets the value at the given grid coordinates.
    fn incorment_version(&mut self);
}

/// A field whose backing storage can be uploaded to the GPU as a flat byte
/// buffer. `Elem` is the raw stored type — f32 for SDF distances, u8 for
/// packed material ids.
pub trait VoxelDataSlice {
    type Elem: bytemuck::Pod + Default;
    fn data_slice(&self) -> &[Self::Elem];
    #[inline]
    /// Converts a value sampled from a grid `ratio`x finer (or, if <1,
    /// coarser) than the destination. Only distances care: SDF values are
    /// in voxels of their own LOD, so they divide by `ratio`. Categorical
    /// data is unchanged. Used by downsampling AND by the neighbour apron.

    fn rescale(v: Self::Elem, _ratio: f32) -> Self::Elem {
        v
    }
}

/// How a field samples its own data at a non-integer (neighbor-chunk)
/// coordinate when building the padding apron. SDF distances interpolate
/// meaningfully (trilinear); material ids are categorical and must never be
/// blended, so they use nearest-neighbor instead.
pub trait ApronSample: VoxelDataSlice {
    fn sample_apron(data: &[Self::Elem], size: u32, x: f32, y: f32, z: f32) -> Self::Elem;
}

/// Optional opt-out checked by `extract_voxel_chunks<T>` before doing the
/// expensive padding/apron work for a chunk. Default: always do it.
/// SDFField/MaterialField<M> use the default via a trivial empty impl —
/// same pattern as AccumulateExt/BlendExt, explicit per-type impls rather
/// than a blanket, since a blanket would make it impossible for
/// VisibilityField to actually override this under Rust's coherence rules.
pub trait ExtractGate {
    fn should_extract(&self) -> bool {
        true
    }
}

pub trait FieldLOD {
    fn lod(&self) -> LOD;
}

/// Reconstructs a field at a given LOD with default/empty contents —
/// the same shape every field type's `new(lod)` already provides. Exists
/// so the LOD-sync system (see field/lod_sync.rs) can reset any field
/// type generically without knowing its concrete type.
pub trait FieldNew {
    fn new_for_lod(lod: LOD) -> Self;
}

impl FieldNew for SDF {
    fn new_for_lod(lod: LOD) -> Self {
        Self::new(lod)
    }
}
impl FieldNew for VisibilityField {
    fn new_for_lod(lod: LOD) -> Self {
        Self::new(lod)
    }
}
impl<M: VoxelMaterial> FieldNew for MaterialField<M> {
    fn new_for_lod(lod: LOD) -> Self {
        Self::new(lod)
    }
}

/// Build a field directly from raw voxel data at `lod` (version 0).
pub trait FieldFromRaw: VoxelDataSlice + Sized {
    /// Optional fields are REMOVED (not defaulted) when the save has no payload.
    const OPTIONAL: bool = false;
    fn from_raw(lod: LOD, data: Vec<Self::Elem>) -> Self;
}

use crate::chunk::{Chunk, EditPin};

/// The only sanctioned way to mutate a chunk field. `get_mut` returns `Some`
/// only when the chunk is at `MaxEditLod` with no load in flight. Otherwise
/// it pins the chunk at max LOD (upgrading it) and returns `None`, so the
/// caller retries next frame. If the field is optional and absent, it is
/// created at its "editable default" (for visibility: fully visible).
#[derive(SystemParam)]
pub struct EditableChunks<'w, 's, F: Component<Mutability = Mutable> + FieldNew> {
    max: Res<'w, MaxEditLod>,
    commands: Commands<'w, 's>,
    chunks: Query<
        'w,
        's,
        (
            Option<&'static LOD>,
            Has<loading::ChunkDataTask>,
            Option<&'static mut EditPin>,
        ),
        With<Chunk>,
    >,
    fields: Query<'w, 's, &'static mut F, With<Chunk>>,
}

impl<F: Component<Mutability = Mutable> + FieldNew> EditableChunks<'_, '_, F> {
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
