use std::marker::PhantomData;

use bevy::prelude::*;

pub mod editor;
pub mod lod;
pub mod material;
pub mod ops;
pub mod plugin;
pub mod sdf;
pub mod systems;
pub mod visibility;
pub use lod::LOD;
pub use material::MaterialField;
pub use material::VoxelMaterial;
pub use plugin::AppFieldExt;
pub use plugin::{FieldSet, FieldsPlugin};
pub use sdf::SDFField;
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
    fn build(&mut self, pos: UVec3) -> T;
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
}

/// How a field samples its own data at a non-integer (neighbor-chunk)
/// coordinate when building the padding apron. SDF distances interpolate
/// meaningfully (trilinear); material ids are categorical and must never be
/// blended, so they use nearest-neighbor instead.
pub trait ApronSample: VoxelDataSlice {
    fn sample_apron(data: &[Self::Elem], size: u32, x: f32, y: f32, z: f32) -> Self::Elem;
}
