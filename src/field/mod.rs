use bevy::prelude::*;

mod consts;
pub mod editor;
mod lod;
mod material;
pub mod ops;
mod plugin;
mod sdf;
pub mod systems;
mod visibility;

pub use lod::Lod;
pub use material::MaterialField;
pub use plugin::{FieldSet, FieldsPlugin};
pub use sdf::SdfField;
pub use visibility::VisibilityField;

/// Core trait representing a 3D grid of data.
pub trait Field<T: Copy + Default>: Component {
    /// Returns the dimensions of this specific field.
    fn size(&self) -> UVec3;
    /// Gets the value at the given grid coordinates.
    fn get(&self, x: u32, y: u32, z: u32) -> T;
    /// Sets the value at the given grid coordinates.
    fn set(&mut self, x: u32, y: u32, z: u32, value: T);
}
