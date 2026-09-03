use std::marker::PhantomData;

// fields/editor.rs
use super::Field;
use bevy::ecs::message::MessageWriter;
use bevy::ecs::system::SystemParam;
use bevy::math::primitives::{Cuboid, Primitive3d, Sphere};
use bevy::prelude::*;

#[derive(Clone, Copy, Debug)]
pub enum EditMode<V> {
    /// Instantly replaces the data with a static value
    Absolute,
    /// Adds/Subtracts a step size smoothly over time (e.g., for sculpting SDFs)
    Accumulate { delta: V },
    /// Smoothly moves current value towards target value with a falloff rate
    Blend { rate: f32 },
}

#[derive(Message)]
pub struct EditFieldMessage<F: Field<V>, S: Primitive3d, V: Copy + Default + Send + Sync + 'static>
{
    pub center: Vec3,
    pub shape: S,
    pub val: V,
    pub mode: EditMode<V>,
    phantom: PhantomData<F>,
}

impl<F: Field<V>, S: Primitive3d, V: Copy + Default + Send + Sync + 'static>
    EditFieldMessage<F, S, V>
{
    pub fn new(center: Vec3, shape: S, val: V, mode: EditMode<V>) -> Self {
        Self {
            center,
            shape,
            val,
            mode,
            phantom: PhantomData,
        }
    }
}

/// Ergonomic, chunk-agnostic entry point for editing a field's topology.
///
/// Take a `WorldEditor<F, V>` as a system parameter and call `sphere`/
/// `cuboid` (or the `fill_*`/`accumulate_*`/`blend_*` shortcuts) with plain
/// world-space coordinates. You never see a chunk position, a chunk entity,
/// or a local coordinate — under the hood this just writes the same
/// `EditFieldMessage`s as before, and `process_sphere_edits`/
/// `process_box_edits` resolve *every* chunk whose bounds overlap the shape,
/// so a brush that straddles a chunk boundary edits both sides seamlessly.
/// Chunks that don't exist yet are simply skipped by those systems.
///
/// ```ignore
/// fn dig_pit(mut sdf: WorldEditor<SDFField, f32>) {
///     sdf.accumulate_sphere(Vec3::new(9.5, 5.0, 5.0), 4.0, -0.5);
/// }
/// ```
#[derive(SystemParam)]
pub struct WorldEditor<'w, F, V>
where
    F: Field<V>,
    V: Copy + Default + Send + Sync + 'static,
{
    sphere_writer: MessageWriter<'w, EditFieldMessage<F, Sphere, V>>,
    box_writer: MessageWriter<'w, EditFieldMessage<F, Cuboid, V>>,
}

impl<'w, F, V> WorldEditor<'w, F, V>
where
    F: Field<V>,
    V: Copy + Default + Send + Sync + 'static,
{
    /// Applies `mode` to every voxel within `radius` of `center` (world space).
    pub fn sphere(&mut self, center: Vec3, radius: f32, val: V, mode: EditMode<V>) {
        self.sphere_writer
            .write(EditFieldMessage::new(center, Sphere { radius }, val, mode));
    }

    /// Instantly sets every voxel within `radius` of `center` to `val`.
    pub fn fill_sphere(&mut self, center: Vec3, radius: f32, val: V) {
        self.sphere(center, radius, val, EditMode::Absolute);
    }

    /// Adds `delta` to every voxel within `radius` of `center` (e.g. sculpting).
    pub fn accumulate_sphere(&mut self, center: Vec3, radius: f32, delta: V) {
        self.sphere(center, radius, V::default(), EditMode::Accumulate { delta });
    }

    /// Blends every voxel within `radius` towards `target` at `rate`, with
    /// linear falloff from the center.
    pub fn blend_sphere(&mut self, center: Vec3, radius: f32, target: V, rate: f32) {
        self.sphere(center, radius, target, EditMode::Blend { rate });
    }

    /// Applies `mode` to every voxel inside the axis-aligned box centered at
    /// `center` with half-extents `half_size` (world space).
    pub fn cuboid(&mut self, center: Vec3, half_size: Vec3, val: V, mode: EditMode<V>) {
        self.box_writer.write(EditFieldMessage::new(
            center,
            Cuboid { half_size },
            val,
            mode,
        ));
    }

    /// Instantly sets every voxel inside the box to `val`.
    pub fn fill_box(&mut self, center: Vec3, half_size: Vec3, val: V) {
        self.cuboid(center, half_size, val, EditMode::Absolute);
    }

    /// Adds `delta` to every voxel inside the box.
    pub fn accumulate_box(&mut self, center: Vec3, half_size: Vec3, delta: V) {
        self.cuboid(
            center,
            half_size,
            V::default(),
            EditMode::Accumulate { delta },
        );
    }

    /// Blends every voxel inside the box towards `target` at a uniform `rate`.
    pub fn blend_box(&mut self, center: Vec3, half_size: Vec3, target: V, rate: f32) {
        self.cuboid(center, half_size, target, EditMode::Blend { rate });
    }
}
