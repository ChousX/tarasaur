//! World-space edit requests. Systems write these messages; the edit
//! systems in `field::systems` resolve them against every chunk they touch.

use std::marker::PhantomData;

use super::ops::{PlaneMode, Sweep};
use super::{DistanceField, Field, SDF, VisibilityField};
use bevy::ecs::message::Message;
use bevy::ecs::message::MessageWriter;
use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use bevy::shape::{Cuboid, Primitive3d, Sphere};

#[derive(Clone, Copy, Debug)]
pub enum EditMode<V> {
    /// Instantly replaces the data with a static value
    Absolute,
    /// Adds/Subtracts a step size smoothly over time (e.g., for sculpting SDFs)
    Accumulate { delta: V },
    /// Smoothly moves current value towards target value with a falloff rate
    Blend { rate: f32 },
}

/// Apply `mode` with `val` to field `F` inside `shape` at `center` (world
/// space). Usually written through `WorldEditor`.
#[derive(Message)]
pub struct EditFieldMessage<
    F: Field<V>,
    S: Primitive3d + Send + Sync + 'static,
    V: Copy + Default + Send + Sync + 'static,
> {
    pub center: Vec3,
    pub shape: S,
    pub val: V,
    pub mode: EditMode<V>,
    phantom: PhantomData<F>,
}

impl<F: Field<V>, S: Primitive3d + Send + Sync + 'static, V: Copy + Default + Send + Sync + 'static>
    EditFieldMessage<F, S, V>
{
    pub const fn new(center: Vec3, shape: S, val: V, mode: EditMode<V>) -> Self {
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
/// or a local coordinate — under the hood this writes `EditFieldMessage`s,
/// and `process_shape_edits` resolves *every* chunk whose bounds overlap the shape,
/// so a brush that straddles a chunk boundary edits both sides seamlessly.
/// Chunks that don't exist yet are simply skipped by those systems.
///
/// ```ignore
/// fn dig_pit(mut sdf: WorldEditor<SDF, f32>) {
///     sdf.accumulate_sphere(Vec3::new(9.5, 5.0, 5.0), 4.0, 0.5); // + = remove
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

impl<F, V> WorldEditor<'_, F, V>
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

/// Overwrites the SDF within `bound_radius` with the exact distance to a
/// sphere of `sdf_radius`. Marks the chunk dirty (JFA reinit follows).
#[derive(Message, Clone, Copy)]
pub struct SdfSphereStamp {
    pub center: Vec3,
    pub bound_radius: f32,
    pub sdf_radius: f32,
}

/// CSG union/subtraction of a sphere, or of the capsule it sweeps;
/// idempotent, no reinit needed. What the built-in brush's Dig and Build
/// send.
#[derive(Message, Clone, Copy)]
pub struct SdfCsgSphere {
    pub center: Vec3,
    pub radius: f32,
    /// `Some(from)`: the edit is the capsule swept from `from` to `center`,
    /// so a stroke that moves far in a frame leaves no gaps. Send the last
    /// edit's center.
    pub sweep_from: Option<Vec3>,
    pub subtract: bool, // true = dig, false = add material
    /// `None`: exact CSG at once. `Some(step)`: the surface moves at most
    /// `step` world units toward the CSG result, the full step at the
    /// center and less toward the rim, so a held brush grows the shape
    /// gradually (send one per frame, scaled by the frame time). Either way
    /// it never goes past the sphere.
    pub max_step: Option<f32>,
    /// Build only: give every voxel this edit turns solid, and its air
    /// neighbours, this palette id (needs a `MaterialFieldPlugin`). The
    /// color follows the shape as it grows instead of landing all at once.
    pub paint: Option<u8>,
}

impl SdfCsgSphere {
    /// The segment the edit covers (world space).
    #[must_use]
    pub fn sweep(&self) -> Sweep {
        self.csg().sweep()
    }
}

/// Hides or reveals an exact sphere, or the capsule it sweeps, in the
/// `VisibilityField`.
///
/// A CSG subtraction from (or union with) the visible region, so the cap
/// closing the cut follows the sphere at sub-voxel precision. Idempotent,
/// no reinit needed. A chunk without the field gets one, fully visible,
/// first.
#[derive(Message, Clone, Copy, Debug)]
pub struct VisibilityCsgSphere {
    pub center: Vec3,
    pub radius: f32,
    /// `Some(from)`: the capsule swept from `from` to `center`, as in
    /// `SdfCsgSphere`.
    pub sweep_from: Option<Vec3>,
    /// `true` hides the sphere, `false` reveals it.
    pub hide: bool,
    /// As in `SdfCsgSphere`: `None` is the exact result at once, `Some(step)`
    /// moves the cap at most `step` world units per message.
    pub max_step: Option<f32>,
}

impl VisibilityCsgSphere {
    /// The segment the edit covers (world space).
    #[must_use]
    pub fn sweep(&self) -> Sweep {
        self.csg().sweep()
    }
}

/// A CSG sphere edit as a distance field applies it; what a `CsgMessage`
/// resolves to.
#[derive(Clone, Copy, Debug)]
pub struct CsgSphere {
    pub center: Vec3,
    pub radius: f32,
    pub sweep_from: Option<Vec3>,
    /// `max(field, -sphere)` (take the sphere out of the inside) rather
    /// than `min(field, sphere)` (add it).
    pub subtract: bool,
    pub max_step: Option<f32>,
    /// Palette id for the voxels the edit turns inside (`SDF` builds only).
    pub paint: Option<u8>,
}

impl CsgSphere {
    /// The segment the edit covers (world space).
    #[must_use]
    pub fn sweep(&self) -> Sweep {
        Sweep {
            from: self.sweep_from.unwrap_or(self.center),
            to: self.center,
        }
    }
}

/// A message that CSG-edits one distance field: `SdfCsgSphere` the `SDF`,
/// `VisibilityCsgSphere` the `VisibilityField`. `process_csg_spheres::<M>`
/// applies it.
pub trait CsgMessage: Message + Copy {
    type Field: DistanceField;
    fn csg(&self) -> CsgSphere;
    /// Whether the chunks it edits get a jump-flood reinit afterwards.
    /// `max`/`min` are exact only outside the shape they add or remove:
    /// revealing a hidden sphere leaves its distances around where it was,
    /// which `max(sdf, visibility)` would read as a dent in the terrain.
    /// The reinit keeps the zero crossing and rebuilds the rest.
    fn reinit_after(&self) -> bool {
        false
    }
}

impl CsgMessage for SdfCsgSphere {
    type Field = SDF;
    fn csg(&self) -> CsgSphere {
        CsgSphere {
            center: self.center,
            radius: self.radius,
            sweep_from: self.sweep_from,
            subtract: self.subtract,
            max_step: self.max_step,
            paint: self.paint,
        }
    }
}

impl CsgMessage for VisibilityCsgSphere {
    type Field = VisibilityField;
    fn csg(&self) -> CsgSphere {
        CsgSphere {
            center: self.center,
            radius: self.radius,
            sweep_from: self.sweep_from,
            subtract: self.hide,
            max_step: self.max_step,
            paint: None,
        }
    }
    /// A reveal: the union leaves the sphere's distances where it was.
    fn reinit_after(&self) -> bool {
        !self.hide
    }
}

/// What an `SdfSculptSphere` does to the distances inside it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SdfSculptOp {
    /// Relax toward the neighbour average: rounds off bumps and creases.
    Smooth,
    /// Pull toward the plane through `point` (world space) with unit
    /// `normal` pointing out of the terrain, on the side(s) `mode` picks.
    Flatten {
        point: Vec3,
        normal: Vec3,
        mode: PlaneMode,
    },
}

/// A gradual SDF brush, applied without a jump-flood reinit.
///
/// Each message moves the voxels inside the sphere `amount` (0..=1) of the
/// way at the center, fading to nothing at the edge, so send one per frame
/// scaled by the frame time.
#[derive(Message, Clone, Copy, Debug)]
pub struct SdfSculptSphere {
    pub center: Vec3,
    pub radius: f32,
    /// `Some(from)`: work on the capsule swept from `from` to `center`, as
    /// in `SdfCsgSphere`.
    pub sweep_from: Option<Vec3>,
    pub amount: f32,
    pub op: SdfSculptOp,
}

impl SdfSculptSphere {
    /// The segment the edit covers (world space).
    #[must_use]
    pub fn sweep(&self) -> Sweep {
        Sweep {
            from: self.sweep_from.unwrap_or(self.center),
            to: self.center,
        }
    }
}

/// Sets the listed voxels (grid coordinates) of one chunk's material field
/// to palette id `material`. Sent by the CSG system for a painting build;
/// applied by `MaterialFieldPlugin<M>`.
#[derive(Message, Clone, Debug)]
pub struct MaterialPaintVoxels {
    pub chunk: Entity,
    pub voxels: Vec<UVec3>,
    pub material: u8,
}

/// Sets every voxel within `radius` of `center` to palette id `material`.
///
/// For code that doesn't know the app's material enum (like the cursor
/// brush); `MaterialFieldPlugin<M>` applies it to `MaterialField<M>`, and
/// without one it's ignored.
#[derive(Message, Clone, Copy, Debug)]
pub struct MaterialPaint {
    pub center: Vec3,
    pub radius: f32,
    pub material: u8,
}
