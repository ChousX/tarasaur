// fields/systems.rs
use bevy::ecs::component::Mutable;
use bevy::ecs::message::MessageReader;
use bevy::math::primitives::{Cuboid, Sphere};
use bevy::prelude::*;

use super::{
    Field, SdfField,
    editor::{EditFieldMessage, EditMode},
    ops::{AccumulateExt, BlendExt, FieldBoxOps, FieldSphereOps},
};

/// Generic system to process spherical edits for any component matching `Field<V>`.
pub fn process_sphere_edits<F, V>(
    mut events: MessageReader<EditFieldMessage<F, Sphere, V>>,
    mut query: Query<&mut F>,
) where
    F: Field<V> + Component<Mutability = Mutable>,
    V: Copy + Default + Send + Sync + 'static + AccumulateExt + BlendExt, // Added bounds here
{
    for event in events.read() {
        for mut field in query.iter_mut() {
            match event.mode {
                EditMode::Absolute => {
                    field.fill_sphere(event.center, event.shape, event.val);
                }
                EditMode::Accumulate { delta } => {
                    field.accumulate_sphere(event.center, event.shape, delta);
                }
                EditMode::Blend { rate } => {
                    field.blend_sphere(event.center, event.shape, event.val, rate);
                }
            }
        }
    }
}

/// Generic system to process cuboid/box edits for any component matching `Field<V>`.
pub fn process_box_edits<F, V>(
    mut events: MessageReader<EditFieldMessage<F, Cuboid, V>>,
    mut query: Query<&mut F>,
) where
    F: Field<V> + Component<Mutability = Mutable>,
    V: Copy + Default + Send + Sync + 'static + AccumulateExt + BlendExt, // Added bounds here
{
    for event in events.read() {
        for mut field in query.iter_mut() {
            match event.mode {
                EditMode::Absolute => {
                    field.fill_box(event.center, event.shape, event.val);
                }
                EditMode::Accumulate { delta } => {
                    field.accumulate_box(event.center, event.shape, delta);
                }
                EditMode::Blend { rate } => {
                    field.blend_box(event.center, event.shape, event.val, rate);
                }
            }
        }
    }
}

/// Runs in `FieldSet::Reinit` to cleanup dirty SDF fields after edits have landed.
pub fn reinit_dirty_sdf(mut query: Query<&mut SdfField>) {
    for mut sdf_field in query.iter_mut() {
        if sdf_field.is_dirty() {
            sdf_field.reinit_if_dirty();
        }
    }
}

