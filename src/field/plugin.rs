// fields/plugin.rs
use bevy::ecs::component::Mutable;
use bevy::math::primitives::{Cuboid, Sphere};
use bevy::prelude::*;

use crate::field::{MaterialField, SDFField, VisibilityField};

use super::{
    Field,
    editor::EditFieldMessage,
    ops::{AccumulateExt, BlendExt},
    systems::{process_box_edits, process_sphere_edits, reinit_dirty_sdf},
};

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum FieldSet {
    /// All edit-producing and edit-applying systems.
    Edit,
    /// Runs once per cycle, after every edit system has run.
    Reinit,
}

pub struct FieldsPlugin;

impl Plugin for FieldsPlugin {
    fn build(&self, app: &mut App) {
        app.configure_sets(Update, (FieldSet::Edit, FieldSet::Reinit).chain());
        app.add_systems(Update, reinit_dirty_sdf.in_set(FieldSet::Reinit));
        app.add_field::<SDFField, f32>()
            .add_field::<MaterialField, u8>()
            .add_field::<VisibilityField, bool>();
    }
}

/// Extension trait to add custom fluent registration APIs onto the Bevy App builder.
pub trait AppFieldExt {
    fn add_field<F, V>(&mut self) -> &mut Self
    where
        F: Field<V> + Component<Mutability = Mutable>,
        V: Copy + Default + Send + Sync + 'static + AccumulateExt + BlendExt;
}

impl AppFieldExt for App {
    fn add_field<F, V>(&mut self) -> &mut Self
    where
        F: Field<V> + Component<Mutability = Mutable>,
        V: Copy + Default + Send + Sync + 'static + AccumulateExt + BlendExt, // Fixed: Removed the semicolon here!
    {
        // 1. Register the custom Messages/Events for this field type
        self.add_message::<EditFieldMessage<F, Sphere, V>>()
            .add_message::<EditFieldMessage<F, Cuboid, V>>();

        // 2. Attach the generic edit execution systems into the Edit schedule set
        self.add_systems(
            Update,
            (process_sphere_edits::<F, V>, process_box_edits::<F, V>).in_set(FieldSet::Edit),
        );

        self
    }
}
