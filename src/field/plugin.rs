// fields/plugin.rs
use bevy::ecs::component::Mutable;
use bevy::ecs::system::command::trigger;
use bevy::math::primitives::{Cuboid, Sphere};
use bevy::prelude::*;

use crate::chunk::NewChunkSpawned;
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

        app.add_observer(sdf_build_on_chunk_spawn)
            .add_observer(material_build_on_chunk_spawn)
            .add_observer(visibility_build_on_chunk_spawn);
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

fn sdf_build_on_chunk_spawn(
    trigger: On<NewChunkSpawned>,
    mut commands: Commands,
    chunk_q: Query<(), With<SDFField>>,
) {
    let NewChunkSpawned {
        entity,
        ..
        //world_position,
        //chunk_position,
    } = trigger.event();
    if chunk_q.get(*entity).is_ok() {
        return;
    }
    let new_sdf = SDFField::default();
    commands.entity(*entity).insert(new_sdf);
}
fn visibility_build_on_chunk_spawn(
    trigger: On<NewChunkSpawned>,
    mut commands: Commands,
    chunk_q: Query<(), With<VisibilityField>>,
) {
    let NewChunkSpawned {
        entity,
        ..
        //world_position,
        //chunk_position,
    } = trigger.event();
    if chunk_q.get(*entity).is_ok() {
        return;
    }
    let new_visibility = SDFField::default();
    commands.entity(*entity).insert(new_visibility);
}
fn material_build_on_chunk_spawn(
    trigger: On<NewChunkSpawned>,
    mut commands: Commands,
    chunk_q: Query<(), With<MaterialField>>,
) {
    let NewChunkSpawned {
        entity,
        ..
        //world_position,
        //chunk_position,
    } = trigger.event();
    if chunk_q.get(*entity).is_ok() {
        return;
    }
    let new_material = SDFField::default();
    commands.entity(*entity).insert(new_material);
}
