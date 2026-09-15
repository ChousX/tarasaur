use std::marker::PhantomData;

use bevy::ecs::component::Mutable;
use bevy::math::primitives::{Cuboid, Sphere};
use bevy::prelude::*;

use crate::LOD;
use crate::chunk::NewChunkSpawned;
use crate::field::material::VoxelMaterial;
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

/// Registers a single material channel (`MaterialField<M>`) with the app —
/// its edit messages/systems, and the observer that attaches it to new
/// chunks. Add one of these per material enum you use; most projects need
/// exactly one, but nothing stops registering a second channel (e.g. a
/// separate cave-stratum material) alongside it.
pub struct MaterialFieldPlugin<M: VoxelMaterial>(PhantomData<M>);

impl<M: VoxelMaterial> MaterialFieldPlugin<M> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<M: VoxelMaterial> Default for MaterialFieldPlugin<M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<M: VoxelMaterial> Plugin for MaterialFieldPlugin<M> {
    fn build(&self, app: &mut App) {
        app.add_field::<MaterialField<M>, M>();
        app.add_observer(material_build_on_chunk_spawn::<M>);
    }
}

/// Registers the fields every chunk always has: SDF (topology) and
/// visibility (cull mask). Material is opt-in via `MaterialFieldPlugin<M>`
/// since only the end user knows their material enum.
pub struct FieldsPlugin;

impl Plugin for FieldsPlugin {
    fn build(&self, app: &mut App) {
        app.configure_sets(Update, (FieldSet::Edit, FieldSet::Reinit).chain());
        app.add_systems(Update, reinit_dirty_sdf.in_set(FieldSet::Reinit));

        app.add_field::<SDFField, f32>()
            .add_field::<VisibilityField, bool>();

        app.add_observer(sdf_build_on_chunk_spawn)
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
        V: Copy + Default + Send + Sync + 'static + AccumulateExt + BlendExt,
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
    lod_q: Query<&LOD>,
    chunk_q: Query<(), With<SDFField>>,
) {
    let NewChunkSpawned { entity, .. } = trigger.event();
    if chunk_q.get(*entity).is_ok() {
        return;
    }
    let lod = lod_q.get(*entity).copied().unwrap_or_default();
    commands.entity(*entity).insert(SDFField::new(lod));
}

fn visibility_build_on_chunk_spawn(
    trigger: On<NewChunkSpawned>,
    mut commands: Commands,
    lod_q: Query<&LOD>,
    chunk_q: Query<(), With<VisibilityField>>,
) {
    let NewChunkSpawned { entity, .. } = trigger.event();
    if chunk_q.get(*entity).is_ok() {
        return;
    }
    let lod = lod_q.get(*entity).copied().unwrap_or_default();
    commands.entity(*entity).insert(VisibilityField::new(lod));
}

fn material_build_on_chunk_spawn<M: VoxelMaterial>(
    trigger: On<NewChunkSpawned>,
    mut commands: Commands,
    lod_q: Query<&LOD>,
    chunk_q: Query<(), With<MaterialField<M>>>,
) {
    let NewChunkSpawned { entity, .. } = trigger.event();
    if chunk_q.get(*entity).is_ok() {
        return;
    }
    let lod = lod_q.get(*entity).copied().unwrap_or_default();
    commands
        .entity(*entity)
        .insert(MaterialField::<M>::new(lod));
}
