use std::marker::PhantomData;

use bevy::ecs::component::Mutable;
use bevy::math::primitives::{Cuboid, Sphere};
use bevy::prelude::*;
use bevy::render::{Render, RenderApp, RenderSystems};

use crate::chunk::NewChunkSpawned;
use crate::editor::SdfSphereStamp;
use crate::field::material::VoxelMaterial;
use crate::field::{MaterialField, SDF, VisibilityField};
use crate::persistence::RegisterSaveableFieldExt;
use crate::systems::{
    clear_dirty_visibility, detect_topology_desync, process_sdf_sphere_stamps,
    resolve_sdf_reinit_tasks, spawn_sdf_reinit_tasks,
};
use crate::voxel::systems::{
    extract_voxel_chunks, prepare_material_for_arena, prepare_voxel_arena,
};
use crate::{FieldLOD, FieldNew, LOD};

use super::{
    Field,
    editor::EditFieldMessage,
    ops::{AccumulateExt, BlendExt},
    systems::process_shape_edits,
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
        app.register_saveable_field::<MaterialField<M>>();

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .add_systems(ExtractSchedule, extract_voxel_chunks::<MaterialField<M>>)
            .add_systems(
                Render,
                prepare_material_for_arena::<M>
                    .after(prepare_voxel_arena::<SDF>)
                    .in_set(RenderSystems::Prepare),
            );
    }
}

/// Registers the fields every chunk always has: SDF (topology) and
/// visibility (cull mask). Material is opt-in via `MaterialFieldPlugin<M>`
/// since only the end user knows their material enum.
pub struct FieldsPlugin;

impl Plugin for FieldsPlugin {
    fn build(&self, app: &mut App) {
        app.configure_sets(Update, (FieldSet::Edit, FieldSet::Reinit).chain());
        app.add_systems(
            Update,
            (
                detect_topology_desync,
                spawn_sdf_reinit_tasks,
                resolve_sdf_reinit_tasks,
                clear_dirty_visibility,
            )
                .chain()
                .in_set(FieldSet::Reinit),
        );

        app.add_field::<SDF, f32>()
            .add_field::<VisibilityField, bool>();

        app.add_observer(sdf_build_on_chunk_spawn)
            .add_observer(visibility_build_on_chunk_spawn);

        app.add_message::<SdfSphereStamp>()
            .add_systems(Update, process_sdf_sphere_stamps.in_set(FieldSet::Edit));
    }
}

/// Extension trait to add custom fluent registration APIs onto the Bevy App builder.
pub trait AppFieldExt {
    fn add_field<F, V>(&mut self) -> &mut Self
    where
        F: Field<V> + FieldLOD + FieldNew + Component<Mutability = Mutable>,
        V: Copy + Default + Send + Sync + 'static + AccumulateExt + BlendExt;
}

impl AppFieldExt for App {
    fn add_field<F, V>(&mut self) -> &mut Self
    where
        F: Field<V> + FieldLOD + FieldNew + Component<Mutability = Mutable>,
        V: Copy + Default + Send + Sync + 'static + AccumulateExt + BlendExt,
    {
        self.add_message::<EditFieldMessage<F, Sphere, V>>()
            .add_message::<EditFieldMessage<F, Cuboid, V>>();

        self.add_systems(
            Update,
            (
                process_shape_edits::<F, Sphere, V>,
                process_shape_edits::<F, Cuboid, V>,
            )
                .in_set(FieldSet::Edit),
        );

        self
    }
}

fn sdf_build_on_chunk_spawn(
    trigger: On<NewChunkSpawned>,
    mut commands: Commands,
    lod_q: Query<&LOD>,
    chunk_q: Query<(), With<SDF>>,
) {
    let NewChunkSpawned { entity, .. } = trigger.event();
    if chunk_q.get(*entity).is_ok() {
        return;
    }
    let lod = lod_q.get(*entity).copied().unwrap_or_default();
    commands.entity(*entity).insert(SDF::new(lod));
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
