//! Field registration: the edit cycle's system sets, the systems every app
//! has (`FieldsPlugin`), and the per-field edit wiring used by
//! `AppVoxelFieldExt::add_voxel_field`.

use bevy::prelude::*;
use bevy::shape::{Cuboid, Sphere};

use crate::chunk::NewChunkSpawned;
use crate::field::editor::{EditFieldMessage, SdfCsgSphere, SdfSphereStamp};
use crate::field::systems::{
    clear_dirty, process_sdf_csg_spheres, process_sdf_sphere_stamps, process_shape_edits,
    resolve_sdf_reinit_tasks, spawn_sdf_reinit_tasks,
};
use crate::field::{Field, LOD, TargetLOD, VoxelField};

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum FieldSet {
    /// All edit-producing and edit-applying systems.
    Edit,
    /// Runs once per cycle, after every edit system has run.
    Reinit,
    /// Ends the cycle: clears the `DirtyField` markers that need no follow-up.
    EndCycle,
}

/// The edit cycle and the SDF's own systems (jump-flood reinit, sphere
/// stamps, CSG spheres). Fields themselves are registered with
/// `add_voxel_field`.
pub struct FieldsPlugin;

impl Plugin for FieldsPlugin {
    fn build(&self, app: &mut App) {
        app.configure_sets(
            Update,
            (FieldSet::Edit, FieldSet::Reinit, FieldSet::EndCycle).chain(),
        );
        app.add_systems(
            Update,
            (spawn_sdf_reinit_tasks, resolve_sdf_reinit_tasks)
                .chain()
                .in_set(FieldSet::Reinit),
        );
        // A development aid only: release builds neither run nor compile it.
        #[cfg(debug_assertions)]
        app.add_systems(
            Update,
            crate::field::systems::detect_topology_desync
                .in_set(FieldSet::Reinit)
                .before(spawn_sdf_reinit_tasks),
        );

        app.add_message::<SdfSphereStamp>()
            .add_systems(Update, process_sdf_sphere_stamps.in_set(FieldSet::Edit));

        app.add_message::<SdfCsgSphere>()
            .add_systems(Update, process_sdf_csg_spheres);
    }
}

/// `F`'s edit messages and systems, its default on new chunks, and the end
/// of its edit cycle.
pub fn register_edits<F>(app: &mut App)
where
    F: VoxelField + Field<F::Value>,
{
    app.add_message::<EditFieldMessage<F, Sphere, F::Value>>()
        .add_message::<EditFieldMessage<F, Cuboid, F::Value>>()
        .add_systems(
            Update,
            (
                process_shape_edits::<F, Sphere, F::Value>,
                process_shape_edits::<F, Cuboid, F::Value>,
            )
                .in_set(FieldSet::Edit),
        )
        .add_observer(build_field_on_chunk_spawn::<F>);
    if F::CLEAR_DIRTY_EACH_CYCLE {
        app.add_systems(
            Update,
            clear_dirty::<F, F::Value>.in_set(FieldSet::EndCycle),
        );
    }
}

/// Gives a newly spawned chunk a default `F` at its LOD, unless it already
/// has one.
///
/// Chunks spawned with a `TargetLOD` and no `LOD` (the `ChunkLoader`'s) are
/// skipped: their load installs every field, so a default would only be
/// built to be thrown away. Until then readers treat them as unloaded.
/// Chunks spawned by hand, with an `LOD`, still get defaults to fill in.
fn build_field_on_chunk_spawn<F: VoxelField>(
    trigger: On<NewChunkSpawned>,
    mut commands: Commands,
    chunks: Query<(Option<&LOD>, Has<TargetLOD>)>,
    has_field: Query<(), With<F>>,
) {
    let entity = trigger.event().entity;
    if has_field.contains(entity) {
        return;
    }
    let Ok((lod, awaiting_load)) = chunks.get(entity) else {
        return;
    };
    if lod.is_none() && awaiting_load {
        return;
    }
    let lod = lod.copied().unwrap_or_default();
    commands.entity(entity).insert(F::new_for_lod(lod));
}
