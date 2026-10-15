//! Edit application and SDF upkeep.
//!
//! Edit systems split each world-space edit across the chunks it overlaps
//! and retry the ones not yet editable (`apply_edits`). Value edits leave the
//! SDF as signs, so `spawn_sdf_reinit_tasks` / `resolve_sdf_reinit_tasks`
//! rebuild real distances with an async jump flood; CSG edits write exact
//! distances and skip that.

use super::{
    EditableChunks, Field, FieldNew, SDF,
    editor::{EditFieldMessage, EditMode},
    ops::{AccumulateExt, BlendExt, ShapeBounds, ShapeScale},
};
use crate::{
    CHUNK_SIZE, ChunkPosition, DirtyField, MaxEditLod, VisibilityField,
    chunk::{ChunkManager, overlapping_chunks},
    editor::{SdfCsgSphere, SdfSphereStamp},
    num::f32_from_u32,
    ops::{FieldSphereOps, ShapeEditOps},
    voxel::resident::ForceResident,
};
use bevy::{
    ecs::{component::Mutable, message::MessageReader},
    prelude::*,
    tasks::{AsyncComputeTaskPool, Task, block_on, poll_once},
};

const MAX_DEFER_FRAMES: u32 = 600;
const MAX_DEFERRED: usize = 4096;

/// One world-space edit queued against one chunk.
pub struct DeferredEdit<E> {
    chunk_pos: IVec3,
    edit: E,
    age: u32,
}

/// Shared retry loop for the edit systems. Runs last frame's retries (oldest
/// first), then `new` edits — `(center, half_extent, edit)` — against every
/// chunk they overlap. `apply` gets the field in grid space: the chunk's
/// world origin and voxel size. An edit whose chunk isn't editable yet (see
/// `EditableChunks`) is retried for `MAX_DEFER_FRAMES`, then handed to
/// `on_drop`; one whose chunk isn't loaded is skipped.
fn apply_edits<F, V, E: Copy>(
    deferred: &mut Vec<DeferredEdit<E>>,
    new: impl IntoIterator<Item = (Vec3, Vec3, E)>,
    chunk_manager: &ChunkManager,
    editable: &mut EditableChunks<F>,
    mut apply: impl FnMut(Entity, &mut F, Vec3, f32, E),
    mut on_drop: impl FnMut(IVec3),
) where
    F: Field<V> + Component<Mutability = Mutable> + FieldNew,
    V: Copy + Default,
{
    let mut work = std::mem::take(deferred);
    for (center, half_extent, edit) in new {
        work.extend(
            overlapping_chunks(center, half_extent).map(|chunk_pos| DeferredEdit {
                chunk_pos,
                edit,
                age: 0,
            }),
        );
    }
    for mut e in work {
        let Some(entity) = chunk_manager.get_chunk(&e.chunk_pos) else {
            continue; // chunk not loaded / gone: nothing to edit
        };
        if let Some(mut field) = editable.get_mut(entity) {
            let origin = e.chunk_pos.as_vec3() * CHUNK_SIZE;
            let voxel_size = CHUNK_SIZE / f32_from_u32(field.size().x);
            apply(entity, &mut field, origin, voxel_size, e.edit);
        } else {
            e.age = e.age.saturating_add(1); // waiting for the chunk to reach max LOD
            if e.age < MAX_DEFER_FRAMES {
                deferred.push(e);
            } else {
                on_drop(e.chunk_pos);
            }
        }
    }
    if deferred.len() > MAX_DEFERRED {
        let excess = deferred.len().saturating_sub(MAX_DEFERRED);
        deferred.drain(..excess); // drop the oldest
    }
}

/// One queued shape edit: (center, shape, value, mode).
type ShapeEdit<S, V> = (Vec3, S, V, EditMode<V>);

pub fn process_shape_edits<F, S, V>(
    mut commands: Commands,
    mut events: MessageReader<EditFieldMessage<F, S, V>>,
    mut editable: EditableChunks<F>,
    chunk_manager: Res<ChunkManager>,
    mut deferred: Local<Vec<DeferredEdit<ShapeEdit<S, V>>>>,
) where
    F: Field<V> + Component<Mutability = Mutable> + ShapeEditOps<V, S> + FieldNew,
    S: Primitive3d + Copy + Send + Sync + 'static + ShapeBounds + ShapeScale,
    V: Copy + Default + Send + Sync + 'static + AccumulateExt + BlendExt,
{
    let new = events.read().map(|ev| {
        (
            ev.center,
            ev.shape.half_extent(),
            (ev.center, ev.shape, ev.val, ev.mode),
        )
    });
    apply_edits(
        &mut deferred,
        new,
        &chunk_manager,
        &mut editable,
        |entity, field, origin, voxel_size, (center, shape, val, mode)| {
            let grid_center = (center - origin) / voxel_size;
            let grid_shape = shape.scaled_by(1.0 / voxel_size);
            match mode {
                EditMode::Absolute => field.fill(grid_center, grid_shape, val),
                EditMode::Accumulate { delta } => field.accumulate(grid_center, grid_shape, delta),
                EditMode::Blend { rate } => field.blend(grid_center, grid_shape, val, rate),
            }
            commands
                .entity(entity)
                .insert(DirtyField::<F, V>::default());
        },
        |_| {},
    );
}

pub fn process_sdf_sphere_stamps(
    mut commands: Commands,
    mut events: MessageReader<SdfSphereStamp>,
    mut editable: EditableChunks<SDF>,
    chunk_manager: Res<ChunkManager>,
    mut deferred: Local<Vec<DeferredEdit<SdfSphereStamp>>>,
) {
    let new = events
        .read()
        .map(|&s| (s.center, Vec3::splat(s.bound_radius), s));
    apply_edits(
        &mut deferred,
        new,
        &chunk_manager,
        &mut editable,
        |entity, sdf, origin, voxel_size, stamp| {
            sdf.stamp_sdf_sphere(
                (stamp.center - origin) / voxel_size,
                stamp.bound_radius / voxel_size,
                stamp.sdf_radius / voxel_size,
            );
            commands
                .entity(entity)
                .insert(DirtyField::<SDF, f32>::default());
        },
        |_| {},
    );
}

/// In-flight async JFA recompute for one chunk's SDF.
///
/// `started_version` is the `SDF::version` this task's input was cloned at — used by
/// `resolve_sdf_reinit_tasks` to detect whether a newer edit landed while
/// the task was running.
#[derive(Component)]
pub struct SdfReinitTask {
    task: Task<Box<[f32]>>,
    started_version: u64,
}

/// Dirty SDF chunks with no reinit in flight.
type NeedsReinit = (With<DirtyField<SDF, f32>>, Without<SdfReinitTask>);

/// Spawns a background JFA task for every dirty SDF chunk that doesn't
/// already have one in flight.
///
/// The `Without<SdfReinitTask>` filter is
/// what prevents re-spawning a second task each frame while the sphere
/// is still resizing — `DirtyField` stays set (see resolve, below) but no
/// new task starts until the current one resolves.
pub fn spawn_sdf_reinit_tasks(mut commands: Commands, query: Query<(Entity, &SDF), NeedsReinit>) {
    for (entity, sdf) in query.iter() {
        let (data, size, started_version) = sdf.reinit_input();

        let task = AsyncComputeTaskPool::get().spawn(async move {
            let mut data = data.into_vec();
            crate::field::sdf::compute_sdf_distances(&mut data, size);
            data.into_boxed_slice()
        });

        commands.entity(entity).insert(SdfReinitTask {
            task,
            started_version,
        });
    }
}

/// Runs every frame; non-blocking poll of every in-flight reinit task.
pub fn resolve_sdf_reinit_tasks(
    mut commands: Commands,
    mut query: Query<(Entity, &mut SDF, &mut SdfReinitTask)>,
) {
    for (entity, mut sdf, mut pending) in query.iter_mut() {
        let Some(data) = block_on(poll_once(&mut pending.task)) else {
            continue;
        };

        if sdf.version != pending.started_version {
            // A newer edit landed while this task was computing — its
            // input is stale and applying it would erase that edit.
            // Drop the result; DirtyField is still set, so next frame's
            // spawn_sdf_reinit_tasks picks up the *current* data instead.
            commands.entity(entity).remove::<SdfReinitTask>();
            continue;
        }

        sdf.apply_reinit_result(data);
        commands
            .entity(entity)
            .remove::<SdfReinitTask>()
            .remove::<DirtyField<SDF, f32>>();
    }
}

/// (entity, SDF edited, visibility edited).
type EditedFields = (
    Entity,
    Has<DirtyField<SDF, f32>>,
    Has<DirtyField<VisibilityField, bool>>,
);

/// Chunks with both fields, at least one edited, and no reinit in flight.
type DesyncCandidate = (
    With<SDF>,
    With<VisibilityField>,
    Without<SdfReinitTask>,
    Or<(
        With<DirtyField<SDF, f32>>,
        With<DirtyField<VisibilityField, bool>>,
    )>,
);

pub fn detect_topology_desync(
    query: Query<EditedFields, DesyncCandidate>,
    positions: Query<&ChunkPosition>,
) {
    for (entity, sdf_dirty, vis_dirty) in query.iter() {
        if sdf_dirty != vis_dirty {
            let pos = positions.get(entity).map(|p| p.0);
            warn!(
                "[topology_desync] chunk {:?} (entity {:?}): SDF edited={}, VisibilityField edited={} — \
                 these usually need to change together (a solid-sphere edit that only touched one field \
                 will render geometry that doesn't match its visibility mask, or vice versa)",
                pos, entity, sdf_dirty, vis_dirty
            );
        }
    }
}

/// Ends the edit cycle for a field whose `DirtyField` needs no follow-up.
///
/// That's every field but `SDF`, which keeps its marker until the jump flood
/// lands. Runs in `FieldSet::Reinit`, after every edit of the cycle.
pub fn clear_dirty<F, V>(mut commands: Commands, query: Query<Entity, With<DirtyField<F, V>>>)
where
    F: Field<V>,
    V: Copy + Default + Send + Sync + 'static,
{
    for entity in query.iter() {
        commands.entity(entity).remove::<DirtyField<F, V>>();
    }
}

const SDF_CSG_REACH_VOXELS: f32 = 4.0;

pub fn process_sdf_csg_spheres(
    mut events: MessageReader<SdfCsgSphere>,
    mut editable: EditableChunks<SDF>,
    chunk_manager: Res<ChunkManager>,
    max: Res<MaxEditLod>,
    mut deferred: Local<Vec<DeferredEdit<SdfCsgSphere>>>,
    mut resident: ResMut<ForceResident>,
) {
    let reach_world = SDF_CSG_REACH_VOXELS * max.0.voxel_size();
    let new = events.read().map(|&op| {
        resident.touch_sphere(op.center, op.radius + reach_world);
        (op.center, Vec3::splat(op.radius + reach_world), op)
    });
    apply_edits(
        &mut deferred,
        new,
        &chunk_manager,
        &mut editable,
        |_, sdf, origin, voxel_size, op| {
            // No DirtyField: CSG output is already a valid distance field.
            // `set` bumps version/inside_count, which drives re-extract.
            sdf.csg_sphere(
                (op.center - origin) / voxel_size,
                op.radius / voxel_size,
                SDF_CSG_REACH_VOXELS,
                op.subtract,
            );
        },
        |chunk_pos| warn!("[csg] chunk {chunk_pos:?} never reached MaxEditLod, edit dropped"),
    );
}
