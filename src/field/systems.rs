// fields/systems.rs
use super::{EditableChunks, FieldNew};
use super::{
    Field, SDF,
    editor::{EditFieldMessage, EditMode},
    ops::{AccumulateExt, BlendExt, ShapeBounds, ShapeScale},
};
use crate::{
    CHUNK_SIZE, ChunkPosition, DirtyField, VisibilityField,
    chunk::{ChunkManager, world_pos_to_chunk_pos},
    editor::SdfSphereStamp,
    ops::{FieldSphereOps, ShapeEditOps},
};
use bevy::{
    ecs::{component::Mutable, message::MessageReader},
    prelude::*,
};

fn overlapping_chunks(center: Vec3, half_extent: Vec3) -> impl Iterator<Item = IVec3> {
    let min_chunk = world_pos_to_chunk_pos(&(center - half_extent));
    let max_chunk = world_pos_to_chunk_pos(&(center + half_extent));

    (min_chunk.x..=max_chunk.x).flat_map(move |x| {
        (min_chunk.y..=max_chunk.y)
            .flat_map(move |y| (min_chunk.z..=max_chunk.z).map(move |z| IVec3::new(x, y, z)))
    })
}

const MAX_DEFER_FRAMES: u32 = 600;
const MAX_DEFERRED: usize = 4096;

pub struct DeferredEdit<S, V> {
    chunk_pos: IVec3,
    center: Vec3,
    shape: S,
    val: V,
    mode: EditMode<V>,
    age: u32,
}

pub fn process_shape_edits<F, S, V>(
    mut commands: Commands,
    mut events: MessageReader<EditFieldMessage<F, S, V>>,
    mut editable: EditableChunks<F>,
    chunk_manager: Res<ChunkManager>,
    mut deferred: Local<Vec<DeferredEdit<S, V>>>,
) where
    F: Field<V> + Component<Mutability = Mutable> + ShapeEditOps<V, S> + FieldNew,
    S: Primitive3d + Copy + Send + Sync + 'static + ShapeBounds + ShapeScale,
    V: Copy + Default + Send + Sync + 'static + AccumulateExt + BlendExt,
{
    // Retries first (oldest first), then this frame's new edits.
    let mut work = std::mem::take(&mut *deferred);
    for ev in events.read() {
        for chunk_pos in overlapping_chunks(ev.center, ev.shape.half_extent()) {
            work.push(DeferredEdit {
                chunk_pos,
                center: ev.center,
                shape: ev.shape,
                val: ev.val,
                mode: ev.mode,
                age: 0,
            });
        }
    }

    for mut e in work {
        let Some(entity) = chunk_manager.get_chunk(&e.chunk_pos) else {
            continue; // chunk not loaded / gone: nothing to edit
        };
        match editable.get_mut(entity) {
            Some(mut field) => {
                let origin = e.chunk_pos.as_vec3() * CHUNK_SIZE;
                let voxel_size = CHUNK_SIZE / field.size().x as f32;
                let grid_center = (e.center - origin) / voxel_size;
                let grid_shape = e.shape.scaled_by(1.0 / voxel_size);
                match e.mode {
                    EditMode::Absolute => field.fill(grid_center, grid_shape, e.val),
                    EditMode::Accumulate { delta } => {
                        field.accumulate(grid_center, grid_shape, delta)
                    }
                    EditMode::Blend { rate } => field.blend(grid_center, grid_shape, e.val, rate),
                }
                commands
                    .entity(entity)
                    .insert(DirtyField::<F, V>::default());
            }
            None => {
                e.age += 1; // waiting for the chunk to reach max LOD
                if e.age < MAX_DEFER_FRAMES {
                    deferred.push(e);
                }
            }
        }
    }
    if deferred.len() > MAX_DEFERRED {
        let excess = deferred.len() - MAX_DEFERRED;
        deferred.drain(..excess); // drop the oldest
    }
}

pub fn process_sdf_sphere_stamps(
    mut commands: Commands,
    mut events: MessageReader<SdfSphereStamp>,
    mut editable: EditableChunks<SDF>,
    chunk_manager: Res<ChunkManager>,
    mut deferred: Local<Vec<(IVec3, SdfSphereStamp, u32)>>,
) {
    let mut work = std::mem::take(&mut *deferred);
    for &stamp in events.read() {
        for p in overlapping_chunks(stamp.center, Vec3::splat(stamp.bound_radius)) {
            work.push((p, stamp, 0));
        }
    }
    for (chunk_pos, stamp, age) in work {
        let Some(entity) = chunk_manager.get_chunk(&chunk_pos) else {
            continue;
        };
        match editable.get_mut(entity) {
            Some(mut sdf) => {
                let origin = chunk_pos.as_vec3() * CHUNK_SIZE;
                let voxel_size = CHUNK_SIZE / sdf.size().x as f32;
                sdf.stamp_sdf_sphere(
                    (stamp.center - origin) / voxel_size,
                    stamp.bound_radius / voxel_size,
                    stamp.sdf_radius / voxel_size,
                );
                commands
                    .entity(entity)
                    .insert(DirtyField::<SDF, f32>::default());
            }
            None if age + 1 < MAX_DEFER_FRAMES => deferred.push((chunk_pos, stamp, age + 1)),
            None => {}
        }
    }
    if deferred.len() > MAX_DEFERRED {
        let excess = deferred.len() - MAX_DEFERRED;
        deferred.drain(..excess);
    }
}

use super::sdf::{PackedCoord, jump_flood_distance_field};
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, poll_once};

/// In-flight async JFA recompute for one chunk's SDF. `started_version`
/// is the SDFField::version this task's input was cloned at — used by
/// resolve_sdf_reinit_tasks to detect whether a newer edit landed while
/// the task was running.
#[derive(Component)]
pub struct SdfReinitTask {
    task: Task<Box<[f32]>>,
    started_version: u64,
}

/// Spawns a background JFA task for every dirty SDF chunk that doesn't
/// already have one in flight. The `Without<SdfReinitTask>` filter is
/// what prevents re-spawning a second task each frame while the sphere
/// is still resizing — DirtyField stays set (see resolve, below) but no
/// new task starts until the current one resolves.
pub fn spawn_sdf_reinit_tasks(
    mut commands: Commands,
    query: Query<(Entity, &SDF), (With<DirtyField<SDF, f32>>, Without<SdfReinitTask>)>,
) {
    for (entity, sdf) in query.iter() {
        let (data, size, started_version) = sdf.reinit_input();

        let task = AsyncComputeTaskPool::get().spawn(async move {
            let volume = data.len();
            let mut data = data.into_vec();
            let mut seeds = vec![PackedCoord::EMPTY; volume];
            let mut scratch = vec![PackedCoord::EMPTY; volume];
            jump_flood_distance_field(&mut data, &mut seeds, &mut scratch, size);
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
pub fn _reinit_dirty_sdf(
    mut commands: Commands,
    mut query: Query<(Entity, &mut SDF), With<DirtyField<SDF, f32>>>,
) {
    for (entity, mut sdf) in query.iter_mut() {
        sdf.reinit();
        commands.entity(entity).remove::<DirtyField<SDF, f32>>();
        info!("[reinit_dirty_sdf] reinitializing entity {:?}", entity);
    }
}

pub fn detect_topology_desync(
    query: Query<
        (
            Entity,
            Has<DirtyField<SDF, f32>>,
            Has<DirtyField<VisibilityField, bool>>,
        ),
        (
            With<SDF>,
            With<VisibilityField>,
            Without<SdfReinitTask>,
            Or<(
                With<DirtyField<SDF, f32>>,
                With<DirtyField<VisibilityField, bool>>,
            )>,
        ),
    >,
    positions: Query<&ChunkPosition>,
) {
    for (entity, sdf_dirty, vis_dirty) in query.iter() {
        if sdf_dirty != vis_dirty {
            let pos = positions.get(entity).map(|p| p.0);
            warn!(
                "[topology_desync] chunk {:?} (entity {:?}): SDFField edited={}, VisibilityField edited={} — \
                 these usually need to change together (a solid-sphere edit that only touched one field \
                 will render geometry that doesn't match its visibility mask, or vice versa)",
                pos, entity, sdf_dirty, vis_dirty
            );
        }
    }
}

pub fn clear_dirty_visibility(
    mut commands: Commands,
    query: Query<Entity, With<DirtyField<VisibilityField, bool>>>,
) {
    for entity in query.iter() {
        commands
            .entity(entity)
            .remove::<DirtyField<VisibilityField, bool>>();
    }
}
