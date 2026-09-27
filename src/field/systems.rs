// fields/systems.rs
use super::{
    Field, SDF,
    editor::{EditFieldMessage, EditMode},
    ops::{AccumulateExt, BlendExt, FieldBoxOps, ShapeBounds, ShapeScale},
};
use crate::{
    CHUNK_SIZE, ChunkPosition, DirtyField, VisibilityField,
    chunk::{ChunkManager, world_pos_to_chunk_pos},
    editor::SdfSphereStamp,
    ops::{FieldSphereOps, ShapeEditOps},
};
use bevy::{
    ecs::{component::Mutable, message::MessageReader},
    math::primitives::Cuboid,
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

pub fn process_shape_edits<F, S, V>(
    mut commands: Commands,
    mut events: MessageReader<EditFieldMessage<F, S, V>>,
    mut query: Query<&mut F>,
    chunk_manager: Res<ChunkManager>,
) where
    F: Field<V> + Component<Mutability = Mutable> + ShapeEditOps<V, S>,
    S: Primitive3d + Copy + Send + Sync + 'static + ShapeBounds + ShapeScale,
    V: Copy + Default + Send + Sync + 'static + AccumulateExt + BlendExt,
{
    for event in events.read() {
        info!(
            "[process_shape_edits] got an edit message for {}",
            std::any::type_name::<F>()
        );
        let EditFieldMessage {
            center,
            shape,
            val,
            mode,
            ..
        } = event;
        // ^ event is &EditFieldMessage<..> here — do NOT write `= *event`.
        // Match ergonomics gives center: &Vec3, shape: &S, val: &V for free.

        for chunk_pos in overlapping_chunks(*center, shape.half_extent()) {
            let Some(entity) = chunk_manager.get_chunk(&chunk_pos) else {
                warn!(
                    "[process_shape_edits] no chunk at {:?} for {}",
                    chunk_pos,
                    std::any::type_name::<F>()
                );
                continue;
            };
            let Ok(mut field) = query.get_mut(entity) else {
                info!(
                    "[process_shape_edits] resolved chunk_pos {:?} -> entity {:?} for {}",
                    chunk_pos,
                    entity,
                    std::any::type_name::<F>()
                );
                continue;
            };

            let chunk_world_origin = chunk_pos.as_vec3() * CHUNK_SIZE;
            let voxel_size = CHUNK_SIZE / field.size().x as f32;
            let grid_center = (*center - chunk_world_origin) / voxel_size;
            let grid_shape = shape.scaled_by(1.0 / voxel_size);

            match mode {
                EditMode::Absolute => field.fill(grid_center, grid_shape, *val),
                EditMode::Accumulate { delta } => field.accumulate(grid_center, grid_shape, *delta),
                EditMode::Blend { rate } => field.blend(grid_center, grid_shape, *val, *rate),
            }

            commands
                .entity(entity)
                .insert(DirtyField::<F, V>::default());
        }
    }
}

pub fn process_box_edits<F, V>(
    mut commands: Commands,
    mut events: MessageReader<EditFieldMessage<F, Cuboid, V>>,
    mut query: Query<&mut F>,
    chunk_manager: Res<ChunkManager>,
) where
    F: Field<V> + Component<Mutability = Mutable>,
    V: Copy + Default + Send + Sync + 'static + AccumulateExt + BlendExt,
{
    for event in events.read() {
        let EditFieldMessage {
            center,
            shape,
            val,
            mode,
            ..
        } = event;

        for chunk_pos in overlapping_chunks(*center, shape.half_extent()) {
            let Some(entity) = chunk_manager.get_chunk(&chunk_pos) else {
                continue;
            };
            let Ok(mut field) = query.get_mut(entity) else {
                continue;
            };

            let chunk_world_origin = chunk_pos.as_vec3() * CHUNK_SIZE;
            let local_center = *center - chunk_world_origin;

            let voxel_size = CHUNK_SIZE / field.size().x as f32;
            let grid_center = local_center / voxel_size;
            let grid_shape = shape.scaled_by(1.0 / voxel_size);

            match mode {
                EditMode::Absolute => field.fill_box(grid_center, grid_shape, *val),
                EditMode::Accumulate { delta } => {
                    field.accumulate_box(grid_center, grid_shape, *delta)
                }
                EditMode::Blend { rate } => field.blend_box(grid_center, grid_shape, *val, *rate),
            }

            commands
                .entity(entity)
                .insert(DirtyField::<F, V>::default());
        }
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

pub fn process_sdf_sphere_stamps(
    mut commands: Commands,
    mut events: MessageReader<SdfSphereStamp>,
    mut query: Query<&mut SDF>,
    chunk_manager: Res<ChunkManager>,
) {
    for event in events.read() {
        let SdfSphereStamp {
            center,
            bound_radius,
            sdf_radius,
        } = *event;

        for chunk_pos in overlapping_chunks(center, Vec3::splat(bound_radius)) {
            let Some(entity) = chunk_manager.get_chunk(&chunk_pos) else {
                continue;
            };
            let Ok(mut sdf) = query.get_mut(entity) else {
                continue;
            };

            let chunk_world_origin = chunk_pos.as_vec3() * CHUNK_SIZE;
            let voxel_size = CHUNK_SIZE / sdf.size().x as f32;
            let grid_center = (center - chunk_world_origin) / voxel_size;

            sdf.stamp_sdf_sphere(
                grid_center,
                bound_radius / voxel_size,
                sdf_radius / voxel_size,
            );

            commands
                .entity(entity)
                .insert(DirtyField::<SDF, f32>::default());
        }
    }
}
