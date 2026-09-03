// fields/systems.rs
use super::{
    Field, SDFField,
    editor::{EditFieldMessage, EditMode},
    ops::{AccumulateExt, BlendExt, FieldBoxOps, FieldSphereOps, ShapeBounds, ShapeScale},
};
use crate::{
    CHUNK_SIZE, DirtyField,
    chunk::{ChunkManager, world_pos_to_chunk_pos},
};
use bevy::{
    ecs::{component::Mutable, message::MessageReader},
    math::primitives::{Cuboid, Sphere},
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

pub fn process_sphere_edits<F, V>(
    mut commands: Commands,
    mut events: MessageReader<EditFieldMessage<F, Sphere, V>>,
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

            // `field`'s storage is a `size()`-voxel grid regardless of
            // CHUNK_SIZE — apply_sphere/apply_box only ever see raw voxel
            // indices, so world-space units must be rescaled here.
            let voxel_size = CHUNK_SIZE / field.size().x as f32;
            let grid_center = local_center / voxel_size;
            let grid_shape = shape.scaled_by(1.0 / voxel_size);

            match mode {
                EditMode::Absolute => field.fill_sphere(grid_center, grid_shape, *val),
                EditMode::Accumulate { delta } => {
                    field.accumulate_sphere(grid_center, grid_shape, *delta)
                }
                EditMode::Blend { rate } => {
                    field.blend_sphere(grid_center, grid_shape, *val, *rate)
                }
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

pub fn reinit_dirty_sdf(
    mut commands: Commands,
    mut query: Query<(Entity, &mut SDFField), With<DirtyField<SDFField, f32>>>,
) {
    for (entity, mut sdf) in query.iter_mut() {
        sdf.reinit();
        commands
            .entity(entity)
            .remove::<DirtyField<SDFField, f32>>();
    }
}
