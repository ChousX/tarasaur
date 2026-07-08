// fields/systems.rs
use super::{
    Field, SDFField,
    editor::{EditFieldMessage, EditMode},
    ops::{AccumulateExt, BlendExt, FieldBoxOps, FieldSphereOps},
};
use crate::chunk::{ChunkManager, world_pos_to_chunk_pos};
use bevy::{
    ecs::{component::Mutable, message::MessageReader},
    math::primitives::{Cuboid, Sphere},
    prelude::*,
};

/// Generic system to process spherical edits for any component matching `Field<V>`.
pub fn process_sphere_edits<F, V>(
    mut events: MessageReader<EditFieldMessage<F, Sphere, V>>,
    mut query: Query<&mut F>,
    chunk_manager: Res<ChunkManager>,
) where
    F: Field<V> + Component<Mutability = Mutable>,
    V: Copy + Default + Send + Sync + 'static + AccumulateExt + BlendExt, // Added bounds here
{
    for event in events.read() {
        let EditFieldMessage {
            center,
            shape,
            val,
            mode,
            ..
        } = event;
        let chunk_pos = world_pos_to_chunk_pos(center);
        if let Some(chunk_id) = chunk_manager.get_chunk(&chunk_pos) {
            let Ok(mut field) = query.get_mut(chunk_id) else {
                return;
            };
            match mode {
                EditMode::Absolute => {
                    field.fill_sphere(*center, *shape, *val);
                }
                EditMode::Accumulate { delta } => {
                    field.accumulate_sphere(*center, *shape, *delta);
                }
                EditMode::Blend { rate } => {
                    field.blend_sphere(*center, *shape, *val, *rate);
                }
            }
        }
    }
}

/// Generic system to process cuboid/box edits for any component matching `Field<V>`.
pub fn process_box_edits<F, V>(
    mut events: MessageReader<EditFieldMessage<F, Cuboid, V>>,
    mut query: Query<&mut F>,
    chunk_manager: Res<ChunkManager>,
) where
    F: Field<V> + Component<Mutability = Mutable>,
    V: Copy + Default + Send + Sync + 'static + AccumulateExt + BlendExt, // Added bounds here
{
    for event in events.read() {
        let EditFieldMessage {
            center,
            shape,
            val,
            mode,
            ..
        } = event;
        let chunk_pos = world_pos_to_chunk_pos(center);
        if let Some(chunk_id) = chunk_manager.get_chunk(&chunk_pos) {
            let Ok(mut field) = query.get_mut(chunk_id) else {
                return;
            };

            match *mode {
                EditMode::Absolute => {
                    field.fill_box(*center, *shape, *val);
                }
                EditMode::Accumulate { delta } => {
                    field.accumulate_box(*center, *shape, delta);
                }
                EditMode::Blend { rate } => {
                    field.blend_box(*center, *shape, *val, rate);
                }
            }
        }
    }
}

/// Runs in `FieldSet::Reinit` to cleanup dirty SDF fields after edits have landed.
pub fn reinit_dirty_sdf(mut query: Query<&mut SDFField>) {
    for mut sdf_field in query.iter_mut() {
        if sdf_field.is_dirty() {
            sdf_field.reinit_if_dirty();
        }
    }
}
