use bevy::prelude::*;

use crate::{
    chunk::ChunkPlugin,
    field::{
        FieldsPlugin, SDFField, VisibilityField,
        persistence::{ChunkPersistencePlugin, RegisterSaveableFieldExt},
    },
    voxel::VoxelRenderPlugin,
};

pub struct TarasaurPlugin;

impl Plugin for TarasaurPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            VoxelRenderPlugin,
            ChunkPlugin,
            FieldsPlugin,
            ChunkPersistencePlugin,
        ))
        .register_saveable_field::<SDFField>()
        .register_saveable_field::<VisibilityField>();
    }
}
