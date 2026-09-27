use bevy::prelude::*;

use crate::{
    chunk::ChunkPlugin,
    field::{
        FieldsPlugin, SDF, VisibilityField,
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
        .register_saveable_field::<SDF>()
        .register_saveable_field::<VisibilityField>();
    }
}
