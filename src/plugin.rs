use bevy::prelude::*;

use crate::{chunk::ChunkPlugin, field::FieldsPlugin, voxel::VoxelRenderPlugin};

pub struct TarasaurPlugin;

impl Plugin for TarasaurPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((VoxelRenderPlugin, ChunkPlugin, FieldsPlugin));
    }
}
