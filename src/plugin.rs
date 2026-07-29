use bevy::prelude::*;

use crate::{
    chunk::ChunkPlugin, field::FieldsPlugin, indirect_draw::VoxelIndirectDrawPlugin,
    voxel_pipeline::VoxelRenderPlugin,
};

pub struct TarasaurPlugin;

impl Plugin for TarasaurPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            ChunkPlugin,
            FieldsPlugin,
            VoxelRenderPlugin,
            VoxelIndirectDrawPlugin,
        ));
    }
}
