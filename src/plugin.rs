use bevy::prelude::*;

use crate::{
    chunk::ChunkPlugin,
    field::{
        FieldsPlugin, SDF, VisibilityField,
        persistence::{ChunkPersistencePlugin, RegisterSaveableFieldExt},
    },
    loading::ChunkLoadingPlugin,
    surface::SurfaceCullPlugin,
    texture_palette::plugin::PalettePlugin,
    voxel::VoxelRenderPlugin,
};

/// Everything the engine needs: chunks, loading, fields, persistence,
/// rendering and palettes. Add `MaterialFieldPlugin<M>` for materials and
/// register generators in `ChunkGeneratorRegistry` for terrain.
pub struct TarasaurPlugin;

impl Plugin for TarasaurPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            VoxelRenderPlugin,
            ChunkPlugin,
            FieldsPlugin,
            ChunkPersistencePlugin,
            ChunkLoadingPlugin,
            PalettePlugin,
            SurfaceCullPlugin,
        ))
        .register_saveable_field::<SDF>()
        .register_saveable_field::<VisibilityField>();
    }
}
