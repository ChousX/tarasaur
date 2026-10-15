//! Where the layers meet: `TarasaurPlugin`, and `add_voxel_field`, which
//! wires one field type into editing, saving and rendering.

use std::marker::PhantomData;

use bevy::prelude::*;

use crate::{
    chunk::ChunkPlugin,
    field::{
        Field, FieldsPlugin, MaterialField, SDF, VisibilityField, VoxelField, VoxelMaterial,
        register_edits, surface::SurfaceCullPlugin,
    },
    storage::{
        loading::ChunkLoadingPlugin,
        persistence::{ChunkPersistencePlugin, register_saveable},
    },
    texture_palette::plugin::PalettePlugin,
    voxel::{VoxelRenderPlugin, register_field_upload},
};

/// Everything the engine needs, with `SDF` and `VisibilityField` registered.
///
/// That's chunks, loading, fields, persistence, rendering and palettes. Add
/// `MaterialFieldPlugin<M>` for materials and register generators in
/// `ChunkGeneratorRegistry` for terrain.
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
        .add_voxel_field::<SDF>()
        .add_voxel_field::<VisibilityField>();
    }
}

/// Registers a field type with every layer.
pub trait AppVoxelFieldExt {
    /// Wires `F` in: its edit messages and systems (`WorldEditor<F, _>`), a
    /// default on new chunks, saving and loading, and, unless its
    /// `GpuChannel` is `None`, extraction and upload into that channel.
    fn add_voxel_field<F>(&mut self) -> &mut Self
    where
        F: VoxelField + Field<F::Value>;
}

impl AppVoxelFieldExt for App {
    fn add_voxel_field<F>(&mut self) -> &mut Self
    where
        F: VoxelField + Field<F::Value>,
    {
        register_edits::<F>(self);
        register_saveable::<F>(self);
        register_field_upload::<F>(self);
        self
    }
}

/// Registers a material channel: `MaterialField<M>` for your material enum.
pub struct MaterialFieldPlugin<M: VoxelMaterial>(PhantomData<M>);

impl<M: VoxelMaterial> MaterialFieldPlugin<M> {
    #[must_use]
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<M: VoxelMaterial> Default for MaterialFieldPlugin<M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<M: VoxelMaterial> Plugin for MaterialFieldPlugin<M> {
    fn build(&self, app: &mut App) {
        app.add_voxel_field::<MaterialField<M>>();
    }
}

#[cfg(test)]
mod tests {
    use bevy::prelude::*;
    use bevy::shape::Sphere;

    use super::MaterialFieldPlugin;
    use crate::{
        chunk::{Chunk, ChunkManager, ChunkPosition},
        field::{
            DirtyField, FieldsPlugin, LOD, MaterialField, VoxelField, VoxelMaterial,
            editor::{EditFieldMessage, EditMode},
            ops::{AccumulateExt, BlendExt},
        },
        storage::persistence::ChunkPersistencePlugin,
        voxel::resident::ForceResident,
    };

    #[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
    enum Mat {
        #[default]
        A,
        B,
    }

    impl VoxelMaterial for Mat {
        const COUNT: u8 = 2;
        fn to_id(self) -> u8 {
            match self {
                Self::A => 0,
                Self::B => 1,
            }
        }
        fn from_id(id: u8) -> Self {
            if id == 1 { Self::B } else { Self::A }
        }
    }

    impl AccumulateExt for Mat {
        fn accumulate(self, delta: Self) -> Self {
            delta
        }
    }

    impl BlendExt for Mat {
        fn blend_towards(self, target: Self, factor: f32) -> Self {
            if factor >= 0.5 { target } else { self }
        }
    }

    #[test]
    fn material_edits_dont_leave_a_dirty_marker() {
        let mut app = App::new();
        app.add_plugins((
            MinimalPlugins,
            ChunkPersistencePlugin,
            FieldsPlugin,
            MaterialFieldPlugin::<Mat>::new(),
        ))
        .init_resource::<ChunkManager>()
        .init_resource::<ForceResident>();
        let chunk = app
            .world_mut()
            .spawn((
                Chunk,
                ChunkPosition(IVec3::ZERO),
                LOD::High,
                MaterialField::<Mat>::new(LOD::High),
            ))
            .id();
        app.world_mut()
            .write_message(EditFieldMessage::<MaterialField<Mat>, Sphere, Mat>::new(
                Vec3::splat(5.0),
                Sphere { radius: 1.0 },
                Mat::B,
                EditMode::Absolute,
            ));
        app.update();

        let world = app.world();
        let edited = world
            .get::<MaterialField<Mat>>(chunk)
            .is_some_and(|f| f.version() > 0);
        assert!(edited, "the edit should have landed this frame");
        assert!(
            world
                .get::<DirtyField<MaterialField<Mat>, Mat>>(chunk)
                .is_none()
        );
    }
}
