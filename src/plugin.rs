//! Where the layers meet: `TarasaurPlugin`, and `add_voxel_field`, which
//! wires one field type into editing, saving and rendering.

use std::{marker::PhantomData, path::PathBuf};

use bevy::{prelude::*, render::RenderApp, shape::Sphere};

use crate::{
    chunk::{ChunkPlugin, EditPinTimeout, MaxChunkOpsPerFrame},
    field::{
        Field, FieldSet, FieldsPlugin, LOD, MaterialField, MaxEditLod, SDF, VisibilityField,
        VoxelField, VoxelMaterial, register_edits,
        surface::SurfaceCullPlugin,
        systems::{
            apply_material_paint_voxels, forward_material_paint, process_sdf_csg_spheres,
            process_shape_edits,
        },
    },
    raycast::RaycastPlugin,
    storage::{
        loading::{ChunkLoadingPlugin, MaxInFlightLoads},
        persistence::{
            AutosaveIntervalSecs, ChunkPersistencePlugin, SaveDirectory, register_saveable,
        },
    },
    texture_palette::plugin::PalettePlugin,
    voxel::{
        ArenaConfig, GuardTelemetry, RemeshBudget, VoxelDepthPrepass, VoxelRenderPlugin,
        arena::VoxelChunkArenaSet, register_field_upload,
    },
};

/// Everything the engine needs, with `SDF` and `VisibilityField` registered.
///
/// That's chunks, loading, fields, persistence, rendering and palettes. Add
/// `MaterialFieldPlugin<M>` for materials and register generators in
/// `ChunkGeneratorRegistry` for terrain.
///
/// The fields are the engine's settings; `TarasaurPlugin::default()` uses
/// the defaults below. Each becomes a resource (render-world ones in the
/// render app), and the main-world ones can still be changed at runtime.
#[derive(Clone, Debug)]
pub struct TarasaurPlugin {
    /// The only LOD at which chunks are edited and saved (`MaxEditLod`).
    /// Default `High`.
    pub max_edit_lod: LOD,
    /// Where chunk saves are read and written (`SaveDirectory`). Default
    /// `./saves/chunks`.
    pub save_dir: PathBuf,
    /// Seconds between autosaves (`AutosaveIntervalSecs`). Default 2.
    pub autosave_secs: f32,
    /// Seconds an edited chunk stays pinned at max LOD after its last edit
    /// (`EditPinTimeout`). Default 2.
    pub edit_pin_secs: f32,
    /// Chunk spawns and LOD retargets applied per frame
    /// (`MaxChunkOpsPerFrame`). Default 64.
    pub max_chunk_ops_per_frame: usize,
    /// Chunk loads running at once (`MaxInFlightLoads`). Default 64.
    pub max_loads_in_flight: usize,
    /// Cells re-meshed per frame; past it, chunks keep their old mesh
    /// (`RemeshBudget`). Default 4M.
    pub remesh_cells_per_frame: u32,
    /// GPU arena sizes and counts per LOD. Render world, fixed once arenas
    /// exist.
    pub arenas: ArenaConfig,
    /// Draw terrain depth before shading it (`VoxelDepthPrepass`). Default
    /// off; it's slower on tile-based GPUs.
    pub depth_prepass: bool,
    /// Read back and log pass3's range guard (`GuardTelemetry`). Default:
    /// on in debug builds.
    pub guard_telemetry: bool,
}

impl Default for TarasaurPlugin {
    fn default() -> Self {
        Self {
            max_edit_lod: MaxEditLod::default().0,
            save_dir: SaveDirectory::default().0,
            autosave_secs: AutosaveIntervalSecs::default().0,
            edit_pin_secs: EditPinTimeout::default().0,
            max_chunk_ops_per_frame: MaxChunkOpsPerFrame::default().0,
            max_loads_in_flight: MaxInFlightLoads::default().0,
            remesh_cells_per_frame: RemeshBudget::default().cells_per_frame,
            arenas: ArenaConfig::default(),
            depth_prepass: VoxelDepthPrepass::default().0,
            guard_telemetry: GuardTelemetry::default().0,
        }
    }
}

impl TarasaurPlugin {
    /// Inserts the settings, over the defaults the sub-plugins installed.
    fn apply_settings(&self, app: &mut App) {
        app.insert_resource(MaxEditLod(self.max_edit_lod))
            .insert_resource(SaveDirectory(self.save_dir.clone()))
            .insert_resource(AutosaveIntervalSecs(self.autosave_secs))
            .insert_resource(EditPinTimeout(self.edit_pin_secs))
            .insert_resource(MaxChunkOpsPerFrame(self.max_chunk_ops_per_frame))
            .insert_resource(MaxInFlightLoads(self.max_loads_in_flight))
            .insert_resource(VoxelDepthPrepass(self.depth_prepass));
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .insert_resource(RemeshBudget {
                cells_per_frame: self.remesh_cells_per_frame,
            })
            .insert_resource(GuardTelemetry(self.guard_telemetry));
        render_app
            .world_mut()
            .resource_mut::<VoxelChunkArenaSet>()
            .config = self.arenas.clone();
    }
}

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
            RaycastPlugin,
        ))
        .add_voxel_field::<SDF>()
        .add_voxel_field::<VisibilityField>();
        self.apply_settings(app);
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
        app.add_voxel_field::<MaterialField<M>>().add_systems(
            Update,
            (
                forward_material_paint::<M>
                    .before(process_shape_edits::<MaterialField<M>, Sphere, M>),
                apply_material_paint_voxels::<M>.after(process_sdf_csg_spheres),
            )
                .in_set(FieldSet::Edit),
        );
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
            DirtyField, Field, FieldsPlugin, LOD, MaterialField, VoxelField, VoxelMaterial,
            editor::{EditFieldMessage, EditMode, MaterialPaint},
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

    #[test]
    fn material_paint_reaches_the_field() {
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
        app.world_mut().write_message(MaterialPaint {
            center: Vec3::splat(5.0),
            radius: 1.0,
            material: 1,
        });
        app.update();

        // World 5.0 is voxel 32 of a 10-unit, 64-voxel chunk.
        let field = app.world().get::<MaterialField<Mat>>(chunk);
        assert_eq!(field.map(|f| f.get(32, 32, 32)), Some(Mat::B));
        assert_eq!(field.map(|f| f.get(0, 0, 0)), Some(Mat::A));
    }
}
