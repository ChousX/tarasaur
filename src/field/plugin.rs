//! Registration: `FieldsPlugin` (SDF + visibility, always on) and
//! `MaterialFieldPlugin<M>` (one material channel, opt-in).

use std::marker::PhantomData;

use bevy::ecs::component::Mutable;
use bevy::prelude::*;
use bevy::render::{Render, RenderApp};
use bevy::shape::{Cuboid, Sphere};

use crate::chunk::NewChunkSpawned;
use crate::editor::{SdfCsgSphere, SdfSphereStamp};
use crate::field::material::VoxelMaterial;
use crate::field::{MaterialField, SDF, VisibilityField};
use crate::persistence::RegisterSaveableFieldExt;
use crate::systems::{
    clear_dirty, detect_topology_desync, process_sdf_csg_spheres, process_sdf_sphere_stamps,
    resolve_sdf_reinit_tasks, spawn_sdf_reinit_tasks,
};
use crate::voxel::{
    extract::add_field_extraction,
    systems::{prepare_material_for_arena, prepare_voxel_arena},
};
use crate::{FieldLOD, FieldNew, LOD, TargetLOD};

use super::{
    Field,
    editor::EditFieldMessage,
    ops::{AccumulateExt, BlendExt},
    systems::process_shape_edits,
};

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum FieldSet {
    /// All edit-producing and edit-applying systems.
    Edit,
    /// Runs once per cycle, after every edit system has run.
    Reinit,
}

/// Registers a single material channel (`MaterialField<M>`) with the app.
///
/// That is its edit messages/systems, and the observer that attaches it to
/// new chunks. Add one of these per material enum you use; most projects need
/// exactly one, but nothing stops registering a second channel (e.g. a
/// separate cave-stratum material) alongside it.
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
        app.add_field::<MaterialField<M>, M>();
        app.add_systems(
            Update,
            clear_dirty::<MaterialField<M>, M>.in_set(FieldSet::Reinit),
        );
        app.add_observer(build_field_on_chunk_spawn::<MaterialField<M>>);
        app.register_saveable_field::<MaterialField<M>>();
        add_field_extraction::<MaterialField<M>>(app);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app.add_systems(
            Render,
            prepare_material_for_arena::<M>
                .after(prepare_voxel_arena)
                .in_set(crate::voxel::VoxelPrepareSet::Upload),
        );
    }
}

/// Registers the fields every chunk always has: SDF (topology) and
/// visibility (cull mask). Material is opt-in via `MaterialFieldPlugin<M>`
/// since only the end user knows their material enum.
pub struct FieldsPlugin;

impl Plugin for FieldsPlugin {
    fn build(&self, app: &mut App) {
        app.configure_sets(Update, (FieldSet::Edit, FieldSet::Reinit).chain());
        app.add_systems(
            Update,
            (
                detect_topology_desync,
                spawn_sdf_reinit_tasks,
                resolve_sdf_reinit_tasks,
                clear_dirty::<VisibilityField, bool>,
            )
                .chain()
                .in_set(FieldSet::Reinit),
        );

        app.add_field::<SDF, f32>()
            .add_field::<VisibilityField, bool>();

        app.add_observer(build_field_on_chunk_spawn::<SDF>)
            .add_observer(build_field_on_chunk_spawn::<VisibilityField>);

        app.add_message::<SdfSphereStamp>()
            .add_systems(Update, process_sdf_sphere_stamps.in_set(FieldSet::Edit));

        app.add_message::<SdfCsgSphere>()
            .add_systems(Update, process_sdf_csg_spheres);
    }
}

/// Extension trait to add custom fluent registration APIs onto the Bevy App builder.
pub trait AppFieldExt {
    fn add_field<F, V>(&mut self) -> &mut Self
    where
        F: Field<V> + FieldLOD + FieldNew + Component<Mutability = Mutable>,
        V: Copy + Default + Send + Sync + 'static + AccumulateExt + BlendExt;
}

impl AppFieldExt for App {
    fn add_field<F, V>(&mut self) -> &mut Self
    where
        F: Field<V> + FieldLOD + FieldNew + Component<Mutability = Mutable>,
        V: Copy + Default + Send + Sync + 'static + AccumulateExt + BlendExt,
    {
        self.add_message::<EditFieldMessage<F, Sphere, V>>()
            .add_message::<EditFieldMessage<F, Cuboid, V>>();

        self.add_systems(
            Update,
            (
                process_shape_edits::<F, Sphere, V>,
                process_shape_edits::<F, Cuboid, V>,
            )
                .in_set(FieldSet::Edit),
        );

        self
    }
}

/// Gives a newly spawned chunk a default `F` at its LOD, unless it already
/// has one.
///
/// Chunks spawned with a `TargetLOD` and no `LOD` (the `ChunkLoader`'s) are
/// skipped: their load installs every field, so a default would only be
/// built to be thrown away. Until then readers treat them as unloaded.
/// Chunks spawned by hand, with an `LOD`, still get defaults to fill in.
fn build_field_on_chunk_spawn<F: Component + FieldNew>(
    trigger: On<NewChunkSpawned>,
    mut commands: Commands,
    chunks: Query<(Option<&LOD>, Has<TargetLOD>)>,
    has_field: Query<(), With<F>>,
) {
    let entity = trigger.event().entity;
    if has_field.contains(entity) {
        return;
    }
    let Ok((lod, awaiting_load)) = chunks.get(entity) else {
        return;
    };
    if lod.is_none() && awaiting_load {
        return;
    }
    let lod = lod.copied().unwrap_or_default();
    commands.entity(entity).insert(F::new_for_lod(lod));
}

#[cfg(test)]
mod tests {
    use bevy::prelude::*;
    use bevy::shape::Sphere;

    use super::{FieldsPlugin, MaterialFieldPlugin};
    use crate::{
        DirtyField, LOD, MaterialField, Versionable, VoxelMaterial,
        chunk::{Chunk, ChunkManager, ChunkPosition},
        editor::{EditFieldMessage, EditMode},
        ops::{AccumulateExt, BlendExt},
        persistence::ChunkPersistencePlugin,
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
