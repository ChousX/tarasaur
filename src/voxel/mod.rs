//! GPU rendering and querying of chunk fields.
//!
//! Per frame, in the render world:
//! - **Upload** (`prepare_voxel_arena` + material/visibility siblings): new
//!   padded volumes go into a slot of a per-LOD arena, with a vertex
//!   range sized from an exact CPU count of active cells.
//! - **Dispatch**: surface nets in two compute passes over dirty slots only.
//!   `surface_nets_pass1` finds active cells and assigns vertices;
//!   `surface_nets_pass3` places vertices, blends materials, stitches LOD
//!   seams and writes indices and draw args.
//! - **Draw** (`draw`): CPU frustum culling per view, a compaction pass,
//!   then one `multi_draw_indexed_indirect` per arena (`voxel_raster.wesl`:
//!   triplanar albedo + ARM, per-pixel visibility discard, Bevy PBR
//!   lighting), optionally after a depth-only prepass (`VoxelDepthPrepass`).
//!
//! GPU ray queries against the arenas live in `raycast::gpu`.

pub mod arena;
pub(crate) mod draw;
pub(crate) mod extract;
pub(crate) mod mesh;
pub(crate) mod pipeline;
pub(crate) mod raster;
pub(crate) mod resident;
pub mod stats;
pub(crate) mod telemetry;
pub(crate) mod types;
pub(crate) mod upload;
pub(crate) mod util;

pub use pipeline::VoxelDepthPrepass;
pub use telemetry::GuardTelemetry;
pub use upload::RemeshBudget;

use bevy::{
    asset::{load_internal_asset, uuid_handle},
    core_pipeline::{Core3d, Core3dSystems},
    prelude::*,
    render::{Render, RenderApp, RenderSystems, extract_resource::ExtractResourcePlugin},
};

use extract::{LiveOutbox, add_field_extraction, release_despawned_chunks, track_live_chunks};
use pipeline::{VoxelComputePipeline, VoxelPipelineLayouts};

use crate::{
    field::{GpuChannel, VoxelField},
    voxel::{
        arena::VoxelChunkArenaSet,
        draw::{
            DRAW_COMPACTION_SHADER_HANDLE, VoxelDrawPipeline, VoxelViewDraws,
            prepare_voxel_view_draws,
        },
        extract::extract_removed,
        mesh::dispatch_voxel_compute_passes_batched,
        pipeline::{
            VoxelMaterialBindGroup, VoxelRasterPipeline, queue_voxel_raster_pipelines,
            update_voxel_material_bind_group,
        },
        raster::voxel_raster_pass,
        resident::{ForceResident, lease_chunks_near_camera, tick_force_resident},
        telemetry::map_guard_readbacks,
        upload::{prepare_mask_upload, prepare_material_upload, prepare_shape_upload},
    },
};

pub const SURFACE_NETS_PASS1_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("9f3a1b2c-4d5e-6f70-8192-a3b4c5d6e7f8");
pub const SURFACE_NETS_PASS3_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("2b3c4d5e-6f70-8192-a3b4-c5d6e7f8091a");
pub const VOXEL_RASTER_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("3c4d5e6f-7081-92a3-b4c5-d6e7f8091a2b");
pub const CURSOR_QUERY_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("5e6f7081-92a3-b4c5-d6e7-f8091a2b3c4d");
/// Ordering inside `RenderSystems::Prepare`.
///
/// Bevy runs `Queue` *before* `Prepare`, so the compute dispatch must not live in `Queue`: it would
/// mesh last frame's uploads while this frame's upload had just reset the
/// draw args, and a chunk updated every frame (a held brush) never drew.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum VoxelPrepareSet {
    /// Writes chunk data, ranges and draw args into the arenas.
    Upload,
    /// Re-meshes what `Upload` dirtied, in the same frame.
    Dispatch,
}

/// Order inside `VoxelPrepareSet::Upload`: the shape upload places chunks,
/// then the material and mask uploads follow them into their slots.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum UploadSet {
    Shape,
    Channels,
}

/// Marks that a `GpuChannel::Shape` field is registered (there can be one).
#[derive(Resource)]
struct ShapeFieldRegistered(&'static str);

/// Hands `F` to the render world and uploads it into its `GpuChannel`.
/// Nothing for `GpuChannel::None`.
pub(crate) fn register_field_upload<F: VoxelField>(app: &mut App) {
    if F::GPU == GpuChannel::None {
        return;
    }
    if F::GPU == GpuChannel::Shape {
        let name = std::any::type_name::<F>();
        if let Some(existing) = app.world().get_resource::<ShapeFieldRegistered>() {
            error!(
                "{name} is a second GpuChannel::Shape field ({} is already registered); not uploading it",
                existing.0
            );
            return;
        }
        app.insert_resource(ShapeFieldRegistered(name));
    }
    add_field_extraction::<F>(app);
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    if F::OPTIONAL {
        render_app.add_systems(
            ExtractSchedule,
            extract_removed::<F>.after(release_despawned_chunks),
        );
    }
    match F::GPU {
        GpuChannel::Shape => {
            render_app.add_systems(Render, prepare_shape_upload::<F>.in_set(UploadSet::Shape));
        }
        GpuChannel::Material => {
            render_app.add_systems(
                Render,
                prepare_material_upload::<F>.in_set(UploadSet::Channels),
            );
        }
        GpuChannel::Mask => {
            render_app.add_systems(Render, prepare_mask_upload::<F>.in_set(UploadSet::Channels));
        }
        GpuChannel::None => {}
    }
}

/// Extraction, arenas, meshing, drawing, ray queries and the cursor.
pub struct VoxelRenderPlugin;

/// Embeds the voxel shaders.
fn load_shaders(app: &mut App) {
    load_internal_asset!(
        app,
        DRAW_COMPACTION_SHADER_HANDLE,
        "shaders/draw_compaction.wgsl",
        Shader::from_wgsl
    );
    load_internal_asset!(
        app,
        SURFACE_NETS_PASS1_SHADER_HANDLE,
        "shaders/surface_nets_pass1.wgsl",
        Shader::from_wgsl
    );
    load_internal_asset!(
        app,
        SURFACE_NETS_PASS3_SHADER_HANDLE,
        "shaders/surface_nets_pass3.wgsl",
        Shader::from_wgsl
    );
    load_internal_asset!(
        app,
        VOXEL_RASTER_SHADER_HANDLE,
        "shaders/voxel_raster.wesl",
        Shader::from_wesl
    );
    load_internal_asset!(
        app,
        CURSOR_QUERY_SHADER_HANDLE,
        "shaders/cursor_query.wgsl",
        Shader::from_wgsl
    );
}

impl Plugin for VoxelRenderPlugin {
    fn build(&self, app: &mut App) {
        load_shaders(app);

        app.init_resource::<VoxelDepthPrepass>()
            .add_plugins(ExtractResourcePlugin::<VoxelDepthPrepass>::default())
            .init_resource::<ForceResident>()
            .add_systems(First, tick_force_resident)
            .add_systems(Update, lease_chunks_near_camera)
            .init_resource::<LiveOutbox>()
            .add_systems(Last, track_live_chunks);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app.init_resource::<VoxelViewDraws>();

        render_app
            .init_resource::<VoxelChunkArenaSet>()
            .init_resource::<RemeshBudget>()
            .init_resource::<GuardTelemetry>()
            .add_systems(ExtractSchedule, release_despawned_chunks)
            .configure_sets(
                Render,
                (VoxelPrepareSet::Upload, VoxelPrepareSet::Dispatch)
                    .chain()
                    .in_set(RenderSystems::Prepare),
            )
            .configure_sets(
                Render,
                (UploadSet::Shape, UploadSet::Channels)
                    .chain()
                    .in_set(VoxelPrepareSet::Upload),
            )
            .add_systems(
                Render,
                (
                    update_voxel_material_bind_group.in_set(RenderSystems::Prepare),
                    queue_voxel_raster_pipelines
                        .in_set(RenderSystems::Prepare)
                        .after(update_voxel_material_bind_group),
                    prepare_voxel_view_draws
                        .in_set(RenderSystems::Prepare)
                        .after(VoxelPrepareSet::Upload),
                    dispatch_voxel_compute_passes_batched.in_set(VoxelPrepareSet::Dispatch),
                    map_guard_readbacks.in_set(RenderSystems::Cleanup),
                ),
            )
            .add_systems(Core3d, voxel_raster_pass.in_set(Core3dSystems::MainPass));
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<VoxelRasterPipeline>()
            .init_resource::<VoxelMaterialBindGroup>()
            .init_resource::<VoxelPipelineLayouts>()
            .init_resource::<VoxelComputePipeline>()
            .init_resource::<VoxelDrawPipeline>();
    }
}
