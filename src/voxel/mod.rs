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
//! - **Draw** ([`draw`]): CPU frustum culling per view, a compaction pass,
//!   then one `multi_draw_indexed_indirect` per arena (`voxel_raster.wesl`:
//!   triplanar albedo + ARM, per-pixel visibility discard, Bevy PBR
//!   lighting), optionally after a depth-only prepass (`VoxelDepthPrepass`).
//! - **Query** ([`query`]): optional GPU ray casts against the arenas, read
//!   back a few frames later.

mod arena;
pub mod cursor;
pub mod draw;
pub mod extract;
pub mod pipeline;
pub mod query;
pub mod resident;
pub mod stats;
pub mod systems;
pub mod terrain_ray;
pub mod types;
pub mod util;

use bevy::{
    asset::{load_internal_asset, uuid_handle},
    core_pipeline::{Core3d, Core3dSystems},
    prelude::*,
    render::{Render, RenderApp, RenderSystems, extract_resource::ExtractResourcePlugin},
};

use extract::{LiveOutbox, add_field_extraction, release_despawned_chunks, track_live_chunks};
use pipeline::{VoxelComputePipeline, VoxelDepthPrepass, VoxelPipelineLayouts};

use crate::{
    SDF, VisibilityField,
    voxel::{
        arena::VoxelChunkArenaSet,
        cursor::CursorPlugin,
        draw::{
            DRAW_COMPACTION_SHADER_HANDLE, VoxelDrawPipeline, VoxelViewDraws,
            prepare_voxel_view_draws,
        },
        pipeline::{
            VoxelMaterialBindGroup, VoxelRasterPipeline, queue_voxel_raster_pipelines,
            update_voxel_material_bind_group,
        },
        query::{
            CollisionLOD, ExtractedVoxelQueries, NextQuerySlot, PendingVoxelQueries,
            VoxelQueryBuffers, VoxelQueryResultChannel, VoxelQueryResultReceiver,
            VoxelQueryResults, clear_pending_voxel_queries, dispatch_voxel_query_pass,
            drain_voxel_query_results, extract_voxel_queries, map_voxel_query_results,
            prepare_voxel_queries,
        },
        resident::{ForceResident, lease_chunks_near_camera, tick_force_resident},
        systems::{
            GuardTelemetry, RemeshBudget, dispatch_voxel_compute_passes_batched,
            extract_visibility_removed, map_guard_readbacks, prepare_visibility_for_arena,
            prepare_voxel_arena, voxel_raster_pass,
        },
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

        let (query_tx, query_rx) = crossbeam_channel::unbounded();
        app.insert_resource(VoxelQueryResultReceiver { receiver: query_rx });
        app.insert_resource(PendingVoxelQueries::default());
        app.insert_resource(VoxelQueryResults::default());
        app.add_systems(
            First,
            (clear_pending_voxel_queries, drain_voxel_query_results),
        );

        app.init_resource::<VoxelDepthPrepass>()
            .add_plugins(ExtractResourcePlugin::<VoxelDepthPrepass>::default())
            .init_resource::<ForceResident>()
            .add_systems(First, tick_force_resident)
            .add_systems(Update, lease_chunks_near_camera)
            .init_resource::<LiveOutbox>()
            .add_systems(Last, track_live_chunks);
        add_field_extraction::<SDF>(app);
        add_field_extraction::<VisibilityField>(app);

        app.add_plugins(CursorPlugin);
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app.insert_resource(VoxelQueryResultChannel { sender: query_tx });
        render_app.init_resource::<CollisionLOD>();
        render_app.init_resource::<VoxelViewDraws>();
        render_app.init_resource::<ExtractedVoxelQueries>();
        render_app.init_resource::<NextQuerySlot>();

        render_app
            .init_resource::<VoxelChunkArenaSet>()
            .init_resource::<RemeshBudget>()
            .init_resource::<GuardTelemetry>()
            .add_systems(
                ExtractSchedule,
                (
                    release_despawned_chunks,
                    (extract_visibility_removed, extract_voxel_queries),
                )
                    .chain(),
            )
            .configure_sets(
                Render,
                (VoxelPrepareSet::Upload, VoxelPrepareSet::Dispatch)
                    .chain()
                    .in_set(RenderSystems::Prepare),
            )
            .add_systems(
                Render,
                (
                    (prepare_voxel_arena, prepare_visibility_for_arena)
                        .chain()
                        .in_set(VoxelPrepareSet::Upload),
                    update_voxel_material_bind_group.in_set(RenderSystems::Prepare),
                    queue_voxel_raster_pipelines
                        .in_set(RenderSystems::Prepare)
                        .after(update_voxel_material_bind_group),
                    prepare_voxel_view_draws
                        .in_set(RenderSystems::Prepare)
                        .after(VoxelPrepareSet::Upload),
                    prepare_voxel_queries
                        .in_set(VoxelPrepareSet::Upload)
                        .after(prepare_visibility_for_arena),
                    dispatch_voxel_compute_passes_batched.in_set(VoxelPrepareSet::Dispatch),
                    dispatch_voxel_query_pass
                        .in_set(VoxelPrepareSet::Dispatch)
                        .after(dispatch_voxel_compute_passes_batched),
                    map_voxel_query_results.in_set(RenderSystems::Cleanup),
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
            .init_resource::<VoxelDrawPipeline>()
            .init_resource::<VoxelQueryBuffers>();
    }
}
