//! Ray casts against the terrain: the CPU `TerrainRay` (same frame, any
//! LOD), GPU ray queries against the arenas (read back a few frames
//! later), and the cursor built on them.

pub mod cursor;
pub mod gpu;
pub mod terrain_ray;

use bevy::{
    prelude::*,
    render::{ExtractSchedule, Render, RenderApp, RenderSystems},
};

use crate::{
    chunk::ChunkSet,
    field::FieldSet,
    voxel::{UploadSet, VoxelPrepareSet, mesh::dispatch_voxel_compute_passes_batched},
};
use cursor::{
    CursorBrush, CursorHit, CursorQueryMode, cursor_brush, cycle_cursor_mode, draw_cursor_brush,
    resolve_cursor_hit, update_cursor_focus,
};
use gpu::{
    ExtractedVoxelQueries, NextQuerySlot, PendingVoxelQueries, VoxelQueryBuffers,
    VoxelQueryResultChannel, VoxelQueryResultReceiver, VoxelQueryResults,
    clear_pending_voxel_queries, dispatch_voxel_query_pass, drain_voxel_query_results,
    extract_voxel_queries, map_voxel_query_results, prepare_voxel_queries,
};

/// The GPU query pipeline, the cursor hit and its chunk focus, and the
/// default brush.
pub struct RaycastPlugin;

impl Plugin for RaycastPlugin {
    fn build(&self, app: &mut App) {
        let (query_tx, query_rx) = crossbeam_channel::unbounded();
        app.insert_resource(VoxelQueryResultReceiver { receiver: query_rx })
            .init_resource::<PendingVoxelQueries>()
            .init_resource::<VoxelQueryResults>()
            .init_resource::<CursorQueryMode>()
            .init_resource::<CursorHit>()
            .init_resource::<CursorBrush>()
            .add_systems(
                First,
                (clear_pending_voxel_queries, drain_voxel_query_results),
            )
            .add_systems(
                Update,
                (
                    cycle_cursor_mode,
                    resolve_cursor_hit.after(cycle_cursor_mode),
                    update_cursor_focus
                        .after(resolve_cursor_hit)
                        .before(ChunkSet::Plan),
                    (cursor_brush, draw_cursor_brush)
                        .after(resolve_cursor_hit)
                        .in_set(FieldSet::Brush)
                        .run_if(|b: Res<CursorBrush>| b.enabled),
                ),
            );

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .insert_resource(VoxelQueryResultChannel { sender: query_tx })
            .init_resource::<ExtractedVoxelQueries>()
            .init_resource::<NextQuerySlot>()
            .add_systems(ExtractSchedule, extract_voxel_queries)
            .add_systems(
                Render,
                (
                    prepare_voxel_queries
                        .in_set(VoxelPrepareSet::Upload)
                        .after(UploadSet::Channels),
                    dispatch_voxel_query_pass
                        .in_set(VoxelPrepareSet::Dispatch)
                        .after(dispatch_voxel_compute_passes_batched),
                    map_voxel_query_results.in_set(RenderSystems::Cleanup),
                ),
            );
    }

    fn finish(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.init_resource::<VoxelQueryBuffers>();
        }
    }
}
