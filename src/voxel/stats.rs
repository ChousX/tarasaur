//! Per-frame voxel counters, published as Bevy diagnostics.
//!
//! GPU/CPU time of the meshing stages and the raster pass is recorded as
//! render diagnostics (`render/voxel_mesh/...`, `render/voxel_raster/...`)
//! whenever Bevy's `RenderDiagnosticsPlugin` is present; this plugin adds
//! the work counts that explain those times.

use std::sync::{Arc, Mutex};

use bevy::{
    diagnostic::{Diagnostic, DiagnosticPath, Diagnostics, RegisterDiagnostic},
    prelude::*,
    render::{Render, RenderApp, RenderSystems},
};

use crate::{
    Chunk, LOD, TargetLOD,
    field::systems::SdfReinitTask,
    loading::ChunkDataTask,
    voxel::arena::{VoxelChunkArena, VoxelChunkArenaSet},
};

/// Render-world counts for one frame.
#[derive(Resource, Default, Clone, Copy, Debug)]
pub struct VoxelFrameStats {
    /// Indirect draws issued (one per resident slot).
    pub draws: u32,
    /// Chunks re-meshed this frame.
    pub meshed_chunks: u32,
    /// Cells the meshing passes ran over (every cell of every re-meshed chunk).
    pub meshed_cells: u64,
    /// SDF uploads waiting on the remesh budget or arena room.
    pub pending_uploads: u32,
    /// Slots still marked for re-meshing after the dispatch (pipelines not
    /// compiled yet).
    pub dirty_slots: u32,
}

/// Hands render-world stats to the main world. The meshing counts
/// accumulate until the main world takes them, so a frame is never lost or
/// counted twice when the two worlds run out of step.
#[derive(Resource, Default, Clone)]
struct VoxelStatsChannel(Arc<Mutex<VoxelFrameStats>>);

pub const DRAWS: DiagnosticPath = DiagnosticPath::const_new("voxel/draws");
pub const MESHED_CHUNKS: DiagnosticPath = DiagnosticPath::const_new("voxel/meshed_chunks");
pub const MESHED_CELLS: DiagnosticPath = DiagnosticPath::const_new("voxel/meshed_cells");
pub const PENDING_UPLOADS: DiagnosticPath = DiagnosticPath::const_new("voxel/pending_uploads");
pub const CHUNKS: DiagnosticPath = DiagnosticPath::const_new("voxel/chunks");
pub const LOADS_IN_FLIGHT: DiagnosticPath = DiagnosticPath::const_new("voxel/loads_in_flight");
/// Chunks whose LOD differs from their target (not yet loaded or retargeted).
pub const LOD_PENDING: DiagnosticPath = DiagnosticPath::const_new("voxel/lod_pending");
/// Chunks with a jump-flood re-init in flight after an edit.
pub const REINITS: DiagnosticPath = DiagnosticPath::const_new("voxel/reinits");
pub const DIRTY_SLOTS: DiagnosticPath = DiagnosticPath::const_new("voxel/dirty_slots");

pub const ALL: [DiagnosticPath; 9] = [
    DRAWS,
    MESHED_CHUNKS,
    MESHED_CELLS,
    PENDING_UPLOADS,
    DIRTY_SLOTS,
    CHUNKS,
    LOADS_IN_FLIGHT,
    LOD_PENDING,
    REINITS,
];

/// Publishes [`VoxelFrameStats`] and main-world chunk counts as diagnostics.
pub struct VoxelDiagnosticsPlugin;

impl Plugin for VoxelDiagnosticsPlugin {
    fn build(&self, app: &mut App) {
        let channel = VoxelStatsChannel::default();
        for path in ALL {
            app.register_diagnostic(Diagnostic::new(path));
        }
        app.insert_resource(channel.clone())
            .add_systems(PostUpdate, publish_voxel_diagnostics);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .insert_resource(channel)
            .init_resource::<VoxelFrameStats>()
            .add_systems(Render, send_frame_stats.in_set(RenderSystems::Cleanup));
    }
}

fn send_frame_stats(
    mut stats: ResMut<VoxelFrameStats>,
    arena_set: Res<VoxelChunkArenaSet>,
    channel: Res<VoxelStatsChannel>,
) {
    let count = |n: usize| u32::try_from(n).unwrap_or(u32::MAX);
    let sum = |f: fn(&VoxelChunkArena) -> usize| {
        count(arena_set.iter().map(f).fold(0, usize::saturating_add))
    };
    if let Ok(mut out) = channel.0.lock() {
        out.draws = sum(|a| a.active_slots.len());
        out.dirty_slots = sum(|a| a.dirty_slots.len());
        out.pending_uploads = count(arena_set.deferred.len());
        out.meshed_chunks = out.meshed_chunks.saturating_add(stats.meshed_chunks);
        out.meshed_cells = out.meshed_cells.saturating_add(stats.meshed_cells);
    }
    // Meshing counts are per frame; the dispatch only writes them when it runs.
    stats.meshed_chunks = 0;
    stats.meshed_cells = 0;
}

type ChunkState = (
    Option<&'static LOD>,
    Option<&'static TargetLOD>,
    Has<ChunkDataTask>,
    Has<SdfReinitTask>,
);

fn publish_voxel_diagnostics(
    mut diagnostics: Diagnostics,
    channel: Res<VoxelStatsChannel>,
    chunks: Query<ChunkState, With<Chunk>>,
) {
    let (mut total, mut loading, mut lod_pending, mut reinits) = (0usize, 0usize, 0usize, 0usize);
    for (lod, target, load, reinit) in &chunks {
        total = total.saturating_add(1);
        loading = loading.saturating_add(usize::from(load));
        reinits = reinits.saturating_add(usize::from(reinit));
        let at_target = matches!((lod, target), (Some(l), Some(t)) if *l == t.0);
        lod_pending = lod_pending.saturating_add(usize::from(!at_target));
    }
    let to_f64 = |v: usize| f64::from(u32::try_from(v).unwrap_or(u32::MAX));
    diagnostics.add_measurement(&CHUNKS, || to_f64(total));
    diagnostics.add_measurement(&LOADS_IN_FLIGHT, || to_f64(loading));
    diagnostics.add_measurement(&LOD_PENDING, || to_f64(lod_pending));
    diagnostics.add_measurement(&REINITS, || to_f64(reinits));

    let Ok(stats) = channel.0.lock().map(|mut s| {
        let taken = *s;
        s.meshed_chunks = 0;
        s.meshed_cells = 0;
        taken
    }) else {
        return;
    };
    diagnostics.add_measurement(&DRAWS, || f64::from(stats.draws));
    diagnostics.add_measurement(&MESHED_CHUNKS, || f64::from(stats.meshed_chunks));
    diagnostics.add_measurement(&MESHED_CELLS, || {
        f64::from(u32::try_from(stats.meshed_cells).unwrap_or(u32::MAX))
    });
    diagnostics.add_measurement(&PENDING_UPLOADS, || f64::from(stats.pending_uploads));
    diagnostics.add_measurement(&DIRTY_SLOTS, || f64::from(stats.dirty_slots));
}
