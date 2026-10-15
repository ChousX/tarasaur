//! The surface-nets dispatch: pass1 and pass3 over each arena's dirty slots,
//! once per frame after the uploads.

use super::util::dispatch;
use std::sync::atomic::Ordering;

use crate::num::usize_from;

use bevy::{
    ecs::system::SystemParam,
    prelude::*,
    render::{
        diagnostic::{DiagnosticsRecorder, RecordDiagnostics},
        render_resource::{
            CommandEncoder, CommandEncoderDescriptor, ComputePipeline, PipelineCache,
        },
        renderer::{RenderDevice, RenderQueue},
    },
};

use crate::voxel::{
    arena::{PASS1_WORKGROUP_EDGE, VoxelChunkArena, VoxelChunkArenaSet},
    stats::VoxelFrameStats,
};

use super::pipeline::VoxelComputePipeline;
use super::telemetry::{GUARD_COPIED, GUARD_FREE, GuardTelemetry};

/// Re-meshes only the dirty slots of each arena. Every slot owns a fixed
/// vertex/index range, so untouched chunks keep their geometry as-is.
pub fn dispatch_voxel_compute_passes_batched(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<VoxelComputePipeline>,
    mut arena_set: ResMut<VoxelChunkArenaSet>,
    telemetry: Res<GuardTelemetry>,
    diagnostics: MeshDiagnostics,
) {
    if !arena_set.iter().any(|a| !a.dirty_slots.is_empty()) {
        return;
    }

    let ids = [
        ("pass1", pipeline.pass1_pipeline_id),
        ("pass3", pipeline.pass3_pipeline_id),
    ];
    let [Some(pass1_pipeline), Some(pass3_pipeline)] =
        ids.map(|(_, id)| pipeline_cache.get_compute_pipeline(id))
    else {
        for (name, id) in ids {
            if pipeline_cache.get_compute_pipeline(id).is_none() {
                warn!(
                    "[dispatch] {name} not ready: {:?}",
                    pipeline_cache.get_compute_pipeline_state(id)
                );
            }
        }
        return; // dirty slots stay dirty until the pipelines are ready
    };

    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("voxel_compute_encoder_batched"),
    });
    let mut dirty: Vec<&mut VoxelChunkArena> = arena_set
        .iter_mut()
        .filter(|a| !a.dirty_slots.is_empty())
        .collect();
    let (mut meshed_chunks, mut meshed_cells) = (0u32, 0u64);
    for arena in &mut dirty {
        arena.dirty_slots.sort_unstable();
        arena.dirty_slots.dedup();
        render_queue.write_buffer(
            &arena.active_slot_map_buffer,
            0,
            bytemuck::cast_slice(&arena.dirty_slots),
        );
        let n = dirty_count(arena);
        meshed_chunks = meshed_chunks.saturating_add(n);
        meshed_cells =
            meshed_cells.saturating_add(u64::from(arena.total_cells).saturating_mul(u64::from(n)));
    }
    if let Some(mut stats) = diagnostics.stats {
        stats.meshed_chunks = meshed_chunks;
        stats.meshed_cells = meshed_cells;
    }
    encode_mesh_passes(
        &mut encoder,
        diagnostics.recorder.as_deref(),
        [pass1_pipeline, pass3_pipeline],
        &dirty,
    );

    for arena in dirty {
        if telemetry.0 && arena.guard_state.load(Ordering::Acquire) == GUARD_FREE {
            encoder.copy_buffer_to_buffer(
                &arena.guard_debug_buffer,
                0,
                &arena.guard_readback_buffer,
                0,
                16,
            );
            encoder.clear_buffer(&arena.guard_debug_buffer, 0, None);
            arena.guard_state.store(GUARD_COPIED, Ordering::Release);
        }
        arena.dirty_slots.clear();
    }
    render_queue.submit(std::iter::once(encoder.finish()));
}

fn dirty_count(arena: &VoxelChunkArena) -> u32 {
    u32::try_from(arena.dirty_slots.len()).unwrap_or(u32::MAX)
}

/// Optional instrumentation for the meshing dispatch.
#[derive(SystemParam)]
pub struct MeshDiagnostics<'w> {
    recorder: Option<Res<'w, DiagnosticsRecorder>>,
    stats: Option<ResMut<'w, VoxelFrameStats>>,
}

/// Must match pass3's `@workgroup_size`.
const PASS3_WORKGROUP: u32 = 64;

/// Records the meshing passes for every dirty arena. `pipelines` is pass1,
/// pass3.
///
/// Stage-major (every arena's pass1, then every arena's pass3) so each
/// stage is one timing span. Arenas share no buffers, so the order between
/// them doesn't matter.
fn encode_mesh_passes(
    encoder: &mut CommandEncoder,
    recorder: Option<&DiagnosticsRecorder>,
    pipelines: [&ComputePipeline; 2],
    dirty: &[&mut VoxelChunkArena],
) {
    let [pass1, pass3] = pipelines;
    let mesh_span = recorder.time_span(encoder, "voxel_mesh");
    timed(recorder, encoder, "pass1", |enc| {
        for arena in dirty {
            // pass1 counts each chunk's active cells into these.
            enc.clear_buffer(&arena.chunk_counts_buffer, 0, None);
            let wg1 = arena.cell_count.div_ceil(PASS1_WORKGROUP_EDGE);
            dispatch(
                enc,
                "surface_nets_pass1_batched",
                pass1,
                &arena.pass1_bind_group,
                (wg1, wg1, wg1.saturating_mul(dirty_count(arena))),
            );
        }
    });
    // x = vertex (active cell) within a chunk, y = chunk. Every dirty
    // chunk's vertex range holds at least its active cells, so the largest
    // range bounds them all; threads past a chunk's count exit.
    timed(recorder, encoder, "pass3", |enc| {
        for arena in dirty {
            let most = arena
                .dirty_slots
                .iter()
                .filter_map(|&s| arena.slot_range.get(usize_from(s)))
                .map(|&(_, capacity)| capacity)
                .max()
                .unwrap_or(0);
            dispatch(
                enc,
                "surface_nets_pass3_batched",
                pass3,
                &arena.pass3_bind_group,
                (most.div_ceil(PASS3_WORKGROUP), dirty_count(arena), 1),
            );
        }
    });
    mesh_span.end(encoder);
}

/// Runs `f` inside a CPU/GPU timing span. A no-op span without
/// `RenderDiagnosticsPlugin`.
fn timed(
    recorder: Option<&DiagnosticsRecorder>,
    encoder: &mut CommandEncoder,
    name: &'static str,
    f: impl FnOnce(&mut CommandEncoder),
) {
    let span = recorder.time_span(encoder, name);
    f(encoder);
    span.end(encoder);
}
