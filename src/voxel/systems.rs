//! Render-world systems: arena uploads, the surface-nets dispatch, the
//! raster pass, and the CPU side of padded volumes (apron fill, seam LOD
//! packing, active-cell counting).

use super::util::dispatch;
use std::{collections::HashMap, marker::PhantomData, sync::atomic::Ordering};

use crate::field::flatten_cube;
use crate::num::{f32_from_u32, u32_from_f32, u64_from_usize, usize_from};

use bevy::{
    diagnostic::FrameCount,
    ecs::system::SystemParam,
    pbr::{MeshViewBindGroup, ViewKeyCache},
    prelude::*,
    render::{
        Extract,
        diagnostic::{DiagnosticsRecorder, RecordDiagnostics},
        render_resource::{
            CommandEncoder, CommandEncoderDescriptor, ComputePipeline, IndexFormat, MapMode,
            PipelineCache, RenderPassDescriptor, RenderPipeline, StoreOp,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery},
        sync_world::MainEntity,
        view::{ExtractedView, ViewDepthStencilTexture, ViewTarget},
    },
    tasks::ComputeTaskPool,
};

use crate::{
    ApronSample, CHUNK_SIZE, ChunkManager, ChunkPosition, LOD, MaterialField, SDF, VisibilityField,
    VoxelDataSlice, VoxelMaterial,
    voxel::{
        arena::{ArenaSlot, ChunkMeta, INDICES_PER_CELL, VoxelChunkArena, VoxelChunkArenaSet},
        draw::{VoxelDrawPipeline, VoxelViewDraws, encode_draw_compaction},
        pipeline::{
            VoxelDepthPrepass, VoxelMaterialBindGroup, VoxelPipelineLayouts, VoxelRasterKey,
            VoxelRasterPipeline, raster_keys,
        },
        stats::VoxelFrameStats,
        types::DrawIndexedIndirectArgs,
    },
};

use super::pipeline::VoxelComputePipeline;

#[derive(Component)]
pub struct ExtractedChunkField<T: Send + Sync + 'static> {
    pub main_entity: MainEntity,
    pub chunk_pos: IVec3,
    pub padded_data: std::sync::Arc<[u8]>,
    pub size: u32,
    pub lod: LOD,
    /// See [`seam_lod_ranks`]; only the SDF upload uses it.
    pub lod_ranks: u32,
    /// Main-world frame it was extracted in. An upload can wait several
    /// frames; when a newer one for the same chunk arrives, the newest wins.
    pub seq: u32,
    pub _type: PhantomData<T>,
}

/// Max cells re-meshed per frame, summed over all arenas.
///
/// Uploads past it wait for the next frame while the chunk keeps its current
/// mesh, so a burst of loads is spread out instead of landing in one frame.
#[derive(Resource)]
pub struct RemeshBudget {
    pub cells_per_frame: u32,
}

impl Default for RemeshBudget {
    fn default() -> Self {
        Self {
            cells_per_frame: 4_000_000, // ~15 High chunks
        }
    }
}

/// Keeps the newest extraction per chunk, despawning older ones and any for
/// chunks that are no longer live. Returns `(waiting_since, entity)`, oldest
/// wait first. The newest data inherits the oldest wait, so a chunk that is
/// re-extracted every frame (e.g. under a held brush) can't starve behind
/// the budget. Anything not despawned stays in the render world and is
/// retried next frame.
fn newest_pending<T: Send + Sync + 'static>(
    arena_set: &VoxelChunkArenaSet,
    extracted: &Query<(Entity, &ExtractedChunkField<T>)>,
    commands: &mut Commands,
) -> Vec<(u32, Entity)> {
    // main entity -> (oldest seq, newest seq, newest entity)
    let mut newest: HashMap<MainEntity, (u32, u32, Entity)> = HashMap::new();
    for (e, x) in extracted.iter() {
        if !arena_set.live.contains(&x.main_entity) {
            commands.entity(e).despawn();
            continue;
        }
        let Some(entry) = newest.get_mut(&x.main_entity) else {
            newest.insert(x.main_entity, (x.seq, x.seq, e));
            continue;
        };
        entry.0 = entry.0.min(x.seq);
        if x.seq > entry.1 {
            commands.entity(entry.2).despawn();
            (entry.1, entry.2) = (x.seq, e);
        } else {
            commands.entity(e).despawn();
        }
    }
    let mut pending: Vec<_> = newest
        .into_values()
        .map(|(since, _, e)| (since, e))
        .collect();
    pending.sort_unstable_by_key(|p| p.0);
    pending
}

/// Active cells in a padded SDF volume, using exactly pass1's test: a cell
/// is active when some but not all of its 8 corners are `<= 0`. Sizes the
/// chunk's vertex range before the GPU meshes it.
fn count_active_cells(padded: &[u8], texture_size: u32) -> u32 {
    let size = usize_from(texture_size);
    // snorm16, as uploaded (see quantize_sdf): same sign test as pass1.
    let inside: Vec<u8> = padded
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&b| u8::from(i16::from_le_bytes(b) <= 0))
        .collect();
    let Some(slab) = size.checked_mul(size) else {
        return 0;
    };
    if size < 2 || Some(inside.len()) != slab.checked_mul(size) {
        return 0;
    }
    // [z][y] -> row of x
    let slabs: Vec<Vec<&[u8]>> = inside
        .chunks_exact(slab)
        .map(|s| s.chunks_exact(size).collect())
        .collect();
    let mut total: u32 = 0;
    for pair in slabs.windows(2) {
        let [lo, hi] = pair else { continue };
        for (lo_rows, hi_rows) in lo.windows(2).zip(hi.windows(2)) {
            let (&[y0z0, y1z0], &[y0z1, y1z1]) = (lo_rows, hi_rows) else {
                continue;
            };
            // Each corner is 0 or 1, so these sums stay tiny.
            let mut columns = y0z0
                .iter()
                .zip(y1z0)
                .zip(y0z1)
                .zip(y1z1)
                .map(|(((a, b), c), d)| a.wrapping_add(*b).wrapping_add(*c).wrapping_add(*d));
            let Some(mut prev) = columns.next() else {
                continue;
            };
            for next in columns {
                let corners = prev.wrapping_add(next);
                total = total.wrapping_add(u32::from(corners > 0 && corners < 8));
                prev = next;
            }
        }
    }
    total
}

/// The render resources `prepare_voxel_arena` creates arenas and writes with.
#[derive(SystemParam)]
pub struct ArenaGpu<'w> {
    render_device: Res<'w, RenderDevice>,
    layouts: Res<'w, VoxelPipelineLayouts>,
    render_queue: Res<'w, RenderQueue>,
}

/// Uploads changed chunks' SDF and gives each an exact-size vertex range.
///
/// Never drops one: past the per-frame budget, or with no room in any arena,
/// the upload waits and the chunk keeps drawing its current mesh (an old
/// LOD is only released once the new one is placed).
pub fn prepare_voxel_arena(
    mut arena_set: ResMut<VoxelChunkArenaSet>,
    gpu: ArenaGpu,
    budget: Res<RemeshBudget>,
    extracted_chunks: Query<(Entity, &ExtractedChunkField<SDF>)>,
    mut commands: Commands,
    mut out_of_room: Local<bool>,
) {
    arena_set.deferred.clear();
    let pending = newest_pending(&arena_set, &extracted_chunks, &mut commands);

    // Chunks already resident at this LOD (edits, neighbour apron changes)
    // always update this frame so edits stay continuous. The budget only
    // paces new placements (fresh loads, LOD changes), oldest first and
    // always at least one.
    let mut cells_left = budget.cells_per_frame;
    let mut placed_new = false;
    let mut batch = Vec::new();
    for (_, e) in pending {
        let Ok((_, x)) = extracted_chunks.get(e) else {
            continue;
        };
        if arena_set.find_slot(x.lod, x.main_entity).is_none() {
            let cells = x.size.saturating_sub(1).saturating_pow(3);
            if placed_new && cells > cells_left {
                arena_set.deferred.insert(x.main_entity);
                continue;
            }
            placed_new = true;
            cells_left = cells_left.saturating_sub(cells);
        }
        batch.push((e, x));
    }

    let mut counts: Vec<(usize, u32)> = ComputeTaskPool::get().scope(|s| {
        for (i, (_, x)) in batch.iter().enumerate() {
            s.spawn(async move { (i, count_active_cells(&x.padded_data, x.size)) });
        }
    });
    counts.sort_unstable_by_key(|c| c.0);

    let mut any_out_of_room = false;
    for ((extracted_entity, x), (_, count)) in batch.into_iter().zip(counts) {
        let (lod, size) = (x.lod, x.size);
        if count == 0 {
            // No geometry: holding a slot (575 KB of SDF at High) would draw
            // nothing. Any old slot drew nothing for this data either.
            arena_set.release(x.main_entity);
            arena_set.empty.insert(x.main_entity);
            commands.entity(extracted_entity).despawn();
            continue;
        }
        arena_set.empty.remove(&x.main_entity);
        let cell_count = size.saturating_sub(1);
        let Some(ArenaSlot { arena_idx, slot }) =
            arena_set.place(lod, x.main_entity, count, || {
                info!(
                    "[VoxelChunkArenaSet] creating arena for {:?} (cell_count={}, texture_size={})",
                    lod, cell_count, size
                );
                VoxelChunkArena::new(&gpu.render_device, &gpu.layouts, cell_count, size)
            })
        else {
            arena_set.deferred.insert(x.main_entity);
            let lod_idx = VoxelChunkArenaSet::lod_index(lod);
            if arena_set
                .arenas
                .get(lod_idx)
                .is_some_and(|arenas| arenas.iter().any(VoxelChunkArena::dirty_full))
            {
                continue; // out of scratch for this frame: just wait
            }
            if !*out_of_room {
                warn!(
                    "[prepare_voxel_arena] no room for LOD {:?} chunk {:?}; it keeps its current mesh until space frees up",
                    lod, x.chunk_pos
                );
            }
            any_out_of_room = true;
            continue;
        };
        let Some(arena) = arena_set.arena_mut(lod, arena_idx) else {
            continue;
        };
        arena.register_chunk_pos(slot, x.chunk_pos);
        debug_assert_eq!(
            size, arena.texture_size,
            "mixed LOD in one arena — bucket by LOD"
        );
        upload_chunk(&gpu.render_queue, arena, slot, x, count);
        commands.entity(extracted_entity).despawn();
    }
    *out_of_room = any_out_of_room;
}

/// Writes a placed chunk's SDF, metadata, vertex range and draw args into
/// its arena slot, and marks the slot for re-meshing.
fn upload_chunk(
    render_queue: &RenderQueue,
    arena: &mut VoxelChunkArena,
    slot: u32,
    x: &ExtractedChunkField<SDF>,
    count: u32,
) {
    let Some(&(base, _)) = arena.slot_range.get(usize_from(slot)) else {
        return;
    };
    let slot64 = u64::from(slot);
    // snorm16: 2 bytes per element
    let sdf_offset_bytes = slot64
        .wrapping_mul(u64::from(arena.sdf_elems_per_chunk))
        .wrapping_mul(2);
    render_queue.write_buffer(&arena.sdf_buffer, sdf_offset_bytes, &x.padded_data);

    let meta = ChunkMeta {
        chunk_world_origin: (x.chunk_pos.as_vec3() * CHUNK_SIZE).into(),
        lod_ranks: x.lod_ranks,
    };
    render_queue.write_buffer(
        &arena.chunk_meta_buffer,
        slot64.wrapping_mul(u64_from_usize(std::mem::size_of::<ChunkMeta>())),
        bytemuck::bytes_of(&meta),
    );

    let slot_word = slot64.wrapping_mul(4);
    render_queue.write_buffer(
        &arena.slot_vertex_base_buffer,
        slot_word,
        bytemuck::bytes_of(&base),
    );
    render_queue.write_buffer(
        &arena.slot_vertex_count_buffer,
        slot_word,
        bytemuck::bytes_of(&count),
    );
    let args = DrawIndexedIndirectArgs {
        index_count: 0, // pass3 fills it in this frame
        instance_count: 1,
        first_index: base.wrapping_mul(INDICES_PER_CELL),
        base_vertex: 0,
        first_instance: 0,
    };
    render_queue.write_buffer(
        &arena.indirect_args_buffer,
        slot64.wrapping_mul(ARGS_STRIDE),
        bytemuck::bytes_of(&args),
    );

    arena.mark_dirty(slot);
}

/// Pending uploads for a channel that rides along with a chunk's SDF
/// (material, visibility), paired with the slot `prepare_voxel_arena` gave
/// it. Ones whose SDF is waiting stay queued; ones whose chunk meshes to
/// nothing or has no slot are despawned.
fn side_channel_uploads<T: Send + Sync + 'static>(
    arena_set: &VoxelChunkArenaSet,
    extracted: &Query<(Entity, &ExtractedChunkField<T>)>,
    commands: &mut Commands,
) -> Vec<(Entity, ArenaSlot)> {
    let mut out = Vec::new();
    for (_, e) in newest_pending(arena_set, extracted, commands) {
        let Ok((_, x)) = extracted.get(e) else {
            continue;
        };
        if arena_set.deferred.contains(&x.main_entity) {
            continue; // waits with its SDF
        }
        if arena_set.empty.contains(&x.main_entity) {
            commands.entity(e).despawn(); // no geometry to texture or mask
            continue;
        }
        let Some(slot) = arena_set.find_slot(x.lod, x.main_entity) else {
            if !x.padded_data.is_empty() {
                warn!(
                    "[{}] no slot yet for chunk {:?}, dropping this frame's upload",
                    std::any::type_name::<T>(),
                    x.chunk_pos
                );
            }
            commands.entity(e).despawn();
            continue;
        };
        debug_assert_eq!(
            Some(x.size),
            arena_set
                .arenas
                .get(VoxelChunkArenaSet::lod_index(x.lod))
                .and_then(|arenas| arenas.get(slot.arena_idx))
                .map(|arena| arena.texture_size),
            "mixed LOD in one arena — bucket by LOD"
        );
        out.push((e, slot));
    }
    out
}

/// Sibling to `prepare_voxel_arena`: uploads material ids into the slot it
/// placed. Must run after it, so `deferred` and new slots are current.
pub fn prepare_material_for_arena<M: VoxelMaterial>(
    mut arena_set: ResMut<VoxelChunkArenaSet>,
    render_queue: Res<RenderQueue>,
    extracted_chunks: Query<(Entity, &ExtractedChunkField<MaterialField<M>>)>,
    mut commands: Commands,
) {
    for (e, ArenaSlot { arena_idx, slot }) in
        side_channel_uploads(&arena_set, &extracted_chunks, &mut commands)
    {
        let Ok((_, x)) = extracted_chunks.get(e) else {
            continue;
        };
        let Some(arena) = arena_set.arena_mut(x.lod, arena_idx) else {
            continue;
        };
        if !arena.can_mark_dirty(slot) {
            continue; // no scratch left this frame; retried next frame
        }
        let offset = u64::from(slot).wrapping_mul(u64::from(arena.sdf_elems_per_chunk));
        render_queue.write_buffer(&arena.material_buffer, offset, &x.padded_data);
        arena.mark_dirty(slot); // pass3 bakes materials into vertices
        commands.entity(e).despawn();
    }
}

pub fn prepare_visibility_for_arena(
    mut arena_set: ResMut<VoxelChunkArenaSet>,
    render_queue: Res<RenderQueue>,
    extracted_chunks: Query<(Entity, &ExtractedChunkField<VisibilityField>)>,
    mut commands: Commands,
) {
    for (e, ArenaSlot { arena_idx, slot }) in
        side_channel_uploads(&arena_set, &extracted_chunks, &mut commands)
    {
        let Ok((_, x)) = extracted_chunks.get(e) else {
            continue;
        };
        let Some(arena) = arena_set.arena_mut(x.lod, arena_idx) else {
            continue;
        };
        // Empty = the chunk went uniform: clear the flag so the raster stops
        // consulting the now-stale mask.
        let has_mask = !x.padded_data.is_empty();
        if has_mask {
            // One byte per voxel -> 1 bit. Each slot gets its own word-aligned
            // region, like sdf/material_buffer, so adjacent slots' writes can't
            // clobber each other's boundary words.
            let words_per_chunk = u64::from(arena.sdf_elems_per_chunk).div_ceil(32);
            let packed = pack_bits(x.padded_data.iter().map(|&b| b != 0));
            render_queue.write_buffer(
                &arena.visibility_mask_buffer,
                u64::from(slot)
                    .wrapping_mul(words_per_chunk)
                    .wrapping_mul(4),
                bytemuck::cast_slice(&packed),
            );
        }
        arena.set_slot_has_mask(slot, has_mask);
        commands.entity(e).despawn();
    }
    for arena in arena_set.iter_mut() {
        let packed = pack_bits(arena.mask_slots.iter().copied());
        render_queue.write_buffer(
            &arena.chunk_has_mask_buffer,
            0,
            bytemuck::cast_slice(&packed),
        );
    }
}

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
            let wg1 = arena.cell_count.div_ceil(4);
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

/// Everything `voxel_raster_pass` draws with, besides the view.
#[derive(SystemParam)]
pub struct RasterResources<'w> {
    view_keys: Res<'w, ViewKeyCache>,
    depth_prepass: Option<Res<'w, VoxelDepthPrepass>>,
    arena_set: Res<'w, VoxelChunkArenaSet>,
    pipeline_cache: Res<'w, PipelineCache>,
    raster_pipeline: Res<'w, VoxelRasterPipeline>,
    voxel_material: Res<'w, VoxelMaterialBindGroup>,
    draws: Res<'w, VoxelViewDraws>,
    draw_pipeline: Res<'w, VoxelDrawPipeline>,
}

impl RasterResources<'_> {
    /// The compiled pipeline for `key`, warning (once) while it isn't.
    fn pipeline(&self, key: VoxelRasterKey) -> Option<&RenderPipeline> {
        let id = self.raster_pipeline.id(key)?;
        let pipeline = self.pipeline_cache.get_render_pipeline(id);
        if pipeline.is_none() {
            warn_once!(
                "raster pipeline: {:?}",
                self.pipeline_cache.get_render_pipeline_state(id)
            );
        }
        pipeline
    }
}

/// Draws the chunks `prepare_voxel_view_draws` found visible in this view.
///
/// One multi-draw per arena, lit by Bevy's PBR with the view's own mesh view
/// bind groups (lights, shadows, fog, light probes). With
/// [`VoxelDepthPrepass`], every arena is first drawn depth-only, so the PBR
/// pass shades each pixel once.
pub fn voxel_raster_pass(
    view: ViewQuery<(
        &ViewTarget,
        &ViewDepthStencilTexture,
        &ExtractedView,
        &MeshViewBindGroup,
    )>,
    res: RasterResources,
    mut ctx: RenderContext,
) {
    let draws = res.draws.get(view.entity());
    let (target, depth_texture, extracted_view, mesh_view) = view.into_inner();
    if draws.iter().all(|d| d.count == 0) {
        return;
    }
    let depth_prepass = res.depth_prepass.as_deref().copied().unwrap_or_default();
    let Some((prepass_key, shade_key)) = raster_keys(
        &res.view_keys,
        extracted_view,
        &res.voxel_material,
        depth_prepass,
    ) else {
        return;
    };
    // A prepass that isn't compiled yet is skipped; shading alone is still
    // correct, since its depth test works without the prepass.
    let prepass = prepass_key.and_then(|key| res.pipeline(key));
    let (Some(shade), Some(compaction)) = (
        res.pipeline(shade_key),
        res.pipeline_cache
            .get_compute_pipeline(res.draw_pipeline.id),
    ) else {
        return;
    };

    encode_draw_compaction(ctx.command_encoder(), compaction, draws);

    // Timed on the encoder, around the pass: on tiling GPUs (Adreno, Mali)
    // timestamps inside a render pass can fire before the tiles are shaded.
    let diagnostics = ctx.diagnostic_recorder();
    let recorder = diagnostics.as_deref();
    let span = recorder.time_span(ctx.command_encoder(), "voxel_raster");

    let mut render_pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("voxel_raster_pass"),
        color_attachments: &[Some(target.get_color_attachment())],
        depth_stencil_attachment: Some(depth_texture.get_attachment(StoreOp::Store)),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    render_pass.set_bind_group(0, &mesh_view.main, &mesh_view.main_offsets);
    render_pass.set_bind_group(1, &mesh_view.binding_array, &[]);
    render_pass.set_bind_group(2, &res.voxel_material.bind_group, &[]);

    // Both pipelines share one layout, so the bind groups stay set.
    for pipeline in prepass.into_iter().chain([shade]) {
        render_pass.set_render_pipeline(pipeline);
        for d in draws.iter().filter(|d| d.count > 0) {
            let Some(arena) = res.arena_set.arenas.get(d.lod).and_then(|a| a.get(d.arena)) else {
                continue;
            };
            render_pass.set_vertex_buffer(0, arena.final_vertex_buffer.slice(..));
            render_pass.set_index_buffer(arena.index_buffer.slice(..), IndexFormat::Uint32);
            render_pass.set_bind_group(3, &arena.raster_chunk_bind_group, &[]);
            render_pass.multi_draw_indexed_indirect(&d.args, 0, d.count);
        }
    }
    drop(render_pass);
    span.end(ctx.command_encoder());
}

/// Byte stride of one slot's entry in `indirect_args_buffer`.
const ARGS_STRIDE: u64 = u64_from_usize(std::mem::size_of::<DrawIndexedIndirectArgs>());

/// The +x/+y/+z neighbours (faces, edges, corner) whose voxels fill a
/// chunk's apron. Indexes `border_versions` and `lod_ranks` too.
pub(crate) const NEIGHBORS_MASK: [IVec3; 7] = [
    ivec3(1, 0, 0),
    ivec3(0, 1, 0),
    ivec3(0, 0, 1),
    ivec3(1, 1, 0),
    ivec3(1, 0, 1),
    ivec3(0, 1, 1),
    ivec3(1, 1, 1),
];

/// Nearest-neighbor apron sampling for categorical byte data — material
/// ids, visibility bits.
///
/// Shared by `MaterialField`'s and `VisibilityField`'s `ApronSample` impls
/// rather than duplicated: unlike SDF distances, these values must never be
/// blended across a chunk boundary. Reads 0 if `data` is shorter than `size^3`.
#[must_use]
pub fn nearest_neighbor_sample(data: &[u8], size: u32, x: f32, y: f32, z: f32) -> u8 {
    let max_coord = f32_from_u32(size.saturating_sub(1));
    let axis = |v: f32| u32_from_f32(v.round().clamp(0.0, max_coord));
    data.get(flatten_cube(axis(x), axis(y), axis(z), size))
        .copied()
        .unwrap_or_default()
}

pub fn extract_visibility_removed(
    mut commands: Commands,
    mut removed: Extract<RemovedComponents<VisibilityField>>,
    chunks: Extract<Query<(&ChunkPosition, &LOD, Has<VisibilityField>)>>,
    frame: Extract<Res<FrameCount>>,
) {
    for entity in removed.read() {
        let Ok((pos, lod, still_has)) = chunks.get(entity) else {
            continue;
        };
        if still_has {
            continue;
        }
        commands.spawn(ExtractedChunkField::<VisibilityField> {
            main_entity: entity.into(),
            chunk_pos: pos.0,
            padded_data: std::sync::Arc::from([]),
            size: lod.size().wrapping_add(PADDING),
            lod: *lod,
            lod_ranks: 0,
            seq: frame.0,
            _type: PhantomData,
        });
    }
}

/// Apron width in voxels on the +x/+y/+z sides of a padded volume.
pub(crate) const PADDING: u32 = 2;

/// Fills the `+x/+y/+z` apron of a padded volume from the 7 neighbours in
/// `NEIGHBORS_MASK`, resampling across LODs.
///
/// A missing neighbour (or one mid-LOD-change, whose data doesn't match its
/// LOD) falls back to this chunk's own edge voxels. Every region writes only
/// apron voxels and reads only interior ones, so the regions are independent.
pub fn fill_apron<T>(
    vol: &mut [T::Elem],
    chunk_pos: IVec3,
    chunk_manager: &ChunkManager,
    query: &Query<(Entity, &ChunkPosition, &T, &LOD)>,
) where
    T: Component + ApronSample,
{
    let Some((_, lod)) = field_at(chunk_pos, chunk_manager, query) else {
        return;
    };
    fill_apron_with::<T>(vol, lod.size(), |offset| {
        apron_source(chunk_pos.wrapping_add(offset), chunk_manager, query)
            .map(|(f, l)| (f.data_slice(), l.size()))
    });
}

fn field_at<'a, T: Component>(
    pos: IVec3,
    chunk_manager: &ChunkManager,
    query: &'a Query<(Entity, &ChunkPosition, &T, &LOD)>,
) -> Option<(&'a T, LOD)> {
    let (_, _, field, lod) = query.get(chunk_manager.get_chunk(&pos)?).ok()?;
    Some((field, *lod))
}

/// The neighbour at `pos` if the apron can read it: present, and its data
/// matching its LOD (not mid-LOD-change).
fn apron_source<'a, T: Component + VoxelDataSlice>(
    pos: IVec3,
    chunk_manager: &ChunkManager,
    query: &'a Query<(Entity, &ChunkPosition, &T, &LOD)>,
) -> Option<(&'a T, LOD)> {
    field_at(pos, chunk_manager, query).filter(|(f, l)| f.data_slice().len() == l.volume())
}

/// The LODs pass3 needs to stitch seams, packed for `ChunkMeta::lod_ranks`:
///
/// - bits 0-1: this chunk's LOD rank;
/// - bits `2 + 2k`: the rank whose voxel units apron region `k`
///   (`NEIGHBORS_MASK` order) holds: the neighbour's, or this chunk's own
///   where the apron fell back to its edge;
/// - bits `16 + 2a`: the rank of the `-a` face neighbour if it is coarser
///   than this chunk, else this chunk's own.
///
/// Keep in sync with `fill_apron` (which neighbours it reads) and the
/// decoding in `surface_nets_pass3.wgsl` and `cursor_query.wgsl`.
#[must_use]
pub fn seam_lod_ranks<T: Component + VoxelDataSlice>(
    chunk_pos: IVec3,
    lod: LOD,
    chunk_manager: &ChunkManager,
    query: &Query<(Entity, &ChunkPosition, &T, &LOD)>,
) -> u32 {
    let mut ranks = lod.rank();
    for (shift, offset) in (2u32..).step_by(2).zip(NEIGHBORS_MASK) {
        let source = apron_source(chunk_pos.wrapping_add(offset), chunk_manager, query)
            .map_or(lod, |(_, l)| l);
        ranks |= source.rank() << shift;
    }
    for (shift, axis) in (16u32..).step_by(2).zip([IVec3::X, IVec3::Y, IVec3::Z]) {
        let below = apron_source(chunk_pos.wrapping_sub(axis), chunk_manager, query)
            .map(|(_, l)| l)
            .filter(|l| l.size() < lod.size())
            .unwrap_or(lod);
        ranks |= below.rank() << shift;
    }
    ranks
}

/// [`fill_apron`] over plain data: `neighbor(offset)` gives that neighbour's
/// voxels and size, or `None` to fall back to this chunk's edge.
///
/// Same-LOD neighbours are copied as-is. Other LODs are resampled at this
/// chunk's spacing but keep the neighbour's voxel units, so SDF values never
/// clip at `SDF_BAND`; pass3 rescales them (see [`seam_lod_ranks`]).
pub(crate) fn fill_apron_with<'a, T>(
    vol: &mut [T::Elem],
    size: u32,
    neighbor: impl Fn(IVec3) -> Option<(&'a [T::Elem], u32)>,
) where
    T: ApronSample,
    T::Elem: 'a,
{
    let padded = size.wrapping_add(PADDING);
    let idx = |c: UVec3| flatten_cube(c.x, c.y, c.z, padded);
    let edge = UVec3::splat(size.saturating_sub(1));
    let size3 = UVec3::splat(size);

    for offset in NEIGHBORS_MASK {
        let dir = offset.as_uvec3();
        // Axes with dir = 1 cover the apron (n..n+PADDING), the rest 0..n.
        let origin = dir.wrapping_mul(size3);
        let end = size3.wrapping_add(dir.wrapping_mul(UVec3::splat(PADDING)));
        let source = neighbor(offset);
        for z in origin.z..end.z {
            for y in origin.y..end.y {
                for x in origin.x..end.x {
                    let cell = UVec3::new(x, y, z);
                    let local = cell.wrapping_sub(origin);
                    let value = match source {
                        // Same LOD: the sample lands exactly on a voxel, so
                        // read it. Equal to the interpolated path below (an
                        // exact lerp by 0), which costs 8 reads per voxel.
                        Some((data, n_size)) if n_size == size => T::rescale(
                            data.get(flatten_cube(local.x, local.y, local.z, n_size))
                                .copied()
                                .unwrap_or_default(),
                            1.0,
                        ),
                        // A coarser neighbour's 2 apron layers are its first
                        // 2 voxels, so a border cell spans one of its cells
                        // and pass3 can snap it to the coarse vertex.
                        Some((data, n_size)) => {
                            let ratio = f32_from_u32(n_size) / f32_from_u32(size);
                            let mut at = local.as_vec3() * ratio;
                            if n_size < size {
                                at = Vec3::select(dir.cmpeq(UVec3::ONE), local.as_vec3(), at);
                            }
                            T::sample_apron(data, n_size, at.x, at.y, at.z)
                        }
                        None => vol.get(idx(cell.min(edge))).copied().unwrap_or_default(),
                    };
                    if let Some(v) = vol.get_mut(idx(cell)) {
                        *v = value;
                    }
                }
            }
        }
    }
}

/// Trilinear neighbor sampling, for continuous data like SDF distances.
/// Categorical fields (material ids, visibility) must never blend; they use
/// `nearest_neighbor_sample` instead.
///
/// Reads 0 if `data` is shorter than `size^3`.
#[must_use]
pub fn sample_neighbor(data: &[f32], size: u32, x: f32, y: f32, z: f32) -> f32 {
    let edge = UVec3::splat(size.saturating_sub(1));
    let pos = Vec3::new(x, y, z).clamp(Vec3::ZERO, edge.as_vec3());
    let p0 = pos.floor().as_uvec3();
    let p1 = p0.saturating_add(UVec3::ONE).min(edge);
    let frac = pos - p0.as_vec3();
    let at = |ix: u32, iy: u32, iz: u32| {
        data.get(flatten_cube(ix, iy, iz, size))
            .copied()
            .unwrap_or_default()
    };
    let lerp = |from: f32, to: f32, w: f32| to.mul_add(w, from * (1.0 - w));
    let plane = |iz: u32| {
        let y0 = lerp(at(p0.x, p0.y, iz), at(p1.x, p0.y, iz), frac.x);
        let y1 = lerp(at(p0.x, p1.y, iz), at(p1.x, p1.y, iz), frac.x);
        lerp(y0, y1, frac.y)
    };
    lerp(plane(p0.z), plane(p1.z), frac.z)
}

fn pack_bits(bits: impl ExactSizeIterator<Item = bool>) -> Vec<u32> {
    let mut packed = vec![0u32; bits.len().div_ceil(32)];
    for (i, b) in bits.enumerate() {
        if b && let Some(word) = packed.get_mut(i / 32) {
            *word |= 1 << (i % 32);
        }
    }
    packed
}

#[cfg(test)]
mod tests {
    use super::count_active_cells;
    use crate::{VoxelDataSlice, field::SDF, field::flatten_cube, num::usize_from};

    fn volume(n: u32, inside: &[(u32, u32, u32)]) -> Vec<u8> {
        let mut v = vec![1.0f32; usize_from(n).pow(3)];
        for &(x, y, z) in inside {
            v[flatten_cube(x, y, z, n)] = -1.0;
        }
        SDF::gpu_bytes(&v)
    }

    /// Each `NEIGHBORS_MASK` neighbour's voxels and size, if present.
    type Neighbors<E> = [Option<(Vec<E>, u32)>; 7];

    /// The original apron fill: always the interpolating `sample_apron`.
    fn reference_apron<T: crate::ApronSample>(
        vol: &mut [T::Elem],
        size: u32,
        neighbors: &Neighbors<T::Elem>,
    ) {
        use super::{NEIGHBORS_MASK, PADDING};
        use crate::num::f32_from_u32;
        use bevy::math::{UVec3, Vec3};
        let padded = size.wrapping_add(PADDING);
        let idx = |c: UVec3| flatten_cube(c.x, c.y, c.z, padded);
        let edge = UVec3::splat(size.wrapping_sub(1));
        for (offset, n) in NEIGHBORS_MASK.iter().zip(neighbors) {
            let dir = offset.as_uvec3();
            let origin = dir.wrapping_mul(UVec3::splat(size));
            let end = UVec3::splat(size).wrapping_add(dir.wrapping_mul(UVec3::splat(PADDING)));
            for z in origin.z..end.z {
                for y in origin.y..end.y {
                    for x in origin.x..end.x {
                        let cell = UVec3::new(x, y, z);
                        vol[idx(cell)] = match n {
                            Some((data, n_size)) => {
                                let ratio = f32_from_u32(*n_size) / f32_from_u32(size);
                                let local = cell.wrapping_sub(origin).as_vec3();
                                // coarser: apron layers are its own voxels 0 and 1
                                let at = if *n_size < size {
                                    Vec3::select(dir.cmpeq(UVec3::ONE), local, local * ratio)
                                } else {
                                    local * ratio
                                };
                                T::sample_apron(data, *n_size, at.x, at.y, at.z)
                            }
                            None => vol[idx(cell.min(edge))],
                        };
                    }
                }
            }
        }
    }

    /// Deterministic noise in `0..bound` for test volumes.
    fn lcg(seed: &mut u64, bound: u32) -> u32 {
        *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        u32::try_from(*seed >> 33)
            .unwrap()
            .checked_rem(bound)
            .unwrap()
    }

    /// Fills a `size`^3 interior and the 7 neighbours, then checks the fast
    /// apron fill against the reference, compared as uploaded bytes.
    fn check_apron<T: crate::ApronSample>(
        size: u32,
        neighbor_sizes: [Option<u32>; 7],
        value: impl Fn(&mut u64) -> T::Elem,
        bytes: impl Fn(&[T::Elem]) -> Vec<u8>,
    ) {
        let mut seed = u64::from(size).wrapping_mul(7919);
        let padded_size = size.wrapping_add(super::PADDING);
        let mut interior = vec![T::Elem::default(); usize_from(padded_size).pow(3)];
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    interior[flatten_cube(x, y, z, padded_size)] = value(&mut seed);
                }
            }
        }
        let neighbors: Neighbors<T::Elem> = neighbor_sizes.map(|s| {
            s.map(|n| {
                let data = (0..usize_from(n).pow(3)).map(|_| value(&mut seed));
                (data.collect(), n)
            })
        });

        let mut expected = interior.clone();
        reference_apron::<T>(&mut expected, size, &neighbors);
        let mut actual = interior;
        super::fill_apron_with::<T>(&mut actual, size, |offset| {
            let i = super::NEIGHBORS_MASK.iter().position(|o| *o == offset)?;
            neighbors[i].as_ref().map(|(d, n)| (d.as_slice(), *n))
        });
        assert_eq!(
            bytes(&actual),
            bytes(&expected),
            "size {size}, neighbours {neighbor_sizes:?}"
        );
    }

    #[test]
    fn fast_apron_matches_interpolated_apron() {
        use crate::{field::VisibilityField, num::f32_from_u32};
        // Equal LOD (fast path), coarser/finer (interpolated) and missing
        // neighbours, mixed.
        let layouts = [
            [Some(16); 7],
            [
                Some(16),
                Some(32),
                Some(4),
                None,
                Some(16),
                Some(64),
                Some(16),
            ],
            [None; 7],
        ];
        // Includes exact zeros of both signs: the fast path must not change
        // what the GPU gets for them.
        let sdf = |s: &mut u64| match lcg(s, 10) {
            0 => 0.0,
            1 => -0.0,
            _ => f32_from_u32(lcg(s, 1601)) / 100.0 - 8.0,
        };
        let id = |s: &mut u64| u8::try_from(lcg(s, 4)).unwrap();
        for layout in layouts {
            check_apron::<SDF>(16, layout, sdf, SDF::gpu_bytes);
            check_apron::<VisibilityField>(16, layout, id, <[u8]>::to_vec);
        }
    }

    #[test]
    fn counts_cells_with_a_sign_change() {
        assert_eq!(count_active_cells(&volume(4, &[]), 4), 0);
        // A centre voxel touches all 8 cells around it.
        assert_eq!(count_active_cells(&volume(3, &[(1, 1, 1)]), 3), 8);
        // A corner voxel touches only the one cell it's a corner of.
        assert_eq!(count_active_cells(&volume(4, &[(0, 0, 0)]), 4), 1);
    }
}

/// Logs pass3's range guard (a chunk skipped because the GPU counted more
/// cells than its CPU-sized range). Should never fire; costs one small
/// readback per arena per frame, so on in debug builds only.
#[derive(Resource)]
pub struct GuardTelemetry(pub bool);

impl Default for GuardTelemetry {
    fn default() -> Self {
        Self(cfg!(debug_assertions))
    }
}

const GUARD_FREE: u8 = 0;
const GUARD_COPIED: u8 = 1;
const GUARD_MAPPED: u8 = 2;

pub fn map_guard_readbacks(telemetry: Res<GuardTelemetry>, arena_set: Res<VoxelChunkArenaSet>) {
    if !telemetry.0 {
        return;
    }
    for (rank, arenas) in arena_set.arenas.iter().enumerate() {
        for (arena_idx, arena) in arenas.iter().enumerate() {
            let state = arena.guard_state.clone();
            if state
                .compare_exchange(
                    GUARD_COPIED,
                    GUARD_MAPPED,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_err()
            {
                continue;
            }
            let buf = arena.guard_readback_buffer.clone();
            let chunk_pos_of_slot = arena.chunk_pos_of_slot.clone();
            buf.clone().slice(..).map_async(MapMode::Read, move |result| {
                if result.is_ok() {
                    // The view must be dropped before `unmap`; it is, at the
                    // end of this statement.
                    let v = buf.slice(..).get_mapped_range().ok().and_then(|data| {
                        let bytes = data.get(..16)?;
                        bytemuck::try_pod_read_unaligned::<[u32; 4]>(bytes).ok()
                    });
                    buf.unmap();
                    if let Some(v) = v.filter(|v| v[0] != 0) {
                        let pos = chunk_pos_of_slot.get(usize_from(v[1])).copied().flatten();
                        warn!(
                            "[guard] LOD rank {rank} arena {arena_idx}: {} chunk(s) skipped; last slot {} at {:?}: gpu_count {} > cpu_count {}",
                            v[0], v[1], pos, v[2], v[3]
                        );
                    }
                }
                state.store(GUARD_FREE, Ordering::Release);
            });
        }
    }
}
