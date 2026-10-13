use super::util::dispatch;
use std::{collections::HashMap, marker::PhantomData, sync::atomic::Ordering};

use bevy::{
    diagnostic::FrameCount,
    prelude::*,
    render::{
        Extract,
        camera::ExtractedCamera,
        render_resource::*,
        renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery},
        sync_world::MainEntity,
        view::{ViewDepthStencilTexture, ViewTarget, ViewUniformOffset, ViewUniforms},
    },
    tasks::ComputeTaskPool,
};

use crate::{
    ApronSample, CHUNK_SIZE, ChunkManager, ChunkPosition, ExtractGate, LOD, MaterialField, SDF,
    Versionable, VisibilityField, VoxelMaterial,
    voxel::{
        arena::{ArenaSlot, ChunkMeta, INDICES_PER_CELL, VoxelChunkArena, VoxelChunkArenaSet},
        pipeline::{VoxelMaterialBindGroup, VoxelPipelineLayouts, VoxelRasterPipeline},
        resident::ForceResident,
        types::DrawIndexedIndirectArgs,
    },
};

use super::pipeline::VoxelComputePipeline;

#[derive(Component)]
pub struct ExtractedChunkField<T: Send + Sync + 'static> {
    pub main_entity: MainEntity,
    pub chunk_pos: IVec3,
    pub padded_data: Vec<u8>,
    pub size: u32,
    pub lod: LOD,
    /// Main-world frame it was extracted in. An upload can wait several
    /// frames; when a newer one for the same chunk arrives, the newest wins.
    pub seq: u32,
    pub _type: PhantomData<T>,
}

/// Max cells re-meshed per frame, summed over all arenas. Uploads past it
/// wait for the next frame while the chunk keeps its current mesh, so a
/// burst of loads is spread out instead of landing in one frame.
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
    let n = texture_size as usize;
    // snorm16, as uploaded (see quantize_sdf): same sign test as pass1.
    let inside: Vec<u8> = padded
        .chunks_exact(2)
        .map(|b| u8::from(i16::from_le_bytes([b[0], b[1]]) <= 0))
        .collect();
    if n < 2 || inside.len() != n * n * n {
        return 0;
    }
    let row = |y: usize, z: usize| (z * n + y) * n;
    let mut total = 0;
    for z in 0..n - 1 {
        for y in 0..n - 1 {
            let (a, b, c, d) = (row(y, z), row(y + 1, z), row(y, z + 1), row(y + 1, z + 1));
            let column = |x: usize| inside[a + x] + inside[b + x] + inside[c + x] + inside[d + x];
            let mut prev = column(0);
            for x in 1..n {
                let next = column(x);
                let corners = prev + next;
                total += u32::from(corners > 0 && corners < 8);
                prev = next;
            }
        }
    }
    total
}

/// Uploads changed chunks' SDF and gives each an exact-size vertex range.
/// Never drops one: past the per-frame budget, or with no room in any arena,
/// the upload waits and the chunk keeps drawing its current mesh (an old
/// LOD is only released once the new one is placed).
pub fn prepare_voxel_arena(
    mut arena_set: ResMut<VoxelChunkArenaSet>,
    render_device: Res<RenderDevice>,
    layouts: Res<VoxelPipelineLayouts>,
    render_queue: Res<RenderQueue>,
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
            let cells = (x.size - 1).pow(3);
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
        let Some(ArenaSlot { arena_idx, slot }) =
            arena_set.place(lod, x.main_entity, count, || {
                info!(
                    "[VoxelChunkArenaSet] creating arena for {:?} (cell_count={}, texture_size={})",
                    lod,
                    size - 1,
                    size
                );
                VoxelChunkArena::new(&render_device, &layouts, size - 1, size)
            })
        else {
            arena_set.deferred.insert(x.main_entity);
            let lod_idx = VoxelChunkArenaSet::lod_index(lod);
            if arena_set.arenas[lod_idx]
                .iter()
                .any(VoxelChunkArena::dirty_full)
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
        let arena = arena_set.arena_mut(lod, arena_idx);
        arena.register_chunk_pos(slot, x.chunk_pos);
        debug_assert_eq!(
            size, arena.texture_size,
            "mixed LOD in one arena — bucket by LOD"
        );

        let sdf_offset_bytes = slot as u64 * arena.sdf_elems_per_chunk as u64 * 2; // snorm16
        render_queue.write_buffer(&arena.sdf_buffer, sdf_offset_bytes, &x.padded_data);

        let meta = ChunkMeta {
            chunk_world_origin: (x.chunk_pos.as_vec3() * CHUNK_SIZE).into(),
            active_list_pos: 0,
        };
        let meta_size = std::mem::size_of::<ChunkMeta>() as u64;
        render_queue.write_buffer(
            &arena.chunk_meta_buffer,
            slot as u64 * meta_size,
            bytemuck::bytes_of(&meta),
        );

        let (base, _) = arena.slot_range[slot as usize];
        let slot_word = slot as u64 * 4;
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
            first_index: base * INDICES_PER_CELL,
            base_vertex: 0,
            first_instance: 0,
        };
        render_queue.write_buffer(
            &arena.indirect_args_buffer,
            slot as u64 * std::mem::size_of::<DrawIndexedIndirectArgs>() as u64,
            bytemuck::bytes_of(&args),
        );

        arena.mark_dirty(slot);
        commands.entity(extracted_entity).despawn();
    }
    *out_of_room = any_out_of_room;
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
            x.size,
            arena_set.arenas[VoxelChunkArenaSet::lod_index(x.lod)][slot.arena_idx].texture_size,
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
        let arena = arena_set.arena_mut(x.lod, arena_idx);
        if !arena.can_mark_dirty(slot) {
            continue; // no scratch left this frame; retried next frame
        }
        let offset = slot as u64 * arena.sdf_elems_per_chunk as u64;
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
        let arena = arena_set.arena_mut(x.lod, arena_idx);
        // Empty = the chunk went uniform: clear the flag so the raster stops
        // consulting the now-stale mask.
        let has_mask = !x.padded_data.is_empty();
        if has_mask {
            // One byte per voxel -> 1 bit. Each slot gets its own word-aligned
            // region, like sdf/material_buffer, so adjacent slots' writes can't
            // clobber each other's boundary words.
            let words_per_chunk = (arena.sdf_elems_per_chunk as u64).div_ceil(32);
            let packed = pack_bits(x.padded_data.iter().map(|&b| b != 0));
            render_queue.write_buffer(
                &arena.visibility_mask_buffer,
                slot as u64 * words_per_chunk * 4,
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
) {
    if !arena_set.iter().any(|a| !a.dirty_slots.is_empty()) {
        return;
    }

    let ids = [
        ("pass1", pipeline.pass1_pipeline_id),
        ("scan", pipeline.stream_compaction_pipeline_id),
        ("scan_block_sums", pipeline.scan_block_sums_pipeline_id),
        ("resolve", pipeline.stream_compaction_resolve_pipeline_id),
        ("pass3", pipeline.pass3_pipeline_id),
    ];
    let [
        Some(pass1_pipeline),
        Some(stream_compaction_pipeline),
        Some(scan_block_sums_pipeline),
        Some(stream_compaction_resolve_pipeline),
        Some(pass3_pipeline),
    ] = ids.map(|(_, id)| pipeline_cache.get_compute_pipeline(id))
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
    for arena in arena_set.iter_mut() {
        if arena.dirty_slots.is_empty() {
            continue;
        }
        arena.dirty_slots.sort_unstable();
        arena.dirty_slots.dedup();
        let n = arena.dirty_slots.len() as u32;
        render_queue.write_buffer(
            &arena.active_slot_map_buffer,
            0,
            bytemuck::cast_slice(&arena.dirty_slots),
        );

        let wg1 = arena.cell_count.div_ceil(4);
        dispatch(
            &mut encoder,
            "surface_nets_pass1_batched",
            pass1_pipeline,
            &arena.pass1_bind_group,
            (wg1, wg1, wg1 * n),
        );
        // Scan/resolve: x = block within a chunk, y = chunk. Keeps x under
        // the 65535-workgroup limit however many chunks are dispatched.
        let wg_scan = arena.blocks_per_chunk;
        dispatch(
            &mut encoder,
            "stream_compaction_scan_batched",
            stream_compaction_pipeline,
            &arena.compaction_bind_group,
            (wg_scan, n, 1),
        );
        dispatch(
            &mut encoder,
            "stream_compaction_scan_block_sums_batched",
            scan_block_sums_pipeline,
            &arena.compaction_bind_group,
            (n, 1, 1),
        );
        dispatch(
            &mut encoder,
            "stream_compaction_resolve_batched",
            stream_compaction_resolve_pipeline,
            &arena.compaction_bind_group,
            (wg_scan, n, 1),
        );
        let wg3 = arena.cell_count.div_ceil(8);
        dispatch(
            &mut encoder,
            "surface_nets_pass3_batched",
            pass3_pipeline,
            &arena.pass3_bind_group,
            (wg3, wg3, wg3 * n),
        );
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

pub fn voxel_raster_pass(
    view: ViewQuery<(
        &ExtractedCamera,
        &ViewTarget,
        &ViewDepthStencilTexture,
        &ViewUniformOffset,
    )>,
    view_uniforms: Res<ViewUniforms>,
    arena_set: Option<Res<VoxelChunkArenaSet>>,
    pipeline_cache: Res<PipelineCache>,
    raster_pipeline: Res<VoxelRasterPipeline>,
    voxel_material: Res<VoxelMaterialBindGroup>,
    mut ctx: RenderContext,
) {
    // Arena set may not exist yet (frame 0, before any chunk has been extracted).
    let Some(arena_set) = arena_set else {
        return;
    };
    if pipeline_cache
        .get_render_pipeline(raster_pipeline.pipeline_id)
        .is_none()
    {
        warn_once!(
            "raster pipeline: {:?}",
            pipeline_cache.get_render_pipeline_state(raster_pipeline.pipeline_id)
        );
    }
    if arena_set.is_empty() {
        return;
    }
    let Some(pipeline) = pipeline_cache.get_render_pipeline(raster_pipeline.pipeline_id) else {
        return;
    };
    let Some(binding) = view_uniforms.uniforms.binding() else {
        return;
    };
    let (_camera, target, depth_texture, view_offset) = view.into_inner();
    let view_bind_group = ctx.render_device().create_bind_group(
        Some("voxel_view_bind_group"),
        &raster_pipeline.view_layout,
        &[BindGroupEntry {
            binding: 0,
            resource: binding,
        }],
    );
    let material_bind_group = &voxel_material.0;

    let mut render_pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("voxel_raster_pass"),
        color_attachments: &[Some(target.get_color_attachment())],
        depth_stencil_attachment: Some(depth_texture.get_attachment(StoreOp::Store)),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });

    render_pass.set_render_pipeline(pipeline);
    render_pass.set_bind_group(0, &view_bind_group, &[view_offset.offset]);
    render_pass.set_bind_group(1, &material_bind_group, &[]);

    const ARGS_STRIDE: u64 = std::mem::size_of::<DrawIndexedIndirectArgs>() as u64;

    for arena in arena_set.iter() {
        if arena.active_slots.is_empty() {
            continue;
        }

        render_pass.set_vertex_buffer(0, arena.final_vertex_buffer.slice(..));
        render_pass.set_index_buffer(arena.index_buffer.slice(..), IndexFormat::Uint32);
        render_pass.set_bind_group(2, &arena.raster_chunk_bind_group, &[]);

        // indirect_args is indexed by slot.
        for &slot in &arena.active_slots {
            render_pass
                .draw_indexed_indirect(&arena.indirect_args_buffer, slot as u64 * ARGS_STRIDE);
        }
    }
}

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
/// ids, visibility bits. Shared by MaterialField's and VisibilityField's
/// ApronSample impls rather than duplicated: unlike SDF distances, these
/// values must never be blended across a chunk boundary.
pub fn nearest_neighbor_sample(data: &[u8], size: u32, x: f32, y: f32, z: f32) -> u8 {
    let max_coord = (size - 1) as f32;
    let xi = x.round().clamp(0.0, max_coord) as usize;
    let yi = y.round().clamp(0.0, max_coord) as usize;
    let zi = z.round().clamp(0.0, max_coord) as usize;
    let s = size as usize;
    data[(zi * s + yi) * s + xi]
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
            padded_data: Vec::new(),
            size: lod.size() + PADDING,
            lod: *lod,
            seq: frame.0,
            _type: PhantomData,
        });
    }
}

pub fn extract_voxel_chunks<T>(
    mut commands: Commands,
    chunk_manager: Extract<Res<ChunkManager>>,
    query: Extract<Query<(Entity, &ChunkPosition, &T, &LOD)>>,
    visibility_query: Extract<Query<Option<&VisibilityField>>>,
    mut last_versions: Local<std::collections::HashMap<Entity, (LOD, [u64; 8])>>,
    surface_query: Extract<Query<(), With<crate::field::surface::HasSurface>>>,
    forced: Extract<Res<ForceResident>>,
    frame: Extract<Res<FrameCount>>,
) where
    T: Send + Sync + 'static + Component + Versionable + ApronSample + ExtractGate,
{
    for (entity, pos, field, lod) in query.iter() {
        if !surface_query.contains(entity) {
            last_versions.remove(&entity); // so gaining a surface re-extracts
            continue;
        }
        if visibility_query
            .get(entity)
            .is_ok_and(|v| v.is_some_and(|vv| vv.is_all_empty()))
            && !forced.contains(&pos.0)
        {
            last_versions.remove(&entity); // so leasing/un-hiding re-extracts
            continue;
        }
        let size = lod.size();
        let expected_elems = lod.volume();
        if field.data_slice().len() != expected_elems {
            unreachable!("can't happen now (LOD and fields install together); defensive");
        }

        let mut versions = [0u64; 8];
        versions[0] = field.version();
        for (i, offset) in NEIGHBORS_MASK.iter().enumerate() {
            if let Some(n_entity) = chunk_manager.get_chunk(&(pos.0 + *offset)) {
                if let Ok((_, _, n_field, n_lod)) = query.get(n_entity) {
                    // At equal LOD the apron reads only the neighbour's facing slab;
                    // across LODs it resamples more of it, so use the full version.
                    let v = if n_lod == lod {
                        n_field.border_version(i)
                    } else {
                        n_field.version()
                    };
                    // fold neighbour LOD in: a reload resets version to 0
                    versions[i + 1] = v ^ ((n_lod.rank() as u64 + 1) << 56);
                }
            }
        }

        // Keyed by entity, not position: a chunk that despawns and respawns at
        // the same spot comes back at version 0 with unchanged neighbours, so a
        // position key matched and the new entity never got an arena slot.
        let key = (*lod, versions);
        if last_versions.get(&entity) == Some(&key) {
            continue;
        }
        last_versions.insert(entity, key);

        let padded_size = size + PADDING;

        let padded_data: Vec<u8> = if field.should_extract() {
            let mut vol =
                vec![T::Elem::default(); (padded_size * padded_size * padded_size) as usize];
            let raw = field.data_slice();
            for z in 0..size {
                for y in 0..size {
                    for x in 0..size {
                        let src = ((z * size + y) * size + x) as usize;
                        let dst = ((z * padded_size + y) * padded_size + x) as usize;
                        vol[dst] = raw[src];
                    }
                }
            }
            fill_apron::<T>(&mut vol, pos.0, &chunk_manager, &query);
            T::gpu_bytes(&vol)
        } else {
            Vec::new()
        };

        commands.spawn(ExtractedChunkField::<T> {
            main_entity: entity.into(),
            chunk_pos: pos.0,
            padded_data,
            size: padded_size,
            lod: *lod,
            seq: frame.0,
            _type: PhantomData,
        });
    }
    last_versions.retain(|e, _| query.contains(*e)); // drop despawned chunks
}

pub(crate) const PADDING: u32 = 2;

/// Fills the `+x/+y/+z` apron of a padded volume from the 7 neighbours in
/// `NEIGHBORS_MASK`, resampling across LODs. A missing neighbour (or one
/// mid-LOD-change, whose data doesn't match its LOD) falls back to this
/// chunk's own edge voxels. Every region writes only apron voxels and reads
/// only interior ones, so the regions are independent.
pub fn fill_apron<T>(
    vol: &mut [T::Elem],
    chunk_pos: IVec3,
    chunk_manager: &ChunkManager,
    query: &Query<(Entity, &ChunkPosition, &T, &LOD)>,
) where
    T: Component + ApronSample,
{
    let field_at = |pos: IVec3| {
        let (_, _, field, lod) = query.get(chunk_manager.get_chunk(&pos)?).ok()?;
        Some((field, *lod))
    };
    let Some((_, lod)) = field_at(chunk_pos) else {
        return;
    };
    let n = lod.size();
    let p = n + PADDING;
    let idx = |c: UVec3| ((c.z * p + c.y) * p + c.x) as usize;

    for offset in NEIGHBORS_MASK {
        let o = offset.as_uvec3();
        // Borrowed, not cloned: this runs in ExtractSchedule, which stalls both worlds.
        let neighbor = field_at(chunk_pos + offset)
            .filter(|(f, l)| f.data_slice().len() == l.volume())
            .map(|(f, l)| (f.data_slice(), l.size()));
        // Axes with o = 1 cover the apron (n..n+PADDING), the rest 0..n.
        let (lo, hi) = (o * n, UVec3::splat(n) + o * PADDING);
        for z in lo.z..hi.z {
            for y in lo.y..hi.y {
                for x in lo.x..hi.x {
                    let c = UVec3::new(x, y, z);
                    vol[idx(c)] = match neighbor {
                        Some((data, n_size)) => {
                            let ratio = n_size as f32 / n as f32;
                            let l = (c - o * n).as_vec3() * ratio;
                            T::rescale(T::sample_apron(data, n_size, l.x, l.y, l.z), ratio)
                        }
                        None => vol[idx(c.min(UVec3::splat(n - 1)))],
                    };
                }
            }
        }
    }
}

/// Trilinear neighbor sampling, for continuous data like SDF distances.
/// Categorical fields (material ids, visibility) must never blend; they use
/// `nearest_neighbor_sample` instead.
pub fn sample_neighbor(data: &[f32], size: u32, x: f32, y: f32, z: f32) -> f32 {
    let p = Vec3::new(x, y, z).clamp(Vec3::ZERO, Vec3::splat((size - 1) as f32));
    let p0 = p.floor().as_uvec3();
    let p1 = (p0 + 1).min(UVec3::splat(size - 1));
    let t = p - p0.as_vec3();
    let s = size as usize;
    let at = |x: u32, y: u32, z: u32| data[(z as usize * s + y as usize) * s + x as usize];
    let lerp = |a: f32, b: f32, t: f32| a * (1.0 - t) + b * t;
    let plane = |z: u32| {
        let y0 = lerp(at(p0.x, p0.y, z), at(p1.x, p0.y, z), t.x);
        let y1 = lerp(at(p0.x, p1.y, z), at(p1.x, p1.y, z), t.x);
        lerp(y0, y1, t.y)
    };
    lerp(plane(p0.z), plane(p1.z), t.z)
}

fn pack_bits(bits: impl ExactSizeIterator<Item = bool>) -> Vec<u32> {
    let mut packed = vec![0u32; bits.len().div_ceil(32)];
    for (i, b) in bits.enumerate() {
        if b {
            packed[i / 32] |= 1 << (i % 32);
        }
    }
    packed
}

pub fn release_despawned_chunks(
    live: Extract<
        Query<
            (Entity, &ChunkPosition, Option<&VisibilityField>),
            (
                With<crate::Chunk>,
                With<LOD>,
                With<crate::field::surface::HasSurface>,
            ),
        >,
    >,
    forced: Extract<Res<ForceResident>>,
    arena_set: Option<ResMut<VoxelChunkArenaSet>>,
) {
    let Some(mut arena_set) = arena_set else {
        return;
    };
    let wanted: std::collections::HashSet<MainEntity> = live
        .iter()
        .filter(|(_, pos, v)| !v.is_some_and(|v| v.is_all_empty()) || forced.contains(&pos.0))
        .map(|(e, _, _)| MainEntity::from(e))
        .collect();
    let stale: Vec<MainEntity> = arena_set
        .iter()
        .flat_map(|a| a.slot_of_main_entity.keys().copied())
        .filter(|e| !wanted.contains(e))
        .collect();
    for e in stale {
        arena_set.release(e);
    }
    arena_set.empty.retain(|e| wanted.contains(e));
    arena_set.live = wanted;
}

#[cfg(test)]
mod tests {
    use super::count_active_cells;
    use crate::{VoxelDataSlice, field::SDF};

    fn volume(n: usize, inside: &[(usize, usize, usize)]) -> Vec<u8> {
        let mut v = vec![1.0f32; n * n * n];
        for &(x, y, z) in inside {
            v[(z * n + y) * n + x] = -1.0;
        }
        SDF::gpu_bytes(&v)
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
                    let v: [u32; 4] = {
                        let Ok(data) = buf.slice(..).get_mapped_range() else {
                            buf.unmap();
                            return;
                        };
                        *bytemuck::from_bytes(&data[..16])
                    };
                    buf.unmap();
                    if v[0] != 0 {
                        let pos = chunk_pos_of_slot.get(v[1] as usize).copied().flatten();
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
