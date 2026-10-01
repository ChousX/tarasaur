use super::util::dispatch;
use std::{
    marker::PhantomData,
    sync::atomic::{AtomicU32, Ordering},
};

use bevy::{
    prelude::*,
    render::{
        Extract,
        camera::ExtractedCamera,
        render_resource::*,
        renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery},
        sync_world::MainEntity,
        view::{ViewDepthTexture, ViewTarget, ViewUniformOffset, ViewUniforms},
    },
};

use crate::{
    ApronSample, CHUNK_SIZE, ChunkManager, ChunkPosition, ExtractGate, LOD, MaterialField,
    Versionable, VisibilityField, VoxelMaterial,
    voxel::{
        arena::{ArenaSlot, ChunkMeta, VoxelChunkArena, VoxelChunkArenaSet},
        pipeline::{VoxelMaterialBindGroup, VoxelPipelineLayouts, VoxelRasterPipeline},
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
    pub _type: PhantomData<T>,
}

pub fn prepare_voxel_arena<T: Send + Sync + 'static>(
    mut arena_set: ResMut<VoxelChunkArenaSet>,
    render_device: Res<RenderDevice>,
    layouts: Res<VoxelPipelineLayouts>,
    render_queue: Res<RenderQueue>,
    extracted_chunks: Query<(Entity, &ExtractedChunkField<T>)>,
    mut commands: Commands,
) {
    for arena in arena_set.iter_mut() {
        arena.dirty_slots.clear();
    }

    for (extracted_entity, extracted_sdf) in extracted_chunks.iter() {
        let lod = extracted_sdf.lod;
        let size = extracted_sdf.size;

        let Some(ArenaSlot { arena_idx, slot }) =
            arena_set.slot_for(lod, extracted_sdf.main_entity, || {
                info!(
                    "[VoxelChunkArenaSet] creating arena for {:?} (cell_count={}, texture_size={})",
                    lod,
                    size - 1,
                    size
                );
                VoxelChunkArena::new(&render_device, &layouts, size - 1, size)
            })
        else {
            warn!(
                "[prepare_voxel_arena] all arenas for LOD {:?} full, dropping chunk {:?}",
                lod, extracted_sdf.chunk_pos
            );
            commands.entity(extracted_entity).despawn();
            continue;
        };
        let arena = arena_set.arena_mut(lod, arena_idx);

        if !arena.active_slots.contains(&slot) {
            arena.active_slots.push(slot);
        }
        arena.register_chunk_pos(slot, extracted_sdf.chunk_pos);
        debug_assert_eq!(
            size, arena.texture_size,
            "mixed LOD in one arena — bucket by LOD"
        );

        // Upload this chunk's padded SDF data into its slot of the shared sdf_buffer.
        let sdf_offset_bytes = slot as u64 * arena.sdf_elems_per_chunk as u64 * 4;
        render_queue.write_buffer(
            &arena.sdf_buffer,
            sdf_offset_bytes,
            &extracted_sdf.padded_data,
        );

        let chunk_world_origin = extracted_sdf.chunk_pos.as_vec3() * CHUNK_SIZE;

        // active_list_pos is unknown until arena.active_slots is finalized for
        // this frame (see the patch loop below, after all chunks are processed).
        // Written as 0 here as a placeholder; patched immediately after.
        let meta = ChunkMeta {
            chunk_world_origin: chunk_world_origin.into(),
            active_list_pos: 0,
        };
        render_queue.write_buffer(
            &arena.chunk_meta_buffer,
            slot as u64 * std::mem::size_of::<ChunkMeta>() as u64,
            bytemuck::bytes_of(&meta),
        );

        arena.dirty_slots.push(slot);
        commands.entity(extracted_entity).despawn();
    }

    // Now that arena.active_slots reflects this frame's final order, patch
    // each active chunk's active_list_pos field (its index into active_slots),
    // populate active_slot_map (the inverse mapping used by compute_chunk_bases
    // and pass3), and refresh the bases-pass uniform's active_count.
    for arena in arena_set.iter_mut() {
        const ACTIVE_LIST_POS_OFFSET: u64 = std::mem::offset_of!(ChunkMeta, active_list_pos) as u64;

        for (pos, &slot) in arena.active_slots.iter().enumerate() {
            let offset =
                slot as u64 * std::mem::size_of::<ChunkMeta>() as u64 + ACTIVE_LIST_POS_OFFSET;
            render_queue.write_buffer(
                &arena.chunk_meta_buffer,
                offset,
                bytemuck::bytes_of(&(pos as u32)),
            );
        }

        let slot_map: Vec<u32> = arena.active_slots.clone();
        render_queue.write_buffer(
            &arena.active_slot_map_buffer,
            0,
            bytemuck::cast_slice(&slot_map),
        );

        // Clear overflow flag for this frame.
        render_queue.write_buffer(&arena.overflow_flag_buffer, 0, bytemuck::bytes_of(&0u32));

        // Update active_count in the chunk-bases uniform (first field of
        // ChunkBasesUniforms; budget fields are set once at construction and
        // never change, so this partial write is safe).
        render_queue.write_buffer(
            &arena.chunk_bases_uniform_buffer,
            0,
            bytemuck::bytes_of(&(arena.active_slots.len() as u32)),
        );
    }
}

/// Sibling to `prepare_voxel_arena`, not a reuse of it — that function's
/// body is SDF-specific (hardcodes `sdf_buffer`/`sdf_elems_per_chunk`/
/// `meta.sdf_offset`) despite being generic over `T`. This system owns the
/// material upload instead of trying to force-fit into that one.
///
/// Must run after `prepare_voxel_arena::<SDFField>` in the same Prepare
/// set — that system owns slot allocation and the initial full `ChunkMeta`
/// write; this one only ever looks a slot up.
pub fn prepare_material_for_arena<M: VoxelMaterial>(
    mut arena_set: ResMut<VoxelChunkArenaSet>,
    render_queue: Res<RenderQueue>,
    extracted_chunks: Query<(Entity, &ExtractedChunkField<MaterialField<M>>)>,
    mut commands: Commands,
) {
    for (extracted_entity, extracted_material) in extracted_chunks.iter() {
        let Some(ArenaSlot { arena_idx, slot }) =
            arena_set.find_slot(extracted_material.lod, extracted_material.main_entity)
        else {
            warn!(
                "[prepare_material_for_arena] no slot yet for chunk {:?}, dropping this frame's material upload",
                extracted_material.chunk_pos
            );
            commands.entity(extracted_entity).despawn();
            continue;
        };
        let arena = arena_set.arena_mut(extracted_material.lod, arena_idx);

        let Some(slot) = arena.existing_slot(extracted_material.main_entity) else {
            warn!(
                "[prepare_material_for_arena] no slot yet for chunk {:?}, dropping this frame's material upload",
                extracted_material.chunk_pos
            );
            commands.entity(extracted_entity).despawn();
            continue;
        };

        debug_assert_eq!(
            extracted_material.size, arena.texture_size,
            "mixed LOD in one arena — bucket by LOD"
        );

        // 1 byte/elem, same element count as the SDF payload for this slot.
        let material_offset_bytes = slot as u64 * arena.sdf_elems_per_chunk as u64;
        render_queue.write_buffer(
            &arena.material_buffer,
            material_offset_bytes,
            &extracted_material.padded_data,
        );

        commands.entity(extracted_entity).despawn();
    }
}

pub fn prepare_visibility_for_arena(
    mut arena_set: ResMut<VoxelChunkArenaSet>,
    render_queue: Res<RenderQueue>,
    extracted_chunks: Query<(Entity, &ExtractedChunkField<VisibilityField>)>,
    mut commands: Commands,
) {
    for (extracted_entity, extracted_mask) in extracted_chunks.iter() {
        let Some(ArenaSlot { arena_idx, slot }) =
            arena_set.find_slot(extracted_mask.lod, extracted_mask.main_entity)
        else {
            if !extracted_mask.padded_data.is_empty() {
                warn!(
                    "[prepare_visibility_for_arena] no slot yet for chunk {:?}, dropping this frame's mask",
                    extracted_mask.chunk_pos
                );
            }
            commands.entity(extracted_entity).despawn();
            continue;
        };
        let arena = arena_set.arena_mut(extracted_mask.lod, arena_idx);

        let Some(slot) = arena.existing_slot(extracted_mask.main_entity) else {
            warn!(
                "[prepare_visibility_for_arena] no slot yet for chunk {:?}, dropping this frame's mask",
                extracted_mask.chunk_pos
            );
            commands.entity(extracted_entity).despawn();
            continue;
        };

        if extracted_mask.padded_data.is_empty() {
            // Chunk transitioned to uniform this frame — clear the flag;
            // the previous frame's mask data, if any, is now stale and
            // pass1 will stop consulting it once chunk_has_mask reflects
            // this.
            arena.set_slot_has_mask(slot, false);
            commands.entity(extracted_entity).despawn();
            continue;
        }

        debug_assert_eq!(
            extracted_mask.size, arena.texture_size,
            "mixed LOD in one arena — bucket by LOD"
        );

        // padded_data is one byte (0/1) per voxel from VisibilityField's
        // byte-mirror ApronSample path — pack to 1 bit/voxel before
        // upload. Each slot gets a word-aligned region (mask_words_per_chunk
        // words), matching sdf_buffer/material_buffer's fixed-stride-per-slot
        // layout — never a flat global bit-packing, which would let
        // adjacent slots' writes clobber each other's boundary words.
        let mask_words_per_chunk = (arena.sdf_elems_per_chunk as usize).div_ceil(32);
        let packed = pack_bits(extracted_mask.padded_data.iter().map(|&b| b != 0));
        let byte_offset = slot as u64 * mask_words_per_chunk as u64 * 4;
        render_queue.write_buffer(
            &arena.visibility_mask_buffer,
            byte_offset,
            bytemuck::cast_slice(&packed),
        );

        arena.set_slot_has_mask(slot, true);
        commands.entity(extracted_entity).despawn();
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
pub fn dispatch_voxel_compute_passes_batched(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<VoxelComputePipeline>,
    arena_set: Res<VoxelChunkArenaSet>,
) {
    if arena_set.arenas.is_empty() {
        return;
    }

    let (
        Some(pass1_pipeline),
        Some(stream_compaction_pipeline),
        Some(scan_block_sums_pipeline),
        Some(stream_compaction_resolve_pipeline),
        Some(write_chunk_active_count_pipeline),
        Some(chunk_bases_pipeline),
        Some(pass3_pipeline),
    ) = (
        pipeline_cache.get_compute_pipeline(pipeline.pass1_pipeline_id),
        pipeline_cache.get_compute_pipeline(pipeline.stream_compaction_pipeline_id),
        pipeline_cache.get_compute_pipeline(pipeline.scan_block_sums_pipeline_id),
        pipeline_cache.get_compute_pipeline(pipeline.stream_compaction_resolve_pipeline_id),
        pipeline_cache.get_compute_pipeline(pipeline.write_chunk_active_count_pipeline_id),
        pipeline_cache.get_compute_pipeline(pipeline.chunk_bases_pipeline_id),
        pipeline_cache.get_compute_pipeline(pipeline.pass3_pipeline_id),
    )
    else {
        let ids = [
            ("pass1", pipeline.pass1_pipeline_id),
            ("scan", pipeline.stream_compaction_pipeline_id),
            ("scan_block_sums", pipeline.scan_block_sums_pipeline_id),
            ("resolve", pipeline.stream_compaction_resolve_pipeline_id),
            (
                "write_active_count",
                pipeline.write_chunk_active_count_pipeline_id,
            ),
            ("chunk_bases", pipeline.chunk_bases_pipeline_id),
            ("pass3", pipeline.pass3_pipeline_id),
        ];
        for (name, id) in ids {
            if pipeline_cache.get_compute_pipeline(id).is_none() {
                warn!(
                    "[dispatch] {name} not ready: {:?}",
                    pipeline_cache.get_compute_pipeline_state(id)
                );
            }
        }
        return;
    };

    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("voxel_compute_encoder_batched"),
    });

    for arena in arena_set.iter() {
        let active = arena.active_chunk_count();
        static LAST: AtomicU32 = AtomicU32::new(0);
        if active != LAST.swap(active, Ordering::Relaxed) {
            info!(
                "[arena] texture_size={} active={} max={}",
                arena.texture_size, active, arena.max_chunks
            );
        }
        if active == 0 {
            continue;
        }

        let wg1 = arena.cell_count.div_ceil(4);
        dispatch(
            &mut encoder,
            "surface_nets_pass1_batched",
            pass1_pipeline,
            &arena.pass1_bind_group,
            (wg1, wg1, wg1 * active),
        );

        let wg_scan = arena.blocks_per_chunk;
        dispatch(
            &mut encoder,
            "stream_compaction_scan_batched",
            stream_compaction_pipeline,
            &arena.compaction_bind_group,
            (wg_scan * active, 1, 1),
        );
        dispatch(
            &mut encoder,
            "stream_compaction_scan_block_sums_batched",
            scan_block_sums_pipeline,
            &arena.compaction_bind_group,
            (active, 1, 1),
        );
        dispatch(
            &mut encoder,
            "stream_compaction_resolve_batched",
            stream_compaction_resolve_pipeline,
            &arena.compaction_bind_group,
            (wg_scan * active, 1, 1),
        );
        dispatch(
            &mut encoder,
            "stream_compaction_write_active_counts",
            write_chunk_active_count_pipeline,
            &arena.compaction_bind_group,
            (active, 1, 1),
        );
        dispatch(
            &mut encoder,
            "compute_chunk_bases",
            chunk_bases_pipeline,
            &arena.chunk_bases_bind_group,
            (1, 1, 1),
        );

        let wg3 = arena.cell_count.div_ceil(8);
        dispatch(
            &mut encoder,
            "surface_nets_pass3_batched",
            pass3_pipeline,
            &arena.pass3_bind_group,
            (wg3, wg3, wg3 * active),
        );

        encoder.copy_buffer_to_buffer(
            &arena.overflow_flag_buffer,
            0,
            &arena.overflow_readback_buffer,
            0,
            4,
        );
    }
    render_queue.submit(std::iter::once(encoder.finish()));
}

pub fn voxel_raster_pass(
    view: ViewQuery<(
        &ExtractedCamera,
        &ViewTarget,
        &ViewDepthTexture,
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

    // Each LOD's arena has its own vertex/index buffers, so the bind-once
    // optimization from the single-arena version now happens once per arena
    // instead of once per frame.
    for arena in arena_set.iter() {
        if arena.active_slots.is_empty() {
            continue;
        }

        render_pass.set_vertex_buffer(0, arena.final_vertex_buffer.slice(..));
        render_pass.set_index_buffer(arena.index_buffer.slice(..), IndexFormat::Uint32);
        render_pass.set_bind_group(2, &arena.raster_chunk_bind_group, &[]);

        for &slot in arena.active_slots.iter() {
            let offset = slot as u64 * ARGS_STRIDE;
            render_pass.draw_indexed_indirect(&arena.indirect_args_buffer, offset);
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
            _type: PhantomData,
        });
    }
}

pub fn extract_voxel_chunks<T>(
    mut commands: Commands,
    chunk_manager: Extract<Res<ChunkManager>>,
    query: Extract<Query<(Entity, &ChunkPosition, &T, &LOD)>>,
    visibility_query: Extract<Query<Option<&VisibilityField>>>,
    mut last_versions: Local<std::collections::HashMap<IVec3, (LOD, [u64; 8])>>,
    surface_query: Extract<Query<(), With<crate::field::surface::HasSurface>>>,
) where
    T: Send + Sync + 'static + Component + Versionable + ApronSample + ExtractGate,
{
    for (entity, pos, field, lod) in query.iter() {
        if !surface_query.contains(entity) {
            last_versions.remove(&pos.0); // so gaining a surface re-extracts
            continue;
        }
        if visibility_query
            .get(entity)
            .is_ok_and(|v| v.is_some_and(|vv| vv.is_all_full() == false))
        {
            last_versions.remove(&pos.0); // so un-hiding re-extracts
            //       I am not sure about this ^
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
                    // fold neighbour LOD in: a reload resets version to 0
                    versions[i + 1] = n_field.version() ^ ((n_lod.rank() as u64 + 1) << 56);
                }
            }
        }

        let key = (*lod, versions);
        if last_versions.get(&pos.0) == Some(&key) {
            continue;
        }
        last_versions.insert(pos.0, key);

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
            bytemuck::cast_slice(&vol).to_vec()
        } else {
            Vec::new()
        };

        commands.spawn(ExtractedChunkField::<T> {
            main_entity: entity.into(),
            chunk_pos: pos.0,
            padded_data,
            size: padded_size,
            lod: *lod,
            _type: PhantomData,
        });
    }
}

const PADDING: u32 = 2;

pub fn fill_apron<T>(
    vol: &mut [T::Elem],
    chunk_pos: IVec3,
    chunk_manager: &ChunkManager,
    query: &Query<(Entity, &ChunkPosition, &T, &LOD)>,
) where
    T: Component + ApronSample,
{
    let Some(current_entity) = chunk_manager.get_chunk(&chunk_pos) else {
        return;
    };
    let Ok((_, _, _, current_lod)) = query.get(current_entity) else {
        return;
    };

    let chunk_size = current_lod.size();
    let padded_size = chunk_size + PADDING;

    let idx =
        |x: u32, y: u32, z: u32| -> usize { ((z * padded_size + y) * padded_size + x) as usize };

    let neighbor_info = |offset: IVec3| -> Option<(Box<[T::Elem]>, u32)> {
        let n_entity = chunk_manager.get_chunk(&(chunk_pos + offset))?;
        let Ok((_, _, n_field, n_lod)) = query.get(n_entity) else {
            return None;
        };
        let n_size = n_lod.size();
        let expected = (n_size * n_size * n_size) as usize;
        if n_field.data_slice().len() != expected {
            // Same LOD/field race as extract_voxel_chunks's own guard, one
            // hop over: this neighbor's LOD was bumped but its field hasn't
            // been resized by sync_field_lod yet this frame. Treat as "no
            // neighbor data available" for this frame — fill_region's None
            // branch already handles that by falling back to the current
            // chunk's own edge data, which is a safe (if slightly less
            // accurate for one frame) apron value.
            return None;
        }
        let data = n_field.data_slice().to_vec().into_boxed_slice();
        Some((data, n_size))
    };

    // --- Faces: +x, +y, +z ---
    for d in 0..PADDING {
        let x_target = chunk_size + d;
        let mut coords = Vec::new();
        for z in 0..chunk_size {
            for y in 0..chunk_size {
                coords.push((x_target, y, z));
            }
        }
        fill_region::<T>(
            vol,
            chunk_size,
            IVec3::new(1, 0, 0),
            &coords,
            |_, cy, cz| idx(chunk_size - 1, cy, cz),
            &neighbor_info,
            &idx,
        );
    }

    for d in 0..PADDING {
        let y_target = chunk_size + d;
        let mut coords = Vec::new();
        for z in 0..chunk_size {
            for x in 0..chunk_size {
                coords.push((x, y_target, z));
            }
        }
        fill_region::<T>(
            vol,
            chunk_size,
            IVec3::new(0, 1, 0),
            &coords,
            |cx, _, cz| idx(cx, chunk_size - 1, cz),
            &neighbor_info,
            &idx,
        );
    }

    for d in 0..PADDING {
        let z_target = chunk_size + d;
        let mut coords = Vec::new();
        for y in 0..chunk_size {
            for x in 0..chunk_size {
                coords.push((x, y, z_target));
            }
        }
        fill_region::<T>(
            vol,
            chunk_size,
            IVec3::new(0, 0, 1),
            &coords,
            |cx, cy, _| idx(cx, cy, chunk_size - 1),
            &neighbor_info,
            &idx,
        );
    }

    // --- Edges ---
    for dx in 0..PADDING {
        for dy in 0..PADDING {
            let mut coords = Vec::new();
            for z in 0..chunk_size {
                coords.push((chunk_size + dx, chunk_size + dy, z));
            }
            fill_region::<T>(
                vol,
                chunk_size,
                IVec3::new(1, 1, 0),
                &coords,
                |_, _, cz| idx(chunk_size - 1, chunk_size - 1, cz),
                &neighbor_info,
                &idx,
            );
        }
    }

    for dx in 0..PADDING {
        for dz in 0..PADDING {
            let mut coords = Vec::new();
            for y in 0..chunk_size {
                coords.push((chunk_size + dx, y, chunk_size + dz));
            }
            fill_region::<T>(
                vol,
                chunk_size,
                IVec3::new(1, 0, 1),
                &coords,
                |_, cy, _| idx(chunk_size - 1, cy, chunk_size - 1),
                &neighbor_info,
                &idx,
            );
        }
    }

    for dy in 0..PADDING {
        for dz in 0..PADDING {
            let mut coords = Vec::new();
            for x in 0..chunk_size {
                coords.push((x, chunk_size + dy, chunk_size + dz));
            }
            fill_region::<T>(
                vol,
                chunk_size,
                IVec3::new(0, 1, 1),
                &coords,
                |cx, _, _| idx(cx, chunk_size - 1, chunk_size - 1),
                &neighbor_info,
                &idx,
            );
        }
    }

    // --- Corner ---
    for dx in 0..PADDING {
        for dy in 0..PADDING {
            for dz in 0..PADDING {
                let coords = vec![(chunk_size + dx, chunk_size + dy, chunk_size + dz)];
                fill_region::<T>(
                    vol,
                    chunk_size,
                    IVec3::new(1, 1, 1),
                    &coords,
                    |_, _, _| idx(chunk_size - 1, chunk_size - 1, chunk_size - 1),
                    &neighbor_info,
                    &idx,
                );
            }
        }
    }
}

fn fill_region<T: ApronSample>(
    vol: &mut [T::Elem],
    chunk_size: u32,
    offset: IVec3,
    iter_ranges: &[(u32, u32, u32)],
    fallback_idx: impl Fn(u32, u32, u32) -> usize,
    neighbor_info: &impl Fn(IVec3) -> Option<(Box<[T::Elem]>, u32)>,
    idx: &impl Fn(u32, u32, u32) -> usize,
) {
    if let Some((nx_data, n_size)) = neighbor_info(offset) {
        let scale_ratio = n_size as f32 / chunk_size as f32;

        for &(cx, cy, cz) in iter_ranges {
            let local_x = if cx >= chunk_size {
                (cx - chunk_size) as f32
            } else {
                cx as f32
            };
            let local_y = if cy >= chunk_size {
                (cy - chunk_size) as f32
            } else {
                cy as f32
            };
            let local_z = if cz >= chunk_size {
                (cz - chunk_size) as f32
            } else {
                cz as f32
            };

            let nx_coord = local_x * scale_ratio;
            let ny_coord = local_y * scale_ratio;
            let nz_coord = local_z * scale_ratio;
            let val = T::rescale(
                T::sample_apron(&nx_data, n_size, nx_coord, ny_coord, nz_coord),
                scale_ratio,
            );
            vol[idx(cx, cy, cz)] = val;
        }
    } else {
        for &(cx, cy, cz) in iter_ranges {
            let src = fallback_idx(cx, cy, cz);
            let val = vol[src];
            let dst = idx(cx, cy, cz);
            vol[dst] = val;
        }
    }
}

/// Trilinear neighbor sampling — meaningful for continuous data like SDF
/// distances. Used by `SDFField`'s `ApronSample` impl. Not exported for
/// general use: categorical fields (e.g. material ids) must not blend
/// samples this way — see `MaterialField`'s nearest-neighbor `ApronSample`
/// impl instead.
pub fn sample_neighbor(data: &[f32], size: u32, x: f32, y: f32, z: f32) -> f32 {
    let max_coord = (size - 1) as f32;
    let x = x.clamp(0.0, max_coord);
    let y = y.clamp(0.0, max_coord);
    let z = z.clamp(0.0, max_coord);

    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let z0 = z.floor() as usize;
    let x1 = (x0 + 1).min(size as usize - 1);
    let y1 = (y0 + 1).min(size as usize - 1);
    let z1 = (z0 + 1).min(size as usize - 1);

    let tx = x - x0 as f32;
    let ty = y - y0 as f32;
    let tz = z - z0 as f32;

    let s = size as usize;
    let sample = |ix: usize, iy: usize, iz: usize| -> f32 { data[(iz * s + iy) * s + ix] };

    // Trilinear interpolation over the 8 surrounding neighbor voxels.
    let c000 = sample(x0, y0, z0);
    let c100 = sample(x1, y0, z0);
    let c010 = sample(x0, y1, z0);
    let c110 = sample(x1, y1, z0);
    let c001 = sample(x0, y0, z1);
    let c101 = sample(x1, y0, z1);
    let c011 = sample(x0, y1, z1);
    let c111 = sample(x1, y1, z1);

    let c00 = c000 * (1.0 - tx) + c100 * tx;
    let c10 = c010 * (1.0 - tx) + c110 * tx;
    let c01 = c001 * (1.0 - tx) + c101 * tx;
    let c11 = c011 * (1.0 - tx) + c111 * tx;

    let c0 = c00 * (1.0 - ty) + c10 * ty;
    let c1 = c01 * (1.0 - ty) + c11 * ty;

    c0 * (1.0 - tz) + c1 * tz
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
            (Entity, Option<&VisibilityField>),
            (
                With<crate::Chunk>,
                With<LOD>,
                With<crate::field::surface::HasSurface>,
            ),
        >,
    >,
    arena_set: Option<ResMut<VoxelChunkArenaSet>>,
) {
    let Some(mut arena_set) = arena_set else {
        return;
    };
    let wanted: std::collections::HashSet<MainEntity> = live
        .iter()
        .filter(|(_, v)| !v.is_some_and(|v| v.is_all_empty()))
        .map(|(e, _)| MainEntity::from(e))
        .collect();
    let stale: Vec<MainEntity> = arena_set
        .iter()
        .flat_map(|a| a.slot_of_main_entity.keys().copied())
        .filter(|e| !wanted.contains(e))
        .collect();
    for e in stale {
        arena_set.release(e);
    }
}
