use std::marker::PhantomData;

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
    CHUNK_SIZE, ChunkManager, ChunkPosition, LOD, Versionable, VoxelDataSlice,
    voxel::{
        arena::{ChunkMeta, VoxelChunkArena},
        pipeline::{VoxelDummyMaterial, VoxelPipelineLayouts, VoxelRasterPipeline},
        types::{CollisionMeshData, MeshReadbackChannel, PendingMeshReadback},
    },
};

use super::{buffers::GpuVoxelChunkBuffers, pipeline::VoxelComputePipeline};

#[derive(Component)]
pub struct ExtractedChunkField<T: Send + Sync + 'static> {
    pub main_entity: MainEntity,
    pub chunk_pos: IVec3,
    pub padded_sdf_data: Vec<u8>,
    pub size: u32,
    pub lod: LOD,
    pub _type: PhantomData<T>,
}

pub fn prepare_voxel_arena<T: Send + Sync + 'static>(
    mut arena: ResMut<VoxelChunkArena>,
    render_queue: Res<RenderQueue>,
    extracted_chunks: Query<(Entity, &ExtractedChunkField<T>)>,
    mut commands: Commands,
) {
    arena.dirty_slots.clear();

    for (extracted_entity, extracted_sdf) in extracted_chunks.iter() {
        let Some(slot) = arena.slot_for(extracted_sdf.main_entity) else {
            warn!(
                "[prepare_voxel_arena] arena full ({} slots), dropping chunk {:?}",
                arena.max_chunks, extracted_sdf.chunk_pos
            );
            commands.entity(extracted_entity).despawn();
            continue;
        };

        if !arena.active_slots.contains(&slot) {
            arena.active_slots.push(slot);
        }

        let size = extracted_sdf.size;
        debug_assert_eq!(
            size, arena.texture_size,
            "mixed LOD in one arena — bucket by LOD"
        );

        // Upload this chunk's padded SDF data into its slot of the shared sdf_buffer.
        let sdf_offset_bytes = slot as u64 * arena.sdf_elems_per_chunk as u64 * 4;
        render_queue.write_buffer(
            &arena.sdf_buffer,
            sdf_offset_bytes,
            &extracted_sdf.padded_sdf_data,
        );

        let chunk_voxels = size - 2;
        let voxel_size = CHUNK_SIZE / chunk_voxels as f32;
        let chunk_world_origin = extracted_sdf.chunk_pos.as_vec3() * CHUNK_SIZE;

        // active_list_pos is unknown until arena.active_slots is finalized for
        // this frame (see the patch loop below, after all chunks are processed).
        // Written as 0 here as a placeholder; patched immediately after.
        let meta = ChunkMeta {
            chunk_world_origin: chunk_world_origin.into(),
            voxel_size,
            sdf_offset: slot * arena.sdf_elems_per_chunk,
            cell_offset: slot * arena.total_cells,
            active_list_pos: 0,
            _pad: 0,
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
    const ACTIVE_LIST_POS_OFFSET: u64 = std::mem::offset_of!(ChunkMeta, active_list_pos) as u64;

    for (pos, &slot) in arena.active_slots.iter().enumerate() {
        let offset = slot as u64 * std::mem::size_of::<ChunkMeta>() as u64 + ACTIVE_LIST_POS_OFFSET;
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

pub fn dispatch_voxel_compute_passes_batched(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<VoxelComputePipeline>,
    arena: Res<VoxelChunkArena>,
    mut debug_requested: Local<bool>,
) {
    let active = arena.active_chunk_count();
    if active == 0 {
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
        warn!("[dispatch_voxel_compute_passes_batched] pipelines not yet compiled, skipping");
        return;
    };

    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("voxel_compute_encoder_batched"),
    });

    let wg_per_chunk_pass1 = arena.cell_count.div_ceil(4);
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("surface_nets_pass1_batched"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pass1_pipeline);
        pass.set_bind_group(0, &arena.pass1_bind_group, &[]);
        pass.dispatch_workgroups(
            wg_per_chunk_pass1,
            wg_per_chunk_pass1,
            wg_per_chunk_pass1 * active,
        );
    }

    //let wg_per_chunk_scan = ((arena.total_cells + 511) / 512).max(1);
    let wg_per_chunk_scan = arena.blocks_per_chunk;
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("stream_compaction_scan_batched"),
            timestamp_writes: None,
        });
        pass.set_pipeline(stream_compaction_pipeline);
        pass.set_bind_group(0, &arena.compaction_bind_group, &[]);
        pass.dispatch_workgroups(wg_per_chunk_scan * active, 1, 1);
        //pass.dispatch_workgroups(active, 1, 1);
    }

    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("stream_compaction_scan_block_sums_batched"),
            timestamp_writes: None,
        });
        pass.set_pipeline(scan_block_sums_pipeline);
        pass.set_bind_group(0, &arena.compaction_bind_group, &[]);
        pass.dispatch_workgroups(active, 1, 1); // one block-sum scan per chunk, indexed by workgroup_id.x
    }

    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("stream_compaction_resolve_batched"),
            timestamp_writes: None,
        });
        pass.set_pipeline(stream_compaction_resolve_pipeline);
        pass.set_bind_group(0, &arena.compaction_bind_group, &[]);
        pass.dispatch_workgroups(wg_per_chunk_scan * active, 1, 1);
    }

    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("stream_compaction_write_active_counts"),
            timestamp_writes: None,
        });
        pass.set_pipeline(write_chunk_active_count_pipeline);
        pass.set_bind_group(0, &arena.compaction_bind_group, &[]);
        pass.dispatch_workgroups(active, 1, 1);
    }

    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("compute_chunk_bases"),
            timestamp_writes: None,
        });
        pass.set_pipeline(chunk_bases_pipeline);
        pass.set_bind_group(0, &arena.chunk_bases_bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    let wg_per_chunk_pass3 = arena.cell_count.div_ceil(8);
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("surface_nets_pass3_batched"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pass3_pipeline);
        pass.set_bind_group(0, &arena.pass3_bind_group, &[]);
        pass.dispatch_workgroups(
            wg_per_chunk_pass3,
            wg_per_chunk_pass3,
            wg_per_chunk_pass3 * active,
        );
    }

    // Single readback copy per resource type, covering every active slot's span at once.
    encoder.copy_buffer_to_buffer(
        &arena.indirect_args_buffer,
        0,
        &arena.readback_indirect_buffer,
        0,
        4 * arena.max_chunks as u64,
    );
    encoder.copy_buffer_to_buffer(
        &arena.final_vertex_buffer,
        0,
        &arena.readback_vertex_buffer,
        0,
        arena.final_vertex_buffer.size(),
    );
    encoder.copy_buffer_to_buffer(
        &arena.index_buffer,
        0,
        &arena.readback_index_buffer,
        0,
        arena.index_buffer.size(),
    );
    encoder.copy_buffer_to_buffer(
        &arena.overflow_flag_buffer,
        0,
        &arena.overflow_readback_buffer,
        0,
        4,
    );

    render_queue.submit(std::iter::once(encoder.finish()));
}

/// Runs one GPU compute pass across every chunk in `chunks`, reusing the
/// same pipeline for all of them. `bind_group_of` and `workgroups_of` let
/// each of the five voxel passes plug in its own per-chunk bind group and
/// dispatch size while sharing the begin/set-pipeline/loop/dispatch shape.
fn run_compute_pass<'a>(
    command_encoder: &mut CommandEncoder,
    label: &'static str,
    pipeline: &ComputePipeline,
    chunks: impl Iterator<Item = &'a GpuVoxelChunkBuffers>,
    bind_group_of: impl Fn(&'a GpuVoxelChunkBuffers) -> &'a BindGroup,
    workgroups_of: impl Fn(&'a GpuVoxelChunkBuffers) -> (u32, u32, u32),
) {
    let mut compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: None,
    });
    compute_pass.set_pipeline(pipeline);
    for chunk in chunks {
        let (x, y, z) = workgroups_of(chunk);
        compute_pass.set_bind_group(0, bind_group_of(chunk), &[]);
        compute_pass.dispatch_workgroups(x, y, z);
    }
}

pub fn dispatch_voxel_compute_passes(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<VoxelComputePipeline>,
    chunk_buffers: Query<&GpuVoxelChunkBuffers>,
    pending_readback: Query<&GpuVoxelChunkBuffers, With<PendingMeshReadback>>,
) {
    if chunk_buffers.is_empty() {
        return;
    }

    let (
        Some(pass1_pipeline),
        Some(stream_compaction_pipeline),
        Some(scan_block_sums_pipeline),
        Some(stream_compaction_resolve_pipeline),
        Some(pass3_pipeline),
    ) = (
        pipeline_cache.get_compute_pipeline(pipeline.pass1_pipeline_id),
        pipeline_cache.get_compute_pipeline(pipeline.stream_compaction_pipeline_id),
        pipeline_cache.get_compute_pipeline(pipeline.scan_block_sums_pipeline_id),
        pipeline_cache.get_compute_pipeline(pipeline.stream_compaction_resolve_pipeline_id),
        pipeline_cache.get_compute_pipeline(pipeline.pass3_pipeline_id),
    )
    else {
        warn!(
            "[dispatch_voxel_compute_passes] one or more pipelines not yet compiled, skipping dispatch"
        );
        return;
    };

    let mut command_encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("voxel_compute_encoder"),
    });

    // Reset index_count to 0 for every chunk, recorded into this same command
    // encoder so it's strictly ordered before the compute passes below.
    //for chunk in chunk_buffers.iter() {
    //command_encoder.clear_buffer(&chunk.indirect_args_buffer, 0, Some(4));
    //}

    // --- Pass 1: surface_nets_pass1, all chunks ---
    run_compute_pass(
        &mut command_encoder,
        "surface_nets_pass1_all_chunks",
        pass1_pipeline,
        chunk_buffers.iter(),
        |c| &c.pass1_surface_bind_group,
        |c| c.dispatch_grid_3d(4).into(),
    );

    // --- Pass 2: stream_compaction_scan, all chunks ---
    run_compute_pass(
        &mut command_encoder,
        "stream_compaction_scan_all_chunks",
        stream_compaction_pipeline,
        chunk_buffers.iter(),
        |c| &c.compaction_bind_group,
        |c| (c.dispatch_blocks_1d(512), 1, 1),
    );

    // --- Pass 2.5: scan_block_sums, all chunks ---
    run_compute_pass(
        &mut command_encoder,
        "stream_compaction_scan_block_sums_all_chunks",
        scan_block_sums_pipeline,
        chunk_buffers.iter(),
        |c| &c.compaction_bind_group,
        |_| (1, 1, 1),
    );

    // --- Pass 3: stream_compaction_resolve, all chunks ---
    run_compute_pass(
        &mut command_encoder,
        "stream_compaction_resolve_all_chunks",
        stream_compaction_resolve_pipeline,
        chunk_buffers.iter(),
        |c| &c.compaction_bind_group,
        |c| (c.dispatch_blocks_1d(512), 1, 1),
    );

    // --- Pass 4: surface_nets_pass3, all chunks ---
    run_compute_pass(
        &mut command_encoder,
        "surface_nets_pass3_all_chunks",
        pass3_pipeline,
        chunk_buffers.iter(),
        |c| &c.pass3_surface_bind_group,
        |c| c.dispatch_grid_3d(8).into(),
    );

    for chunk in pending_readback.iter() {
        command_encoder.copy_buffer_to_buffer(
            &chunk.indirect_args_buffer,
            0,
            &chunk.readback_indirect_buffer,
            0,
            4,
        );
        command_encoder.copy_buffer_to_buffer(
            &chunk.final_vertex_buffer,
            0,
            &chunk.readback_vertex_buffer,
            0,
            chunk.final_vertex_buffer.size(),
        );
        command_encoder.copy_buffer_to_buffer(
            &chunk.index_buffer,
            0,
            &chunk.readback_index_buffer,
            0,
            chunk.index_buffer.size(),
        );
    }

    render_queue.submit(std::iter::once(command_encoder.finish()));
}

pub fn voxel_raster_pass(
    view: ViewQuery<(
        &ExtractedCamera,
        &ViewTarget,
        &ViewDepthTexture,
        &ViewUniformOffset,
    )>,
    view_uniforms: Res<ViewUniforms>,
    arena: Option<Res<VoxelChunkArena>>,
    pipeline_cache: Res<PipelineCache>,
    raster_pipeline: Res<VoxelRasterPipeline>,
    voxel_material: Res<VoxelDummyMaterial>,
    mut ctx: RenderContext,
) {
    // Arena may not exist yet (frame 0, before any chunk has been extracted).
    let Some(arena) = arena else {
        return;
    };
    if arena.active_slots.is_empty() {
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

    let material_bind_group = &voxel_material.bind_group;

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

    // Buffers are shared across all chunks now — bind once, not per-chunk.
    render_pass.set_vertex_buffer(0, arena.final_vertex_buffer.slice(..));
    render_pass.set_index_buffer(arena.index_buffer.slice(..), IndexFormat::Uint32);

    const ARGS_STRIDE: u64 = std::mem::size_of::<DrawIndexedIndirectArgs>() as u64;
    for &slot in arena.active_slots.iter() {
        let offset = slot as u64 * ARGS_STRIDE;
        render_pass.draw_indexed_indirect(&arena.indirect_args_buffer, offset);
    }
}

pub fn queue_mesh_readback_maps(
    mut commands: Commands,
    chunk_buffers: Query<(
        Entity,
        &GpuVoxelChunkBuffers,
        &PendingMeshReadback,
        &crate::chunk::ChunkPosition,
    )>,
    channel: Res<MeshReadbackChannel>,
) {
    for (entity, chunk, pending, pos) in chunk_buffers.iter() {
        let sender = channel.sender.clone();
        let chunk_pos = pos.0;
        let generation = pending.get_val();

        let vb = chunk.readback_vertex_buffer.clone();
        let ib = chunk.readback_index_buffer.clone();
        let cb = chunk.readback_indirect_buffer.clone();

        let cb_for_slice = cb.clone();
        cb_for_slice
            .slice(..)
            .map_async(MapMode::Read, move |result| {
                if result.is_err() {
                    return;
                }
                let index_count = {
                    let data = cb.slice(..).get_mapped_range();
                    u32::from_ne_bytes(data[0..4].try_into().unwrap())
                };
                cb.unmap();

                let vb2 = vb.clone();
                let ib2 = ib.clone();
                let sender2 = sender.clone();

                let ib2_for_slice = ib2.clone();
                ib2_for_slice.slice(..(index_count as u64 * 4)).map_async(
                    MapMode::Read,
                    move |r| {
                        if r.is_err() {
                            return;
                        }
                        let indices: Vec<u32> = {
                            let data = ib2.slice(..(index_count as u64 * 4)).get_mapped_range();
                            bytemuck::cast_slice(&data).to_vec()
                        };
                        ib2.unmap();

                        let max_vert = indices.iter().copied().max().unwrap_or(0) as u64 + 1;
                        let vb3 = vb2.clone();
                        let sender3 = sender2.clone();

                        let vb3_for_slice = vb3.clone();
                        vb3_for_slice
                            .slice(..(max_vert * 32))
                            .map_async(MapMode::Read, move |r| {
                                if r.is_err() {
                                    return;
                                }
                                let vertices: Vec<[f32; 3]> = {
                                    let data = vb3.slice(..(max_vert * 32)).get_mapped_range();
                                    data.chunks_exact(32)
                                        .map(|v| {
                                            let x = f32::from_ne_bytes(v[0..4].try_into().unwrap());
                                            let y = f32::from_ne_bytes(v[4..8].try_into().unwrap());
                                            let z =
                                                f32::from_ne_bytes(v[8..12].try_into().unwrap());
                                            [x, y, z]
                                        })
                                        .collect()
                                };
                                vb3.unmap();

                                let _ = sender3.send(CollisionMeshData {
                                    chunk_pos,
                                    generation,
                                    vertices,
                                    indices: indices.clone(),
                                });
                            });
                    },
                );
            });

        commands.entity(entity).remove::<PendingMeshReadback>();
    }
}

const NEIGHBORS_MASK: [IVec3; 7] = [
    ivec3(1, 0, 0),
    ivec3(0, 1, 0),
    ivec3(0, 0, 1),
    ivec3(1, 1, 0),
    ivec3(1, 0, 1),
    ivec3(0, 1, 1),
    ivec3(1, 1, 1),
];

pub fn extract_voxel_chunks<T>(
    mut commands: Commands,
    chunk_manager: Extract<Res<ChunkManager>>,
    query: Extract<Query<(Entity, &ChunkPosition, &T, &LOD)>>,
    mut last_versions: Local<std::collections::HashMap<IVec3, [u64; 8]>>,
) where
    T: Send + Sync + 'static + Component + Versionable + VoxelDataSlice,
{
    for (entity, pos, sdf, lod) in query.iter() {
        let size = lod.size();

        let mut versions = [0u64; 8];
        versions[0] = sdf.version(); // Updated from field access to Versionable trait method
        for (i, offset) in NEIGHBORS_MASK.iter().enumerate() {
            if let Some(n_entity) = chunk_manager.get_chunk(&(pos.0 + *offset)) {
                if let Ok((_, _, n_sdf, _)) = query.get(n_entity) {
                    versions[i + 1] = n_sdf.version(); // Updated from field access to Versionable trait method
                }
            }
        }

        if last_versions.get(&pos.0) == Some(&versions) {
            continue;
        }
        last_versions.insert(pos.0, versions);

        let padded_size = size + PADDING;
        let mut vol = vec![0.0f32; (padded_size * padded_size * padded_size) as usize];

        let raw = sdf.data_slice();
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let src = ((z * size + y) * size + x) as usize;
                    let dst = ((z * padded_size + y) * padded_size + x) as usize;
                    vol[dst] = raw[src];
                }
            }
        }

        fill_apron(&mut vol, pos.0, &chunk_manager, &query);

        let padded_sdf_data: Vec<u8> = bytemuck::cast_slice(&vol).to_vec();
        commands.spawn(ExtractedChunkField::<T> {
            main_entity: entity.into(),
            chunk_pos: pos.0,
            padded_sdf_data,
            size: padded_size,
            _type: PhantomData,
        });
    }
}

const PADDING: u32 = 2;

pub fn fill_apron<T>(
    vol: &mut [f32],
    chunk_pos: IVec3,
    chunk_manager: &ChunkManager,
    query: &Query<(Entity, &ChunkPosition, &T, &LOD)>,
) where
    T: Component + VoxelDataSlice,
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

    let neighbor_info = |offset: IVec3| -> Option<(Box<[f32]>, u32)> {
        let n_entity = chunk_manager.get_chunk(&(chunk_pos + offset))?;
        let Ok((_, _, n_sdf, n_lod)) = query.get(n_entity) else {
            return None;
        };
        let data = n_sdf.data_slice().to_vec().into_boxed_slice();
        Some((data, n_lod.size()))
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
        fill_region(
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
        fill_region(
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
        fill_region(
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
            fill_region(
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
            fill_region(
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
            fill_region(
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
                fill_region(
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

fn fill_region(
    vol: &mut [f32],
    chunk_size: u32,
    offset: IVec3,
    iter_ranges: &[(u32, u32, u32)],
    fallback_idx: impl Fn(u32, u32, u32) -> usize,
    neighbor_info: &impl Fn(IVec3) -> Option<(Box<[f32]>, u32)>,
    idx: &impl Fn(u32, u32, u32) -> usize,
) {
    if let Some((nx_data, n_size)) = neighbor_info(offset) {
        //let scale_ratio = chunk_size as f32 / n_size as f32;
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

            let val = sample_neighbor(&nx_data, n_size, nx_coord, ny_coord, nz_coord);
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

fn sample_neighbor(data: &[f32], size: u32, x: f32, y: f32, z: f32) -> f32 {
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

/// Runs before `prepare_voxel_arena`. Creates and inserts `VoxelChunkArena`
/// the first time chunk data appears — we can't build it earlier because
/// `cell_count`/`texture_size` come from the chunk's `size`, which isn't
/// known until extraction has produced at least one `ExtractedChunkField`.
pub fn init_voxel_arena<T: Send + Sync + 'static>(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    layouts: Res<VoxelPipelineLayouts>,
    extracted_chunks: Query<&ExtractedChunkField<T>>,
    arena: Option<Res<VoxelChunkArena>>,
) {
    if arena.is_some() {
        return;
    }
    let Some(first) = extracted_chunks.iter().next() else {
        return; // nothing extracted yet this frame, try again next frame
    };

    let size = first.size;
    let chunk_voxels = size - 2;
    let cell_count = chunk_voxels + 1;

    commands.insert_resource(VoxelChunkArena::new(
        &render_device,
        &layouts,
        cell_count,
        size,
    ));
}
