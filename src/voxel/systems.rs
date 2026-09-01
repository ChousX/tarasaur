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
    CHUNK_SIZE, ChunkManager, ChunkPosition, SDFField,
    voxel::{
        pipeline::{VoxelDummyMaterial, VoxelRasterPipeline},
        types::{
            CollisionMeshData, MeshReadbackChannel, Pass1Uniforms, Pass3Uniforms,
            PendingMeshReadback,
        },
    },
};

use super::{
    buffers::{ExtractedChunkSdf, GpuVoxelChunkBuffers},
    pipeline::{VoxelComputePipeline, VoxelPipelineLayouts},
    types::{CompactionUniforms, DrawIndexedIndirectArgs},
};
pub fn prepare_voxel_chunk_buffers(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    layouts: Res<VoxelPipelineLayouts>,
    extracted_chunks: Query<(Entity, &ExtractedChunkSdf)>,
    mut existing_buffers: Query<(Entity, &MainEntity, &mut GpuVoxelChunkBuffers)>,
) {
    for (extracted_entity, extracted_sdf) in extracted_chunks.iter() {
        let size = extracted_sdf.size;
        let chunk_voxels = size - 2;
        let cell_count = chunk_voxels + 1;
        let total_cells = (cell_count * cell_count * cell_count) as usize;
        let unpadded_bytes_per_row = size * std::mem::size_of::<f32>() as u32;
        let padded_bytes_per_row = (unpadded_bytes_per_row + 255) & !255;
        let mut found_existing = false;

        for (buf_entity, main_entity, mut gpu_buffers) in existing_buffers.iter_mut() {
            if *main_entity == extracted_sdf.main_entity {
                render_queue.write_texture(
                    TexelCopyTextureInfo {
                        texture: &gpu_buffers.sdf_texture,
                        mip_level: 0,
                        origin: Origin3d::ZERO,
                        aspect: TextureAspect::All,
                    },
                    &extracted_sdf.padded_sdf_data,
                    TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded_bytes_per_row),
                        rows_per_image: Some(size),
                    },
                    Extent3d {
                        width: size,
                        height: size,
                        depth_or_array_layers: size,
                    },
                );
                gpu_buffers.mesh_generation += 1;
                commands
                    .entity(buf_entity)
                    .insert(PendingMeshReadback::new(gpu_buffers.mesh_generation));
                found_existing = true;
                break;
            }
        }

        if !found_existing {
            let sdf_texture = render_device.create_texture(&TextureDescriptor {
                label: Some("chunk_sdf_texture"),
                size: Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: size,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D3,
                format: TextureFormat::R32Float,
                usage: TextureUsages::STORAGE_BINDING
                    | TextureUsages::COPY_DST
                    | TextureUsages::COPY_SRC,
                view_formats: &[],
            });

            render_queue.write_texture(
                TexelCopyTextureInfo {
                    texture: &sdf_texture,
                    mip_level: 0,
                    origin: Origin3d::ZERO,
                    aspect: TextureAspect::All,
                },
                &extracted_sdf.padded_sdf_data,
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(size),
                },
                Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: size,
                },
            );

            let sdf_view = sdf_texture.create_view(&TextureViewDescriptor::default());

            let flags_buffer = render_device.create_buffer(&BufferDescriptor {
                label: Some("chunk_flags_buffer"),
                size: (total_cells * std::mem::size_of::<u32>()) as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let compacted_offsets_buffer = render_device.create_buffer(&BufferDescriptor {
                label: Some("chunk_compacted_offsets_buffer"),
                size: (total_cells * std::mem::size_of::<u32>()) as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let scattered_vertex_buffer = render_device.create_buffer(&BufferDescriptor {
                label: Some("chunk_scattered_vertex_buffer"),
                size: (total_cells * 32) as u64,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            });

            let final_vertex_buffer = render_device.create_buffer(&BufferDescriptor {
                label: Some("chunk_final_vertex_buffer"),
                size: (total_cells * 32) as u64,
                usage: BufferUsages::STORAGE | BufferUsages::VERTEX | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let index_buffer = render_device.create_buffer(&BufferDescriptor {
                label: Some("chunk_index_buffer"),
                size: (total_cells * 18 * std::mem::size_of::<u32>()) as u64,
                usage: BufferUsages::STORAGE | BufferUsages::INDEX | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let initial_indirect_args = DrawIndexedIndirectArgs {
                index_count: 0,
                instance_count: 1,
                first_index: 0,
                base_vertex: 0,
                first_instance: 0,
            };

            let indirect_args_buffer =
                render_device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("chunk_indirect_args_buffer"),
                    contents: bytemuck::bytes_of(&initial_indirect_args),
                    usage: BufferUsages::STORAGE
                        | BufferUsages::INDIRECT
                        | BufferUsages::COPY_DST
                        | BufferUsages::COPY_SRC,
                });

            let compaction_uniforms = CompactionUniforms {
                chunk_size: cell_count,
                total_cells: total_cells as u32,
                _pad0: 0,
                _pad1: 0,
            };

            let compaction_uniform_buffer =
                render_device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("chunk_compaction_uniform_buffer"),
                    contents: bytemuck::bytes_of(&compaction_uniforms),
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                });

            let voxel_size = CHUNK_SIZE / chunk_voxels as f32;
            let chunk_world_origin = extracted_sdf.chunk_pos.as_vec3() * CHUNK_SIZE;

            let pass3_uniforms = Pass3Uniforms {
                cell_count,
                texture_size: size,
                voxel_size,
                _pad0: [0; 1],
                chunk_world_origin: chunk_world_origin.into(),
                _pad1: 0,
            };

            let pass3_uniform_buffer =
                render_device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("chunk_pass3_uniform_buffer"),
                    contents: bytemuck::bytes_of(&pass3_uniforms),
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                });

            let workgroup_capacity = 512;
            let num_blocks = ((total_cells + workgroup_capacity - 1) / workgroup_capacity) as u64;
            let block_sums_buffer = render_device.create_buffer(&BufferDescriptor {
                label: Some("chunk_block_sums_buffer"),
                size: num_blocks * std::mem::size_of::<u32>() as u64,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            });

            let pass1_uniforms = Pass1Uniforms {
                cell_count,
                texture_size: size,
                _pad0: 0,
                _pad1: 0,
            };
            let pass1_uniform_buffer =
                render_device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("chunk_pass1_uniform_buffer"),
                    contents: bytemuck::bytes_of(&pass1_uniforms),
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                });

            let pass1_surface_bind_group = render_device.create_bind_group(
                Some("chunk_pass1_surface_bind_group"),
                &layouts.pass1_surface_layout,
                &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&sdf_view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: flags_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: compacted_offsets_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: scattered_vertex_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 4,
                        resource: final_vertex_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 5,
                        resource: indirect_args_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 6,
                        resource: pass1_uniform_buffer.as_entire_binding(),
                    },
                ],
            );

            let pass3_surface_bind_group = render_device.create_bind_group(
                Some("chunk_pass3_surface_bind_group"),
                &layouts.pass3_surface_layout,
                &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&sdf_view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: flags_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: compacted_offsets_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: final_vertex_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 4,
                        resource: index_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 5,
                        resource: indirect_args_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 6,
                        resource: pass3_uniform_buffer.as_entire_binding(),
                    },
                ],
            );

            let compaction_bind_group = render_device.create_bind_group(
                Some("chunk_compaction_bind_group"),
                &layouts.compaction_bind_group_layout,
                &[
                    BindGroupEntry {
                        binding: 0,
                        resource: compaction_uniform_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: flags_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: compacted_offsets_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: block_sums_buffer.as_entire_binding(),
                    },
                ],
            );
            // systems.rs, inside prepare_voxel_chunk_buffers, alongside the other buffer creation
            let readback_vertex_buffer = render_device.create_buffer(&BufferDescriptor {
                label: Some("chunk_readback_vertex_buffer"),
                size: (total_cells * 32) as u64,
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            let readback_index_buffer = render_device.create_buffer(&BufferDescriptor {
                label: Some("chunk_readback_index_buffer"),
                size: (total_cells * 18 * std::mem::size_of::<u32>()) as u64,
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            let readback_indirect_buffer = render_device.create_buffer(&BufferDescriptor {
                label: Some("chunk_readback_indirect_buffer"),
                size: 4, // just index_count
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            commands.spawn((
                extracted_sdf.main_entity,
                GpuVoxelChunkBuffers {
                    chunk_coord: extracted_sdf.chunk_pos,
                    lod: size,
                    chunk_voxels,
                    sdf_texture,
                    sdf_view,
                    flags_buffer,
                    compacted_offsets_buffer,
                    scattered_vertex_buffer,
                    final_vertex_buffer,
                    index_buffer,
                    indirect_args_buffer,
                    compaction_uniform_buffer,
                    block_sums_buffer,
                    pass_uniform_buffer: pass3_uniform_buffer,
                    pass1_surface_bind_group,
                    pass3_surface_bind_group,
                    compaction_bind_group,
                    readback_vertex_buffer,
                    readback_index_buffer,
                    readback_indirect_buffer,
                    mesh_generation: 0,
                },
                PendingMeshReadback::new(0),
            ));
        }

        commands.entity(extracted_entity).despawn();
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
    for chunk in chunk_buffers.iter() {
        command_encoder.clear_buffer(&chunk.indirect_args_buffer, 0, Some(4));
    }

    // --- Pass 1: surface_nets_pass1, all chunks in one compute pass ---
    {
        let mut compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("surface_nets_pass1_all_chunks"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(pass1_pipeline);
        for chunk in chunk_buffers.iter() {
            let cell_count = chunk.chunk_voxels + 1;
            let p1_grid = (cell_count + 3) / 4;
            compute_pass.set_bind_group(0, &chunk.pass1_surface_bind_group, &[]);
            compute_pass.dispatch_workgroups(p1_grid, p1_grid, p1_grid);
        }
    }

    // --- Pass 2: stream_compaction_scan, all chunks ---
    {
        let mut compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("stream_compaction_scan_all_chunks"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(stream_compaction_pipeline);
        for chunk in chunk_buffers.iter() {
            // Pass 2 (stream_compaction_scan)
            let workgroup_size = 512;
            let cell_count = chunk.chunk_voxels + 1;
            let total_cells = cell_count * cell_count * cell_count;
            let num_blocks = (total_cells + workgroup_size - 1) / workgroup_size;
            compute_pass.set_bind_group(0, &chunk.compaction_bind_group, &[]);
            compute_pass.dispatch_workgroups(num_blocks, 1, 1);
        }
    }

    // --- Pass 2.5: scan_block_sums, all chunks ---
    {
        let mut compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("stream_compaction_scan_block_sums_all_chunks"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(scan_block_sums_pipeline);
        for chunk in chunk_buffers.iter() {
            compute_pass.set_bind_group(0, &chunk.compaction_bind_group, &[]);
            compute_pass.dispatch_workgroups(1, 1, 1);
        }
    }

    // --- Pass 3: stream_compaction_resolve, all chunks ---
    {
        let mut compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("stream_compaction_resolve_all_chunks"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(stream_compaction_resolve_pipeline);
        for chunk in chunk_buffers.iter() {
            let workgroup_size = 512;
            let cell_count = chunk.chunk_voxels + 1;
            let total_cells = cell_count * cell_count * cell_count;
            let num_blocks = (total_cells + workgroup_size - 1) / workgroup_size;
            compute_pass.set_bind_group(0, &chunk.compaction_bind_group, &[]);
            compute_pass.dispatch_workgroups(num_blocks, 1, 1);
        }
    }

    // --- Pass 4: surface_nets_pass3, all chunks ---
    {
        let mut compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("surface_nets_pass3_all_chunks"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(pass3_pipeline);
        for chunk in chunk_buffers.iter() {
            let cell_count = chunk.chunk_voxels + 1;
            let p3_grid = (cell_count + 7) / 8;
            compute_pass.set_bind_group(0, &chunk.pass3_surface_bind_group, &[]);
            compute_pass.dispatch_workgroups(p3_grid, p3_grid, p3_grid);
        }
    }
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
    render_queue: Res<RenderQueue>,
    chunk_buffers: Query<&GpuVoxelChunkBuffers>,
    pipeline_cache: Res<PipelineCache>,
    raster_pipeline: Res<VoxelRasterPipeline>,
    voxel_material: Res<VoxelDummyMaterial>,
    mut ctx: RenderContext,
) {
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

    for chunk in chunk_buffers.iter() {
        render_pass.set_vertex_buffer(0, chunk.final_vertex_buffer.slice(..));
        render_pass.set_index_buffer(chunk.index_buffer.slice(..), IndexFormat::Uint32);
        render_pass.draw_indexed_indirect(&chunk.indirect_args_buffer, 0);
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

pub fn extract_voxel_chunks(
    mut commands: Commands,
    chunk_manager: Extract<Res<ChunkManager>>,
    query: Extract<Query<(Entity, &ChunkPosition, &SDFField)>>,
    mut last_versions: Local<std::collections::HashMap<IVec3, [u64; 8]>>,
) {
    for (entity, pos, sdf) in query.iter() {
        let size = sdf.lod.size();

        let mut versions = [0u64; 8];
        versions[0] = sdf.version;
        for (i, offset) in NEIGHBORS_MASK.iter().enumerate() {
            if let Some(n_entity) = chunk_manager.get_chunk(&(pos.0 + *offset)) {
                if let Ok((_, _, n_sdf)) = query.get(n_entity) {
                    versions[i + 1] = n_sdf.version;
                }
            }
            // else: leave as 0 — "no neighbor loaded yet" is itself a state,
            // so when that neighbor later appears with version >= 1, the
            // mismatch against our cached `last_versions` entry will
            // correctly trigger a re-extraction.
        }

        if last_versions.get(&pos.0) == Some(&versions) {
            continue;
        }
        last_versions.insert(pos.0, versions);

        let padded_size = size + 2;
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

        fill_apron(&mut vol, size, padded_size, pos.0, &chunk_manager, &query);

        let unpadded_bytes_per_row = padded_size * 4;
        let padded_bytes_per_row = (unpadded_bytes_per_row + 255) & !255;
        let padding_per_row = (padded_bytes_per_row - unpadded_bytes_per_row) as usize;
        let mut padded_sdf_data =
            Vec::with_capacity((padded_bytes_per_row * padded_size * padded_size) as usize);
        for z in 0..padded_size {
            for y in 0..padded_size {
                let start = ((z * padded_size + y) * padded_size) as usize;
                let end = start + padded_size as usize;
                padded_sdf_data.extend_from_slice(bytemuck::cast_slice(&vol[start..end]));
                padded_sdf_data.resize(padded_sdf_data.len() + padding_per_row, 0);
            }
        }

        commands.spawn(ExtractedChunkSdf {
            main_entity: entity.into(),
            chunk_pos: pos.0,
            padded_sdf_data,
            size: padded_size,
        });
    }
}

fn fill_apron(
    vol: &mut [f32],
    size: u32,
    padded_size: u32,
    chunk_pos: IVec3,
    chunk_manager: &ChunkManager,
    query: &Query<(Entity, &ChunkPosition, &SDFField)>,
) {
    let idx =
        |x: u32, y: u32, z: u32| -> usize { ((z * padded_size + y) * padded_size + x) as usize };
    let n_idx = |x: u32, y: u32, z: u32| -> usize { ((z * size + y) * size + x) as usize };

    let neighbor_raw = |offset: IVec3| -> Option<Box<[f32]>> {
        let n_entity = chunk_manager.get_chunk(&(chunk_pos + offset))?;
        let (_, _, n_sdf) = query.get(n_entity).ok()?;
        Some(n_sdf.data_slice().to_vec().into_boxed_slice())
    };

    // --- Faces: +x, +y, +z, 2 layers deep ---
    if let Some(nx) = neighbor_raw(IVec3::new(1, 0, 0)) {
        for d in 0..2u32 {
            for z in 0..size {
                for y in 0..size {
                    vol[idx(size + d, y, z)] = nx[n_idx(d, y, z)];
                }
            }
        }
    } else {
        for d in 0..2u32 {
            for z in 0..size {
                for y in 0..size {
                    vol[idx(size + d, y, z)] = vol[idx(size - 1, y, z)];
                }
            }
        }
    }

    if let Some(ny) = neighbor_raw(IVec3::new(0, 1, 0)) {
        for d in 0..2u32 {
            for z in 0..size {
                for x in 0..size {
                    vol[idx(x, size + d, z)] = ny[n_idx(x, d, z)];
                }
            }
        }
    } else {
        for d in 0..2u32 {
            for z in 0..size {
                for x in 0..size {
                    vol[idx(x, size + d, z)] = vol[idx(x, size - 1, z)];
                }
            }
        }
    }

    if let Some(nz) = neighbor_raw(IVec3::new(0, 0, 1)) {
        for d in 0..2u32 {
            for y in 0..size {
                for x in 0..size {
                    vol[idx(x, y, size + d)] = nz[n_idx(x, y, d)];
                }
            }
        }
    } else {
        for d in 0..2u32 {
            for y in 0..size {
                for x in 0..size {
                    vol[idx(x, y, size + d)] = vol[idx(x, y, size - 1)];
                }
            }
        }
    }

    // --- Edges: +x+y, +x+z, +y+z — 2x2 block, 2 layers on each of the two axes ---
    if let Some(nxy) = neighbor_raw(IVec3::new(1, 1, 0)) {
        for dx in 0..2u32 {
            for dy in 0..2u32 {
                for z in 0..size {
                    vol[idx(size + dx, size + dy, z)] = nxy[n_idx(dx, dy, z)];
                }
            }
        }
    } else {
        for dx in 0..2u32 {
            for dy in 0..2u32 {
                for z in 0..size {
                    vol[idx(size + dx, size + dy, z)] = vol[idx(size - 1, size - 1, z)];
                }
            }
        }
    }

    if let Some(nxz) = neighbor_raw(IVec3::new(1, 0, 1)) {
        for dx in 0..2u32 {
            for dz in 0..2u32 {
                for y in 0..size {
                    vol[idx(size + dx, y, size + dz)] = nxz[n_idx(dx, y, dz)];
                }
            }
        }
    } else {
        for dx in 0..2u32 {
            for dz in 0..2u32 {
                for y in 0..size {
                    vol[idx(size + dx, y, size + dz)] = vol[idx(size - 1, y, size - 1)];
                }
            }
        }
    }

    if let Some(nyz) = neighbor_raw(IVec3::new(0, 1, 1)) {
        for dy in 0..2u32 {
            for dz in 0..2u32 {
                for x in 0..size {
                    vol[idx(x, size + dy, size + dz)] = nyz[n_idx(x, dy, dz)];
                }
            }
        }
    } else {
        for dy in 0..2u32 {
            for dz in 0..2u32 {
                for x in 0..size {
                    vol[idx(x, size + dy, size + dz)] = vol[idx(x, size - 1, size - 1)];
                }
            }
        }
    }

    // --- Corner: +x+y+z — full 2x2x2 block ---
    if let Some(nxyz) = neighbor_raw(IVec3::new(1, 1, 1)) {
        for dx in 0..2u32 {
            for dy in 0..2u32 {
                for dz in 0..2u32 {
                    vol[idx(size + dx, size + dy, size + dz)] = nxyz[n_idx(dx, dy, dz)];
                }
            }
        }
    } else {
        for dx in 0..2u32 {
            for dy in 0..2u32 {
                for dz in 0..2u32 {
                    vol[idx(size + dx, size + dy, size + dz)] =
                        vol[idx(size - 1, size - 1, size - 1)];
                }
            }
        }
    }
}
