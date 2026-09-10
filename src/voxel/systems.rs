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
    CHUNK_SIZE, ChunkManager, ChunkPosition, LOD, SDFField,
    voxel::{
        pipeline::{VoxelDummyMaterial, VoxelRasterPipeline},
        types::{
            CollisionMeshData, MeshReadbackChannel, Pass1Uniforms, Pass3Uniforms,
            PendingMeshReadback,
        },
    },
};

use super::{
    buffers::GpuVoxelChunkBuffers,
    pipeline::{VoxelComputePipeline, VoxelPipelineLayouts},
    types::{CompactionUniforms, DrawIndexedIndirectArgs},
};

#[derive(Component)]
pub struct ExtractedChunkField<T: Send + Sync + 'static> {
    pub main_entity: MainEntity,
    pub chunk_pos: IVec3,
    pub padded_sdf_data: Vec<u8>,
    pub size: u32,
    pub _type: PhantomData<T>,
}

pub fn prepare_voxel_chunk_buffers<T: Send + Sync + 'static>(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    layouts: Res<VoxelPipelineLayouts>,
    extracted_chunks: Query<(Entity, &ExtractedChunkField<T>)>,
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
    query: Extract<Query<(Entity, &ChunkPosition, &SDFField, &LOD)>>,
    mut last_versions: Local<std::collections::HashMap<IVec3, [u64; 8]>>,
) {
    for (entity, pos, sdf, lod) in query.iter() {
        let size = lod.size();

        let mut versions = [0u64; 8];
        versions[0] = sdf.version;
        for (i, offset) in NEIGHBORS_MASK.iter().enumerate() {
            if let Some(n_entity) = chunk_manager.get_chunk(&(pos.0 + *offset)) {
                if let Ok((_, _, n_sdf, _)) = query.get(n_entity) {
                    versions[i + 1] = n_sdf.version;
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

        commands.spawn(ExtractedChunkField::<SDFField> {
            main_entity: entity.into(),
            chunk_pos: pos.0,
            padded_sdf_data,
            size: padded_size,
            _type: PhantomData,
        });
    }
}

const PADDING: u32 = 2;

pub fn fill_apron(
    vol: &mut [f32],
    chunk_pos: IVec3,
    chunk_manager: &ChunkManager,
    query: &Query<(Entity, &ChunkPosition, &SDFField, &LOD)>,
) {
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
