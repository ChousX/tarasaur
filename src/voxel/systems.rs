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
    CHUNK_SIZE,
    voxel::{pipeline::VoxelRasterPipeline, types::Pass3Uniforms},
};

use super::{
    buffers::{ExtractedChunkSdf, GpuVoxelChunkBuffers},
    pipeline::{VoxelComputePipeline, VoxelPipelineLayouts},
    types::{CompactionUniforms, DrawIndexedIndirectArgs},
};

pub fn extract_voxel_chunks(
    mut commands: Commands,
    query: Extract<Query<(Entity, &crate::chunk::ChunkPosition, &crate::SDFField)>>,
) {
    for (entity, pos, sdf) in query.iter() {
        let size = sdf.lod.size();
        let raw_slice = sdf.data_slice();
        let expected_len = (size * size * size) as usize;
        if raw_slice.len() != expected_len {
            // SDF data not yet populated for this chunk (e.g. still generating async) — skip this frame.
            continue;
        }

        let unpadded_bytes_per_row = size * std::mem::size_of::<f32>() as u32;
        let padded_bytes_per_row = (unpadded_bytes_per_row + 255) & !255;
        let padding_per_row = (padded_bytes_per_row - unpadded_bytes_per_row) as usize;

        let mut padded_sdf_data = Vec::with_capacity((padded_bytes_per_row * size * size) as usize);

        for z in 0..size {
            for y in 0..size {
                let start_idx = ((z * size + y) * size) as usize;
                let end_idx = start_idx + size as usize;
                let row_bytes: &[u8] = bytemuck::cast_slice(&raw_slice[start_idx..end_idx]);

                padded_sdf_data.extend_from_slice(row_bytes);
                padded_sdf_data.resize(padded_sdf_data.len() + padding_per_row, 0);
            }
        }

        commands.spawn(ExtractedChunkSdf {
            main_entity: entity.into(),
            chunk_pos: pos.0,
            padded_sdf_data,
            size,
        });
    }
}

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
        let total_cells = (size * size * size) as usize;
        let unpadded_bytes_per_row = size * std::mem::size_of::<f32>() as u32;
        let padded_bytes_per_row = (unpadded_bytes_per_row + 255) & !255;
        let mut found_existing = false;

        for (_, main_entity, gpu_buffers) in existing_buffers.iter_mut() {
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

                let initial_indirect_args = DrawIndexedIndirectArgs {
                    index_count: 0,
                    instance_count: 1,
                    first_index: 0,
                    base_vertex: 0,
                    first_instance: 0,
                };
                render_queue.write_buffer(
                    &gpu_buffers.indirect_args_buffer,
                    0,
                    bytemuck::bytes_of(&initial_indirect_args),
                );

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
                usage: BufferUsages::STORAGE | BufferUsages::VERTEX,
                mapped_at_creation: false,
            });

            let index_buffer = render_device.create_buffer(&BufferDescriptor {
                label: Some("chunk_index_buffer"),
                size: (total_cells * 18 * std::mem::size_of::<u32>()) as u64,
                usage: BufferUsages::STORAGE | BufferUsages::INDEX,
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
                chunk_size: size,
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

            let voxel_size = CHUNK_SIZE / size as f32;
            let chunk_world_origin = extracted_sdf.chunk_pos.as_vec3() * CHUNK_SIZE;

            let pass3_uniforms = Pass3Uniforms {
                chunk_size: size,
                voxel_size,
                _pad0: [0; 2],
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

            commands.spawn((
                extracted_sdf.main_entity,
                GpuVoxelChunkBuffers {
                    chunk_coord: extracted_sdf.chunk_pos,
                    lod: size,
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
                },
            ));
        }

        commands.entity(extracted_entity).despawn();
    }
}

pub fn dispatch_voxel_compute_passes(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>, // 1. Added PipelineCache resource
    pipeline: Res<VoxelComputePipeline>,
    chunk_buffers: Query<&GpuVoxelChunkBuffers>,
) {
    if chunk_buffers.is_empty() {
        return;
    }

    // 2. Resolve cached pipeline IDs into actual compiled ComputePipelines
    let (
        Some(pass1_pipeline),
        Some(stream_compaction_pipeline),
        Some(stream_compaction_resolve_pipeline),
        Some(pass3_pipeline),
    ) = (
        pipeline_cache.get_compute_pipeline(pipeline.pass1_pipeline_id),
        pipeline_cache.get_compute_pipeline(pipeline.stream_compaction_pipeline_id),
        pipeline_cache.get_compute_pipeline(pipeline.stream_compaction_resolve_pipeline_id),
        pipeline_cache.get_compute_pipeline(pipeline.pass3_pipeline_id),
    )
    else {
        // Shaders are still compiling on a background thread; safely skip this frame
        return;
    };

    let initial_indirect_args = DrawIndexedIndirectArgs {
        index_count: 0,
        instance_count: 1,
        first_index: 0,
        base_vertex: 0,
        first_instance: 0,
    };

    for chunk in chunk_buffers.iter() {
        render_queue.write_buffer(
            &chunk.indirect_args_buffer,
            0,
            bytemuck::bytes_of(&initial_indirect_args),
        );
    }

    let mut command_encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("voxel_compute_encoder"),
    });

    for chunk in chunk_buffers.iter() {
        let size = chunk.lod;
        let total_cells = size * size * size;

        // Pass 1: Surface Nets Classification
        {
            let p1_grid = (size + 3) / 4;
            let mut compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("surface_nets_pass1"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(pass1_pipeline); // Updated reference
            compute_pass.set_bind_group(0, &chunk.pass1_surface_bind_group, &[]);
            compute_pass.dispatch_workgroups(p1_grid, p1_grid, p1_grid);
        }

        // Pass 2: Stream Compaction
        {
            let workgroup_size = 512;
            let num_blocks = (total_cells + workgroup_size - 1) / workgroup_size;

            let mut compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("stream_compaction_scan"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(stream_compaction_pipeline); // Updated reference
            compute_pass.set_bind_group(0, &chunk.compaction_bind_group, &[]);
            compute_pass.dispatch_workgroups(num_blocks, 1, 1);
        }

        {
            let workgroup_size = 512;
            let num_blocks = (total_cells + workgroup_size - 1) / workgroup_size;

            let mut compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("stream_compaction_resolve"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(stream_compaction_resolve_pipeline); // Updated reference
            compute_pass.set_bind_group(0, &chunk.compaction_bind_group, &[]);
            compute_pass.dispatch_workgroups(num_blocks, 1, 1);
        }

        // Pass 3: Vertex and Index Generation
        {
            //let p3_grid = (size + 3) / 4;
            let p3_grid = (size + 7) / 8; // Resolves to 4 workgroups (4 * 8 = 32 threads)[cite: 5, 9]
            let mut compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("surface_nets_pass3"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(pass3_pipeline); // Updated reference
            compute_pass.set_bind_group(0, &chunk.pass3_surface_bind_group, &[]);
            compute_pass.dispatch_workgroups(p3_grid, p3_grid, p3_grid);
        }
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
    chunk_buffers: Query<&GpuVoxelChunkBuffers>,
    pipeline_cache: Res<PipelineCache>,
    raster_pipeline: Res<VoxelRasterPipeline>,
    mut ctx: RenderContext,
) {
    let Some(pipeline) = pipeline_cache.get_render_pipeline(raster_pipeline.pipeline_id) else {
        return;
    };

    let Some(binding) = view_uniforms.uniforms.binding() else {
        return;
    };

    let (_camera, target, depth_texture, view_offset) = view.into_inner();

    // Group 0: View
    let view_bind_group = ctx.render_device().create_bind_group(
        Some("voxel_view_bind_group"),
        &raster_pipeline.view_layout,
        &[BindGroupEntry {
            binding: 0,
            resource: binding,
        }],
    );

    // 1. Create dummy 1x1 texture
    let dummy_texture = ctx.render_device().create_texture(&TextureDescriptor {
        label: Some("voxel_dummy_material_texture"),
        size: Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8UnormSrgb,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let dummy_texture_view = dummy_texture.create_view(&TextureViewDescriptor::default());

    // 2. Create dummy sampler
    let dummy_sampler = ctx.render_device().create_sampler(&SamplerDescriptor {
        label: Some("voxel_dummy_sampler"),
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        ..default()
    });

    // 3. Create Group 1 bind group with both entries
    let material_bind_group = ctx.render_device().create_bind_group(
        Some("voxel_dummy_material_bind_group"),
        &raster_pipeline.material_layout,
        &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&dummy_texture_view),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Sampler(&dummy_sampler),
            },
        ],
    );
    let mut render_pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("voxel_raster_pass"),
        color_attachments: &[Some(target.get_color_attachment())],
        depth_stencil_attachment: Some(depth_texture.get_attachment(StoreOp::Store)),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });

    render_pass.set_render_pipeline(pipeline);

    // Bind both groups
    render_pass.set_bind_group(0, &view_bind_group, &[view_offset.offset]);
    render_pass.set_bind_group(1, &material_bind_group, &[]);

    for chunk in chunk_buffers.iter() {
        render_pass.set_vertex_buffer(0, chunk.final_vertex_buffer.slice(..));
        render_pass.set_index_buffer(chunk.index_buffer.slice(..), IndexFormat::Uint32);
        render_pass.draw_indexed_indirect(&chunk.indirect_args_buffer, 0);
    }
}
