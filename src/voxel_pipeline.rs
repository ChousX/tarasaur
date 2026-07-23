use crate::SDFField;
use bevy::{
    prelude::*,
    render::{
        Extract, Render, RenderApp, RenderSystems,
        render_resource::*,
        renderer::{RenderContext, RenderDevice, RenderQueue},
        sync_world::MainEntity,
    },
};

pub const CHUNK_SIZE: u32 = 32;
pub const UNPADDED_BYTES_PER_ROW: u32 = CHUNK_SIZE * std::mem::size_of::<f32>() as u32; // 128
pub const PADDED_BYTES_PER_ROW: u32 = 256; // WebGPU alignment requirement

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawIndexedIndirectArgs {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub first_instance: u32,
}

// Uniform structure matching stream_compaction.wgsl binding 0
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CompactionUniforms {
    pub chunk_size: u32,
    pub total_cells: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

#[derive(Component)]
pub struct GpuVoxelChunkBuffers {
    pub chunk_coord: IVec3,
    pub lod: u32, // Storing active LOD grid dimension (e.g. 4, 16, 32, 64)
    pub sdf_texture: Texture,
    pub sdf_view: TextureView,

    pub flags_buffer: Buffer,              // Pass 1 Output
    pub compacted_offsets_buffer: Buffer,  // Pass 2 Output
    pub scattered_vertex_buffer: Buffer,   // Pass 3 Temporary
    pub final_vertex_buffer: Buffer,       // Pass 3 Packed Output
    pub index_buffer: Buffer,              // Pass 4 Index Buffer
    pub indirect_args_buffer: Buffer,      // Pass 4 Indirect Buffer
    pub compaction_uniform_buffer: Buffer, // Pass 2 Uniforms
    pub block_sums_buffer: Buffer,         // Pass 2 Inter-workgroup block reductions

    pub pass1_surface_bind_group: BindGroup,
    pub pass3_surface_bind_group: BindGroup,
    pub compaction_bind_group: BindGroup,
}

#[derive(Resource)]
pub struct VoxelPipelineLayouts {
    pub pass1_surface_layout: BindGroupLayout,
    pub pass3_surface_layout: BindGroupLayout,
    pub compaction_bind_group_layout: BindGroupLayout,
}

impl FromWorld for VoxelPipelineLayouts {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        // Layout for Pass 1: Binding 1 is Read-Write
        let pass1_surface_layout = render_device.create_bind_group_layout(
            Some("voxel_surface_pass1_layout"),
            &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::ReadOnly,
                        format: TextureFormat::R32Float,
                        view_dimension: TextureViewDimension::D3,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE | ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        );

        // Layout for Pass 3 (Index & Indirect Arg Generation)
        let pass3_surface_layout = render_device.create_bind_group_layout(
            Some("voxel_surface_pass3_layout"),
            &[
                // Binding 0: Read-Only SDF Storage Texture
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::ReadOnly,
                        format: TextureFormat::R32Float,
                        view_dimension: TextureViewDimension::D3,
                    },
                    count: None,
                },
                // Binding 1: Flags Buffer (Read-Only)
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 2: Compacted Offsets Buffer (Read-Only)
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 3: Index Buffer (Read-Write)
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 4: Indirect Draw Args Buffer (Read-Write)
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE | ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        );

        // Layout for Pass 2 (Stream Compaction)
        let compaction_bind_group_layout = render_device.create_bind_group_layout(
            Some("voxel_compaction_layout"),
            &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        );

        Self {
            pass1_surface_layout,
            pass3_surface_layout,
            compaction_bind_group_layout,
        }
    }
}

// --- Compute Pipeline Resource ---

#[derive(Resource)]
pub struct VoxelComputePipeline {
    pub pass1_pipeline: ComputePipeline,
    pub stream_compaction_pipeline: ComputePipeline,
    pub pass3_pipeline: ComputePipeline,
}

impl FromWorld for VoxelComputePipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let layouts = world.resource::<VoxelPipelineLayouts>();

        let pass1_pipeline_layout =
            render_device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("voxel_pass1_pipeline_layout"),
                bind_group_layouts: &[Some(&layouts.pass1_surface_layout)],
                immediate_size: 0,
            });

        let pass3_pipeline_layout =
            render_device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("voxel_pass3_pipeline_layout"),
                bind_group_layouts: &[Some(&layouts.pass3_surface_layout)],
                immediate_size: 0,
            });

        let compaction_pipeline_layout =
            render_device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("voxel_compaction_pipeline_layout"),
                bind_group_layouts: &[Some(&layouts.compaction_bind_group_layout)],
                immediate_size: 0,
            });

        // Load shader modules
        let shader_pass1 = unsafe {
            render_device.create_shader_module(ShaderModuleDescriptor {
                label: Some("surface_nets_pass1_shader"),
                source: ShaderSource::Wgsl(include_str!("shaders/surface_nets_pass1.wgsl").into()),
            })
        };

        let shader_compaction = unsafe {
            render_device.create_shader_module(ShaderModuleDescriptor {
                label: Some("stream_compaction_shader"),
                source: ShaderSource::Wgsl(include_str!("shaders/stream_compaction.wgsl").into()),
            })
        };

        let shader_pass3 = unsafe {
            render_device.create_shader_module(ShaderModuleDescriptor {
                label: Some("surface_nets_pass3_shader"),
                source: ShaderSource::Wgsl(include_str!("shaders/surface_nets_pass3.wgsl").into()),
            })
        };

        // Compute Pipelines
        let pass1_pipeline = render_device.create_compute_pipeline(&RawComputePipelineDescriptor {
            label: Some("surface_nets_pass1_pipeline"),
            layout: Some(&pass1_pipeline_layout),
            module: &shader_pass1,
            entry_point: Some("cs_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        let stream_compaction_pipeline =
            render_device.create_compute_pipeline(&RawComputePipelineDescriptor {
                label: Some("stream_compaction_pipeline"),
                layout: Some(&compaction_pipeline_layout),
                module: &shader_compaction,
                entry_point: Some("scan_workgroup"),
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            });

        let pass3_pipeline = render_device.create_compute_pipeline(&RawComputePipelineDescriptor {
            label: Some("surface_nets_pass3_pipeline"),
            layout: Some(&pass3_pipeline_layout),
            module: &shader_pass3,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pass1_pipeline,
            stream_compaction_pipeline,
            pass3_pipeline,
        }
    }
}

/// Extracted SDF data copied during `ExtractSchedule` to prepare GPU buffers.
#[derive(Component)]
pub struct ExtractedChunkSdf {
    pub main_entity: MainEntity,
    pub chunk_pos: IVec3,
    pub padded_sdf_data: Vec<u8>,
    pub size: u32,
}

pub struct VoxelRenderPlugin;

impl Plugin for VoxelRenderPlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_systems(ExtractSchedule, extract_voxel_chunks)
            .add_systems(
                Render,
                (
                    prepare_voxel_chunk_buffers.in_set(RenderSystems::Prepare),
                    dispatch_voxel_compute_passes
                        .in_set(RenderSystems::Render)
                        .run_if(resource_exists::<VoxelComputePipeline>),
                )
                    .chain()
                    .run_if(resource_exists::<VoxelPipelineLayouts>),
            );
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<VoxelPipelineLayouts>()
            .init_resource::<VoxelComputePipeline>();
    }
}

// --- Extraction System ---

pub fn extract_voxel_chunks(
    mut commands: Commands,
    query: Extract<Query<(Entity, &crate::chunk::ChunkPosition, &SDFField)>>,
) {
    for (entity, pos, sdf) in query.iter() {
        let size = sdf.lod.size();
        let raw_slice = sdf.data_slice();

        let mut padded_sdf_data = Vec::with_capacity((PADDED_BYTES_PER_ROW * size * size) as usize);

        for z in 0..size {
            for y in 0..size {
                let start_idx = ((z * size + y) * size) as usize;
                let end_idx = start_idx + size as usize;
                let row_bytes: &[u8] = bytemuck::cast_slice(&raw_slice[start_idx..end_idx]);

                padded_sdf_data.extend_from_slice(row_bytes);
                padded_sdf_data.resize(
                    padded_sdf_data.len()
                        + (PADDED_BYTES_PER_ROW - UNPADDED_BYTES_PER_ROW) as usize,
                    0,
                );
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

// --- GPU Buffer System ---

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
                        bytes_per_row: Some(PADDED_BYTES_PER_ROW),
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
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_DST,
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
                    bytes_per_row: Some(PADDED_BYTES_PER_ROW),
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

            let workgroup_capacity = 512;
            let num_blocks = ((total_cells + workgroup_capacity - 1) / workgroup_capacity) as u64;
            let block_sums_buffer = render_device.create_buffer(&BufferDescriptor {
                label: Some("chunk_block_sums_buffer"),
                size: num_blocks * std::mem::size_of::<u32>() as u64,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            });

            // Bind Group for Pass 1 (Read-Write layout)
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

            // Bind Group for Pass 3 (Read-Only layout)
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
                        resource: index_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 4,
                        resource: indirect_args_buffer.as_entire_binding(),
                    },
                ],
            );

            // Bind Group for Stream Compaction (Pass 2)
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
                    pass1_surface_bind_group,
                    pass3_surface_bind_group,
                    compaction_bind_group,
                },
            ));
        }

        commands.entity(extracted_entity).despawn();
    }
}

// --- Compute Pass Execution System ---

pub fn dispatch_voxel_compute_passes(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline: Res<VoxelComputePipeline>,
    chunk_buffers: Query<&GpuVoxelChunkBuffers>,
) {
    if chunk_buffers.is_empty() {
        return;
    }

    let mut command_encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("voxel_compute_encoder"),
    });

    for chunk in chunk_buffers.iter() {
        // --- Pass 1: Surface Nets Voxel Classification ---
        {
            let mut compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("surface_nets_pass1"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline.pass1_pipeline);
            compute_pass.set_bind_group(0, &chunk.pass1_surface_bind_group, &[]);
            compute_pass.dispatch_workgroups(CHUNK_SIZE / 4, CHUNK_SIZE / 4, CHUNK_SIZE / 4);
        }

        // --- Pass 2: Stream Compaction (Prefix Sum Scan) ---
        let elements_per_workgroup = 512;
        let total_cells = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
        let dispatch_x = (total_cells + elements_per_workgroup - 1) / elements_per_workgroup;

        {
            let mut compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("stream_compaction_pass2"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline.stream_compaction_pipeline);
            compute_pass.set_bind_group(0, &chunk.compaction_bind_group, &[]);
            compute_pass.dispatch_workgroups(dispatch_x, 1, 1);
        }

        // --- Pass 3: Surface Nets Mesh Generation ---
        {
            let mut compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("surface_nets_pass3"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline.pass3_pipeline);
            compute_pass.set_bind_group(0, &chunk.pass3_surface_bind_group, &[]);
            compute_pass.dispatch_workgroups(CHUNK_SIZE / 8, CHUNK_SIZE / 8, CHUNK_SIZE / 8);
        }
    }

    render_queue.submit(std::iter::once(command_encoder.finish()));
}
