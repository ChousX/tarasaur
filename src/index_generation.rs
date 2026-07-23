//src/index_generation.rs
use bevy::{
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        render_resource::*,
        renderer::{RenderContext, RenderDevice},
    },
};

// Distinct System Set ordering markers to satisfy modern Bevy schedule execution rules
#[derive(SystemSet, Debug, Hash, PartialEq, Eq, Clone)]
pub struct VoxelIndexGenerationSet;

#[derive(Resource)]
pub struct Pass3PipelineCache {
    pub pipeline: ComputePipeline,
    pub bind_group_layout: BindGroupLayout,
}

pub struct VoxelIndexGenerationPlugin;

impl Plugin for VoxelIndexGenerationPlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .configure_sets(
                Render,
                VoxelIndexGenerationSet.in_set(RenderSystems::Prepare),
            )
            .add_systems(
                Render,
                dispatch_index_generation_pass.in_set(VoxelIndexGenerationSet),
            );
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        let render_device = render_app.world().resource::<RenderDevice>();

        // Establish strict binding constraints to align layout expectations directly
        let bind_group_layout = render_device.create_bind_group_layout(
            Some("voxel_pass3_bind_group_layout"),
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
                // Binding 1: Scattered Cell Flags (Read-Only in Pass 3)
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
                // Binding 2: Compacted Cell Prefix Sum Offsets (Read-Only in Pass 3)
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
                // Binding 3: Destination Target Index Buffer
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
                // Binding 4: Dynamic Indirect Call Arguments (Read-Write)
                BindGroupLayoutEntry {
                    binding: 4,
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

        // 1. Create the Pipeline Layout using modern wgpu fields
        let pipeline_layout = render_device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("voxel_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        // 2. Safely embed the WGSL shader code directly via raw string
        let shader = unsafe {
            render_device.create_shader_module(ShaderModuleDescriptor {
                label: Some("surface_nets_pass3_shader"),
                source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                    "shaders/surface_nets_pass3.wgsl"
                ))),
            })
        };

        // 3. Build using RawComputePipelineDescriptor to match RenderDevice requirements
        let pipeline = render_device.create_compute_pipeline(&RawComputePipelineDescriptor {
            label: Some("voxel_pass3_compute_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        render_app.insert_resource(Pass3PipelineCache {
            pipeline,
            bind_group_layout,
        });
    }
}

/// Query and iterate across components inside the RenderApp schedule
fn dispatch_index_generation_pass(
    mut render_context: RenderContext,
    pipeline_cache: Res<Pass3PipelineCache>,
    chunk_buffers_query: Query<&crate::voxel_pipeline::GpuVoxelChunkBuffers>,
) {
    // Grab the device before initializing the encoder to resolve overlapping borrow constraints
    let render_device = render_context.render_device().clone();
    let command_encoder = render_context.command_encoder();
    for chunk in &chunk_buffers_query {
        // Build dedicated transient bind group targeting the specific buffers allocation
        let pass3_bind_group = render_device.create_bind_group(
            Some("voxel_pass3_chunk_bind_group"),
            &pipeline_cache.bind_group_layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&chunk.sdf_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: chunk.flags_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: chunk.compacted_offsets_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: chunk.index_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: chunk.indirect_args_buffer.as_entire_binding(),
                },
            ],
        );

        // Instantiate a distinct Compute Pass scope
        let mut compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("surface_nets_pass3_index_generation"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipeline_cache.pipeline);
        compute_pass.set_bind_group(0, &pass3_bind_group, &[]);

        // Dynamic workgroup calculation supporting LOD scales from 4 up to 64
        // (Assumes `chunk.chunk_size` stores the active u32 dimension length per axis)
        let workgroup_size = 8;
        let chunk_dim = chunk.lod;
        let dispatch_dim = (chunk_dim + workgroup_size - 1) / workgroup_size;

        compute_pass.dispatch_workgroups(dispatch_dim, dispatch_dim, dispatch_dim);
    }
}
