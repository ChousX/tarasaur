use bevy::{
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        render_resource::*,
        renderer::{RenderContext, RenderDevice},
    },
};
use std::borrow::Cow;

#[derive(SystemSet, Debug, Hash, PartialEq, Eq, Clone)]
pub struct StreamCompactionSet;

pub struct StreamCompactionPlugin;

impl Plugin for StreamCompactionPlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<StreamCompactionPipeline>()
            .configure_sets(Render, StreamCompactionSet.in_set(RenderSystems::Render))
            .add_systems(
                Render,
                dispatch_stream_compaction_pass.in_set(StreamCompactionSet),
            );
    }
}

#[derive(Resource)]
pub struct StreamCompactionPipeline {
    pub bind_group_layout: BindGroupLayout,
    // Store the CachedPipelineId instead of the compiled pipeline object
    pub scan_pipeline_id: CachedComputePipelineId,
    pub resolve_pipeline_id: CachedComputePipelineId,
}

impl FromWorld for StreamCompactionPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        // We look up the PipelineCache resource where Bevy manages compiled modules
        let pipeline_cache = world.resource::<PipelineCache>();

        let bind_group_layout = render_device.create_bind_group_layout(
            Some("stream_compaction_layout"),
            &[
                // Binding 0: Compaction Uniform Parameters
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
                // Binding 1: Cell Validity Flags Array
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
                // Binding 2: Destination Compacted Offsets
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
                // Binding 3: Inter-workgroup block reduction allocation
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

        let shader = world.load_asset("shaders/stream_compaction.wgsl");

        // 1. Scan Pipeline Setup
        let scan_pipeline_id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(Cow::Borrowed("stream_compaction_scan_pipeline")),
            layout: vec![BindGroupLayoutDescriptor {
                entries: vec![
                    // Binding 0: Compaction Uniform Parameters
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
                    // Binding 1: Cell Validity Flags Array
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
                    // Binding 2: Destination Compacted Offsets
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
                    // Binding 3: Inter-workgroup block reduction allocation
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
                label: Cow::Borrowed("stream_compaction_layout_desc"),
            }],
            shader: shader.clone(),
            entry_point: Some(Cow::Borrowed("scan_workgroup")),
            shader_defs: vec![],
            immediate_size: 0,
            zero_initialize_workgroup_memory: false,
        });

        // 2. Resolve Pipeline Setup
        let resolve_pipeline_id =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some(Cow::Borrowed("stream_compaction_resolve_pipeline")),
                // We pass the identical descriptor array structure here so layouts match completely
                layout: vec![BindGroupLayoutDescriptor {
                    entries: vec![
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
                    label: Cow::Borrowed("stream_compaction_layout_desc"),
                }],
                shader,
                entry_point: Some(Cow::Borrowed("resolve_block_offsets")),
                shader_defs: vec![],
                immediate_size: 0,
                zero_initialize_workgroup_memory: false,
            });
        Self {
            bind_group_layout,
            scan_pipeline_id,
            resolve_pipeline_id,
        }
    }
}

#[derive(Component)]
pub struct CompactionChunkResources {
    pub uniform_buffer: Buffer,
    pub flags_buffer: Buffer,
    pub offsets_buffer: Buffer,
    pub block_sums_buffer: Buffer,
    pub bind_group: BindGroup,
    pub total_cells: u32,
}

fn dispatch_stream_compaction_pass(
    pipeline: Res<StreamCompactionPipeline>,
    pipeline_cache: Res<PipelineCache>, // Pull in the cache to retrieve ready pipelines
    query: Query<&CompactionChunkResources>,
    mut render_context: RenderContext,
) {
    // Safely look up the asynchronous pipelines from the cache
    let (Some(scan_pipeline), Some(resolve_pipeline)) = (
        pipeline_cache.get_compute_pipeline(pipeline.scan_pipeline_id),
        pipeline_cache.get_compute_pipeline(pipeline.resolve_pipeline_id),
    ) else {
        // Skip execution if shaders are still compiling or loading this frame
        return;
    };

    // Removed 'mut' to resolve the warning wrapper
    let command_encoder = render_context.command_encoder();

    for chunk in query.iter() {
        let workgroup_size_elements = 256 * 2;
        let dispatch_x =
            (chunk.total_cells + workgroup_size_elements - 1) / workgroup_size_elements;

        // Pass 2A: Local Workgroup Scan Pass
        {
            let mut compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("stream_compaction_scan_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(scan_pipeline);
            compute_pass.set_bind_group(0, &chunk.bind_group, &[]);
            compute_pass.dispatch_workgroups(dispatch_x, 1, 1);
        }

        // Pass 2B: Global Offset Modifier Resolution Pass
        {
            let mut compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("stream_compaction_resolve_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(resolve_pipeline);
            compute_pass.set_bind_group(0, &chunk.bind_group, &[]);
            compute_pass.dispatch_workgroups(dispatch_x, 1, 1);
        }
    }
}

