use crate::voxel_pipeline::{GpuVoxelChunkBuffers, VoxelPipelineLayouts};
use bevy::{
    prelude::*,
    render::{
        Render, RenderApp, RenderSet,
        render_resource::*,
        renderer::{RenderContext, RenderDevice},
    },
};

#[derive(SystemSet, Debug, Hash, PartialEq, Eq, Clone)]
pub struct VoxelIndirectDrawSet;

#[derive(Resource)]
pub struct VoxelRenderPipeline {
    pub pipeline: RenderPipeline,
}

pub struct VoxelIndirectDrawPlugin;

impl Plugin for VoxelIndirectDrawPlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .configure_sets(Render, VoxelIndirectDrawSet.in_set(RenderSet::Render))
            .add_systems(Render, draw_voxels_indirect.in_set(VoxelIndirectDrawSet));
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        let render_device = render_app.world().resource::<RenderDevice>();

        // 1. Describe the memory layout of our GPU-generated packed storage vertices.
        // Match our WGSL struct Vertex { position: vec4<f32>, normal: vec4<f32> }
        let vertex_layout = VertexBufferLayout {
            array_stride: std::mem::size_of::<f32>() as u64 * 8, // 4 floats position + 4 floats normal
            step_mode: VertexStepMode::Vertex,
            attributes: vec![
                // Position attribute
                VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 0,
                },
                // Normal attribute
                VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: std::mem::size_of::<f32>() as u64 * 4,
                    shader_location: 1,
                },
            ],
        };

        // For Milestone 5, we hook up a minimal placeholder shader path for testing geometry topologies.
        // Milestone 6 replaces this with the full Triplanar Fragment Shader module.
        let shader = render_device.create_shader_module(ShaderModuleDescriptor {
            label: Some("voxel_render_shader"),
            source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "../assets/shaders/voxel_raster.wgsl"
            ))),
        });

        // Fetch layouts set up in Milestone 1/Review phases
        let layouts = render_app.world().resource::<VoxelPipelineLayouts>();

        let pipeline = render_device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("voxel_indirect_draw_pipeline"),
            layout: Some(
                &render_device.create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: Some("voxel_render_pipeline_layout"),
                    bind_group_layouts: &[&layouts.chunk_bind_group_layout],
                    push_constant_ranges: &[],
                }),
            ),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: vec![vertex_layout],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::Bgra8UnormSrgb, // Aligning with standard Bevy target formats
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: Some(Face::Back),
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Greater, // Bevy uses reversed-Z conventions
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState::default(),
            multiview: None,
        });

        render_app.insert_resource(VoxelRenderPipeline { pipeline });
    }
}

/// Pure GPU-driven Indirect Drawing System execution pass
fn draw_voxels_indirect(
    query: Query<&GpuVoxelChunkBuffers>,
    pipeline_cache: Res<VoxelRenderPipeline>,
    render_context: Res<RenderContext>,
) {
    // 2. Fetch the base command encoder directly from Bevy's execution framing structure
    let command_encoder = render_context.command_encoder();

    // Loop through extracted chunk buffer arrays and issue indirect rendering operations natively
    for chunk in query.iter() {
        // Prepare attachments manually matching target specifications
        // We initialize a standalone visual layout attachment to prevent pipeline synchronization bubbles
        let mut render_pass = command_encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("voxel_indirect_drawing_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &chunk.sdf_view, // Replace with Bevy's active frame view target when mapping to main engine passes
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                },
            })],
            // Setup standardized depth attachment configurations matching core engine configurations
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        // 3. Establish hardware pipeline configurations and stream buffer registers
        render_pass.set_pipeline(&pipeline_cache.pipeline);

        // Bind chunk-specific data (uniform state mappings, textures, etc.)
        render_pass.set_bind_group(0, &chunk.chunk_bind_group, &[]);

        // Bind our packed GPU-generated buffer to the vertex buffer slot 0
        render_pass.set_vertex_buffer(0, chunk.final_vertex_buffer.slice(..));

        // Bind index parameters using standard 32-bit unsigned spatial indexing layouts
        render_pass.set_index_buffer(chunk.index_buffer.slice(..), IndexFormat::Uint32);

        // 4. Issue the zero-latency indirect graphics execution command block
        // This instructs WebGPU to read all triangle draw metric parameters directly from memory
        // without passing data configurations or counters back to the CPU.
        render_pass.draw_indexed_indirect(&chunk.indirect_args_buffer, 0);
    }
}
