use crate::voxel_pipeline::GpuVoxelChunkBuffers;
use bevy::{
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        render_resource::*,
        renderer::{RenderContext, RenderDevice},
        texture::GpuImage,
        view::{ExtractedView, ViewDepthTexture, ViewTarget},
    },
};

pub struct VoxelIndirectDrawPlugin;

impl Plugin for VoxelIndirectDrawPlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<ExtractedVoxelChunks>()
            .init_resource::<ExtractedVoxelMaterial>()
            .add_systems(
                ExtractSchedule,
                (extract_voxel_chunks, extract_voxel_material),
            )
            .add_systems(
                Render,
                (
                    prepare_voxel_draw_pipeline.in_set(RenderSystems::Prepare),
                    render_voxel_chunks_indirect.in_set(RenderSystems::Render),
                ),
            );
    }
}

#[derive(Resource, Default)]
pub struct ExtractedVoxelChunks {
    pub chunks: Vec<GpuChunkDrawData>,
}

pub struct GpuChunkDrawData {
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub indirect_args_buffer: Buffer,
}

#[derive(Resource, Default)]
pub struct ExtractedVoxelMaterial {
    pub material_bind_group: Option<BindGroup>,
}

// Handle for a terrain texture asset managed on the Main App side
#[derive(Resource)]
pub struct VoxelMaterialAsset {
    pub texture_handle: Handle<Image>,
}

fn extract_voxel_chunks(
    query: Query<&GpuVoxelChunkBuffers>,
    mut extracted: ResMut<ExtractedVoxelChunks>,
) {
    extracted.chunks.clear();
    for chunk in query.iter() {
        extracted.chunks.push(GpuChunkDrawData {
            vertex_buffer: chunk.final_vertex_buffer.clone(),
            index_buffer: chunk.index_buffer.clone(),
            indirect_args_buffer: chunk.indirect_args_buffer.clone(),
        });
    }
}

fn extract_voxel_material(
    render_device: Res<RenderDevice>,
    pipeline: Option<Res<VoxelDrawPipeline>>,
    gpu_images: Res<bevy::render::render_asset::RenderAssets<GpuImage>>,
    material_asset: Option<Res<VoxelMaterialAsset>>,
    mut extracted_material: ResMut<ExtractedVoxelMaterial>,
) {
    if let (Some(pipeline), Some(material)) = (pipeline, material_asset) {
        if extracted_material.material_bind_group.is_none() {
            if let Some(gpu_image) = gpu_images.get(&material.texture_handle) {
                let bind_group = render_device.create_bind_group(
                    Some("voxel_material_bind_group"),
                    &pipeline.material_bind_group_layout,
                    &[
                        BindGroupEntry {
                            binding: 0,
                            resource: BindingResource::TextureView(&gpu_image.texture_view),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: BindingResource::Sampler(&gpu_image.sampler),
                        },
                    ],
                );
                extracted_material.material_bind_group = Some(bind_group);
            }
        }
    }
}

#[derive(Resource)]
pub struct VoxelDrawPipeline {
    pub pipeline_id: RenderPipeline,
    pub view_bind_group_layout: BindGroupLayout,
    pub material_bind_group_layout: BindGroupLayout,
}

fn prepare_voxel_draw_pipeline(mut commands: Commands, render_device: Res<RenderDevice>) {
    let shader = unsafe {
        render_device.create_shader_module(ShaderModuleDescriptor {
            label: Some("voxel_draw_shader"),
            source: ShaderSource::Wgsl(include_str!("../assets/shaders/voxel_raster.wgsl").into()),
        })
    };

    let view_bind_group_layout = render_device.create_bind_group_layout(
        Some("voxel_view_layout"),
        &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    );

    // Layout configuration for Triplanar Mapping textures
    let material_bind_group_layout = render_device.create_bind_group_layout(
        Some("voxel_material_layout"),
        &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
        ],
    );

    let pipeline_layout = render_device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("voxel_draw_pipeline_layout"),
        bind_group_layouts: &[
            Some(&view_bind_group_layout),
            Some(&material_bind_group_layout),
        ],
        immediate_size: 0,
    });

    let vertex_buffers = [RawVertexBufferLayout {
        array_stride: 32, // Accommodating float32x4 mappings
        step_mode: VertexStepMode::Vertex,
        attributes: &[
            VertexAttribute {
                format: VertexFormat::Float32x4,
                offset: 0,
                shader_location: 0,
            },
            VertexAttribute {
                format: VertexFormat::Float32x4,
                offset: 16,
                shader_location: 1,
            },
        ],
    }];

    let pipeline_id = render_device.create_render_pipeline(&RawRenderPipelineDescriptor {
        label: Some("voxel_indirect_draw_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: RawVertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &vertex_buffers,
            compilation_options: PipelineCompilationOptions::default(),
        },
        fragment: Some(RawFragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(ColorTargetState {
                format: TextureFormat::Bgra8UnormSrgb,
                blend: Some(BlendState::REPLACE),
                write_mask: ColorWrites::ALL,
            })],
            compilation_options: PipelineCompilationOptions::default(),
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
            depth_write_enabled: Some(true),
            depth_compare: Some(CompareFunction::GreaterEqual),
            stencil: StencilState::default(),
            bias: DepthBiasState::default(),
        }),
        multisample: MultisampleState::default(),
        cache: None,
        multiview_mask: None,
    });

    commands.insert_resource(VoxelDrawPipeline {
        pipeline_id,
        view_bind_group_layout,
        material_bind_group_layout,
    });
}

fn render_voxel_chunks_indirect(
    mut render_context: RenderContext,
    pipeline: Res<VoxelDrawPipeline>,
    extracted_chunks: Res<ExtractedVoxelChunks>,
    extracted_material: Res<ExtractedVoxelMaterial>,
    view_query: Query<(&ViewTarget, &ViewDepthTexture, &ExtractedView)>,
) {
    if extracted_chunks.chunks.is_empty() {
        return;
    }

    let Some(material_bind_group) = &extracted_material.material_bind_group else {
        return; // Guard against unallocated textures
    };

    for (view_target, depth_texture, _view) in view_query.iter() {
        let render_pass_desc = RenderPassDescriptor {
            label: Some("voxel_indirect_draw_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: view_target.main_texture_view(),
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: depth_texture.view(),
                depth_ops: Some(Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        };

        let mut render_pass = render_context
            .command_encoder()
            .begin_render_pass(&render_pass_desc);

        render_pass.set_pipeline(&pipeline.pipeline_id);

        // Bind the material textures to Group 1
        render_pass.set_bind_group(1, material_bind_group, &[]);

        for chunk in &extracted_chunks.chunks {
            // Note: Ensure Group 0 (View Uniforms) is set outside or wrapped here depending on your view binding strategy
            render_pass.set_vertex_buffer(0, (*chunk.vertex_buffer).slice(..));
            render_pass.set_index_buffer((*chunk.index_buffer).slice(..), IndexFormat::Uint32);
            render_pass.draw_indexed_indirect(&chunk.indirect_args_buffer, 0);
        }
    }
}

