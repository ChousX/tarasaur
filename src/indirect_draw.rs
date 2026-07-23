use crate::{VoxelRasterShader, voxel_pipeline::GpuVoxelChunkBuffers};
use bevy::{
    core_pipeline::{Core3d, Core3dSystems, core_3d::main_opaque_pass_3d},
    mesh::VertexBufferLayout,
    prelude::*,
    render::{
        ExtractSchedule, Render, RenderApp, RenderSystems,
        render_asset::RenderAssets,
        render_resource::*,
        renderer::{RenderContext, RenderDevice, ViewQuery},
        texture::GpuImage,
        view::{ViewDepthTexture, ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms},
    },
};
use std::borrow::Cow;

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
                prepare_voxel_draw_pipeline
                    .in_set(RenderSystems::Prepare)
                    .run_if(resource_exists::<VoxelRasterShader>),
            )
            .add_systems(
                Core3d,
                render_voxel_chunks_system
                    .after(main_opaque_pass_3d)
                    .in_set(Core3dSystems::MainPass)
                    .run_if(resource_exists::<VoxelDrawPipeline>),
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

pub fn extract_voxel_material(
    render_device: Res<RenderDevice>,
    pipeline: Option<Res<VoxelDrawPipeline>>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    material_asset: Option<Res<VoxelMaterialAsset>>,
    mut extracted_material: ResMut<ExtractedVoxelMaterial>,
) {
    let (Some(pipeline), Some(material)) = (pipeline, material_asset) else {
        extracted_material.material_bind_group = None;
        return;
    };

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

#[derive(Resource)]
pub struct VoxelDrawPipeline {
    pub pipeline_id: CachedRenderPipelineId,
    pub view_bind_group_layout: BindGroupLayout,
    pub material_bind_group_layout: BindGroupLayout,
}

pub fn prepare_voxel_draw_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    shader_res: Res<VoxelRasterShader>,
    mut pipeline_cache: ResMut<PipelineCache>,
    existing_pipeline: Option<Res<VoxelDrawPipeline>>,
) {
    if existing_pipeline.is_some() {
        return;
    }

    // PipelineCache needs an asset Handle<Shader> (not a raw wgpu::ShaderModule) so it
    // can build the pipeline lazily and support hot-reload. Requires this file to exist
    // under your assets root, e.g. `assets/shaders/voxel_raster.wgsl`.
    let shader: Handle<Shader> = shader_res.0.clone();
    // These are the *real* layout objects we'll reuse later to build actual bind groups
    // (in extract_voxel_material and render_voxel_chunks_system).
    let view_bind_group_layout = render_device.create_bind_group_layout(
        Some("voxel_view_layout"),
        &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: Some(ViewUniform::min_size()),
            },
            count: None,
        }],
    );

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

    let vertex_buffers = vec![VertexBufferLayout {
        array_stride: 32,
        step_mode: VertexStepMode::Vertex,
        attributes: vec![
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

    // PipelineCache wants descriptors of the bind group layouts (so it can build/rebuild
    // the pipeline layout itself), not a pre-built PipelineLayout object. These must
    // describe the SAME bindings as the real layouts above, in the same group order.
    let layout_descriptors = vec![
        BindGroupLayoutDescriptor {
            label: Cow::Borrowed("voxel_view_layout"),
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: Some(ViewUniform::min_size()),
                },
                count: None,
            }],
        },
        BindGroupLayoutDescriptor {
            label: Cow::Borrowed("voxel_material_layout"),
            entries: vec![
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
        },
    ];

    let pipeline_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("voxel_indirect_draw_pipeline".into()),
        layout: layout_descriptors,
        vertex: VertexState {
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some("vs_main".into()),
            buffers: vertex_buffers,
        },
        fragment: Some(FragmentState {
            shader,
            shader_defs: vec![],
            entry_point: Some("fs_main".into()),
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::Bgra8UnormSrgb,
                blend: Some(BlendState::REPLACE),
                write_mask: ColorWrites::ALL,
            })],
        }),
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: FrontFace::Ccw,
            cull_mode: None,
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
        immediate_size: 0,
        zero_initialize_workgroup_memory: false,
    });

    commands.insert_resource(VoxelDrawPipeline {
        pipeline_id,
        view_bind_group_layout,
        material_bind_group_layout,
    });
}

pub fn render_voxel_chunks_system(
    view_query: ViewQuery<(&ViewTarget, &ViewDepthTexture, &ViewUniformOffset)>,
    mut render_context: RenderContext,
    pipeline: Res<VoxelDrawPipeline>,
    pipeline_cache: Res<PipelineCache>,
    extracted_chunks: Res<ExtractedVoxelChunks>,
    extracted_material: Res<ExtractedVoxelMaterial>,
    view_uniforms: Res<ViewUniforms>,
) {
    if extracted_chunks.chunks.is_empty() {
        return;
    }

    let Some(loaded_pipeline) = pipeline_cache.get_render_pipeline(pipeline.pipeline_id) else {
        return;
    };

    let Some(material_bind_group) = &extracted_material.material_bind_group else {
        return;
    };

    let Some(view_binding) = view_uniforms.uniforms.binding() else {
        return;
    };

    let view_bind_group = render_context.render_device().create_bind_group(
        Some("voxel_view_bind_group"),
        &pipeline.view_bind_group_layout,
        &[BindGroupEntry {
            binding: 0,
            resource: view_binding,
        }],
    );

    let (view_target, depth_texture, view_uniform_offset) = view_query.into_inner();

    let render_pass_desc = RenderPassDescriptor {
        label: Some("voxel_indirect_draw_pass"),
        color_attachments: &[Some(RenderPassColorAttachment {
            view: view_target.main_texture_view(),
            depth_slice: None,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Load,
                store: StoreOp::Store,
            },
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

    let mut tracked_pass = render_context.begin_tracked_render_pass(render_pass_desc);

    tracked_pass.set_render_pipeline(loaded_pipeline);
    tracked_pass.set_bind_group(0, &view_bind_group, &[view_uniform_offset.offset]);
    tracked_pass.set_bind_group(1, material_bind_group, &[]);

    for chunk in &extracted_chunks.chunks {
        tracked_pass.set_vertex_buffer(0, chunk.vertex_buffer.slice(..));
        tracked_pass.set_index_buffer(chunk.index_buffer.slice(..), IndexFormat::Uint32);
        tracked_pass.draw_indexed_indirect(&chunk.indirect_args_buffer, 0);
    }
}
