mod layouts;
use bevy::{
    mesh::VertexBufferLayout,
    prelude::*,
    render::{
        render_asset::RenderAssets,
        render_resource::*,
        renderer::{RenderDevice, RenderQueue},
        texture::GpuImage,
    },
};
use std::borrow::Cow;

use crate::{
    texture_palette::{MaterialPropertiesGpu, plugin::ExtractedPalette},
    voxel::{
        CURSOR_QUERY_SHADER_HANDLE, STREAM_COMPACTION_SHADER_HANDLE,
        SURFACE_NETS_PASS1_SHADER_HANDLE, SURFACE_NETS_PASS3_SHADER_HANDLE,
    },
};

#[derive(Resource)]
pub struct VoxelPipelineLayouts {
    pub pass1_surface_layout: BindGroupLayout,
    pub pass3_surface_layout: BindGroupLayout,
    pub compaction_bind_group_layout: BindGroupLayout,
    pub raster_chunk_visibility_layout: BindGroupLayout,
    pub query_layout: BindGroupLayout,
}

impl FromWorld for VoxelPipelineLayouts {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let layout = |label, entries: Vec<BindGroupLayoutEntry>| {
            render_device.create_bind_group_layout(Some(label), &entries)
        };
        Self {
            pass1_surface_layout: layout(
                "voxel_surface_pass1_layout",
                layouts::pass1_surface_entries(),
            ),
            pass3_surface_layout: layout(
                "voxel_surface_pass3_layout",
                layouts::pass3_surface_entries(),
            ),
            compaction_bind_group_layout: layout(
                "voxel_compaction_layout",
                layouts::compaction_entries(),
            ),
            raster_chunk_visibility_layout: layout(
                "voxel_raster_chunk_visibility_layout",
                layouts::raster_chunk_visibility_entries(),
            ),
            query_layout: layout("voxel_query_layout", layouts::query_entries()),
        }
    }
}

#[derive(Resource)]
pub struct VoxelComputePipeline {
    pub pass1_pipeline_id: CachedComputePipelineId,
    pub stream_compaction_pipeline_id: CachedComputePipelineId,
    pub scan_block_sums_pipeline_id: CachedComputePipelineId,
    pub stream_compaction_resolve_pipeline_id: CachedComputePipelineId,
    pub pass3_pipeline_id: CachedComputePipelineId,
    pub query_pipeline_id: CachedComputePipelineId,
}

impl FromWorld for VoxelComputePipeline {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let queue = |label: &'static str,
                     layout_label: &'static str,
                     entries: Vec<BindGroupLayoutEntry>,
                     shader: Handle<Shader>,
                     entry_point: &'static str| {
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some(Cow::Borrowed(label)),
                layout: vec![BindGroupLayoutDescriptor {
                    label: Cow::Borrowed(layout_label),
                    entries,
                }],
                shader,
                entry_point: Some(Cow::Borrowed(entry_point)),
                shader_defs: vec![],
                immediate_size: 0,
                zero_initialize_workgroup_memory: false,
            })
        };
        let compaction = |label, entry_point| {
            queue(
                label,
                "voxel_compaction_pipeline_layout",
                layouts::compaction_entries(),
                STREAM_COMPACTION_SHADER_HANDLE,
                entry_point,
            )
        };

        Self {
            // Edge classification & vertex generation.
            pass1_pipeline_id: queue(
                "surface_nets_pass1_pipeline",
                "voxel_pass1_pipeline_layout",
                layouts::pass1_surface_entries(),
                SURFACE_NETS_PASS1_SHADER_HANDLE,
                "cs_main",
            ),
            stream_compaction_pipeline_id: compaction(
                "stream_compaction_scan_pipeline",
                "scan_workgroup",
            ),
            scan_block_sums_pipeline_id: compaction("scan_block_sums", "scan_block_sums"),
            stream_compaction_resolve_pipeline_id: compaction(
                "stream_compaction_resolve_pipeline",
                "resolve_block_offsets",
            ),
            // Index assembly & indirect arguments.
            pass3_pipeline_id: queue(
                "surface_nets_pass3_pipeline",
                "voxel_pass3_pipeline_layout",
                layouts::pass3_surface_entries(),
                SURFACE_NETS_PASS3_SHADER_HANDLE,
                "cs_main",
            ),
            // Cursor/raycast query: single-thread workgroups, one per query.
            query_pipeline_id: queue(
                "voxel_cursor_query_pipeline",
                "voxel_query_pipeline_layout",
                layouts::query_entries(),
                CURSOR_QUERY_SHADER_HANDLE,
                "cs_main",
            ),
        }
    }
}

#[derive(Resource)]
pub struct VoxelRasterPipeline {
    pub pipeline_id: CachedRenderPipelineId,
    pub view_layout: BindGroupLayout,
    pub material_layout: BindGroupLayout,
}

impl FromWorld for VoxelRasterPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let pipeline_cache = world.resource::<PipelineCache>();

        // Group 0: View Layout
        let view_layout_entries = vec![BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: None,
            },
            count: None,
        }];

        let view_layout =
            render_device.create_bind_group_layout(Some("voxel_view_layout"), &view_layout_entries);

        // Group 1: Material Layout (WGSL expects texture_2d_array<f32>)
        let material_layout_entries = vec![
            // Binding 0: Texture
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            // Binding 1: Sampler
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
            // Binding 2: MaterialPropertiesArray — storage, not uniform, since
            // it's runtime-sized (materials.len() varies per palette).
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ];

        let material_layout = render_device
            .create_bind_group_layout(Some("voxel_material_layout"), &material_layout_entries);

        let pipeline_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("voxel_raster_pipeline".into()),
            layout: vec![
                BindGroupLayoutDescriptor {
                    label: Cow::Borrowed("voxel_view_layout"),
                    entries: view_layout_entries,
                },
                BindGroupLayoutDescriptor {
                    label: Cow::Borrowed("voxel_material_layout"),
                    entries: material_layout_entries,
                },
                BindGroupLayoutDescriptor {
                    label: Cow::Borrowed("voxel_raster_chunk_visibility_layout"),
                    entries: layouts::raster_chunk_visibility_entries(),
                },
            ],
            vertex: VertexState {
                shader: crate::voxel::VOXEL_RASTER_SHADER_HANDLE,
                entry_point: Some("vs_main".into()),
                shader_defs: vec![],
                buffers: vec![VertexBufferLayout {
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
                }],
            },
            fragment: Some(FragmentState {
                shader: crate::voxel::VOXEL_RASTER_SHADER_HANDLE,
                entry_point: Some("fs_main".into()),
                shader_defs: vec![],
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::Rgba8UnormSrgb,
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                cull_mode: Some(Face::Back),
                ..default()
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(CompareFunction::GreaterEqual),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState {
                count: Msaa::default().samples(),
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            zero_initialize_workgroup_memory: false,
            immediate_size: 0,
        });

        Self {
            pipeline_id,
            view_layout,
            material_layout,
        }
    }
}

fn material_bind_group(
    render_device: &RenderDevice,
    layout: &BindGroupLayout,
    label: &'static str,
    texture_view: &TextureView,
    sampler: &Sampler,
    properties: &Buffer,
) -> BindGroup {
    render_device.create_bind_group(
        Some(label),
        layout,
        &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(texture_view),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Sampler(sampler),
            },
            BindGroupEntry {
                binding: 2,
                resource: properties.as_entire_binding(),
            },
        ],
    )
}

/// The bind group actually used by `voxel_raster_pass`. Starts out as a
/// 1x1 white dummy; `update_voxel_material_bind_group` replaces it once a
/// real palette's albedo texture has finished loading.
#[derive(Resource)]
pub struct VoxelMaterialBindGroup(pub BindGroup);

impl FromWorld for VoxelMaterialBindGroup {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let render_queue = world.resource::<RenderQueue>();
        let raster_pipeline = world.resource::<VoxelRasterPipeline>();

        let extent = Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        };
        let dummy_texture = render_device.create_texture(&TextureDescriptor {
            label: Some("voxel_dummy_material_texture"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        render_queue.write_texture(
            TexelCopyTextureInfo {
                texture: &dummy_texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &[255u8, 255, 255, 255],
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            extent,
        );
        // A 1-layer D2 texture defaults to a D2 view; the shader wants D2Array.
        let texture_view = dummy_texture.create_view(&TextureViewDescriptor {
            dimension: Some(TextureViewDimension::D2Array),
            ..Default::default()
        });
        let sampler = render_device.create_sampler(&SamplerDescriptor {
            label: Some("voxel_dummy_sampler"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..default()
        });
        // One entry in MaterialPropertiesGpu's layout: texture_scale,
        // blend_sharpness, roughness_override, metallic_override.
        let properties = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("voxel_dummy_material_properties_buffer"),
            contents: bytemuck::cast_slice(&[0.1f32, 4.0, -1.0, -1.0]),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        Self(material_bind_group(
            render_device,
            &raster_pipeline.material_layout,
            "voxel_dummy_material_bind_group",
            &texture_view,
            &sampler,
            &properties,
        ))
    }
}

/// Runs in RenderSystems::Prepare. Rebuilds VoxelMaterialBindGroup once
/// the active palette's albedo image is loaded, and never again after
/// that for the same albedo handle — in-place edits to palette.materials
/// after the initial build won't hot-reload; only a handle change
/// (different palette asset) triggers a rebuild.
pub fn update_voxel_material_bind_group(
    render_device: Res<RenderDevice>,
    raster_pipeline: Res<VoxelRasterPipeline>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    extracted: Option<Res<ExtractedPalette>>,
    mut bind_group: ResMut<VoxelMaterialBindGroup>,
    mut built_for: Local<Option<AssetId<Image>>>,
) {
    let Some(extracted) = extracted else {
        return; // No PalettePlugin registered — keep the dummy.
    };

    let albedo_id = extracted.albedo.id();
    if *built_for == Some(albedo_id) {
        return; // Already built for this exact albedo — nothing changed.
    }
    let Some(gpu_image) = gpu_images.get(&extracted.albedo) else {
        return; // Not loaded/converted to GPU form yet — try again next frame.
    };

    // A zero-length storage buffer is invalid, so an empty palette gets one
    // default entry.
    let mut materials: Vec<MaterialPropertiesGpu> = extracted
        .materials
        .iter()
        .map(MaterialPropertiesGpu::from)
        .collect();
    if materials.is_empty() {
        materials.push(MaterialPropertiesGpu::default());
    }
    let properties = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("voxel_material_properties_buffer"),
        contents: bytemuck::cast_slice(&materials),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    let sampler = render_device.create_sampler(&SamplerDescriptor {
        label: Some("voxel_material_sampler"),
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        address_mode_u: AddressMode::Repeat,
        address_mode_v: AddressMode::Repeat,
        address_mode_w: AddressMode::Repeat,
        ..default()
    });

    bind_group.0 = material_bind_group(
        &render_device,
        &raster_pipeline.material_layout,
        "voxel_material_bind_group",
        &gpu_image.texture_view,
        &sampler,
        &properties,
    );
    *built_for = Some(albedo_id);
}
