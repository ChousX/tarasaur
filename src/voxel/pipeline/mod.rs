pub(crate) mod layouts;
use bevy::{
    mesh::VertexBufferLayout,
    prelude::*,
    render::{
        camera::ExtractedCamera,
        render_asset::RenderAssets,
        render_resource::{
            AddressMode, BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
            BindGroupLayoutEntry, BindingResource, BindingType, BlendState, Buffer,
            BufferBindingType, BufferInitDescriptor, BufferUsages, CachedComputePipelineId,
            CachedRenderPipelineId, ColorTargetState, ColorWrites, CompareFunction,
            ComputePipelineDescriptor, DepthBiasState, DepthStencilState, Extent3d, Face,
            FilterMode, FragmentState, MultisampleState, Origin3d, PipelineCache, PrimitiveState,
            PrimitiveTopology, RenderPipelineDescriptor, Sampler, SamplerBindingType,
            SamplerDescriptor, ShaderStages, StencilState, TexelCopyBufferLayout,
            TexelCopyTextureInfo, TextureAspect, TextureDescriptor, TextureDimension,
            TextureFormat, TextureSampleType, TextureUsages, TextureView, TextureViewDescriptor,
            TextureViewDimension, VertexAttribute, VertexFormat, VertexState, VertexStepMode,
        },
        renderer::{RenderDevice, RenderQueue},
        texture::GpuImage,
    },
};
use std::{borrow::Cow, collections::HashMap};

use crate::{
    texture_palette::{MaterialPropertiesGpu, plugin::ExtractedPalette},
    voxel::{
        CURSOR_QUERY_SHADER_HANDLE, SURFACE_NETS_PASS1_SHADER_HANDLE,
        SURFACE_NETS_PASS3_SHADER_HANDLE,
    },
};

#[derive(Resource)]
pub struct VoxelPipelineLayouts {
    pub pass1_surface_layout: BindGroupLayout,
    pub pass3_surface_layout: BindGroupLayout,
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
                constants: vec![],
                immediate_size: 0,
                zero_initialize_workgroup_memory: false,
            })
        };
        Self {
            // Active cells and their vertex slots.
            pass1_pipeline_id: queue(
                "surface_nets_pass1_pipeline",
                "voxel_pass1_pipeline_layout",
                layouts::pass1_surface_entries(),
                SURFACE_NETS_PASS1_SHADER_HANDLE,
                "cs_main",
            ),
            // Vertices, indices and indirect arguments, per active cell.
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

/// The voxel raster pipeline, one per MSAA sample count in use: a view's
/// pipeline must match its target's sample count.
#[derive(Resource)]
pub struct VoxelRasterPipeline {
    pipelines: HashMap<u32, CachedRenderPipelineId>,
    pub view_layout: BindGroupLayout,
    pub material_layout: BindGroupLayout,
}

impl VoxelRasterPipeline {
    /// The pipeline for `samples`, once `queue_voxel_raster_pipelines` has
    /// queued it.
    #[must_use]
    pub fn id(&self, samples: u32) -> Option<CachedRenderPipelineId> {
        self.pipelines.get(&samples).copied()
    }

    fn queue(&mut self, pipeline_cache: &PipelineCache, samples: u32) {
        self.pipelines.entry(samples).or_insert_with(|| {
            pipeline_cache.queue_render_pipeline(raster_pipeline_descriptor(
                view_layout_entries(),
                material_layout_entries(),
                samples,
            ))
        });
    }
}

impl FromWorld for VoxelRasterPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let view_layout = render_device
            .create_bind_group_layout(Some("voxel_view_layout"), &view_layout_entries());
        let material_layout = render_device
            .create_bind_group_layout(Some("voxel_material_layout"), &material_layout_entries());
        let mut pipeline = Self {
            pipelines: HashMap::new(),
            view_layout,
            material_layout,
        };
        // Start compiling the common case before any view asks for it.
        pipeline.queue(world.resource::<PipelineCache>(), Msaa::default().samples());
        pipeline
    }
}

/// Queues a raster pipeline for every camera's MSAA sample count.
pub fn queue_voxel_raster_pipelines(
    views: Query<&Msaa, With<ExtractedCamera>>,
    pipeline_cache: Res<PipelineCache>,
    mut pipeline: ResMut<VoxelRasterPipeline>,
) {
    for msaa in &views {
        pipeline.queue(&pipeline_cache, msaa.samples());
    }
}

/// Group 0: the view uniform.
fn view_layout_entries() -> Vec<BindGroupLayoutEntry> {
    vec![BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: None,
        },
        count: None,
    }]
}

/// Group 1: the palette (WGSL expects `texture_2d_array<f32>`).
fn material_layout_entries() -> Vec<BindGroupLayoutEntry> {
    vec![
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
    ]
}

fn raster_pipeline_descriptor(
    view_entries: Vec<BindGroupLayoutEntry>,
    material_entries: Vec<BindGroupLayoutEntry>,
    samples: u32,
) -> RenderPipelineDescriptor {
    RenderPipelineDescriptor {
        label: Some("voxel_raster_pipeline".into()),
        layout: vec![
            BindGroupLayoutDescriptor {
                label: Cow::Borrowed("voxel_view_layout"),
                entries: view_entries,
            },
            BindGroupLayoutDescriptor {
                label: Cow::Borrowed("voxel_material_layout"),
                entries: material_entries,
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
            constants: vec![],
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
            constants: vec![],
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
            count: samples,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        zero_initialize_workgroup_memory: false,
        immediate_size: 0,
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

/// Runs in `RenderSystems::Prepare`.
///
/// Rebuilds `VoxelMaterialBindGroup` once the active palette's albedo image
/// is loaded, and never again after
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
