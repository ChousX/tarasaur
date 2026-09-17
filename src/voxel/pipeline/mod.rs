// mod.rs
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
        COMPUTE_CHUNK_BASES_SHADER_HANDLE, STREAM_COMPACTION_SHADER_HANDLE,
        SURFACE_NETS_PASS1_SHADER_HANDLE, SURFACE_NETS_PASS3_SHADER_HANDLE,
    },
};
#[derive(Resource)]
pub struct VoxelPipelineLayouts {
    pub pass1_surface_layout: BindGroupLayout,
    pub pass3_surface_layout: BindGroupLayout,
    pub compaction_bind_group_layout: BindGroupLayout,
    pub chunk_bases_layout: BindGroupLayout,
}

impl FromWorld for VoxelPipelineLayouts {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let pass1_surface_layout = render_device.create_bind_group_layout(
            Some("voxel_surface_pass1_layout"),
            &layouts::pass1_surface_entries(),
        );

        let pass3_surface_layout = render_device.create_bind_group_layout(
            Some("voxel_surface_pass3_layout"),
            &layouts::pass3_surface_entries(),
        );

        let compaction_bind_group_layout = render_device.create_bind_group_layout(
            Some("voxel_compaction_layout"),
            &layouts::compaction_entries(),
        );

        let chunk_bases_layout = render_device.create_bind_group_layout(
            Some("voxel_chunk_bases_layout"),
            &layouts::chunk_bases_entries(),
        );

        Self {
            pass1_surface_layout,
            pass3_surface_layout,
            compaction_bind_group_layout,
            chunk_bases_layout,
        }
    }
}

#[derive(Resource)]
pub struct VoxelComputePipeline {
    pub pass1_pipeline_id: CachedComputePipelineId,
    pub stream_compaction_pipeline_id: CachedComputePipelineId,
    pub scan_block_sums_pipeline_id: CachedComputePipelineId,
    pub stream_compaction_resolve_pipeline_id: CachedComputePipelineId,
    pub write_chunk_active_count_pipeline_id: CachedComputePipelineId,
    pub chunk_bases_pipeline_id: CachedComputePipelineId,
    pub pass3_pipeline_id: CachedComputePipelineId,
}

impl FromWorld for VoxelComputePipeline {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();

        // 1. Pass 1 Pipeline (Edge Classification & Vertex Generation)
        let pass1_pipeline_id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(Cow::Borrowed("surface_nets_pass1_pipeline")),
            layout: vec![BindGroupLayoutDescriptor {
                label: Cow::Borrowed("voxel_pass1_pipeline_layout"),
                entries: layouts::pass1_surface_entries(),
            }],
            shader: SURFACE_NETS_PASS1_SHADER_HANDLE,
            entry_point: Some(Cow::Borrowed("cs_main")),
            shader_defs: vec![],
            immediate_size: 0,
            zero_initialize_workgroup_memory: false,
        });

        // 2. Stream Compaction Scan Pipeline
        let stream_compaction_pipeline_id =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some(Cow::Borrowed("stream_compaction_scan_pipeline")),
                layout: vec![BindGroupLayoutDescriptor {
                    label: Cow::Borrowed("voxel_compaction_pipeline_layout"),
                    entries: layouts::compaction_entries(),
                }],
                shader: STREAM_COMPACTION_SHADER_HANDLE.clone(),
                entry_point: Some(Cow::Borrowed("scan_workgroup")),
                shader_defs: vec![],
                immediate_size: 0,
                zero_initialize_workgroup_memory: false,
            });

        // 2.5
        let scan_block_sums_pipeline_id =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some(Cow::Borrowed("scan_block_sums")),
                layout: vec![BindGroupLayoutDescriptor {
                    label: Cow::Borrowed("voxel_compaction_pipeline_layout"),
                    entries: layouts::compaction_entries(),
                }],
                shader: STREAM_COMPACTION_SHADER_HANDLE.clone(),
                entry_point: Some(Cow::Borrowed("scan_block_sums")),
                shader_defs: vec![],
                immediate_size: 0,
                zero_initialize_workgroup_memory: false,
            });

        // 3. Stream Compaction Resolve Pipeline
        let stream_compaction_resolve_pipeline_id =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some(Cow::Borrowed("stream_compaction_resolve_pipeline")),
                layout: vec![BindGroupLayoutDescriptor {
                    label: Cow::Borrowed("voxel_compaction_pipeline_layout"),
                    entries: layouts::compaction_entries(),
                }],
                shader: STREAM_COMPACTION_SHADER_HANDLE.clone(),
                entry_point: Some(Cow::Borrowed("resolve_block_offsets")),
                shader_defs: vec![],
                immediate_size: 0,
                zero_initialize_workgroup_memory: false,
            });

        // 3.5 Write per-chunk active cell counts (one thread per active chunk)
        let write_chunk_active_count_pipeline_id =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some(Cow::Borrowed("write_chunk_active_count_pipeline")),
                layout: vec![BindGroupLayoutDescriptor {
                    label: Cow::Borrowed("voxel_compaction_pipeline_layout"),
                    entries: layouts::compaction_entries(),
                }],
                shader: STREAM_COMPACTION_SHADER_HANDLE,
                entry_point: Some(Cow::Borrowed("write_chunk_active_count")),
                shader_defs: vec![],
                immediate_size: 0,
                zero_initialize_workgroup_memory: false,
            });

        // 3.75 Compute per-chunk dynamic vertex/index bases (single workgroup, serial)
        let chunk_bases_pipeline_id =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some(Cow::Borrowed("compute_chunk_bases_pipeline")),
                layout: vec![BindGroupLayoutDescriptor {
                    label: Cow::Borrowed("voxel_chunk_bases_pipeline_layout"),
                    entries: layouts::chunk_bases_entries(),
                }],
                shader: COMPUTE_CHUNK_BASES_SHADER_HANDLE,
                entry_point: Some(Cow::Borrowed("cs_main")),
                shader_defs: vec![],
                immediate_size: 0,
                zero_initialize_workgroup_memory: false,
            });

        // 4. Pass 3 Pipeline (Index Assembly & Indirect Arguments)
        let pass3_pipeline_id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(Cow::Borrowed("surface_nets_pass3_pipeline")),
            layout: vec![BindGroupLayoutDescriptor {
                label: Cow::Borrowed("voxel_pass3_pipeline_layout"),
                entries: layouts::pass3_surface_entries(),
            }],
            shader: SURFACE_NETS_PASS3_SHADER_HANDLE,
            entry_point: Some(Cow::Borrowed("cs_main")),
            shader_defs: vec![],
            immediate_size: 0,
            zero_initialize_workgroup_memory: false,
        });

        Self {
            pass1_pipeline_id,
            stream_compaction_pipeline_id,
            scan_block_sums_pipeline_id,
            stream_compaction_resolve_pipeline_id,
            write_chunk_active_count_pipeline_id,
            chunk_bases_pipeline_id,
            pass3_pipeline_id,
        }
    }
}

#[derive(Resource)]
pub struct VoxelDummyMaterial {
    pub texture_view: TextureView,
    pub sampler: Sampler,
    pub properties_buffer: Buffer,
    pub bind_group: BindGroup,
}

impl FromWorld for VoxelDummyMaterial {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let render_queue = world.resource::<RenderQueue>();
        let raster_pipeline = world.resource::<VoxelRasterPipeline>();

        let dummy_texture = render_device.create_texture(&TextureDescriptor {
            label: Some("voxel_dummy_material_texture"),
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1, // 1-layer array, not a plain 2D texture
            },
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
            Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        // ASSUMPTION: default_view creates a D2Array view for a texture
        // with dimension D2 + array_layer_count 1 when array_layer_count
        // is explicit — TextureViewDescriptor's dimension field defaults
        // to inferring from the texture, which for a D2 texture with
        // depth_or_array_layers > 0 declared as an array... WGPU actually
        // needs the *view* dimension set explicitly to D2Array here, since
        // a 1-layer D2 texture defaults its view to D2, not D2Array.
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

        // Single dummy material entry — matches MaterialPropertiesGpu's
        // layout exactly so the real buffer can later replace this
        // 1-for-1 with no shader-side changes.
        let dummy_properties = [0.1f32, 4.0, -1.0, -1.0]; // texture_scale, blend_sharpness, roughness_override, metallic_override
        let properties_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("voxel_dummy_material_properties_buffer"),
            contents: bytemuck::cast_slice(&dummy_properties),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        let bind_group = render_device.create_bind_group(
            Some("voxel_dummy_material_bind_group"),
            &raster_pipeline.material_layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&texture_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&sampler),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: properties_buffer.as_entire_binding(),
                },
            ],
        );

        Self {
            texture_view,
            sampler,
            properties_buffer,
            bind_group,
        }
    }
}
#[derive(Resource)]
pub struct VoxelRasterPipeline {
    pub pipeline_id: CachedRenderPipelineId,
    pub view_layout: BindGroupLayout,
    pub material_layout: BindGroupLayout, // Stored for dummy bind group creation
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

        // Group 1: Material Layout (WGSL expects texture_2d<f32>)
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
                //cull_mode: None,
                ..default()
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(CompareFunction::GreaterEqual),
                //depth_compare: Some(CompareFunction::LessEqual),
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

/// The bind group actually used by `voxel_raster_pass`. Starts out cloned
/// from `VoxelDummyMaterial`'s bind group; `update_voxel_material_bind_group`
/// replaces it once a real palette's albedo texture has finished loading.
/// A separate resource rather than mutating `VoxelDummyMaterial` in place —
/// the dummy stays available as a known-good fallback value to reset to
/// if that's ever useful (e.g. palette hot-swap failing validation).
#[derive(Resource)]
pub struct VoxelMaterialBindGroup(pub BindGroup);

impl FromWorld for VoxelMaterialBindGroup {
    fn from_world(world: &mut World) -> Self {
        // ASSUMPTION: BindGroup is Clone (an Arc-wrapped wgpu handle, per
        // Bevy's usual render_resource wrapper convention). If this
        // doesn't compile, the fallback is restructuring VoxelDummyMaterial
        // to build a bind group once and have both resources hold a
        // reference to the same underlying wgpu::BindGroup instead.
        let dummy = world.resource::<VoxelDummyMaterial>();
        Self(dummy.bind_group.clone())
    }
}

/// Runs in RenderSystems::Prepare. Rebuilds VoxelMaterialBindGroup once
/// the active palette's albedo image is loaded, and never again after
/// that for the same albedo handle — in-place edits to palette.materials
/// after the initial build won't hot-reload; only a handle change
/// (different palette asset) triggers a rebuild. Acceptable for now since
/// palette hot-editing isn't a stated requirement.
pub fn update_voxel_material_bind_group(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
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

    // ASSUMPTION: GpuImage exposes `texture_view: TextureView` — field
    // name based on Bevy's typical GpuImage shape; verify against actual
    // compiler output if this doesn't match.
    let Some(gpu_image) = gpu_images.get(&extracted.albedo) else {
        return; // Not loaded/converted to GPU form yet — try again next frame.
    };

    let material_data: Vec<MaterialPropertiesGpu> = extracted
        .materials
        .iter()
        .map(MaterialPropertiesGpu::from)
        .collect();
    // Guard against an empty palette — a zero-length storage buffer is
    // invalid to create/bind. Falls back to a single dummy entry so the
    // bind group is always valid even if `materials` is empty.
    let material_bytes: Vec<MaterialPropertiesGpu> = if material_data.is_empty() {
        vec![MaterialPropertiesGpu::default()]
    } else {
        material_data
    };

    let properties_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("voxel_material_properties_buffer"),
        contents: bytemuck::cast_slice(&material_bytes),
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

    let new_bind_group = render_device.create_bind_group(
        Some("voxel_material_bind_group"),
        &raster_pipeline.material_layout,
        &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&gpu_image.texture_view),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Sampler(&sampler),
            },
            BindGroupEntry {
                binding: 2,
                resource: properties_buffer.as_entire_binding(),
            },
        ],
    );

    bind_group.0 = new_bind_group;
    *built_for = Some(albedo_id);
    let _ = &render_queue; // kept for symmetry with other prepare systems; unused directly here since create_buffer_with_data already uploads
}
