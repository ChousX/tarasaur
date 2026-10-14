//! Pipelines and bind groups: the meshing and query compute pipelines, the
//! PBR raster pipeline (one per view key) and the palette material.

pub(crate) mod layouts;
use bevy::{
    mesh::VertexBufferLayout,
    pbr::{
        MeshPipeline, MeshPipelineKey, TONEMAPPING_LUT_SAMPLER_BINDING_INDEX,
        TONEMAPPING_LUT_TEXTURE_BINDING_INDEX, ViewKeyCache,
    },
    prelude::*,
    render::{
        RenderApp,
        camera::ExtractedCamera,
        extract_resource::ExtractResource,
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
        view::ExtractedView,
    },
    shader::ShaderDefVal,
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

/// Draw the voxel terrain's depth first, then shade only the front-most surface.
///
/// Saves PBR shading on hidden terrain (overdraw) at the cost of a second,
/// depth-only draw. Set it in the main world; it's extracted.
#[derive(Resource, Clone, Copy, Debug, Default, ExtractResource)]
#[extract_app(RenderApp)]
pub struct VoxelDepthPrepass(pub bool);

/// Which of the voxel raster pipelines a key is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RasterPass {
    /// PBR shading that writes depth (no prepass).
    Shade,
    /// Depth only, for the prepass.
    Prepass,
    /// PBR shading against the prepass's depth, without writing it.
    ShadeAfterPrepass,
}

/// What a raster pipeline is specialized on.
///
/// The view's mesh pipeline key (MSAA, target format, tonemapping, prepasses,
/// fog, light probes, ...), whether the palette has an ARM texture to
/// sample, and which pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VoxelRasterKey {
    pub view: MeshPipelineKey,
    pub arm: bool,
    pub pass: RasterPass,
}

/// The voxel raster pipelines, one per [`VoxelRasterKey`] in use.
///
/// Lighting is Bevy's PBR (`apply_pbr_lighting`), so groups 0 and 1 are
/// Bevy's mesh view bind groups, and each pipeline must be specialized with
/// the same view key the view's bind group was built for. Like Bevy's
/// deferred lighting pass, this reads that key from `ViewKeyCache`.
#[derive(Resource)]
pub struct VoxelRasterPipeline {
    pipelines: HashMap<VoxelRasterKey, CachedRenderPipelineId>,
    pub material_layout: BindGroupLayout,
}

impl VoxelRasterPipeline {
    /// The pipeline for `key`, once `queue_voxel_raster_pipelines` has
    /// queued it.
    #[must_use]
    pub fn id(&self, key: VoxelRasterKey) -> Option<CachedRenderPipelineId> {
        self.pipelines.get(&key).copied()
    }
}

impl FromWorld for VoxelRasterPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        Self {
            pipelines: HashMap::new(),
            material_layout: render_device.create_bind_group_layout(
                Some("voxel_material_layout"),
                &material_layout_entries(),
            ),
        }
    }
}

/// The keys a view draws with, in order: the prepass (if on), then the
/// shading pass. `None` until Bevy has cached the view's mesh pipeline key
/// (it does so in `PrepareAssets`).
#[must_use]
pub fn raster_keys(
    view_keys: &ViewKeyCache,
    view: &ExtractedView,
    material: &VoxelMaterialBindGroup,
    depth_prepass: VoxelDepthPrepass,
) -> Option<(Option<VoxelRasterKey>, VoxelRasterKey)> {
    let &view_key = view_keys.get(&view.retained_view_entity)?;
    let key = |arm, pass| VoxelRasterKey {
        view: view_key,
        arm,
        pass,
    };
    Some(if depth_prepass.0 {
        (
            Some(key(false, RasterPass::Prepass)),
            key(material.has_arm, RasterPass::ShadeAfterPrepass),
        )
    } else {
        (None, key(material.has_arm, RasterPass::Shade))
    })
}

/// Queues the raster pipelines every camera's view needs.
pub fn queue_voxel_raster_pipelines(
    views: Query<&ExtractedView, With<ExtractedCamera>>,
    view_keys: Res<ViewKeyCache>,
    material: Res<VoxelMaterialBindGroup>,
    depth_prepass: Option<Res<VoxelDepthPrepass>>,
    mesh_pipeline: Res<MeshPipeline>,
    pipeline_cache: Res<PipelineCache>,
    mut pipeline: ResMut<VoxelRasterPipeline>,
) {
    let depth_prepass = depth_prepass.map_or_else(VoxelDepthPrepass::default, |p| *p);
    for view in &views {
        let Some((prepass, shade)) = raster_keys(&view_keys, view, &material, depth_prepass) else {
            continue;
        };
        for key in prepass.into_iter().chain([shade]) {
            pipeline.pipelines.entry(key).or_insert_with(|| {
                pipeline_cache
                    .queue_render_pipeline(raster_pipeline_descriptor(&mesh_pipeline, key))
            });
        }
    }
}

/// Group 2: the palette (WGSL expects `texture_2d_array<f32>`).
fn material_layout_entries() -> Vec<BindGroupLayoutEntry> {
    let texture_array = |binding| BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: true },
            view_dimension: TextureViewDimension::D2Array,
            multisampled: false,
        },
        count: None,
    };
    vec![
        // Binding 0: albedo
        texture_array(0),
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
        // Binding 3: ARM (a 1x1 dummy when the palette has none; the shader
        // only samples it under `VOXEL_ARM`)
        texture_array(3),
    ]
}

/// Shader defs for the parts of Bevy's PBR lighting that depend on the
/// view. Mirrors Bevy's deferred lighting pipeline, minus what only applies
/// to meshes in Bevy's prepasses (the voxel terrain isn't in them).
fn view_shader_defs(mesh_pipeline: &MeshPipeline, key: MeshPipelineKey) -> Vec<ShaderDefVal> {
    let mut defs: Vec<ShaderDefVal> = Vec::new();
    let mut flag = |on: bool, name: &'static str| {
        if on {
            defs.push(name.into());
        }
    };

    flag(
        key.contains(MeshPipelineKey::ENVIRONMENT_MAP),
        "ENVIRONMENT_MAP",
    );
    flag(
        key.contains(MeshPipelineKey::IRRADIANCE_VOLUME) && IRRADIANCE_VOLUMES_ARE_USABLE,
        "IRRADIANCE_VOLUME",
    );
    flag(
        key.contains(MeshPipelineKey::NORMAL_PREPASS),
        "NORMAL_PREPASS",
    );
    flag(
        key.contains(MeshPipelineKey::DEPTH_PREPASS),
        "DEPTH_PREPASS",
    );
    flag(
        key.contains(MeshPipelineKey::MOTION_VECTOR_PREPASS),
        "MOTION_VECTOR_PREPASS",
    );
    flag(
        key.contains(MeshPipelineKey::DEFERRED_PREPASS),
        "DEFERRED_PREPASS",
    );
    flag(key.contains(MeshPipelineKey::DISTANCE_FOG), "DISTANCE_FOG");
    flag(key.contains(MeshPipelineKey::ATMOSPHERE), "ATMOSPHERE");
    flag(key.msaa_samples() > 1, "MULTISAMPLED");

    let projection = key.intersection(MeshPipelineKey::VIEW_PROJECTION_RESERVED_BITS);
    for (bits, name) in [
        (
            MeshPipelineKey::VIEW_PROJECTION_NONSTANDARD,
            "VIEW_PROJECTION_NONSTANDARD",
        ),
        (
            MeshPipelineKey::VIEW_PROJECTION_PERSPECTIVE,
            "VIEW_PROJECTION_PERSPECTIVE",
        ),
        (
            MeshPipelineKey::VIEW_PROJECTION_ORTHOGRAPHIC,
            "VIEW_PROJECTION_ORTHOGRAPHIC",
        ),
    ] {
        flag(projection == bits, name);
    }

    let shadow_filter = key.intersection(MeshPipelineKey::SHADOW_FILTER_METHOD_RESERVED_BITS);
    flag(
        shadow_filter == MeshPipelineKey::SHADOW_FILTER_METHOD_HARDWARE_2X2,
        "SHADOW_FILTER_METHOD_HARDWARE_2X2",
    );
    flag(
        shadow_filter == MeshPipelineKey::SHADOW_FILTER_METHOD_GAUSSIAN,
        "SHADOW_FILTER_METHOD_GAUSSIAN",
    );
    flag(
        shadow_filter == MeshPipelineKey::SHADOW_FILTER_METHOD_TEMPORAL,
        "SHADOW_FILTER_METHOD_TEMPORAL",
    );

    flag(
        mesh_pipeline.binding_arrays_are_usable,
        "MULTIPLE_LIGHT_PROBES_IN_ARRAY",
    );
    flag(
        mesh_pipeline.binding_arrays_are_usable,
        "MULTIPLE_LIGHTMAPS_IN_ARRAY",
    );
    flag(
        IRRADIANCE_VOLUMES_ARE_USABLE,
        "IRRADIANCE_VOLUMES_ARE_USABLE",
    );
    flag(
        mesh_pipeline.clustered_decals_are_usable,
        "CLUSTERED_DECALS_ARE_USABLE",
    );
    defs.extend(tonemap_shader_defs(key));
    defs
}

/// Defs for tonemapping in the shader (non-HDR cameras): the method, the
/// LUT bindings and debanding. Empty when the view tonemaps elsewhere.
fn tonemap_shader_defs(key: MeshPipelineKey) -> Vec<ShaderDefVal> {
    if !key.contains(MeshPipelineKey::TONEMAP_IN_SHADER) {
        return Vec::new();
    }
    let method = key.intersection(MeshPipelineKey::TONEMAP_METHOD_RESERVED_BITS);
    let mut defs: Vec<ShaderDefVal> = vec![
        "TONEMAP_IN_SHADER".into(),
        ShaderDefVal::UInt(
            "TONEMAPPING_LUT_TEXTURE_BINDING_INDEX".into(),
            TONEMAPPING_LUT_TEXTURE_BINDING_INDEX,
        ),
        ShaderDefVal::UInt(
            "TONEMAPPING_LUT_SAMPLER_BINDING_INDEX".into(),
            TONEMAPPING_LUT_SAMPLER_BINDING_INDEX,
        ),
    ];
    defs.extend(
        TONEMAP_METHODS
            .into_iter()
            .filter(|&(bits, _)| method == bits)
            .map(|(_, name)| name.into()),
    );
    // Debanding is tied to tonemapping in the shader.
    if key.contains(MeshPipelineKey::DEBAND_DITHER) {
        defs.push("DEBAND_DITHER".into());
    }
    defs
}

/// `MeshPipelineKey` tonemapping methods and the shader def for each.
const TONEMAP_METHODS: [(MeshPipelineKey, &str); 9] = [
    (
        MeshPipelineKey::TONEMAP_METHOD_LINEAR,
        "TONEMAP_METHOD_LINEAR",
    ),
    (
        MeshPipelineKey::TONEMAP_METHOD_REINHARD,
        "TONEMAP_METHOD_REINHARD",
    ),
    (
        MeshPipelineKey::TONEMAP_METHOD_REINHARD_LUMINANCE,
        "TONEMAP_METHOD_REINHARD_LUMINANCE",
    ),
    (
        MeshPipelineKey::TONEMAP_METHOD_ACES_FITTED,
        "TONEMAP_METHOD_ACES_FITTED",
    ),
    (MeshPipelineKey::TONEMAP_METHOD_AGX, "TONEMAP_METHOD_AGX"),
    (
        MeshPipelineKey::TONEMAP_METHOD_SOMEWHAT_BORING_DISPLAY_TRANSFORM,
        "TONEMAP_METHOD_SOMEWHAT_BORING_DISPLAY_TRANSFORM",
    ),
    (
        MeshPipelineKey::TONEMAP_METHOD_BLENDER_FILMIC,
        "TONEMAP_METHOD_BLENDER_FILMIC",
    ),
    (
        MeshPipelineKey::TONEMAP_METHOD_TONY_MC_MAPFACE,
        "TONEMAP_METHOD_TONY_MC_MAPFACE",
    ),
    (
        MeshPipelineKey::TONEMAP_METHOD_PBR_NEUTRAL,
        "TONEMAP_METHOD_PBR_NEUTRAL",
    ),
];
/// Bevy's `IRRADIANCE_VOLUMES_ARE_USABLE`, which it keeps crate-private.
const IRRADIANCE_VOLUMES_ARE_USABLE: bool = cfg!(not(target_arch = "wasm32"));

fn raster_pipeline_descriptor(
    mesh_pipeline: &MeshPipeline,
    key: VoxelRasterKey,
) -> RenderPipelineDescriptor {
    let mut shader_defs = view_shader_defs(mesh_pipeline, key.view);
    if key.arm {
        shader_defs.push("VOXEL_ARM".into());
    }
    // The prepass writes depth only; the shading pass after it tests against
    // that depth without writing it, so each pixel is shaded once.
    let (label, fragment_entry, color_writes, depth_write) = match key.pass {
        RasterPass::Shade => ("voxel_raster_pipeline", "fs_main", ColorWrites::ALL, true),
        RasterPass::Prepass => (
            "voxel_depth_prepass_pipeline",
            "fs_prepass",
            ColorWrites::empty(),
            true,
        ),
        RasterPass::ShadeAfterPrepass => {
            ("voxel_raster_pipeline", "fs_main", ColorWrites::ALL, false)
        }
    };
    let view_layout = mesh_pipeline.get_view_layout(key.view.into());
    let cull_mode = if key.view.contains(MeshPipelineKey::INVERT_CULLING) {
        Face::Front
    } else {
        Face::Back
    };

    RenderPipelineDescriptor {
        label: Some(label.into()),
        layout: vec![
            view_layout.main_layout,
            view_layout.binding_array_layout,
            BindGroupLayoutDescriptor {
                label: Cow::Borrowed("voxel_material_layout"),
                entries: material_layout_entries(),
            },
            BindGroupLayoutDescriptor {
                label: Cow::Borrowed("voxel_raster_chunk_visibility_layout"),
                entries: layouts::raster_chunk_visibility_entries(),
            },
        ],
        vertex: VertexState {
            shader: crate::voxel::VOXEL_RASTER_SHADER_HANDLE,
            entry_point: Some("vs_main".into()),
            shader_defs: shader_defs.clone(),
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
            entry_point: Some(fragment_entry.into()),
            shader_defs,
            constants: vec![],
            targets: vec![Some(ColorTargetState {
                format: key.view.target_format(),
                blend: Some(BlendState::REPLACE),
                write_mask: color_writes,
            })],
        }),
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleList,
            cull_mode: Some(cull_mode),
            ..default()
        },
        depth_stencil: Some(DepthStencilState {
            format: TextureFormat::Depth32Float,
            depth_write_enabled: Some(depth_write),
            depth_compare: Some(CompareFunction::GreaterEqual),
            stencil: StencilState::default(),
            bias: DepthBiasState::default(),
        }),
        multisample: MultisampleState {
            count: key.view.msaa_samples(),
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
    albedo: &TextureView,
    arm: &TextureView,
    sampler: &Sampler,
    properties: &Buffer,
) -> BindGroup {
    render_device.create_bind_group(
        Some(label),
        layout,
        &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(albedo),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Sampler(sampler),
            },
            BindGroupEntry {
                binding: 2,
                resource: properties.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: BindingResource::TextureView(arm),
            },
        ],
    )
}

/// A 1x1, one-layer texture array of `texel`, standing in for a palette
/// texture that isn't there (yet).
fn dummy_texture_array(
    render_device: &RenderDevice,
    render_queue: &RenderQueue,
    label: &'static str,
    format: TextureFormat,
    texel: [u8; 4],
) -> TextureView {
    let extent = Extent3d {
        width: 1,
        height: 1,
        depth_or_array_layers: 1,
    };
    let texture = render_device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    render_queue.write_texture(
        TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        &texel,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        extent,
    );
    // A 1-layer D2 texture defaults to a D2 view; the shader wants D2Array.
    texture.create_view(&TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2Array),
        ..Default::default()
    })
}

/// The bind group actually used by `voxel_raster_pass`. Starts out as a
/// 1x1 white dummy; `update_voxel_material_bind_group` replaces it once a
/// real palette's textures have finished loading.
#[derive(Resource)]
pub struct VoxelMaterialBindGroup {
    pub bind_group: BindGroup,
    /// Whether `bind_group` holds a real ARM texture (selects `VOXEL_ARM`).
    pub has_arm: bool,
    /// Stands in for a palette without an ARM texture.
    dummy_arm: TextureView,
}

impl FromWorld for VoxelMaterialBindGroup {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let render_queue = world.resource::<RenderQueue>();
        let raster_pipeline = world.resource::<VoxelRasterPipeline>();

        let albedo = dummy_texture_array(
            render_device,
            render_queue,
            "voxel_dummy_albedo_texture",
            TextureFormat::Rgba8UnormSrgb,
            [255, 255, 255, 255],
        );
        // Unoccluded, fully rough, not metallic. Never sampled (no
        // `VOXEL_ARM`), but the layout needs a texture here.
        let dummy_arm = dummy_texture_array(
            render_device,
            render_queue,
            "voxel_dummy_arm_texture",
            TextureFormat::Rgba8Unorm,
            [255, 255, 0, 255],
        );
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

        Self {
            bind_group: material_bind_group(
                render_device,
                &raster_pipeline.material_layout,
                "voxel_dummy_material_bind_group",
                &albedo,
                &dummy_arm,
                &sampler,
                &properties,
            ),
            has_arm: false,
            dummy_arm,
        }
    }
}

/// The textures a material bind group was built from: (albedo, ARM).
type PaletteIds = (AssetId<Image>, Option<AssetId<Image>>);

/// Runs in `RenderSystems::Prepare`.
///
/// Rebuilds `VoxelMaterialBindGroup` once the active palette's albedo (and
/// ARM, if it has one) image is loaded, and never again after that for the
/// same handles — in-place edits to palette.materials after the initial
/// build won't hot-reload; only a handle change (different palette asset)
/// triggers a rebuild.
pub fn update_voxel_material_bind_group(
    render_device: Res<RenderDevice>,
    raster_pipeline: Res<VoxelRasterPipeline>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    extracted: Option<Res<ExtractedPalette>>,
    mut material: ResMut<VoxelMaterialBindGroup>,
    mut built_for: Local<Option<PaletteIds>>,
) {
    let Some(extracted) = extracted else {
        return; // No PalettePlugin registered — keep the dummy.
    };

    let ids = (
        extracted.albedo.id(),
        extracted.arm.as_ref().map(Handle::id),
    );
    if *built_for == Some(ids) {
        return; // Already built for these exact textures — nothing changed.
    }
    let Some(albedo) = gpu_images.get(&extracted.albedo) else {
        return; // Not loaded/converted to GPU form yet — try again next frame.
    };
    let arm = match &extracted.arm {
        Some(handle) => match gpu_images.get(handle) {
            Some(image) => Some(&image.texture_view),
            None => return, // wait for it, so the first build already has it
        },
        None => None,
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

    let material = &mut *material;
    material.bind_group = material_bind_group(
        &render_device,
        &raster_pipeline.material_layout,
        "voxel_material_bind_group",
        &albedo.texture_view,
        arm.unwrap_or(&material.dummy_arm),
        &sampler,
        &properties,
    );
    material.has_arm = arm.is_some();
    *built_for = Some(ids);
}
