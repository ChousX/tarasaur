//! GPU rendering and querying of chunk fields.
//!
//! Per frame, in the render world:
//! - **Upload** (`prepare_voxel_arena` + material/visibility siblings): new
//!   padded volumes go into a slot of a per-LOD arena, with a vertex
//!   range sized from an exact CPU count of active cells.
//! - **Dispatch**: surface nets in two compute passes over dirty slots only.
//!   `surface_nets_pass1` finds active cells and assigns vertices;
//!   `surface_nets_pass3` places vertices, blends materials, stitches LOD
//!   seams and writes indices and draw args.
//! - **Draw** (`draw`): CPU frustum culling per view, a compaction pass,
//!   then one `multi_draw_indexed_indirect` per arena (`voxel_raster.wesl`:
//!   triplanar albedo + ARM, per-pixel visibility discard, Bevy PBR
//!   lighting), optionally after a depth-only prepass (`VoxelDepthPrepass`).
//!
//! GPU ray queries against the arenas live in `raycast::gpu`.

pub(crate) mod arena;
pub(crate) mod draw;
pub(crate) mod extract;
pub mod inspect;
pub mod layer;
pub(crate) mod mesh;
pub(crate) mod pipeline;
pub(crate) mod raster;
pub(crate) mod resident;
pub mod stats;
pub(crate) mod telemetry;
pub(crate) mod types;
pub(crate) mod upload;
pub(crate) mod util;

pub use arena::ArenaConfig;
pub use layer::{
    LayerDesc, LayerIndex, LayerMaterial, LayerPass, LayerRaster, LayerResidency, NoChannel,
    SideChannel, SubSurface, SurfaceLayer, VoxelLayers,
};
pub use pipeline::{PaletteRaster, VoxelDepthPrepass};
pub use telemetry::GuardTelemetry;
pub use upload::RemeshBudget;

use bevy::{
    asset::{load_internal_asset, uuid_handle},
    core_pipeline::{
        Core3d, Core3dSystems,
        core_3d::{main_opaque_pass_3d, main_transparent_pass_3d},
    },
    prelude::*,
    render::{Render, RenderApp, RenderSystems, extract_resource::ExtractResourcePlugin},
    shader::load_shader_library,
};

use extract::{
    LiveOutbox, Surfaces, add_field_extraction, release_despawned_chunks, single_surface,
    track_live_chunks,
};
use pipeline::{VoxelComputePipeline, VoxelPipelineLayouts};

use crate::{
    field::{GpuChannel, VoxelField},
    plugin::AppVoxelFieldExt,
    voxel::{
        draw::{
            DRAW_COMPACTION_SHADER_HANDLE, VoxelDrawPipeline, VoxelViewDraws,
            prepare_voxel_view_draws,
        },
        extract::extract_removed,
        layer::{FieldLayer, create_layer_layouts, update_layer_materials},
        mesh::dispatch_voxel_compute_passes_batched,
        pipeline::{VoxelRasterPipeline, queue_voxel_raster_pipelines},
        raster::{voxel_raster_pass_opaque, voxel_raster_pass_transparent},
        resident::{ForceResident, lease_chunks_near_camera, tick_force_resident},
        telemetry::map_guard_readbacks,
        upload::{prepare_mask_upload, prepare_material_upload, prepare_shape_upload},
    },
};

pub const SURFACE_NETS_PASS1_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("9f3a1b2c-4d5e-6f70-8192-a3b4c5d6e7f8");
pub const SURFACE_NETS_PASS3_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("2b3c4d5e-6f70-8192-a3b4-c5d6e7f8091a");
pub const VOXEL_RASTER_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("3c4d5e6f-7081-92a3-b4c5-d6e7f8091a2b");
pub const CURSOR_QUERY_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("5e6f7081-92a3-b4c5-d6e7-f8091a2b3c4d");
/// Ordering inside `RenderSystems::Prepare`.
///
/// Bevy runs `Queue` *before* `Prepare`, so the compute dispatch must not live in `Queue`: it would
/// mesh last frame's uploads while this frame's upload had just reset the
/// draw args, and a chunk updated every frame (a held brush) never drew.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum VoxelPrepareSet {
    /// Writes chunk data, ranges and draw args into the arenas.
    Upload,
    /// Re-meshes what `Upload` dirtied, in the same frame.
    Dispatch,
}

/// Order inside `VoxelPrepareSet::Upload`: the shape upload places chunks,
/// then the material and mask uploads follow them into their slots.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum UploadSet {
    Shape,
    Channels,
}

/// Registers a surface layer: a shape field meshed and drawn on the GPU.
pub trait AppSurfaceLayerExt {
    /// Wires `L` in: its shape field (as `add_voxel_field` does, if not
    /// already), its material and mask fields, and the layer's arenas,
    /// uploads, meshing, culling and raster pipelines. Without a render
    /// app (headless), only the fields are registered.
    fn add_surface_layer<L: SurfaceLayer>(&mut self) -> &mut Self;
}

impl AppSurfaceLayerExt for App {
    fn add_surface_layer<L: SurfaceLayer>(&mut self) -> &mut Self {
        self.add_voxel_field::<L::Shape>();
        if <L::Shape as VoxelField>::GPU != GpuChannel::Shape {
            error!(
                "{}'s shape field {} is not a GpuChannel::Shape field; not drawing it",
                std::any::type_name::<L>(),
                std::any::type_name::<L::Shape>()
            );
            return self;
        }
        let Some(render_app) = self.get_sub_app_mut(RenderApp) else {
            return self;
        };
        if render_app
            .world()
            .contains_resource::<FieldLayer<L::Shape>>()
        {
            error!(
                "{} is already a layer's shape field; not registering {} again",
                std::any::type_name::<L::Shape>(),
                std::any::type_name::<L>()
            );
            return self;
        }
        let layer = render_app.world_mut().resource_mut::<VoxelLayers>().add(
            LayerDesc::of::<L::Raster>(),
            L::RESIDENCY,
            L::QUERYABLE,
        );
        add_shape_upload::<L::Shape>(self, layer, L::RESIDENCY, L::surfaces);
        L::Material::register(self, layer, L::RESIDENCY);
        L::Mask::register(self, layer, L::RESIDENCY);
        self
    }
}

/// Hands the shape field `F` to the render world and uploads its surfaces
/// into `layer`'s arenas.
fn add_shape_upload<F: VoxelField>(
    app: &mut App,
    layer: LayerIndex,
    residency: LayerResidency,
    surfaces: Surfaces<F>,
) {
    add_field_extraction::<F>(app, residency, surfaces);
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app.insert_resource(FieldLayer::<F>::new(layer));
    if F::OPTIONAL {
        render_app.add_systems(
            ExtractSchedule,
            extract_removed::<F>.after(release_despawned_chunks),
        );
    }
    render_app.add_systems(Render, prepare_shape_upload::<F>.in_set(UploadSet::Shape));
}

/// Hands the material or mask field `F` to the render world and uploads it
/// into the slots of `layer`'s chunks. Nothing for a field without a GPU
/// channel.
pub(crate) fn add_channel_upload<F: VoxelField>(
    app: &mut App,
    layer: LayerIndex,
    residency: LayerResidency,
) {
    let system = match F::GPU {
        GpuChannel::Material => prepare_material_upload::<F>.in_set(UploadSet::Channels),
        GpuChannel::Mask => prepare_mask_upload::<F>.in_set(UploadSet::Channels),
        GpuChannel::Shape | GpuChannel::None => {
            error!(
                "{} is not a material or mask field; not uploading it as a side channel",
                std::any::type_name::<F>()
            );
            return;
        }
    };
    add_field_extraction::<F>(app, residency, single_surface::<F>);
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app.insert_resource(FieldLayer::<F>::new(layer));
    if F::OPTIONAL {
        render_app.add_systems(
            ExtractSchedule,
            extract_removed::<F>.after(release_despawned_chunks),
        );
    }
    render_app.add_systems(Render, system);
}

/// Extraction, arenas, meshing, drawing, ray queries and the cursor.
pub struct VoxelRenderPlugin;

/// Embeds the voxel shaders. `voxel_common` is a shader library the others
/// import as `tarasaur::voxel::shaders::voxel_common`; registering it as an
/// embedded asset is what gives it that path.
fn load_shaders(app: &mut App) {
    load_shader_library!(app, "shaders/voxel_common.wesl");
    load_shader_library!(app, "shaders/voxel_raster_vertex.wesl");
    load_internal_asset!(
        app,
        DRAW_COMPACTION_SHADER_HANDLE,
        "shaders/draw_compaction.wesl",
        Shader::from_wesl
    );
    load_internal_asset!(
        app,
        SURFACE_NETS_PASS1_SHADER_HANDLE,
        "shaders/surface_nets_pass1.wesl",
        Shader::from_wesl
    );
    load_internal_asset!(
        app,
        SURFACE_NETS_PASS3_SHADER_HANDLE,
        "shaders/surface_nets_pass3.wesl",
        Shader::from_wesl
    );
    load_internal_asset!(
        app,
        VOXEL_RASTER_SHADER_HANDLE,
        "shaders/voxel_raster.wesl",
        Shader::from_wesl
    );
    load_internal_asset!(
        app,
        CURSOR_QUERY_SHADER_HANDLE,
        "shaders/cursor_query.wesl",
        Shader::from_wesl
    );
}

impl Plugin for VoxelRenderPlugin {
    fn build(&self, app: &mut App) {
        load_shaders(app);

        app.init_resource::<VoxelDepthPrepass>()
            .add_plugins(ExtractResourcePlugin::<VoxelDepthPrepass>::default())
            .init_resource::<ForceResident>()
            .add_systems(First, tick_force_resident)
            .add_systems(Update, lease_chunks_near_camera)
            .init_resource::<LiveOutbox>()
            .add_systems(Last, track_live_chunks);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app.init_resource::<VoxelViewDraws>();

        render_app
            .init_resource::<VoxelLayers>()
            .init_resource::<RemeshBudget>()
            .init_resource::<GuardTelemetry>()
            .add_systems(ExtractSchedule, release_despawned_chunks)
            .configure_sets(
                Render,
                (VoxelPrepareSet::Upload, VoxelPrepareSet::Dispatch)
                    .chain()
                    .in_set(RenderSystems::Prepare),
            )
            .configure_sets(
                Render,
                (UploadSet::Shape, UploadSet::Channels)
                    .chain()
                    .in_set(VoxelPrepareSet::Upload),
            )
            .add_systems(
                Render,
                (
                    update_layer_materials.in_set(RenderSystems::Prepare),
                    queue_voxel_raster_pipelines
                        .in_set(RenderSystems::Prepare)
                        .after(update_layer_materials),
                    prepare_voxel_view_draws
                        .in_set(RenderSystems::Prepare)
                        .after(VoxelPrepareSet::Upload),
                    dispatch_voxel_compute_passes_batched.in_set(VoxelPrepareSet::Dispatch),
                    map_guard_readbacks.in_set(RenderSystems::Cleanup),
                ),
            )
            // Opaque layers: after Bevy's opaque pass, and before its
            // transparent pass so blended meshes and gizmos land on top of
            // them. Transparent layers: after Bevy's transparent pass.
            .add_systems(
                Core3d,
                (
                    voxel_raster_pass_opaque
                        .after(main_opaque_pass_3d)
                        .before(main_transparent_pass_3d),
                    voxel_raster_pass_transparent.after(main_transparent_pass_3d),
                )
                    .in_set(Core3dSystems::MainPass),
            );
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<VoxelRasterPipeline>()
            .init_resource::<VoxelPipelineLayouts>()
            .init_resource::<VoxelComputePipeline>()
            .init_resource::<VoxelDrawPipeline>();
        create_layer_layouts(render_app.world_mut());
    }
}
