//! The raster passes: each view's visible chunks, one multi-draw per arena,
//! in the pass the layer asked for. Opaque layers are lit by Bevy's PBR
//! after its opaque pass; transparent ones draw after its transparent pass.

use bevy::{
    ecs::system::SystemParam,
    pbr::{MeshViewBindGroup, ViewKeyCache},
    prelude::*,
    render::{
        diagnostic::RecordDiagnostics,
        render_resource::{
            CachedPipelineState, IndexFormat, PipelineCache, RenderPassDescriptor,
            RenderPipeline, StoreOp,
        },
        renderer::{RenderContext, ViewQuery},
        view::{ExtractedView, ViewDepthStencilTexture, ViewTarget},
    },
};

use crate::voxel::{
    draw::{ArenaDraws, VoxelDrawPipeline, VoxelViewDraws, encode_draw_compaction},
    layer::{Layer, LayerIndex, LayerPass, VoxelLayers},
    pipeline::{VoxelDepthPrepass, VoxelRasterKey, VoxelRasterPipeline, raster_keys},
};

/// Everything the raster passes draw with, besides the view.
#[derive(SystemParam)]
pub struct RasterResources<'w> {
    view_keys: Res<'w, ViewKeyCache>,
    depth_prepass: Option<Res<'w, VoxelDepthPrepass>>,
    layers: Res<'w, VoxelLayers>,
    pipeline_cache: Res<'w, PipelineCache>,
    raster_pipeline: Res<'w, VoxelRasterPipeline>,
    draws: Res<'w, VoxelViewDraws>,
    draw_pipeline: Res<'w, VoxelDrawPipeline>,
}

impl RasterResources<'_> {
    /// The compiled pipeline for `key`: `None` while it is still queued or
    /// compiling (normal for the first frames, so silent), warning once if
    /// its compile failed.
    fn pipeline(&self, key: VoxelRasterKey) -> Option<&RenderPipeline> {
        let id = self.raster_pipeline.id(key)?;
        let pipeline = self.pipeline_cache.get_render_pipeline(id);
        if pipeline.is_none()
            && let CachedPipelineState::Err(err) = self.pipeline_cache.get_render_pipeline_state(id)
        {
            warn_once!("raster pipeline {key:?} failed to compile: {err}");
        }
        pipeline
    }

    /// The layers drawn in `pass` that have a material to draw with.
    fn layers_in(&self, pass: LayerPass) -> impl Iterator<Item = (LayerIndex, &Layer)> {
        self.layers
            .iter()
            .filter(move |(_, l)| l.raster.pass == pass && l.material.is_some())
    }
}

/// Draws the opaque layers' chunks `prepare_voxel_view_draws` found visible
/// in this view: after Bevy's opaque pass and before its transparent pass,
/// so blended meshes and gizmos land on top. With [`VoxelDepthPrepass`],
/// every arena is first drawn depth-only, so the PBR pass shades each pixel
/// once.
pub fn voxel_raster_pass_opaque(
    view: ViewQuery<(
        &ViewTarget,
        &ViewDepthStencilTexture,
        &ExtractedView,
        &MeshViewBindGroup,
    )>,
    res: RasterResources,
    ctx: RenderContext,
) {
    raster_pass(LayerPass::Opaque, view, &res, ctx);
}

/// Draws the transparent layers' chunks, after Bevy's transparent pass.
pub fn voxel_raster_pass_transparent(
    view: ViewQuery<(
        &ViewTarget,
        &ViewDepthStencilTexture,
        &ExtractedView,
        &MeshViewBindGroup,
    )>,
    res: RasterResources,
    ctx: RenderContext,
) {
    raster_pass(LayerPass::Transparent, view, &res, ctx);
}

/// One pipeline to draw a group of layers' draws with: a (layer, key) per
/// stage, every layer's prepass before any layer's shading.
struct Stage {
    layer: LayerIndex,
    key: VoxelRasterKey,
}

fn raster_pass(
    pass: LayerPass,
    view: ViewQuery<(
        &ViewTarget,
        &ViewDepthStencilTexture,
        &ExtractedView,
        &MeshViewBindGroup,
    )>,
    res: &RasterResources,
    mut ctx: RenderContext,
) {
    let draws = res.draws.get(view.entity());
    let (target, depth_texture, extracted_view, mesh_view) = view.into_inner();
    let in_pass = |d: &ArenaDraws| {
        d.count > 0
            && res
                .layers
                .get(d.layer)
                .is_some_and(|l| l.raster.pass == pass)
    };
    if !draws.iter().any(in_pass) {
        return;
    }
    let depth_prepass = res.depth_prepass.as_deref().copied().unwrap_or_default();
    let Some(compaction) = res
        .pipeline_cache
        .get_compute_pipeline(res.draw_pipeline.id)
    else {
        return;
    };

    let (prepasses, shades) = stages(res, pass, extracted_view, depth_prepass);
    if shades.is_empty() {
        return;
    }

    encode_draw_compaction(
        ctx.command_encoder(),
        compaction,
        draws.iter().filter(|d| in_pass(d)),
    );

    // Timed on the encoder, around the pass: on tiling GPUs (Adreno, Mali)
    // timestamps inside a render pass can fire before the tiles are shaded.
    let diagnostics = ctx.diagnostic_recorder();
    let recorder = diagnostics.as_deref();
    let (label, span_name) = match pass {
        LayerPass::Opaque => ("voxel_raster_pass", "voxel_raster"),
        LayerPass::Transparent => ("voxel_raster_pass_transparent", "voxel_raster_transparent"),
    };
    let span = recorder.time_span(ctx.command_encoder(), span_name);

    let mut render_pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(target.get_color_attachment())],
        depth_stencil_attachment: Some(depth_texture.get_attachment(StoreOp::Store)),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    render_pass.set_bind_group(0, &mesh_view.main, &mesh_view.main_offsets);
    render_pass.set_bind_group(1, &mesh_view.binding_array, &[]);

    // Every pipeline of a layer shares one layout, so its bind groups stay
    // set; switching layer rebinds group 2.
    let mut bound = None;
    let mut bound_layer = None;
    for Stage { layer, key: base } in prepasses.into_iter().chain(shades) {
        let Some(l) = res.layers.get(layer) else {
            continue;
        };
        let Some(material) = &l.material else {
            continue;
        };
        if bound_layer != Some(layer) {
            render_pass.set_bind_group(2, &material.bind_group, &[]);
            bound_layer = Some(layer);
        }
        for d in draws.iter().filter(|d| d.layer == layer && d.count > 0) {
            let narrow = base.for_draw(d.variant);
            let key = if res.pipeline(narrow).is_some() {
                narrow
            } else {
                base
            };
            if bound != Some(key) {
                let Some(pipeline) = res.pipeline(key) else {
                    continue;
                };
                render_pass.set_render_pipeline(pipeline);
                bound = Some(key);
            }
            let Some(arena) = l.set.arenas.get(d.lod).and_then(|a| a.get(d.arena)) else {
                continue;
            };
            render_pass.set_vertex_buffer(0, arena.final_vertex_buffer.slice(..));
            render_pass.set_index_buffer(arena.index_buffer.slice(..), IndexFormat::Uint32);
            render_pass.set_bind_group(3, &arena.raster_chunk_bind_group, &[]);
            render_pass.multi_draw_indexed_indirect(&d.args, 0, d.count);
        }
    }
    drop(render_pass);
    span.end(ctx.command_encoder());
}

/// The (layer, key) pairs to draw `pass` with for `view`: every layer's
/// prepass, then every layer's shading.
///
/// A prepass that isn't compiled yet is skipped; shading alone is still
/// correct, since its depth test works without the prepass. The full keys
/// (mask test, normal maps) draw every chunk correctly, so they stand in
/// for narrower variants still compiling.
fn stages(
    res: &RasterResources,
    pass: LayerPass,
    view: &ExtractedView,
    depth_prepass: VoxelDepthPrepass,
) -> (Vec<Stage>, Vec<Stage>) {
    let (mut prepasses, mut shades) = (Vec::new(), Vec::new());
    for (layer, l) in res.layers_in(pass) {
        let Some(material) = &l.material else {
            continue;
        };
        let with_prepass = depth_prepass.0 && pass == LayerPass::Opaque;
        let Some((prepass_key, shade_key)) =
            raster_keys(&res.view_keys, view, layer, material, with_prepass)
        else {
            continue;
        };
        if res.pipeline(shade_key).is_none() {
            continue;
        }
        if let Some(key) = prepass_key.filter(|&key| res.pipeline(key).is_some()) {
            prepasses.push(Stage { layer, key });
        }
        shades.push(Stage {
            layer,
            key: shade_key,
        });
    }
    (prepasses, shades)
}
