//! The raster pass: draws each view's visible chunks, one multi-draw per
//! arena, lit by Bevy's PBR.

use bevy::{
    ecs::system::SystemParam,
    pbr::{MeshViewBindGroup, ViewKeyCache},
    prelude::*,
    render::{
        diagnostic::RecordDiagnostics,
        render_resource::{
            IndexFormat, PipelineCache, RenderPassDescriptor, RenderPipeline, StoreOp,
        },
        renderer::{RenderContext, ViewQuery},
        view::{ExtractedView, ViewDepthStencilTexture, ViewTarget},
    },
};

use crate::voxel::{
    draw::{VoxelDrawPipeline, VoxelViewDraws, encode_draw_compaction},
    layer::VoxelLayers,
    pipeline::{
        VoxelDepthPrepass, VoxelMaterialBindGroup, VoxelRasterKey, VoxelRasterPipeline, raster_keys,
    },
};

/// Everything `voxel_raster_pass` draws with, besides the view.
#[derive(SystemParam)]
pub struct RasterResources<'w> {
    view_keys: Res<'w, ViewKeyCache>,
    depth_prepass: Option<Res<'w, VoxelDepthPrepass>>,
    layers: Res<'w, VoxelLayers>,
    pipeline_cache: Res<'w, PipelineCache>,
    raster_pipeline: Res<'w, VoxelRasterPipeline>,
    voxel_material: Res<'w, VoxelMaterialBindGroup>,
    draws: Res<'w, VoxelViewDraws>,
    draw_pipeline: Res<'w, VoxelDrawPipeline>,
}

impl RasterResources<'_> {
    /// The compiled pipeline for `key`, warning (once) while it isn't.
    fn pipeline(&self, key: VoxelRasterKey) -> Option<&RenderPipeline> {
        let id = self.raster_pipeline.id(key)?;
        let pipeline = self.pipeline_cache.get_render_pipeline(id);
        if pipeline.is_none() {
            warn_once!(
                "raster pipeline: {:?}",
                self.pipeline_cache.get_render_pipeline_state(id)
            );
        }
        pipeline
    }
}

/// Draws the chunks `prepare_voxel_view_draws` found visible in this view.
///
/// One multi-draw per arena, lit by Bevy's PBR with the view's own mesh view
/// bind groups (lights, shadows, fog, light probes). With
/// [`VoxelDepthPrepass`], every arena is first drawn depth-only, so the PBR
/// pass shades each pixel once.
pub fn voxel_raster_pass(
    view: ViewQuery<(
        &ViewTarget,
        &ViewDepthStencilTexture,
        &ExtractedView,
        &MeshViewBindGroup,
    )>,
    res: RasterResources,
    mut ctx: RenderContext,
) {
    let draws = res.draws.get(view.entity());
    let (target, depth_texture, extracted_view, mesh_view) = view.into_inner();
    if draws.iter().all(|d| d.count == 0) {
        return;
    }
    let depth_prepass = res.depth_prepass.as_deref().copied().unwrap_or_default();
    let Some((prepass_key, shade_key)) = raster_keys(
        &res.view_keys,
        extracted_view,
        &res.voxel_material,
        depth_prepass,
    ) else {
        return;
    };
    // A prepass that isn't compiled yet is skipped; shading alone is still
    // correct, since its depth test works without the prepass. The full
    // keys (mask test, normal maps) draw every chunk correctly, so they
    // stand in for narrower variants still compiling.
    let prepass_key = prepass_key.filter(|&key| res.pipeline(key).is_some());
    let (Some(_), Some(compaction)) = (
        res.pipeline(shade_key),
        res.pipeline_cache
            .get_compute_pipeline(res.draw_pipeline.id),
    ) else {
        return;
    };

    encode_draw_compaction(ctx.command_encoder(), compaction, draws);

    // Timed on the encoder, around the pass: on tiling GPUs (Adreno, Mali)
    // timestamps inside a render pass can fire before the tiles are shaded.
    let diagnostics = ctx.diagnostic_recorder();
    let recorder = diagnostics.as_deref();
    let span = recorder.time_span(ctx.command_encoder(), "voxel_raster");

    let mut render_pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("voxel_raster_pass"),
        color_attachments: &[Some(target.get_color_attachment())],
        depth_stencil_attachment: Some(depth_texture.get_attachment(StoreOp::Store)),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    render_pass.set_bind_group(0, &mesh_view.main, &mesh_view.main_offsets);
    render_pass.set_bind_group(1, &mesh_view.binding_array, &[]);
    render_pass.set_bind_group(2, &res.voxel_material.bind_group, &[]);

    // All the pipelines share one layout, so the bind groups stay set.
    for base in prepass_key.into_iter().chain([shade_key]) {
        let mut bound = None;
        for d in draws.iter().filter(|d| d.count > 0) {
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
            let Some(arena) = res
                .layers
                .get(d.layer)
                .and_then(|set| set.arenas.get(d.lod))
                .and_then(|a| a.get(d.arena))
            else {
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
