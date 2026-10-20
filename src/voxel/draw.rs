//! Per-view culling and batched drawing of the voxel arenas.
//!
//! Each frame, every camera's frustum culls the resident chunks on the CPU
//! (a box test per chunk). The visible slots go to a tiny compute pass that
//! packs their draw arguments, which pass3 writes on the GPU, into one
//! buffer per view and arena, so the raster pass draws a whole arena with a
//! single `multi_draw_indexed_indirect`.
//!
//! Buffers are per view: `write_buffer`s all land before the frame's command
//! buffers run, so views sharing one buffer would all see the last write.

use std::borrow::Cow;

use bevy::{
    asset::uuid_handle,
    camera::primitives::{Aabb, Frustum},
    ecs::{entity::EntityHashMap, system::SystemParam},
    prelude::*,
    render::{
        camera::ExtractedCamera,
        render_resource::{
            BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
            BindGroupLayoutEntry, Buffer, BufferUsages, CachedComputePipelineId, CommandEncoder,
            ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, PipelineCache,
            ShaderStages,
        },
        renderer::{RenderDevice, RenderQueue},
        view::ExtractedView,
    },
};

use crate::{
    chunk::CHUNK_SIZE,
    num::{f32_from_u32, u64_from_usize, usize_from},
    voxel::{
        arena::VoxelChunkArena,
        layer::{LayerIndex, LayerPass, VoxelLayers},
        pipeline::layouts::storage_buffer_entry,
        types::{CreateStorage, DrawIndexedIndirectArgs},
    },
};

pub const DRAW_COMPACTION_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("6f708192-a3b4-c5d6-e7f8-091a2b3c4d5e");

/// Must match `WORKGROUP_SIZE` in `draw_compaction.wesl`.
const WORKGROUP_SIZE: u32 = 64;

const ARGS_BYTES: u64 = u64_from_usize(std::mem::size_of::<DrawIndexedIndirectArgs>());

fn compaction_entries() -> Vec<BindGroupLayoutEntry> {
    vec![
        storage_buffer_entry(0, ShaderStages::COMPUTE, true), // visible slots
        storage_buffer_entry(1, ShaderStages::COMPUTE, true), // per-slot args
        storage_buffer_entry(2, ShaderStages::COMPUTE, false), // packed args
    ]
}

#[derive(Resource)]
pub struct VoxelDrawPipeline {
    pub id: CachedComputePipelineId,
    layout: BindGroupLayout,
}

impl FromWorld for VoxelDrawPipeline {
    fn from_world(world: &mut World) -> Self {
        let layout = world
            .resource::<RenderDevice>()
            .create_bind_group_layout(Some("voxel_draw_compaction_layout"), &compaction_entries());
        let id =
            world
                .resource::<PipelineCache>()
                .queue_compute_pipeline(ComputePipelineDescriptor {
                    label: Some(Cow::Borrowed("voxel_draw_compaction_pipeline")),
                    layout: vec![BindGroupLayoutDescriptor {
                        label: Cow::Borrowed("voxel_draw_compaction_layout"),
                        entries: compaction_entries(),
                    }],
                    shader: DRAW_COMPACTION_SHADER_HANDLE,
                    entry_point: Some(Cow::Borrowed("cs_main")),
                    shader_defs: vec![],
                    constants: vec![],
                    immediate_size: 0,
                    zero_initialize_workgroup_memory: false,
                });
        Self { id, layout }
    }
}

/// Which raster pipeline a group of chunks needs (see
/// `VoxelRasterKey::for_draw`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DrawVariant {
    /// The chunk has a visibility mask: test it per pixel, and discard.
    pub masked: bool,
    /// The chunk is entirely past every material's normal fade
    /// (`Layer::normals_end`): skip normal maps.
    pub far: bool,
}

impl DrawVariant {
    pub const ALL: [Self; 4] = [
        Self {
            masked: false,
            far: false,
        },
        Self {
            masked: false,
            far: true,
        },
        Self {
            masked: true,
            far: false,
        },
        Self {
            masked: true,
            far: true,
        },
    ];

    const fn index(self) -> usize {
        match (self.masked, self.far) {
            (false, false) => 0,
            (false, true) => 1,
            (true, false) => 2,
            (true, true) => 3,
        }
    }
}

/// One arena's visible chunks of one `DrawVariant`, for one view.
pub struct ArenaDraws {
    /// Where the arena is: its layer, then (LOD, index) in that layer's
    /// `VoxelChunkArenaSet::arenas`. Layers and arenas are only ever
    /// appended, so this stays valid.
    pub layer: LayerIndex,
    pub lod: usize,
    pub arena: usize,
    pub variant: DrawVariant,
    /// `[count, slot, slot, ...]`, CPU-written each frame.
    visible: Buffer,
    /// Packed draw arguments of the visible slots, GPU-written.
    pub args: Buffer,
    bind_group: BindGroup,
    pub count: u32,
}

impl ArenaDraws {
    fn new(
        device: &RenderDevice,
        layout: &BindGroupLayout,
        arena: &VoxelChunkArena,
        layer: LayerIndex,
        lod: usize,
        index: usize,
        variant: DrawVariant,
    ) -> Self {
        let max = u64::from(arena.max_chunks);
        let visible = device.storage(
            "voxel_visible_slots_buffer",
            max.saturating_add(1).saturating_mul(4),
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        let args = device.storage(
            "voxel_packed_draw_args_buffer",
            max.saturating_mul(ARGS_BYTES),
            BufferUsages::STORAGE | BufferUsages::INDIRECT,
        );
        let bind_group = device.create_bind_group(
            Some("voxel_draw_compaction_bind_group"),
            layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: visible.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: arena.indirect_args_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: args.as_entire_binding(),
                },
            ],
        );
        Self {
            layer,
            lod,
            arena: index,
            variant,
            visible,
            args,
            bind_group,
            count: 0,
        }
    }
}

/// Visible chunks per camera view.
#[derive(Resource, Default)]
pub struct VoxelViewDraws(EntityHashMap<Vec<ArenaDraws>>);

impl VoxelViewDraws {
    #[must_use]
    pub fn get(&self, view: Entity) -> &[ArenaDraws] {
        self.0.get(&view).map_or(&[], Vec::as_slice)
    }

    /// Draws issued across all views.
    #[must_use]
    pub fn total(&self) -> u32 {
        self.0
            .values()
            .flatten()
            .fold(0, |sum, d| sum.saturating_add(d.count))
    }
}

/// How far a resident chunk's mesh is from `eye`, or `None` if it's out of
/// view. A slot with no known position is kept, at distance 0.
fn chunk_distance(arena: &VoxelChunkArena, slot: u32, frustum: &Frustum, eye: Vec3) -> Option<f32> {
    let Some(Some(pos)) = arena.chunk_pos_of_slot.get(usize_from(slot)) else {
        return Some(0.0);
    };
    // Surface nets puts vertices up to one voxel past the chunk's far faces.
    let voxel = CHUNK_SIZE / f32_from_u32(arena.cell_count.saturating_sub(1).max(1));
    let min = pos.as_vec3() * CHUNK_SIZE;
    let max = min + Vec3::splat(CHUNK_SIZE + voxel);
    frustum
        .intersects_obb_identity(&Aabb::from_min_max(min, max))
        .then(|| eye.clamp(min, max).distance(eye))
}

/// What `prepare_voxel_view_draws` reads.
#[derive(SystemParam)]
pub struct DrawInputs<'w> {
    layers: Res<'w, VoxelLayers>,
    pipeline: Res<'w, VoxelDrawPipeline>,
    device: Res<'w, RenderDevice>,
    queue: Res<'w, RenderQueue>,
}

/// Culls every arena against every camera's frustum, sorts the visible
/// chunks front to back (back to front for a transparent layer) and splits
/// them by `DrawVariant`, and uploads them. Runs after the arenas' uploads
/// for the frame.
pub fn prepare_voxel_view_draws(
    views: Query<(Entity, &Frustum, &ExtractedView), With<ExtractedCamera>>,
    inputs: DrawInputs,
    mut draws: ResMut<VoxelViewDraws>,
    mut visible: Local<[Vec<u32>; 4]>,
    mut by_distance: Local<Vec<(f32, u32)>>,
) {
    let DrawInputs {
        layers,
        pipeline,
        device,
        queue,
    } = inputs;
    draws.0.retain(|view, _| views.contains(*view));
    for (view, frustum, extracted) in &views {
        let eye = extracted.world_from_view.translation();
        let per_view = draws.0.entry(view).or_default();
        let arenas = layers.iter().flat_map(|(layer, l)| {
            l.set
                .arenas
                .iter()
                .enumerate()
                .flat_map(move |(lod, arenas)| {
                    arenas
                        .iter()
                        .enumerate()
                        .map(move |(index, arena)| (layer, l, lod, index, arena))
                })
        });
        for (layer, l, lod, index, arena) in arenas {
            {
                // Front to back, so nearer terrain can hide what's behind
                // it before that is shaded; a transparent layer the other
                // way, so what's behind shows through what's in front.
                by_distance.clear();
                by_distance.extend(arena.active_slots.iter().filter_map(|&slot| {
                    chunk_distance(arena, slot, frustum, eye).map(|d| (d, slot))
                }));
                by_distance.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
                if l.raster.pass == LayerPass::Transparent {
                    by_distance.reverse();
                }
                let normals_end = l.normals_end();
                for list in &mut *visible {
                    list.clear();
                    list.push(0); // count, filled in below
                }
                for &(distance, slot) in &*by_distance {
                    let variant = DrawVariant {
                        masked: arena
                            .mask_slots
                            .get(usize_from(slot))
                            .copied()
                            .unwrap_or(true),
                        far: distance >= normals_end,
                    };
                    if let Some(list) = visible.get_mut(variant.index()) {
                        list.push(slot);
                    }
                }

                for (variant, list) in DrawVariant::ALL.into_iter().zip(visible.iter_mut()) {
                    let count = u32::try_from(list.len().saturating_sub(1)).unwrap_or(u32::MAX);
                    if let Some(first) = list.first_mut() {
                        *first = count;
                    }
                    let is_this = |d: &ArenaDraws| {
                        d.layer == layer && d.lod == lod && d.arena == index && d.variant == variant
                    };
                    let at = per_view.iter().position(is_this);
                    let at = match at {
                        Some(at) => at,
                        // Nothing to draw yet: don't make buffers for it.
                        None if count == 0 => continue,
                        None => {
                            per_view.push(ArenaDraws::new(
                                &device,
                                &pipeline.layout,
                                arena,
                                layer,
                                lod,
                                index,
                                variant,
                            ));
                            // Draw order: by layer; within one, chunks
                            // without a mask first (they fill the early
                            // depth buffer), then the finest LOD and nearer
                            // chunks first.
                            per_view.sort_by_key(|d| {
                                (
                                    d.layer,
                                    d.variant.masked,
                                    std::cmp::Reverse(d.lod),
                                    d.variant.far,
                                    d.arena,
                                )
                            });
                            per_view.iter().position(is_this).unwrap_or_default()
                        }
                    };
                    if let Some(d) = per_view.get_mut(at) {
                        if count > 0 {
                            queue.write_buffer(&d.visible, 0, bytemuck::cast_slice(list));
                        }
                        d.count = count;
                    }
                }
            }
        }
    }
}

/// Packs one view's visible draw arguments. Record it before the view's
/// render pass, in the same encoder.
pub fn encode_draw_compaction<'a>(
    encoder: &mut CommandEncoder,
    pipeline: &ComputePipeline,
    draws: impl Iterator<Item = &'a ArenaDraws>,
) {
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("voxel_draw_compaction"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    for d in draws.filter(|d| d.count > 0) {
        pass.set_bind_group(0, &d.bind_group, &[]);
        pass.dispatch_workgroups(d.count.div_ceil(WORKGROUP_SIZE), 1, 1);
    }
}
