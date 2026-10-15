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
    ecs::entity::EntityHashMap,
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
    },
};

use crate::{
    chunk::CHUNK_SIZE,
    num::{f32_from_u32, u64_from_usize, usize_from},
    voxel::{
        arena::{VoxelChunkArena, VoxelChunkArenaSet},
        pipeline::layouts::storage_buffer_entry,
        types::{CreateStorage, DrawIndexedIndirectArgs},
    },
};

pub const DRAW_COMPACTION_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("6f708192-a3b4-c5d6-e7f8-091a2b3c4d5e");

/// Must match `WORKGROUP_SIZE` in `draw_compaction.wgsl`.
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

/// One arena's visible chunks for one view.
pub struct ArenaDraws {
    /// Where the arena is in `VoxelChunkArenaSet::arenas` (LOD, index).
    /// Arenas are only ever appended, so this stays valid.
    pub lod: usize,
    pub arena: usize,
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
        lod: usize,
        index: usize,
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
            lod,
            arena: index,
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

/// Whether a resident chunk's mesh can be in view. A slot with no known
/// position is kept.
fn chunk_visible(arena: &VoxelChunkArena, slot: u32, frustum: &Frustum) -> bool {
    let Some(Some(pos)) = arena.chunk_pos_of_slot.get(usize_from(slot)) else {
        return true;
    };
    // Surface nets puts vertices up to one voxel past the chunk's far faces.
    let voxel = CHUNK_SIZE / f32_from_u32(arena.cell_count.saturating_sub(1).max(1));
    let min = pos.as_vec3() * CHUNK_SIZE;
    let aabb = Aabb::from_min_max(min, min + Vec3::splat(CHUNK_SIZE + voxel));
    frustum.intersects_obb_identity(&aabb)
}

/// Culls every arena against every camera's frustum and uploads the
/// visible slots. Runs after the arenas' uploads for the frame.
pub fn prepare_voxel_view_draws(
    views: Query<(Entity, &Frustum), With<ExtractedCamera>>,
    arena_set: Res<VoxelChunkArenaSet>,
    pipeline: Res<VoxelDrawPipeline>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut draws: ResMut<VoxelViewDraws>,
    mut visible: Local<Vec<u32>>,
) {
    draws.0.retain(|view, _| views.contains(*view));
    for (view, frustum) in &views {
        let per_view = draws.0.entry(view).or_default();
        for (lod, arenas) in arena_set.arenas.iter().enumerate() {
            for (index, arena) in arenas.iter().enumerate() {
                visible.clear();
                visible.push(0); // count, filled in below
                visible.extend(
                    arena
                        .active_slots
                        .iter()
                        .copied()
                        .filter(|&slot| chunk_visible(arena, slot, frustum)),
                );
                let count = u32::try_from(visible.len().saturating_sub(1)).unwrap_or(u32::MAX);
                if let Some(first) = visible.first_mut() {
                    *first = count;
                }

                let at = per_view
                    .iter()
                    .position(|d| d.lod == lod && d.arena == index)
                    .unwrap_or_else(|| {
                        per_view.push(ArenaDraws::new(
                            &device,
                            &pipeline.layout,
                            arena,
                            lod,
                            index,
                        ));
                        per_view.len().saturating_sub(1)
                    });
                if let Some(d) = per_view.get_mut(at) {
                    queue.write_buffer(&d.visible, 0, bytemuck::cast_slice(&visible));
                    d.count = count;
                }
            }
        }
    }
}

/// Packs one view's visible draw arguments. Record it before the view's
/// render pass, in the same encoder.
pub fn encode_draw_compaction(
    encoder: &mut CommandEncoder,
    pipeline: &ComputePipeline,
    draws: &[ArenaDraws],
) {
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("voxel_draw_compaction"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    for d in draws.iter().filter(|d| d.count > 0) {
        pass.set_bind_group(0, &d.bind_group, &[]);
        pass.dispatch_workgroups(d.count.div_ceil(WORKGROUP_SIZE), 1, 1);
    }
}
