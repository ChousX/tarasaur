use bevy::{
    prelude::*,
    render::{
        Extract,
        render_resource::*,
        renderer::{RenderDevice, RenderQueue},
    },
};
use bytemuck::{Pod, Zeroable};

use crate::{
    CHUNK_SIZE, LOD,
    voxel::{
        arena::VoxelChunkArenaSet,
        pipeline::{VoxelComputePipeline, VoxelPipelineLayouts},
        types::{HitResult, RayQuery},
    },
};

pub const MAX_QUERIES_PER_FRAME: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct QueryUniforms {
    pub query_count: u32,
    pub chunk_size: f32,
    pub voxel_size: f32,
    pub texture_size: u32,
    pub lookup_capacity: u32,
    pub max_dda_steps: u32,
    pub max_sphere_steps: u32,
    pub surface_epsilon: f32,
}

/// Which LOD's arena the query shader runs against. Only one arena is
/// tested per query today — see the multi-LOD caveat from the design
/// writeup if you need queries to hit lower-detail terrain too.
#[derive(Resource)]
pub struct CollisionLOD(pub LOD);

impl Default for CollisionLOD {
    fn default() -> Self {
        Self(LOD::High)
    }
}

/// Main-world: gameplay code pushes ray queries here each frame.
#[derive(Resource, Default)]
pub struct PendingVoxelQueries(pub Vec<RayQuery>);

/// Main-world: latest delivered batch of results, keyed positionally to
/// whatever was in `PendingVoxelQueries` ~2 frames ago (1 GPU frame +
/// channel delivery). Match on `RayQuery::user_id` if you need to
/// correlate results back to specific requests across drops.
#[derive(Resource, Default)]
pub struct VoxelQueryResults(pub Vec<HitResult>);

/// Render-world mirror of `PendingVoxelQueries`, capped to
/// `MAX_QUERIES_PER_FRAME`.
#[derive(Resource, Default)]
pub struct ExtractedVoxelQueries(pub Vec<RayQuery>);

/// 0 or 1, flips every render frame. Frame N dispatches into
/// `hit_buffers[parity]`; the *other* readback buffer, copied a full
/// frame earlier, is safe to `map_async` this frame with no stall.
#[derive(Resource)]
pub struct FrameParity(pub usize);

impl Default for FrameParity {
    fn default() -> Self {
        Self(0)
    }
}

/// Render-world sender half. Mirrors `MeshReadbackChannel`'s pattern.
#[derive(Resource)]
pub struct VoxelQueryResultChannel {
    pub sender: crossbeam_channel::Sender<Vec<HitResult>>,
}

/// Main-world receiver half.
#[derive(Resource)]
pub struct VoxelQueryResultReceiver {
    pub receiver: crossbeam_channel::Receiver<Vec<HitResult>>,
}

/// Static buffers + double-buffered bind groups for the cursor/raycast
/// query pass. Buffers themselves don't depend on any arena and are built
/// eagerly in `VoxelRenderPlugin::finish()`; the bind groups DO reference
/// the collision arena's `sdf_buffer`/`chunk_meta_buffer`/
/// `chunk_lookup_buffer`, which don't exist yet at `finish()` time (arenas
/// are created lazily on first chunk extraction) — so bind group creation
/// is deferred to `prepare_voxel_queries`, the first frame the collision
/// arena shows up, and cached forever after since arena buffers never
/// change identity once created.
#[derive(Resource)]
pub struct VoxelQueryBuffers {
    pub bind_groups: [Option<BindGroup>; 2],
    pub query_buffer: Buffer,
    pub hit_buffers: [Buffer; 2],
    pub readback_buffers: [Buffer; 2],
    pub query_uniform_buffer: Buffer,
}

impl FromWorld for VoxelQueryBuffers {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let query_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("voxel_query_buffer"),
            size: std::mem::size_of::<RayQuery>() as u64 * MAX_QUERIES_PER_FRAME as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let hit_buffer_size =
            std::mem::size_of::<HitResult>() as u64 * MAX_QUERIES_PER_FRAME as u64;

        let make_hit_buffer = |label: &str| {
            render_device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: hit_buffer_size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let make_readback_buffer = |label: &str| {
            render_device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: hit_buffer_size,
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        };

        let hit_buffers = [
            make_hit_buffer("voxel_hit_buffer_a"),
            make_hit_buffer("voxel_hit_buffer_b"),
        ];
        let readback_buffers = [
            make_readback_buffer("voxel_hit_readback_a"),
            make_readback_buffer("voxel_hit_readback_b"),
        ];

        let query_uniform_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("voxel_query_uniform_buffer"),
            size: std::mem::size_of::<QueryUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            bind_groups: [None, None],
            query_buffer,
            hit_buffers,
            readback_buffers,
            query_uniform_buffer,
        }
    }
}

// --- Main-world systems ---

/// Runs in `First`, before gameplay code appends this frame's requests —
/// clears out last frame's already-extracted queries so callers don't
/// have to manage the buffer's lifetime themselves.
pub fn clear_pending_voxel_queries(mut pending: ResMut<PendingVoxelQueries>) {
    pending.0.clear();
}

/// Runs in `First`. Non-blocking drain of whatever `map_voxel_query_results`
/// has sent since last frame. At most one batch is expected per frame under
/// normal timing; if several are queued (e.g. after a hitch) the latest wins.
pub fn drain_voxel_query_results(
    receiver: Res<VoxelQueryResultReceiver>,
    mut results: ResMut<VoxelQueryResults>,
) {
    while let Ok(batch) = receiver.receiver.try_recv() {
        results.0 = batch;
    }
}

// --- Render-world systems ---

pub fn extract_voxel_queries(
    mut extracted: ResMut<ExtractedVoxelQueries>,
    pending: Extract<Res<PendingVoxelQueries>>,
) {
    extracted.0.clear();
    extracted.0.extend(
        pending
            .0
            .iter()
            .copied()
            .take(MAX_QUERIES_PER_FRAME as usize),
    );
}

/// Runs in `RenderSystems::Prepare`, after `prepare_visibility_for_arena`
/// (so the collision arena's `slot_of_chunk_pos` reflects this frame's
/// allocations before the lookup table is rebuilt from it).
pub fn prepare_voxel_queries(
    render_queue: Res<RenderQueue>,
    render_device: Res<RenderDevice>,
    extracted: Res<ExtractedVoxelQueries>,
    arena_set: Res<VoxelChunkArenaSet>,
    collision_lod: Res<CollisionLOD>,
    layouts: Res<VoxelPipelineLayouts>,
    mut query_buffers: ResMut<VoxelQueryBuffers>,
) {
    let Some(arena) = arena_set.arenas.get(&collision_lod.0) else {
        return; // collision arena doesn't exist yet — nothing to query against
    };

    arena.rebuild_chunk_lookup(&render_queue);

    if !extracted.0.is_empty() {
        render_queue.write_buffer(
            &query_buffers.query_buffer,
            0,
            bytemuck::cast_slice(&extracted.0),
        );
    }

    let uniforms = QueryUniforms {
        query_count: extracted.0.len() as u32,
        chunk_size: CHUNK_SIZE,
        voxel_size: CHUNK_SIZE / (arena.texture_size - 2) as f32,
        texture_size: arena.texture_size,
        lookup_capacity: arena.lookup_capacity,
        max_dda_steps: 64,
        max_sphere_steps: 48,
        surface_epsilon: 0.01,
    };
    render_queue.write_buffer(
        &query_buffers.query_uniform_buffer,
        0,
        bytemuck::bytes_of(&uniforms),
    );

    if query_buffers.bind_groups[0].is_none() {
        for p in 0..2 {
            let bind_group = render_device.create_bind_group(
                Some("voxel_query_bind_group"),
                &layouts.query_layout,
                &[
                    BindGroupEntry {
                        binding: 0,
                        resource: query_buffers.query_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: query_buffers.hit_buffers[p].as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: query_buffers.query_uniform_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: arena.chunk_lookup_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 4,
                        resource: arena.sdf_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 5,
                        resource: arena.chunk_meta_buffer.as_entire_binding(),
                    },
                ],
            );
            query_buffers.bind_groups[p] = Some(bind_group);
        }
    }
}

/// Runs in `RenderSystems::Queue`, after `dispatch_voxel_compute_passes_batched`
/// so this frame's `sdf_buffer`/`chunk_meta_buffer` writes are already queued
/// ahead of the query dispatch on the same command stream.
pub fn dispatch_voxel_query_pass(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    compute_pipeline: Res<VoxelComputePipeline>,
    query_buffers: Res<VoxelQueryBuffers>,
    extracted: Res<ExtractedVoxelQueries>,
    mut parity: ResMut<FrameParity>,
) {
    if extracted.0.is_empty() {
        return;
    }
    let (Some(bg0), Some(bg1)) = (&query_buffers.bind_groups[0], &query_buffers.bind_groups[1])
    else {
        return; // collision arena not ready yet this frame
    };
    let Some(pipeline) = pipeline_cache.get_compute_pipeline(compute_pipeline.query_pipeline_id)
    else {
        return;
    };

    let p = parity.0;
    let bind_group = if p == 0 { bg0 } else { bg1 };

    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("voxel_query_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("voxel_cursor_query"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(extracted.0.len() as u32, 1, 1);
    }
    encoder.copy_buffer_to_buffer(
        &query_buffers.hit_buffers[p],
        0,
        &query_buffers.readback_buffers[p],
        0,
        query_buffers.readback_buffers[p].size(),
    );
    render_queue.submit(std::iter::once(encoder.finish()));

    parity.0 = 1 - p;
}

/// Runs in `RenderSystems::Cleanup`. Maps the buffer NOT written this
/// frame — the one copied into a readback buffer a full frame ago — so
/// `map_async` resolves without stalling the render thread.
pub fn map_voxel_query_results(
    query_buffers: Res<VoxelQueryBuffers>,
    parity: Res<FrameParity>,
    channel: Res<VoxelQueryResultChannel>,
) {
    let stale = 1 - parity.0;
    let buf = query_buffers.readback_buffers[stale].clone();
    let sender = channel.sender.clone();
    let buf2 = buf.clone();
    buf2.slice(..).map_async(MapMode::Read, move |result| {
        if result.is_err() {
            return;
        }
        let out: Vec<HitResult> = {
            let data = buf.slice(..).get_mapped_range();
            bytemuck::cast_slice(&data).to_vec()
        };
        buf.unmap();
        let _ = sender.send(out);
    });
}

/// Multiplicative hash over a signed chunk coordinate, used both to build
/// and to probe the GPU chunk-lookup table — must match `hash_chunk` in
/// cursor_query.wgsl exactly.
pub fn hash_chunk_pos(pos: IVec3) -> u32 {
    let x = pos.x as u32;
    let y = pos.y as u32;
    let z = pos.z as u32;
    (x.wrapping_mul(0x9E37_79B1)) ^ (y.wrapping_mul(0x85EB_CA77)) ^ (z.wrapping_mul(0xC2B2_AE3D))
}
