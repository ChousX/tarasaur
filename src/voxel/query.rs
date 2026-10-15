//! GPU ray queries against the arenas (`cursor_query.wgsl`).
//!
//! Push `RayQuery`s into `PendingVoxelQueries`; results land in
//! `VoxelQueryResults` 2-3 frames later via a triple-buffered readback and
//! a channel back to the main world. For a same-frame answer use the CPU
//! `TerrainRay` instead.

use std::sync::{
    Arc,
    atomic::{AtomicU8, AtomicU32, Ordering},
};

use bevy::{
    prelude::*,
    render::{
        Extract,
        diagnostic::{DiagnosticsRecorder, RecordDiagnostics},
        render_resource::{
            BindGroup, Buffer, BufferDescriptor, BufferInitDescriptor, BufferUsages,
            CommandEncoderDescriptor, ComputePassDescriptor, MapMode, PipelineCache,
        },
        renderer::{RenderDevice, RenderQueue},
    },
};
use bytemuck::{Pod, Zeroable};

use crate::{
    chunk::CHUNK_SIZE,
    field::LOD,
    num::{f32_from_u32, u64_from_usize, usize_from},
    voxel::{
        arena::VoxelChunkArenaSet,
        pipeline::{VoxelComputePipeline, VoxelPipelineLayouts},
        types::{CreateStorage, HitResult, RayQuery, entries},
    },
};

/// Ray queries per frame. Sizes the fixed query/hit/readback buffers;
/// extraction drops anything past it, since writing more would overflow them.
pub const QUERY_CAPACITY: u32 = 64;
pub const QUERY_BUFFER_SLOTS: usize = 3;

const SLOT_FREE: u8 = 0;
const SLOT_COPIED: u8 = 1; // GPU copy submitted, ready to map
const SLOT_MAPPED: u8 = 2; // map_async in flight, do not touch buffer

/// Round-robin index into the `QUERY_BUFFER_SLOTS` pool.
///
/// Replaces the old 2-buffer `FrameParity` scheme, which reused a buffer every 2 frames
/// regardless of whether its previous `map_async` callback had actually
/// fired — on slower drivers that raced against `Queue::submit` and
/// produced "buffer still mapped" validation errors. With N=3 slots and
/// per-slot state tracking, a slot whose map hasn't completed is simply
/// skipped rather than reused.
#[derive(Resource, Default)]
pub struct NextQuerySlot(pub usize);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct QueryUniforms {
    pub query_count: u32,
    pub chunk_size: f32,
    pub lookup_capacity: u32,
    pub max_dda_steps: u32,
    pub max_sphere_steps: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    pub lod_voxel_size: [f32; 4],
    pub lod_texture_size: [u32; 4],
    pub lod_surface_epsilon: [f32; 4],
}
/// Main-world: gameplay code pushes ray queries here each frame.
#[derive(Resource, Default)]
pub struct PendingVoxelQueries(pub Vec<RayQuery>);

/// Main-world: latest delivered batch of results.
///
/// Keyed positionally to whatever was in `PendingVoxelQueries` ~2 frames ago (1 GPU frame +
/// channel delivery). Match on `RayQuery::user_id` if you need to
/// correlate results back to specific requests across drops.
#[derive(Resource, Default)]
pub struct VoxelQueryResults(pub Vec<HitResult>);

/// Render-world mirror of `PendingVoxelQueries`, capped to
/// `QUERY_CAPACITY`.
#[derive(Resource, Default)]
pub struct ExtractedVoxelQueries(pub Vec<RayQuery>);

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

/// Static buffers + triple-buffered bind groups for the cursor/raycast
/// query pass.
///
/// The query runs once per arena index: pass `j` binds arena `j` of every
/// LOD (a dummy where a LOD has fewer) and traces only the chunks stored
/// there; a later pass keeps its hit only if it is closer. The bind groups
/// reference the arenas' buffers, which are created lazily, so they're built
/// in `prepare_voxel_queries` and rebuilt only when a LOD gains an arena or
/// the lookup table grows (arena buffers never change identity).
#[derive(Resource)]
pub struct VoxelQueryBuffers {
    /// Per readback slot: one bind group per pass (arena index).
    pub bind_groups: [Vec<BindGroup>; QUERY_BUFFER_SLOTS],
    /// Per pass: a uniform holding that pass's arena index.
    pub pass_uniforms: Vec<Buffer>,
    pub query_buffer: Buffer,
    pub hit_buffers: [Buffer; QUERY_BUFFER_SLOTS],
    pub readback_buffers: [Buffer; QUERY_BUFFER_SLOTS],
    pub query_uniform_buffer: Buffer,
    pub slot_state: [Arc<AtomicU8>; QUERY_BUFFER_SLOTS],
    /// Queries dispatched into each slot; its readback holds that many
    /// results, the rest of the buffer is left over from older dispatches.
    pub slot_query_count: [AtomicU32; QUERY_BUFFER_SLOTS],
    pub active_count: u32,
    pub merged_lookup_buffer: Buffer,
    pub merged_lookup_capacity: u32,
    pub dummy_sdf_buffer: Buffer,
    pub dummy_meta_buffer: Buffer,
    /// Arenas per LOD the bind groups were built for.
    pub bound_arena_counts: [usize; LOD::COUNT],
}

fn slot_buffers(
    render_device: &RenderDevice,
    labels: [&'static str; QUERY_BUFFER_SLOTS],
    capacity: u32,
    usage: BufferUsages,
) -> [Buffer; QUERY_BUFFER_SLOTS] {
    let size = u64_from_usize(std::mem::size_of::<HitResult>()).saturating_mul(u64::from(capacity));
    labels.map(|label| render_device.storage(label, size, usage))
}

impl FromWorld for VoxelQueryBuffers {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let capacity = QUERY_CAPACITY;

        let query_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("voxel_query_buffer"),
            size: u64_from_usize(std::mem::size_of::<RayQuery>())
                .saturating_mul(u64::from(capacity)),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let hit_buffers = slot_buffers(
            render_device,
            [
                "voxel_hit_buffer_a",
                "voxel_hit_buffer_b",
                "voxel_hit_buffer_c",
            ],
            capacity,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );
        let readback_buffers = slot_buffers(
            render_device,
            [
                "voxel_hit_readback_a",
                "voxel_hit_readback_b",
                "voxel_hit_readback_c",
            ],
            capacity,
            BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        );

        let query_uniform_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("voxel_query_uniform_buffer"),
            size: u64_from_usize(std::mem::size_of::<QueryUniforms>()),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let initial_lookup_capacity = 64u32;
        let empty_lookup: Vec<[u32; 4]> = vec![[u32::MAX; 4]; usize_from(initial_lookup_capacity)];
        let merged_lookup_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("voxel_merged_chunk_lookup_buffer"),
            contents: bytemuck::cast_slice(&empty_lookup),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        // Fallback buffers bound wherever a LOD rank has no live arena yet
        // — a bind group can't reference a missing resource, so every
        // rank always gets *some* valid binding, real or dummy.
        let dummy_sdf_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("voxel_query_dummy_sdf_buffer"),
            contents: bytemuck::bytes_of(&0.0f32),
            usage: BufferUsages::STORAGE,
        });
        let dummy_meta_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("voxel_query_dummy_meta_buffer"),
            contents: bytemuck::bytes_of(&super::arena::ChunkMeta::default()),
            usage: BufferUsages::STORAGE,
        });

        Self {
            bind_groups: Default::default(),
            pass_uniforms: Vec::new(),
            query_buffer,
            hit_buffers,
            readback_buffers,
            query_uniform_buffer,
            slot_state: std::array::from_fn(|_| Arc::new(AtomicU8::new(SLOT_FREE))),
            slot_query_count: std::array::from_fn(|_| AtomicU32::new(0)),
            active_count: 0,
            merged_lookup_buffer,
            merged_lookup_capacity: initial_lookup_capacity,
            dummy_sdf_buffer,
            dummy_meta_buffer,
            bound_arena_counts: [0; LOD::COUNT],
        }
    }
}

impl VoxelQueryBuffers {
    /// Rebuilds the bind groups if never built, or a LOD gained an arena
    /// (its buffers replace a dummy, or it needs another pass).
    fn update_bind_groups(
        &mut self,
        device: &RenderDevice,
        layouts: &VoxelPipelineLayouts,
        arena_set: &VoxelChunkArenaSet,
    ) {
        let counts = arena_set.arenas.each_ref().map(Vec::len);
        if !self.bind_groups[0].is_empty() && self.bound_arena_counts == counts {
            return;
        }
        let passes = counts.iter().copied().max().unwrap_or(0).max(1);
        while self.pass_uniforms.len() < passes {
            let pass = u32::try_from(self.pass_uniforms.len()).unwrap_or(u32::MAX);
            self.pass_uniforms
                .push(device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("voxel_query_pass_uniform"),
                    contents: bytemuck::cast_slice(&[pass, 0, 0, 0]),
                    usage: BufferUsages::UNIFORM,
                }));
        }
        let qb = &*self;
        let bind_groups = qb.hit_buffers.each_ref().map(|hit| {
            qb.pass_uniforms
                .iter()
                .take(passes)
                .enumerate()
                .map(|(pass, pass_uniform)| {
                    let mut bufs = vec![
                        (0, &qb.query_buffer),
                        (1, hit),
                        (2, &qb.query_uniform_buffer),
                        (3, &qb.merged_lookup_buffer),
                        (12, pass_uniform),
                    ];
                    // Bindings 4, 5 for rank 0; 6, 7 for rank 1; ...
                    for (base, arenas) in (4u32..).step_by(2).zip(&arena_set.arenas) {
                        let (sdf, meta) = arenas
                            .get(pass)
                            .map_or((&qb.dummy_sdf_buffer, &qb.dummy_meta_buffer), |a| {
                                (&a.sdf_buffer, &a.chunk_meta_buffer)
                            });
                        bufs.extend([(base, sdf), (base.wrapping_add(1), meta)]);
                    }
                    device.create_bind_group(
                        Some("voxel_query_bind_group"),
                        &layouts.query,
                        &entries(&bufs),
                    )
                })
                .collect()
        });
        self.bind_groups = bind_groups;
        self.bound_arena_counts = counts;
    }
}

// --- Main-world systems ---

/// Runs in `First`, before gameplay code appends this frame's requests —
/// clears out last frame's already-extracted queries so callers don't
/// have to manage the buffer's lifetime themselves.
pub fn clear_pending_voxel_queries(mut pending: ResMut<PendingVoxelQueries>) {
    pending.0.clear();
}

/// Runs in `First`.
///
/// Non-blocking drain of whatever `map_voxel_query_results`
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

pub fn extract_voxel_queries(
    mut extracted: ResMut<ExtractedVoxelQueries>,
    pending: Extract<Res<PendingVoxelQueries>>,
) {
    extracted.0.clear();
    if pending.0.len() > usize_from(QUERY_CAPACITY) {
        warn!(
            "[extract_voxel_queries] {} queries requested this frame, capping to QUERY_CAPACITY={}",
            pending.0.len(),
            QUERY_CAPACITY
        );
    }
    extracted
        .0
        .extend(pending.0.iter().copied().take(usize_from(QUERY_CAPACITY)));
}

/// Runs in `RenderSystems::Prepare`, after `prepare_visibility_for_arena`
/// (so the collision arena's `slot_of_chunk_pos` reflects this frame's
/// allocations before the lookup table is rebuilt from it).
pub fn prepare_voxel_queries(
    render_queue: Res<RenderQueue>,
    render_device: Res<RenderDevice>,
    extracted: Res<ExtractedVoxelQueries>,
    arena_set: Res<VoxelChunkArenaSet>,
    layouts: Res<VoxelPipelineLayouts>,
    mut query_buffers: ResMut<VoxelQueryBuffers>,
) {
    let this_frame_count = u32::try_from(extracted.0.len()).unwrap_or(u32::MAX);
    query_buffers.active_count = this_frame_count;

    if this_frame_count == 0 {
        return;
    }

    render_queue.write_buffer(
        &query_buffers.query_buffer,
        0,
        bytemuck::cast_slice(&extracted.0),
    );
    // Grow-only merged lookup, sized from every live arena's max_chunks.
    let needed = arena_set
        .total_max_chunks()
        .saturating_mul(2)
        .next_power_of_two()
        .max(4);
    if needed > query_buffers.merged_lookup_capacity {
        let empty: Vec<[u32; 4]> = vec![[u32::MAX; 4]; usize_from(needed)];
        query_buffers.merged_lookup_buffer =
            render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("voxel_merged_chunk_lookup_buffer"),
                contents: bytemuck::cast_slice(&empty),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            });
        query_buffers.merged_lookup_capacity = needed;
        query_buffers.bind_groups = Default::default(); // buffer identity changed
    }
    arena_set.rebuild_merged_lookup(
        &render_queue,
        &query_buffers.merged_lookup_buffer,
        query_buffers.merged_lookup_capacity,
    );

    let mut lod_voxel_size = [0.0f32; 4];
    let mut lod_texture_size = [0u32; 4];
    let mut lod_surface_epsilon = [0.0f32; 4];

    for (r, arena) in arena_set.query_arenas() {
        let voxel_size = CHUNK_SIZE / f32_from_u32(arena.texture_size.saturating_sub(2));
        let (Some(size), Some(texture), Some(epsilon)) = (
            lod_voxel_size.get_mut(r),
            lod_texture_size.get_mut(r),
            lod_surface_epsilon.get_mut(r),
        ) else {
            continue;
        };
        *size = voxel_size;
        *texture = arena.texture_size;
        *epsilon = voxel_size * 0.05;
    }

    let uniforms = QueryUniforms {
        query_count: this_frame_count,
        chunk_size: CHUNK_SIZE,
        lookup_capacity: query_buffers.merged_lookup_capacity,
        max_dda_steps: 64,
        max_sphere_steps: 256,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
        lod_voxel_size,
        lod_texture_size,
        lod_surface_epsilon,
    };
    render_queue.write_buffer(
        &query_buffers.query_uniform_buffer,
        0,
        bytemuck::bytes_of(&uniforms),
    );

    query_buffers.update_bind_groups(&render_device, &layouts, &arena_set);
}

/// Runs in `VoxelPrepareSet::Dispatch`, after the meshing dispatch.
///
/// So this frame's `sdf_buffer`/`chunk_meta_buffer` writes are already
/// queued ahead of the query dispatch on the same command stream.
pub fn dispatch_voxel_query_pass(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    compute_pipeline: Res<VoxelComputePipeline>,
    query_buffers: Res<VoxelQueryBuffers>,
    mut next_slot: ResMut<NextQuerySlot>,
    recorder: Option<Res<DiagnosticsRecorder>>,
) {
    let count = query_buffers.active_count;
    if count == 0 {
        return;
    }
    let Some(pipeline) = pipeline_cache.get_compute_pipeline(compute_pipeline.query) else {
        return;
    };

    let slot = next_slot.0;
    next_slot.0 = slot.wrapping_add(1) % QUERY_BUFFER_SLOTS;

    let qb = &*query_buffers;
    let (Some(state), Some(slot_count), Some(bind_groups), Some(hit), Some(readback)) = (
        qb.slot_state.get(slot),
        qb.slot_query_count.get(slot),
        qb.bind_groups.get(slot),
        qb.hit_buffers.get(slot),
        qb.readback_buffers.get(slot),
    ) else {
        return;
    };
    if bind_groups.is_empty() || state.load(Ordering::Acquire) != SLOT_FREE {
        return;
    }

    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("voxel_query_encoder"),
    });
    let recorder = recorder.as_deref();
    let span = recorder.time_span(&mut encoder, "voxel_query");
    // One pass per arena index, in order: pass 0 writes every result, later
    // passes only a closer hit.
    for bind_group in bind_groups {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("voxel_cursor_query"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(count, 1, 1);
    }
    span.end(&mut encoder);
    encoder.copy_buffer_to_buffer(hit, 0, readback, 0, hit.size());
    render_queue.submit(std::iter::once(encoder.finish()));

    slot_count.store(count, Ordering::Relaxed);
    state.store(SLOT_COPIED, Ordering::Release);
}

/// Runs in `RenderSystems::Cleanup`.
///
/// Maps every slot currently in the
/// Copied state — i.e. whose GPU copy was submitted (in some earlier
/// frame, or possibly this one) and hasn't been picked up for mapping
/// yet. `compare_exchange` guards against a slot being mapped twice if this
/// system somehow ran with a state already claimed.
pub fn map_voxel_query_results(
    query_buffers: Res<VoxelQueryBuffers>,
    channel: Res<VoxelQueryResultChannel>,
) {
    let slots = query_buffers
        .slot_state
        .iter()
        .zip(&query_buffers.slot_query_count)
        .zip(&query_buffers.readback_buffers);
    for ((state, slot_count), buf) in slots {
        let state = state.clone();
        if state
            .compare_exchange(
                SLOT_COPIED,
                SLOT_MAPPED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            continue;
        }
        // Set before the slot was marked copied; the Acquire above sees it.
        let count = usize_from(slot_count.load(Ordering::Relaxed));

        let buf = buf.clone();
        let sender = channel.sender.clone();
        let buf2 = buf.clone();
        buf2.slice(..).map_async(MapMode::Read, move |result| {
            if result.is_err() {
                // Release the slot even on failure — otherwise a single
                // failed map permanently strands this buffer slot.
                state.store(SLOT_FREE, Ordering::Release);
                return;
            }
            let out: Vec<HitResult> = {
                let Ok(data) = buf.slice(..).get_mapped_range() else {
                    buf.unmap();
                    state.store(SLOT_FREE, Ordering::Release);
                    return;
                };
                let hits: &[HitResult] = bytemuck::cast_slice(&data);
                hits.iter().take(count).copied().collect()
            };
            buf.unmap();
            let _ = sender.send(out);
            state.store(SLOT_FREE, Ordering::Release);
        });
    }
}
