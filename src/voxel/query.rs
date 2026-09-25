use std::sync::{
    Arc,
    atomic::{AtomicU8, Ordering},
};

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
        util::next_pow2,
    },
};

pub const MAX_QUERIES_PER_FRAME: u32 = 64;
pub const INITIAL_QUERY_CAPACITY: u32 = 64;
/// Safety ceiling, not a target — growth stops here regardless of how
/// many queries are requested in a single frame, so a runaway caller
/// can't force unbounded GPU buffer allocation.
pub const MAX_QUERY_CAPACITY: u32 = 8192;
pub const QUERY_BUFFER_SLOTS: usize = 3;

const SLOT_FREE: u8 = 0;
const SLOT_COPIED: u8 = 1; // GPU copy submitted, ready to map
const SLOT_MAPPED: u8 = 2; // map_async in flight, do not touch buffer

/// Round-robin index into the QUERY_BUFFER_SLOTS pool. Replaces the old
/// 2-buffer FrameParity scheme, which reused a buffer every 2 frames
/// regardless of whether its previous map_async callback had actually
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
    pub lod_active_mask: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub lod_voxel_size: [f32; 4],
    pub lod_texture_size: [u32; 4],
    pub lod_surface_epsilon: [f32; 4],
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
    pub bind_groups: [Option<BindGroup>; QUERY_BUFFER_SLOTS],
    pub query_buffer: Buffer,
    pub hit_buffers: [Buffer; QUERY_BUFFER_SLOTS],
    pub readback_buffers: [Buffer; QUERY_BUFFER_SLOTS],
    pub query_uniform_buffer: Buffer,
    pub slot_state: [Arc<AtomicU8>; QUERY_BUFFER_SLOTS],
    pub capacity: u32,
    pub pending_capacity: Option<u32>,
    pub active_count: u32,
    pub merged_lookup_buffer: Buffer,
    pub merged_lookup_capacity: u32,
    pub dummy_sdf_buffer: Buffer,
    pub dummy_meta_buffer: Buffer,
    pub bound_lod_presence: [bool; LOD::COUNT],
}

fn make_hit_buffers(render_device: &RenderDevice, capacity: u32) -> [Buffer; QUERY_BUFFER_SLOTS] {
    let size = std::mem::size_of::<HitResult>() as u64 * capacity as u64;
    std::array::from_fn(|i| {
        render_device.create_buffer(&BufferDescriptor {
            label: Some(match i {
                0 => "voxel_hit_buffer_a",
                1 => "voxel_hit_buffer_b",
                _ => "voxel_hit_buffer_c",
            }),
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    })
}

fn make_readback_buffers(
    render_device: &RenderDevice,
    capacity: u32,
) -> [Buffer; QUERY_BUFFER_SLOTS] {
    let size = std::mem::size_of::<HitResult>() as u64 * capacity as u64;
    std::array::from_fn(|i| {
        render_device.create_buffer(&BufferDescriptor {
            label: Some(match i {
                0 => "voxel_hit_readback_a",
                1 => "voxel_hit_readback_b",
                _ => "voxel_hit_readback_c",
            }),
            size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    })
}

impl VoxelQueryBuffers {
    fn all_slots_free(&self) -> bool {
        self.slot_state
            .iter()
            .all(|s| s.load(Ordering::Acquire) == SLOT_FREE)
    }

    /// Reallocates query/hit/readback buffers to `new_capacity` and drops
    /// the cached bind groups so `prepare_voxel_queries` rebuilds them
    /// against the new buffers next call. Caller must have already
    /// confirmed `all_slots_free()` — growing while a slot is
    /// Copied/Mapped would free a buffer a pending GPU copy or
    /// `map_async` callback still references.
    fn grow(&mut self, render_device: &RenderDevice, new_capacity: u32) {
        self.query_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("voxel_query_buffer"),
            size: std::mem::size_of::<RayQuery>() as u64 * new_capacity as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.hit_buffers = make_hit_buffers(render_device, new_capacity);
        self.readback_buffers = make_readback_buffers(render_device, new_capacity);
        self.bind_groups = [None, None, None];
        self.capacity = new_capacity;
    }
}

impl FromWorld for VoxelQueryBuffers {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let capacity = INITIAL_QUERY_CAPACITY;

        let query_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("voxel_query_buffer"),
            size: std::mem::size_of::<RayQuery>() as u64 * capacity as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let hit_buffers = make_hit_buffers(render_device, capacity);
        let readback_buffers = make_readback_buffers(render_device, capacity);

        let query_uniform_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("voxel_query_uniform_buffer"),
            size: std::mem::size_of::<QueryUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let initial_lookup_capacity = 64u32;
        let empty_lookup: Vec<[u32; 4]> = vec![[u32::MAX; 4]; initial_lookup_capacity as usize];
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
            bind_groups: [None, None, None],
            query_buffer,
            hit_buffers,
            readback_buffers,
            query_uniform_buffer,
            slot_state: std::array::from_fn(|_| Arc::new(AtomicU8::new(SLOT_FREE))),
            capacity,
            pending_capacity: None,
            active_count: 0,
            merged_lookup_buffer,
            merged_lookup_capacity: initial_lookup_capacity,
            dummy_sdf_buffer,
            dummy_meta_buffer,
            bound_lod_presence: [false; LOD::COUNT],
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

pub fn extract_voxel_queries(
    mut extracted: ResMut<ExtractedVoxelQueries>,
    pending: Extract<Res<PendingVoxelQueries>>,
) {
    extracted.0.clear();
    if pending.0.len() as u32 > MAX_QUERY_CAPACITY {
        warn!(
            "[extract_voxel_queries] {} queries requested this frame, capping to MAX_QUERY_CAPACITY={}",
            pending.0.len(),
            MAX_QUERY_CAPACITY
        );
    }
    extracted
        .0
        .extend(pending.0.iter().copied().take(MAX_QUERY_CAPACITY as usize));
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
    // 1. Update active_count from extracted queries
    let this_frame_count = extracted.0.len() as u32;
    query_buffers.active_count = this_frame_count;

    if this_frame_count == 0 {
        return;
    }

    // 2. Upload extracted query structs to GPU query_buffer
    render_queue.write_buffer(
        &query_buffers.query_buffer,
        0,
        bytemuck::cast_slice(&extracted.0),
    );
    // Grow-only merged lookup, sized from every live arena's max_chunks —
    // mirrors each arena's own lookup_capacity policy (fixed once known,
    // never shrinks) rather than reacting to frame-to-frame occupancy.
    let needed = next_pow2(arena_set.total_max_chunks().saturating_mul(2)).max(4);
    if needed > query_buffers.merged_lookup_capacity {
        let empty: Vec<[u32; 4]> = vec![[u32::MAX; 4]; needed as usize];
        query_buffers.merged_lookup_buffer =
            render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("voxel_merged_chunk_lookup_buffer"),
                contents: bytemuck::cast_slice(&empty),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            });
        query_buffers.merged_lookup_capacity = needed;
        query_buffers.bind_groups = [None, None, None]; // buffer identity changed
    }
    arena_set.rebuild_merged_lookup(
        &render_queue,
        &query_buffers.merged_lookup_buffer,
        query_buffers.merged_lookup_capacity,
    );

    // ...unchanged: requested/capacity growth logic for query_buffer, then
    // write extracted.0 into query_buffers.query_buffer as before...

    let mut lod_voxel_size = [0.0f32; 4];
    let mut lod_texture_size = [0u32; 4];
    let mut lod_surface_epsilon = [0.0f32; 4];
    let mut lod_active_mask = 0u32;
    let mut current_presence = [false; LOD::COUNT];

    for (&lod, arena) in arena_set.arenas.iter() {
        let r = lod.rank() as usize;
        let voxel_size = CHUNK_SIZE / (arena.texture_size - 2) as f32;
        lod_voxel_size[r] = voxel_size;
        lod_texture_size[r] = arena.texture_size;
        lod_surface_epsilon[r] = voxel_size * 0.05;
        lod_active_mask |= 1 << r;
        current_presence[r] = true;
    }

    let uniforms = QueryUniforms {
        query_count: query_buffers.active_count, // set as before from this_frame_count
        chunk_size: CHUNK_SIZE,
        lookup_capacity: query_buffers.merged_lookup_capacity,
        max_dda_steps: 64,
        max_sphere_steps: 96,
        lod_active_mask,
        _pad0: 0,
        _pad1: 0,
        lod_voxel_size,
        lod_texture_size,
        lod_surface_epsilon,
    };
    render_queue.write_buffer(
        &query_buffers.query_uniform_buffer,
        0,
        bytemuck::bytes_of(&uniforms),
    );

    // Rebuild bind groups if never built, or the live-LOD set changed
    // (a brand-new arena's buffers need to replace the dummy fallback).
    if query_buffers.bind_groups[0].is_none()
        || query_buffers.bound_lod_presence != current_presence
    {
        for p in 0..QUERY_BUFFER_SLOTS {
            let mut entries = vec![
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
                    resource: query_buffers.merged_lookup_buffer.as_entire_binding(),
                },
            ];
            for r in 0..LOD::COUNT as u32 {
                let arena = LOD::from_rank(r).and_then(|lod| arena_set.arenas.get(&lod));
                let (sdf, meta) = match arena {
                    Some(a) => (&a.sdf_buffer, &a.chunk_meta_buffer),
                    None => (
                        &query_buffers.dummy_sdf_buffer,
                        &query_buffers.dummy_meta_buffer,
                    ),
                };
                let base = 4 + r * 2;
                entries.push(BindGroupEntry {
                    binding: base,
                    resource: sdf.as_entire_binding(),
                });
                entries.push(BindGroupEntry {
                    binding: base + 1,
                    resource: meta.as_entire_binding(),
                });
            }
            query_buffers.bind_groups[p] = Some(render_device.create_bind_group(
                Some("voxel_query_bind_group"),
                &layouts.query_layout,
                &entries,
            ));
        }
        query_buffers.bound_lod_presence = current_presence;
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
    mut next_slot: ResMut<NextQuerySlot>,
) {
    let count = query_buffers.active_count; // was: extracted.0.len() as u32
    if count == 0 {
        return;
    }
    let Some(pipeline) = pipeline_cache.get_compute_pipeline(compute_pipeline.query_pipeline_id)
    else {
        return;
    };

    let slot = next_slot.0;
    next_slot.0 = (slot + 1) % QUERY_BUFFER_SLOTS;

    if query_buffers.slot_state[slot].load(Ordering::Acquire) != SLOT_FREE {
        return;
    }
    let Some(bind_group) = &query_buffers.bind_groups[slot] else {
        return;
    };

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
        pass.dispatch_workgroups(count, 1, 1); // was: extracted.0.len() as u32
    }
    encoder.copy_buffer_to_buffer(
        &query_buffers.hit_buffers[slot],
        0,
        &query_buffers.readback_buffers[slot],
        0,
        query_buffers.hit_buffers[slot].size(), // was: readback_buffers[slot].size() — same value, but now correctly tracks capacity, not a stale MAX_QUERIES_PER_FRAME assumption
    );
    render_queue.submit(std::iter::once(encoder.finish()));

    query_buffers.slot_state[slot].store(SLOT_COPIED, Ordering::Release);
}

/// Runs in `RenderSystems::Cleanup`. Maps every slot currently in the
/// Copied state — i.e. whose GPU copy was submitted (in some earlier
/// frame, or possibly this one) and hasn't been picked up for mapping
/// yet. compare_exchange guards against a slot being mapped twice if this
/// system somehow ran with a state already claimed.
pub fn map_voxel_query_results(
    query_buffers: Res<VoxelQueryBuffers>,
    channel: Res<VoxelQueryResultChannel>,
) {
    for slot in 0..QUERY_BUFFER_SLOTS {
        let state = query_buffers.slot_state[slot].clone();
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

        let buf = query_buffers.readback_buffers[slot].clone();
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
                let data = buf.slice(..).get_mapped_range();
                bytemuck::cast_slice(&data).to_vec()
            };
            buf.unmap();
            let _ = sender.send(out);
            state.store(SLOT_FREE, Ordering::Release);
        });
    }
}

fn dispatch(
    encoder: &mut CommandEncoder,
    label: &'static str,
    pipeline: &ComputePipeline,
    bind_group: &BindGroup,
    (x, y, z): (u32, u32, u32),
) {
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(x, y, z);
}
