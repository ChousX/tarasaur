//! Render-world uploads into the arenas.
//!
//! Each changed chunk's SDF gets a slot and an exact-size vertex range (from
//! a CPU count of active cells), and its material/visibility data follows
//! it into that slot.

use std::{
    any::TypeId,
    collections::{HashMap, HashSet},
    sync::Arc,
};

use crate::num::{u64_from_usize, usize_from};

use bevy::{
    ecs::system::SystemParam,
    prelude::*,
    render::{
        renderer::{RenderDevice, RenderQueue},
        sync_world::MainEntity,
    },
    tasks::ComputeTaskPool,
};

use crate::{
    chunk::CHUNK_SIZE,
    field::{LOD, VoxelField},
    voxel::extract::ExtractedChunkField,
    voxel::{
        arena::{ArenaSlot, ChunkMeta, INDICES_PER_CELL, VoxelChunkArena, VoxelChunkArenaSet},
        pipeline::VoxelPipelineLayouts,
        stats::VoxelFrameStats,
        types::DrawIndexedIndirectArgs,
    },
};

/// Max cells re-meshed per frame, summed over all arenas.
///
/// Uploads past it wait for the next frame while the chunk keeps its current
/// mesh, so a burst of loads is spread out instead of landing in one frame.
#[derive(Resource)]
pub struct RemeshBudget {
    pub cells_per_frame: u32,
}

impl Default for RemeshBudget {
    fn default() -> Self {
        Self {
            cells_per_frame: 4_000_000, // ~15 High chunks
        }
    }
}

/// One chunk's upload to make this frame.
struct Pending {
    /// Main-world frame of the oldest extraction it replaces.
    since: u32,
    /// The newest extraction with data (or, if none is pending, the newest).
    entity: Entity,
    /// The newest seam ranks, which a newer metadata-only extraction may
    /// have updated (see `PaddedChunk::meta_only`).
    seam_ranks: [u32; 2],
    /// That newer metadata-only extraction, folded into `entity`: despawn it
    /// along with `entity` once uploaded.
    folded: Option<Entity>,
}

impl Pending {
    /// Done with it: despawns it and anything folded into it.
    fn despawn(&self, commands: &mut Commands) {
        commands.entity(self.entity).despawn();
        if let Some(folded) = self.folded {
            commands.entity(folded).despawn();
        }
    }
}

/// Keeps the newest extraction per chunk, despawning older ones and any for
/// chunks that are no longer live, oldest wait first. The newest data
/// inherits the oldest wait, so a chunk that is re-extracted every frame
/// (e.g. under a held brush) can't starve behind the budget. Anything not
/// despawned stays in the render world and is retried next frame.
///
/// A metadata-only extraction newer than one with data doesn't replace it
/// (the data is still current): it lends it its seam ranks.
fn newest_pending<T: Send + Sync + 'static>(
    arena_set: &VoxelChunkArenaSet,
    extracted: &Query<(Entity, &ExtractedChunkField<T>)>,
    commands: &mut Commands,
) -> Vec<Pending> {
    #[derive(Clone, Copy)]
    struct Newest {
        since: u32,
        /// Newest with data, and newest metadata-only: (seq, entity).
        data: Option<(u32, Entity)>,
        meta: Option<(u32, Entity)>,
    }
    let mut newest: HashMap<MainEntity, Newest> = HashMap::new();
    for (e, x) in extracted.iter() {
        if !arena_set.live.contains(&x.main_entity) {
            commands.entity(e).despawn();
            continue;
        }
        let entry = newest.entry(x.main_entity).or_insert(Newest {
            since: x.seq,
            data: None,
            meta: None,
        });
        entry.since = entry.since.min(x.seq);
        let slot = if x.meta_only {
            &mut entry.meta
        } else {
            &mut entry.data
        };
        match *slot {
            Some((seq, _)) if seq >= x.seq => {
                commands.entity(e).despawn();
            }
            old => {
                if let Some((_, old_e)) = old {
                    commands.entity(old_e).despawn();
                }
                *slot = Some((x.seq, e));
            }
        }
    }
    let mut pending: Vec<Pending> = newest
        .into_values()
        .filter_map(|n| {
            // Data newer than the metadata already carries newer ranks.
            let (meta, folded) = match (n.data, n.meta) {
                (Some((d, _)), Some((m, me))) if m <= d => {
                    commands.entity(me).despawn();
                    (None, None)
                }
                (Some(_), Some((_, me))) => (Some(me), Some(me)),
                (_, meta) => (meta.map(|m| m.1), None),
            };
            let entity = n.data.map(|d| d.1).or(meta)?;
            let ranks_from = meta.unwrap_or(entity);
            let seam_ranks = extracted.get(ranks_from).map_or([0; 2], |(_, x)| x.seam_ranks);
            Some(Pending {
                since: n.since,
                entity,
                seam_ranks,
                folded,
            })
        })
        .collect();
    pending.sort_unstable_by_key(|p| p.since);
    pending
}

/// Active cells in a padded SDF volume, using exactly pass1's test: a cell
/// is active when some but not all of its 8 corners are `<= 0`. Sizes the
/// chunk's vertex range before the GPU meshes it.
fn count_active_cells(padded: &[u8], texture_size: u32) -> u32 {
    let size = usize_from(texture_size);
    // snorm16, as uploaded (see quantize_sdf): same sign test as pass1.
    let inside: Vec<u8> = padded
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&b| u8::from(i16::from_le_bytes(b) <= 0))
        .collect();
    let Some(slab) = size.checked_mul(size) else {
        return 0;
    };
    if size < 2 || Some(inside.len()) != slab.checked_mul(size) {
        return 0;
    }
    // [z][y] -> row of x
    let slabs: Vec<Vec<&[u8]>> = inside
        .chunks_exact(slab)
        .map(|s| s.chunks_exact(size).collect())
        .collect();
    let mut total: u32 = 0;
    for pair in slabs.windows(2) {
        let [lo, hi] = pair else { continue };
        for (lo_rows, hi_rows) in lo.windows(2).zip(hi.windows(2)) {
            let (&[y0z0, y1z0], &[y0z1, y1z1]) = (lo_rows, hi_rows) else {
                continue;
            };
            // Each corner is 0 or 1, so these sums stay tiny.
            let mut columns = y0z0
                .iter()
                .zip(y1z0)
                .zip(y0z1)
                .zip(y1z1)
                .map(|(((a, b), c), d)| a.wrapping_add(*b).wrapping_add(*c).wrapping_add(*d));
            let Some(mut prev) = columns.next() else {
                continue;
            };
            for next in columns {
                let corners = prev.wrapping_add(next);
                total = total.wrapping_add(u32::from(corners > 0 && corners < 8));
                prev = next;
            }
        }
    }
    total
}

/// The render resources `prepare_voxel_arena` creates arenas and writes with.
#[derive(SystemParam)]
pub struct ArenaGpu<'w> {
    render_device: Res<'w, RenderDevice>,
    layouts: Res<'w, VoxelPipelineLayouts>,
    render_queue: Res<'w, RenderQueue>,
}

/// This frame's uploads, in `pending`'s order (oldest wait first).
///
/// Pending chunks already on screen that touch (26-neighbours) form a
/// cluster, and a cluster updates all in one frame or waits whole: a chunk
/// re-extracted to stitch to its neighbour's new LOD must not show before
/// that neighbour's new mesh does, nor after. Showing one side first cracked
/// the seam until the other landed (the brief black lines next to fresh
/// LODs). A chunk loading in for the first time has no old mesh to clash
/// with, so it stands alone: clustering those too made the whole first load
/// one cluster too big to ever fit, and nothing loaded.
///
/// Clusters go oldest first while they fit the budget, which paces new
/// placements (fresh loads, LOD changes), and their LODs' re-mesh scratch.
/// The first cluster of a frame goes even over the budget, and if it needs
/// more scratch than is left, as much of it as fits goes, so nothing waits
/// forever. Waiting new placements go in `deferred`.
fn admit<'a, F: VoxelField>(
    arena_set: &mut VoxelChunkArenaSet,
    cells_per_frame: u32,
    pending: Vec<Pending>,
    extracted: &'a Query<(Entity, &ExtractedChunkField<F>)>,
) -> Vec<(Pending, &'a ExtractedChunkField<F>)> {
    let items: Vec<_> = pending
        .into_iter()
        .filter_map(|p| {
            let (_, x) = extracted.get(p.entity).ok()?;
            let new = !x.meta_only && arena_set.find_slot(x.lod, x.main_entity).is_none();
            let shown = LOD::ALL
                .iter()
                .any(|&l| arena_set.find_slot(l, x.main_entity).is_some());
            Some((p, x, new, shown))
        })
        .collect();
    let cluster = clusters(
        &items
            .iter()
            .map(|(_, x, _, shown)| shown.then_some(x.chunk_pos))
            .collect::<Vec<_>>(),
    );

    // Re-mesh scratch left per LOD this frame, over its arenas. A LOD with
    // none yet makes one with room.
    let mut scratch_left = arena_set.arenas.each_ref().map(|arenas| {
        if arenas.is_empty() {
            return usize::MAX;
        }
        arenas
            .iter()
            .map(|a| usize_from(a.scratch_chunks).saturating_sub(a.dirty_slots.len()))
            .sum()
    });
    let mut cells_left = cells_per_frame;
    let mut admitted_any = false;
    let mut admitted = vec![false; items.len()];
    // Clusters in order of their oldest member, which comes first.
    let mut order = Vec::new();
    let mut members: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &c) in cluster.iter().enumerate() {
        members.entry(c).or_insert_with(|| {
            order.push(c);
            Vec::new()
        });
        if let Some(m) = members.get_mut(&c) {
            m.push(i);
        }
    }
    for c in order {
        let members = members.remove(&c).unwrap_or_default();
        let mut cells = 0u32;
        let mut scratch = [0usize; LOD::COUNT];
        for &i in &members {
            let Some((_, x, new, _)) = items.get(i) else {
                continue;
            };
            if *new {
                cells = cells.saturating_add(x.size.saturating_sub(1).saturating_pow(3));
            }
            if let Some(n) = scratch.get_mut(VoxelChunkArenaSet::lod_index(x.lod)) {
                *n = n.saturating_add(1);
            }
        }
        let fits_scratch = scratch
            .iter()
            .zip(&scratch_left)
            .all(|(need, left)| need <= left);
        if fits_scratch && !(admitted_any && cells > cells_left) {
            cells_left = cells_left.saturating_sub(cells);
            for (left, need) in scratch_left.iter_mut().zip(scratch) {
                *left = left.saturating_sub(need);
            }
        } else if admitted_any {
            continue;
        } else {
            // Too big for this frame's scratch even alone: as much as fits.
            for &i in &members {
                let lod_idx = items
                    .get(i)
                    .map(|(_, x, _, _)| VoxelChunkArenaSet::lod_index(x.lod));
                let left = lod_idx.and_then(|l| scratch_left.get_mut(l));
                let go = left.is_some_and(|left| {
                    let room = *left > 0;
                    *left = left.saturating_sub(1);
                    room
                });
                if let Some(a) = admitted.get_mut(i) {
                    *a = go;
                }
            }
            admitted_any = true;
            continue;
        }
        admitted_any = true;
        for i in members {
            if let Some(a) = admitted.get_mut(i) {
                *a = true;
            }
        }
    }
    items
        .into_iter()
        .zip(admitted)
        .filter_map(|((p, x, new, _), go)| {
            if !go && new {
                arena_set.deferred.insert(x.main_entity);
            }
            go.then_some((p, x))
        })
        .collect()
}

/// Each position's cluster: the index of the first position it connects to
/// through a chain of 26-neighbours. `None` stands alone.
fn clusters(positions: &[Option<IVec3>]) -> Vec<usize> {
    let index: HashMap<IVec3, usize> = positions
        .iter()
        .enumerate()
        .rev()
        .filter_map(|(i, p)| p.map(|p| (p, i)))
        .collect();
    let mut cluster = vec![usize::MAX; positions.len()];
    for start in 0..positions.len() {
        if cluster.get(start) != Some(&usize::MAX) {
            continue;
        }
        let mut stack = vec![start];
        while let Some(i) = stack.pop() {
            match cluster.get_mut(i) {
                Some(c) if *c == usize::MAX => *c = start,
                _ => continue,
            }
            let Some(&Some(pos)) = positions.get(i) else {
                continue;
            };
            for k in 0u32..27 {
                let d = UVec3::new(k % 3, k / 3 % 3, k / 9)
                    .as_ivec3()
                    .wrapping_sub(IVec3::ONE);
                if let Some(&j) = index.get(&pos.saturating_add(d)) {
                    stack.push(j);
                }
            }
        }
    }
    cluster
}

/// The `GpuChannel::Shape` upload: changed chunks' SDF, each given a slot
/// and an exact-size vertex range.
///
/// Never drops one: past the per-frame budget, or with no room in any arena,
/// the upload waits and the chunk keeps drawing its current mesh (an old
/// LOD is only released once the new one is placed).
pub fn prepare_shape_upload<F: VoxelField>(
    mut arena_set: ResMut<VoxelChunkArenaSet>,
    gpu: ArenaGpu,
    budget: Res<RemeshBudget>,
    extracted_chunks: Query<(Entity, &ExtractedChunkField<F>)>,
    mut commands: Commands,
    mut out_of_room: Local<bool>,
    stats: Option<ResMut<VoxelFrameStats>>,
) {
    arena_set.deferred.clear();
    arena_set.reslotted.clear();
    let pending = newest_pending(&arena_set, &extracted_chunks, &mut commands);

    let mut batch = Vec::new();
    for (p, x) in admit(
        &mut arena_set,
        budget.cells_per_frame,
        pending,
        &extracted_chunks,
    ) {
        if x.meta_only {
            if restitch(&mut arena_set, &gpu.render_queue, x, p.seam_ranks) {
                commands.entity(p.entity).despawn();
            }
            continue;
        }
        batch.push((p, x));
    }

    let mut counts: Vec<(usize, u32)> = ComputeTaskPool::get().scope(|s| {
        for (i, (_, x)) in batch.iter().enumerate() {
            s.spawn(async move { (i, count_active_cells(&x.padded_data, x.size)) });
        }
    });
    counts.sort_unstable_by_key(|c| c.0);

    let mut any_out_of_room = false;
    for ((p, x), (_, count)) in batch.into_iter().zip(counts) {
        let (lod, size) = (x.lod, x.size);
        if count == 0 {
            // No geometry: holding a slot (575 KB of SDF at High) would draw
            // nothing. Any old slot drew nothing for this data either.
            arena_set.release(x.main_entity);
            arena_set.empty.insert(x.main_entity);
            p.despawn(&mut commands);
            continue;
        }
        arena_set.empty.remove(&x.main_entity);
        let cell_count = size.saturating_sub(1);
        let before = arena_set.find_slot(lod, x.main_entity);
        let max_buffer_bytes = arena_set.config.max_buffer_bytes;
        let Some(placed @ ArenaSlot { arena_idx, slot }) =
            arena_set.place(lod, x.main_entity, count, || {
                info!(
                    "[VoxelChunkArenaSet] creating arena for {:?} (cell_count={}, texture_size={})",
                    lod, cell_count, size
                );
                VoxelChunkArena::new(
                    &gpu.render_device,
                    &gpu.layouts,
                    max_buffer_bytes,
                    cell_count,
                    size,
                )
            })
        else {
            arena_set.deferred.insert(x.main_entity);
            let lod_idx = VoxelChunkArenaSet::lod_index(lod);
            if arena_set
                .arenas
                .get(lod_idx)
                .is_some_and(|arenas| arenas.iter().any(VoxelChunkArena::dirty_full))
            {
                continue; // out of scratch for this frame: just wait
            }
            if !*out_of_room {
                warn!(
                    "[prepare_voxel_arena] no room for LOD {:?} chunk {:?}; it keeps its current mesh until space frees up",
                    lod, x.chunk_pos
                );
            }
            any_out_of_room = true;
            continue;
        };
        if before != Some(placed) {
            arena_set.reslotted.insert(x.main_entity);
        }
        let Some(arena) = arena_set.arena_mut(lod, arena_idx) else {
            continue;
        };
        arena.register_chunk_pos(slot, x.chunk_pos);
        debug_assert_eq!(
            size, arena.texture_size,
            "mixed LOD in one arena — bucket by LOD"
        );
        upload_chunk(&gpu.render_queue, arena, slot, x, p.seam_ranks, count);
        p.despawn(&mut commands);
    }
    *out_of_room = any_out_of_room;
    if let Some(mut stats) = stats {
        let n = u32::try_from(arena_set.reslotted.len()).unwrap_or(u32::MAX);
        stats.reslotted = stats.reslotted.saturating_add(n);
    }
}

/// Writes a placed chunk's SDF, metadata, vertex range and draw args into
/// its arena slot, and marks the slot for re-meshing.
fn upload_chunk<F: Send + Sync + 'static>(
    render_queue: &RenderQueue,
    arena: &mut VoxelChunkArena,
    slot: u32,
    x: &ExtractedChunkField<F>,
    seam_ranks: [u32; 2],
    count: u32,
) {
    let Some(&(base, _)) = arena.slot_range.get(usize_from(slot)) else {
        return;
    };
    let slot64 = u64::from(slot);
    // snorm16: 2 bytes per element
    let sdf_offset_bytes = slot64
        .wrapping_mul(u64::from(arena.sdf_elems_per_chunk))
        .wrapping_mul(2);
    render_queue.write_buffer(&arena.sdf_buffer, sdf_offset_bytes, &x.padded_data);

    write_chunk_meta(render_queue, arena, slot, x.chunk_pos, seam_ranks);

    let slot_word = slot64.wrapping_mul(4);
    render_queue.write_buffer(
        &arena.slot_vertex_base_buffer,
        slot_word,
        bytemuck::bytes_of(&base),
    );
    render_queue.write_buffer(
        &arena.slot_vertex_count_buffer,
        slot_word,
        bytemuck::bytes_of(&count),
    );
    let args = DrawIndexedIndirectArgs {
        index_count: 0, // pass3 fills it in this frame
        instance_count: 1,
        first_index: base.wrapping_mul(INDICES_PER_CELL),
        base_vertex: 0,
        first_instance: 0,
    };
    render_queue.write_buffer(
        &arena.indirect_args_buffer,
        slot64.wrapping_mul(ARGS_STRIDE),
        bytemuck::bytes_of(&args),
    );

    arena.mark_dirty(slot);
}

/// Applies a metadata-only extraction: the chunk's data is unchanged, its
/// seam ranks aren't, so rewrite its slot's metadata and re-mesh it. No slot
/// means it meshed to nothing, which the ranks can't change. `false`: out of
/// re-mesh scratch this frame, retry next.
fn restitch<F: Send + Sync + 'static>(
    arena_set: &mut VoxelChunkArenaSet,
    render_queue: &RenderQueue,
    x: &ExtractedChunkField<F>,
    seam_ranks: [u32; 2],
) -> bool {
    let Some(ArenaSlot { arena_idx, slot }) = arena_set.find_slot(x.lod, x.main_entity) else {
        return true;
    };
    let Some(arena) = arena_set.arena_mut(x.lod, arena_idx) else {
        return true;
    };
    if !arena.can_mark_dirty(slot) {
        return false;
    }
    write_chunk_meta(render_queue, arena, slot, x.chunk_pos, seam_ranks);
    arena.mark_dirty(slot);
    true
}

/// Writes a slot's `ChunkMeta`.
fn write_chunk_meta(
    render_queue: &RenderQueue,
    arena: &VoxelChunkArena,
    slot: u32,
    chunk_pos: IVec3,
    seam_ranks: [u32; 2],
) {
    let meta = ChunkMeta {
        chunk_world_origin: (chunk_pos.as_vec3() * CHUNK_SIZE).into(),
        seam_ranks,
        ..ChunkMeta::default()
    };
    render_queue.write_buffer(
        &arena.chunk_meta_buffer,
        u64::from(slot).wrapping_mul(u64_from_usize(std::mem::size_of::<ChunkMeta>())),
        bytemuck::bytes_of(&meta),
    );
}

/// One side-channel write into a chunk's slot. `extracted` is the upload it
/// came from, despawned once written; `None` when refilled from the cache.
struct SideUpload {
    lod: LOD,
    slot: ArenaSlot,
    data: Arc<[u8]>,
    extracted: Option<Entity>,
}

/// This frame's writes for a channel that rides along with a chunk's SDF
/// (material, visibility).
///
/// Every upload not waiting on its SDF (`deferred`) is cached in
/// `side_data`, then written if the chunk holds a slot; a chunk that meshes
/// to nothing, or isn't placed yet, gets it from the cache once it is.
/// Chunks that moved slot this frame (`reslotted`) with no fresh upload are
/// refilled from the cache, since the main world only re-sends a field when
/// it changes.
fn side_channel_uploads<T: Send + Sync + 'static>(
    arena_set: &mut VoxelChunkArenaSet,
    extracted: &Query<(Entity, &ExtractedChunkField<T>)>,
    commands: &mut Commands,
) -> Vec<SideUpload> {
    let mut fresh = Vec::new();
    for p in newest_pending(arena_set, extracted, commands) {
        let e = p.entity;
        let Ok((_, x)) = extracted.get(e) else {
            continue;
        };
        if arena_set.deferred.contains(&x.main_entity) {
            continue; // waits with its SDF
        }
        fresh.push((e, x.main_entity, x.lod, Arc::clone(&x.padded_data)));
    }
    let cache = arena_set.side_data.entry(TypeId::of::<T>()).or_default();
    for (_, main, lod, data) in &fresh {
        cache.insert(*main, (*lod, Arc::clone(data)));
    }

    let mut out = Vec::new();
    let mut written = HashSet::new();
    for (e, main, lod, data) in fresh {
        let Some(slot) = arena_set.find_slot(lod, main) else {
            commands.entity(e).despawn(); // cached; written once the chunk has a slot
            continue;
        };
        written.insert(main);
        out.push(SideUpload {
            lod,
            slot,
            data,
            extracted: Some(e),
        });
    }
    if let Some(cache) = arena_set.side_data.get(&TypeId::of::<T>()) {
        for main in arena_set.reslotted.iter().filter(|m| !written.contains(*m)) {
            let Some((lod, data)) = cache.get(main) else {
                continue;
            };
            let Some(slot) = arena_set.find_slot(*lod, *main) else {
                continue; // cached at another LOD: that LOD's upload is on its way
            };
            out.push(SideUpload {
                lod: *lod,
                slot,
                data: Arc::clone(data),
                extracted: None,
            });
        }
    }
    out
}

/// Adds the uploads refilled from the cache to `VoxelFrameStats::side_refills`.
fn count_refills(stats: Option<ResMut<VoxelFrameStats>>, uploads: &[SideUpload]) {
    if let Some(mut stats) = stats {
        let n = uploads.iter().filter(|u| u.extracted.is_none()).count();
        let n = u32::try_from(n).unwrap_or(u32::MAX);
        stats.side_refills = stats.side_refills.saturating_add(n);
    }
}

/// The `GpuChannel::Material` upload: material ids into the slot the shape
/// upload placed. Runs after it (`UploadSet::Channels`), so `deferred`,
/// `reslotted` and new slots are current.
pub fn prepare_material_upload<F: VoxelField>(
    mut arena_set: ResMut<VoxelChunkArenaSet>,
    render_queue: Res<RenderQueue>,
    extracted_chunks: Query<(Entity, &ExtractedChunkField<F>)>,
    mut commands: Commands,
    stats: Option<ResMut<VoxelFrameStats>>,
) {
    let uploads = side_channel_uploads(&mut arena_set, &extracted_chunks, &mut commands);
    count_refills(stats, &uploads);
    for up in uploads {
        let slot = up.slot.slot;
        let Some(arena) = arena_set.arena_mut(up.lod, up.slot.arena_idx) else {
            continue;
        };
        // A reslotted chunk's slot was just marked dirty by its SDF upload,
        // so only fresh uploads can be turned away here.
        if !arena.can_mark_dirty(slot) {
            continue; // no scratch left this frame; retried next frame
        }
        let offset = u64::from(slot).wrapping_mul(u64::from(arena.sdf_elems_per_chunk));
        render_queue.write_buffer(&arena.material_buffer, offset, &up.data);
        arena.mark_dirty(slot); // pass3 bakes materials into vertices
        if let Some(e) = up.extracted {
            commands.entity(e).despawn();
        }
    }
}

/// The `GpuChannel::Mask` upload: each chunk's mask as one bit per voxel,
/// plus the per-slot "has a mask" bits the raster pass checks first.
pub fn prepare_mask_upload<F: VoxelField>(
    mut arena_set: ResMut<VoxelChunkArenaSet>,
    render_queue: Res<RenderQueue>,
    extracted_chunks: Query<(Entity, &ExtractedChunkField<F>)>,
    mut commands: Commands,
    stats: Option<ResMut<VoxelFrameStats>>,
) {
    let uploads = side_channel_uploads(&mut arena_set, &extracted_chunks, &mut commands);
    count_refills(stats, &uploads);
    for up in uploads {
        let slot = up.slot.slot;
        let Some(arena) = arena_set.arena_mut(up.lod, up.slot.arena_idx) else {
            continue;
        };
        // Empty = the chunk went uniform: clear the flag so the raster stops
        // consulting the now-stale mask.
        let has_mask = !up.data.is_empty();
        if has_mask {
            // One byte per voxel -> 1 bit. Each slot gets its own word-aligned
            // region, like sdf/material_buffer, so adjacent slots' writes can't
            // clobber each other's boundary words.
            let words_per_chunk = u64::from(arena.sdf_elems_per_chunk).div_ceil(32);
            let packed = pack_bits(up.data.iter().map(|&b| b != 0));
            render_queue.write_buffer(
                &arena.visibility_mask_buffer,
                u64::from(slot)
                    .wrapping_mul(words_per_chunk)
                    .wrapping_mul(4),
                bytemuck::cast_slice(&packed),
            );
        }
        arena.set_slot_has_mask(slot, has_mask);
        if let Some(e) = up.extracted {
            commands.entity(e).despawn();
        }
    }
    for arena in arena_set.iter_mut() {
        let packed = pack_bits(arena.mask_slots.iter().copied());
        render_queue.write_buffer(
            &arena.chunk_has_mask_buffer,
            0,
            bytemuck::cast_slice(&packed),
        );
    }
}

/// Byte stride of one slot's entry in `indirect_args_buffer`.
const ARGS_STRIDE: u64 = u64_from_usize(std::mem::size_of::<DrawIndexedIndirectArgs>());

fn pack_bits(bits: impl ExactSizeIterator<Item = bool>) -> Vec<u32> {
    let mut packed = vec![0u32; bits.len().div_ceil(32)];
    for (i, b) in bits.enumerate() {
        if b && let Some(word) = packed.get_mut(i / 32) {
            *word |= 1 << (i % 32);
        }
    }
    packed
}

#[cfg(test)]
mod tests {
    use super::count_active_cells;
    use crate::{field::SDF, field::VoxelField, grid::flatten_cube, num::usize_from};

    fn volume(n: u32, inside: &[(u32, u32, u32)]) -> Vec<u8> {
        let mut v = vec![1.0f32; usize_from(n).pow(3)];
        for &(x, y, z) in inside {
            v[flatten_cube(x, y, z, n)] = -1.0;
        }
        SDF::gpu_bytes(&v)
    }

    #[test]
    fn counts_cells_with_a_sign_change() {
        assert_eq!(count_active_cells(&volume(4, &[]), 4), 0);
        // A centre voxel touches all 8 cells around it.
        assert_eq!(count_active_cells(&volume(3, &[(1, 1, 1)]), 3), 8);
        // A corner voxel touches only the one cell it's a corner of.
        assert_eq!(count_active_cells(&volume(4, &[(0, 0, 0)]), 4), 1);
    }
}
