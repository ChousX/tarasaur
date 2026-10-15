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
    field::{LOD, MaterialField, SDF, VisibilityField, VoxelMaterial},
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

/// Keeps the newest extraction per chunk, despawning older ones and any for
/// chunks that are no longer live. Returns `(waiting_since, entity)`, oldest
/// wait first. The newest data inherits the oldest wait, so a chunk that is
/// re-extracted every frame (e.g. under a held brush) can't starve behind
/// the budget. Anything not despawned stays in the render world and is
/// retried next frame.
fn newest_pending<T: Send + Sync + 'static>(
    arena_set: &VoxelChunkArenaSet,
    extracted: &Query<(Entity, &ExtractedChunkField<T>)>,
    commands: &mut Commands,
) -> Vec<(u32, Entity)> {
    // main entity -> (oldest seq, newest seq, newest entity)
    let mut newest: HashMap<MainEntity, (u32, u32, Entity)> = HashMap::new();
    for (e, x) in extracted.iter() {
        if !arena_set.live.contains(&x.main_entity) {
            commands.entity(e).despawn();
            continue;
        }
        let Some(entry) = newest.get_mut(&x.main_entity) else {
            newest.insert(x.main_entity, (x.seq, x.seq, e));
            continue;
        };
        entry.0 = entry.0.min(x.seq);
        if x.seq > entry.1 {
            commands.entity(entry.2).despawn();
            (entry.1, entry.2) = (x.seq, e);
        } else {
            commands.entity(e).despawn();
        }
    }
    let mut pending: Vec<_> = newest
        .into_values()
        .map(|(since, _, e)| (since, e))
        .collect();
    pending.sort_unstable_by_key(|p| p.0);
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

/// Uploads changed chunks' SDF and gives each an exact-size vertex range.
///
/// Never drops one: past the per-frame budget, or with no room in any arena,
/// the upload waits and the chunk keeps drawing its current mesh (an old
/// LOD is only released once the new one is placed).
pub fn prepare_voxel_arena(
    mut arena_set: ResMut<VoxelChunkArenaSet>,
    gpu: ArenaGpu,
    budget: Res<RemeshBudget>,
    extracted_chunks: Query<(Entity, &ExtractedChunkField<SDF>)>,
    mut commands: Commands,
    mut out_of_room: Local<bool>,
    stats: Option<ResMut<VoxelFrameStats>>,
) {
    arena_set.deferred.clear();
    arena_set.reslotted.clear();
    let pending = newest_pending(&arena_set, &extracted_chunks, &mut commands);

    // Chunks already resident at this LOD (edits, neighbour apron changes)
    // always update this frame so edits stay continuous. The budget only
    // paces new placements (fresh loads, LOD changes), oldest first and
    // always at least one.
    let mut cells_left = budget.cells_per_frame;
    let mut placed_new = false;
    let mut batch = Vec::new();
    for (_, e) in pending {
        let Ok((_, x)) = extracted_chunks.get(e) else {
            continue;
        };
        if arena_set.find_slot(x.lod, x.main_entity).is_none() {
            let cells = x.size.saturating_sub(1).saturating_pow(3);
            if placed_new && cells > cells_left {
                arena_set.deferred.insert(x.main_entity);
                continue;
            }
            placed_new = true;
            cells_left = cells_left.saturating_sub(cells);
        }
        batch.push((e, x));
    }

    let mut counts: Vec<(usize, u32)> = ComputeTaskPool::get().scope(|s| {
        for (i, (_, x)) in batch.iter().enumerate() {
            s.spawn(async move { (i, count_active_cells(&x.padded_data, x.size)) });
        }
    });
    counts.sort_unstable_by_key(|c| c.0);

    let mut any_out_of_room = false;
    for ((extracted_entity, x), (_, count)) in batch.into_iter().zip(counts) {
        let (lod, size) = (x.lod, x.size);
        if count == 0 {
            // No geometry: holding a slot (575 KB of SDF at High) would draw
            // nothing. Any old slot drew nothing for this data either.
            arena_set.release(x.main_entity);
            arena_set.empty.insert(x.main_entity);
            commands.entity(extracted_entity).despawn();
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
        upload_chunk(&gpu.render_queue, arena, slot, x, count);
        commands.entity(extracted_entity).despawn();
    }
    *out_of_room = any_out_of_room;
    if let Some(mut stats) = stats {
        let n = u32::try_from(arena_set.reslotted.len()).unwrap_or(u32::MAX);
        stats.reslotted = stats.reslotted.saturating_add(n);
    }
}

/// Writes a placed chunk's SDF, metadata, vertex range and draw args into
/// its arena slot, and marks the slot for re-meshing.
fn upload_chunk(
    render_queue: &RenderQueue,
    arena: &mut VoxelChunkArena,
    slot: u32,
    x: &ExtractedChunkField<SDF>,
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

    let meta = ChunkMeta {
        chunk_world_origin: (x.chunk_pos.as_vec3() * CHUNK_SIZE).into(),
        lod_ranks: x.lod_ranks,
    };
    render_queue.write_buffer(
        &arena.chunk_meta_buffer,
        slot64.wrapping_mul(u64_from_usize(std::mem::size_of::<ChunkMeta>())),
        bytemuck::bytes_of(&meta),
    );

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
    for (_, e) in newest_pending(arena_set, extracted, commands) {
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

/// Sibling to `prepare_voxel_arena`: uploads material ids into the slot it
/// placed. Must run after it, so `deferred`, `reslotted` and new slots are
/// current.
pub fn prepare_material_for_arena<M: VoxelMaterial>(
    mut arena_set: ResMut<VoxelChunkArenaSet>,
    render_queue: Res<RenderQueue>,
    extracted_chunks: Query<(Entity, &ExtractedChunkField<MaterialField<M>>)>,
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

pub fn prepare_visibility_for_arena(
    mut arena_set: ResMut<VoxelChunkArenaSet>,
    render_queue: Res<RenderQueue>,
    extracted_chunks: Query<(Entity, &ExtractedChunkField<VisibilityField>)>,
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
    use crate::{field::SDF, field::VoxelDataSlice, grid::flatten_cube, num::usize_from};

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
