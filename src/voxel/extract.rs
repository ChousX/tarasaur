//! Change-driven extraction of chunk fields into the render world.
//!
//! Only chunks whose inputs may have changed this frame are looked at: their
//! own field, a neighbour their apron reads, or their residency (surface,
//! visibility, lease). Scanning every loaded chunk each frame cost ~2 ms per
//! field type at ~30k chunks, inside `ExtractSchedule`, which stalls both
//! worlds.
//!
//! Debug builds re-check every chunk every [`AUDIT_INTERVAL`] frames and warn
//! about (and repair) anything the change tracking missed.

use std::marker::PhantomData;

use bevy::{
    diagnostic::FrameCount,
    ecs::{entity::EntityHashMap, system::SystemParam},
    platform::collections::HashSet,
    prelude::*,
    render::{Extract, sync_world::MainEntity},
};

use crate::{
    ApronSample, Chunk, ChunkManager, ChunkPosition, ExtractGate, LOD, Versionable,
    VisibilityField,
    field::{flatten_cube, surface::HasSurface},
    num::usize_from,
    voxel::{
        arena::VoxelChunkArenaSet,
        resident::ForceResident,
        systems::{ExtractedChunkField, NEIGHBORS_MASK, PADDING, fill_apron},
    },
};

/// Frames between full consistency checks in debug builds.
const AUDIT_INTERVAL: u32 = 120;

/// What a chunk's extraction depends on: its LOD, its own version and each
/// `NEIGHBORS_MASK` neighbour's (border) version with that neighbour's LOD
/// folded in. Unchanged key = unchanged padded volume.
type ExtractKey = (LOD, [u64; 8]);

type FieldQuery<'w, 's, T> =
    Query<'w, 's, (Entity, &'static ChunkPosition, &'static T, &'static LOD)>;

/// Marks `pos` and every chunk whose apron reads `pos` (`pos - offset`).
fn dirty_with_dependents(dirty: &mut HashSet<IVec3>, pos: IVec3) {
    dirty.insert(pos);
    dirty.extend(NEIGHBORS_MASK.iter().map(|off| pos.wrapping_sub(*off)));
}

// ---------------------------------------------------------------------------
// Residency: which chunks are extracted at all
// ---------------------------------------------------------------------------

/// Which chunks `extract_voxel_chunks` skips: no surface, or fully hidden
/// (unless forced resident).
#[derive(SystemParam)]
pub struct ExtractFilter<'w, 's> {
    surface: Extract<'w, 's, Query<'static, 'static, (), With<HasSurface>>>,
    visibility: Extract<'w, 's, Query<'static, 'static, Option<&'static VisibilityField>>>,
    forced: Extract<'w, 's, Res<'static, ForceResident>>,
}

impl ExtractFilter<'_, '_> {
    fn skips(&self, entity: Entity, pos: IVec3) -> bool {
        let hidden = self
            .visibility
            .get(entity)
            .is_ok_and(|v| v.is_some_and(VisibilityField::is_all_empty));
        !self.surface.contains(entity) || (hidden && !self.forced.contains(&pos))
    }
}

/// Chunks whose [`ExtractFilter`] result may have flipped this frame.
#[derive(SystemParam)]
pub struct ResidencyChanges<'w, 's> {
    positions: Extract<'w, 's, Query<'static, 'static, &'static ChunkPosition>>,
    surface_added:
        Extract<'w, 's, Query<'static, 'static, &'static ChunkPosition, Added<HasSurface>>>,
    surface_removed: Extract<'w, 's, RemovedComponents<'static, 'static, HasSurface>>,
    visibility_changed:
        Extract<'w, 's, Query<'static, 'static, &'static ChunkPosition, Changed<VisibilityField>>>,
    visibility_removed: Extract<'w, 's, RemovedComponents<'static, 'static, VisibilityField>>,
    forced: Extract<'w, 's, Res<'static, ForceResident>>,
}

impl ResidencyChanges<'_, '_> {
    /// Adds the positions of chunks whose residency may have changed.
    /// Despawned chunks are left to the caller.
    fn collect(&mut self, out: &mut HashSet<IVec3>) {
        out.extend(self.surface_added.iter().map(|p| p.0));
        out.extend(self.visibility_changed.iter().map(|p| p.0));
        let removed: Vec<Entity> = self
            .surface_removed
            .read()
            .chain(self.visibility_removed.read())
            .collect();
        out.extend(
            removed
                .into_iter()
                .filter_map(|e| self.positions.get(e).ok().map(|p| p.0)),
        );
        out.extend(self.forced.changed().iter().copied());
    }
}

// ---------------------------------------------------------------------------
// extract_voxel_chunks
// ---------------------------------------------------------------------------

/// Everything that can make chunks holding `T` need re-extraction.
#[derive(SystemParam)]
pub struct FieldChanges<'w, 's, T: Component> {
    changed: Extract<'w, 's, Query<'static, 'static, (Entity, &'static ChunkPosition), Changed<T>>>,
    removed: Extract<'w, 's, RemovedComponents<'static, 'static, T>>,
    residency: ResidencyChanges<'w, 's>,
}

#[derive(Default)]
pub struct ExtractState {
    /// Key of the last extraction of each chunk currently extracted.
    last: EntityHashMap<ExtractKey>,
    /// Position of every chunk holding the field, so removing it can dirty
    /// the neighbours whose apron read it.
    positions: EntityHashMap<IVec3>,
    dirty: HashSet<IVec3>,
}

impl ExtractState {
    fn collect<T: Component>(&mut self, changes: &mut FieldChanges<T>) {
        self.dirty.clear();
        // Removals first: a field removed and re-added this frame shows up in
        // both, and must end up recorded.
        for e in changes.removed.read() {
            self.last.remove(&e);
            if let Some(pos) = self.positions.remove(&e) {
                dirty_with_dependents(&mut self.dirty, pos);
            }
        }
        for (e, pos) in changes.changed.iter() {
            self.positions.insert(e, pos.0);
            dirty_with_dependents(&mut self.dirty, pos.0);
        }
        changes.residency.collect(&mut self.dirty);
    }
}

fn extract_key<T: Component + Versionable>(
    field: &T,
    pos: IVec3,
    lod: LOD,
    chunk_manager: &ChunkManager,
    query: &FieldQuery<T>,
) -> ExtractKey {
    let mut versions = [0u64; 8];
    let (own, neighbor_versions) = versions.split_at_mut(1);
    if let Some(own) = own.first_mut() {
        *own = field.version();
    }
    for ((i, offset), slot) in NEIGHBORS_MASK.iter().enumerate().zip(neighbor_versions) {
        if let Some(n_entity) = chunk_manager.get_chunk(&pos.wrapping_add(*offset))
            && let Ok((_, _, n_field, n_lod)) = query.get(n_entity)
        {
            // At equal LOD the apron reads only the neighbour's facing slab;
            // across LODs it resamples more of it, so use the full version.
            let v = if *n_lod == lod {
                n_field.border_version(i)
            } else {
                n_field.version()
            };
            // fold neighbour LOD in: a reload resets version to 0
            *slot = v ^ (u64::from(n_lod.rank()).wrapping_add(1) << 56);
        }
    }
    (lod, versions)
}

/// The chunk's field plus its `+x/+y/+z` apron, as uploaded.
fn padded_bytes<T>(
    field: &T,
    pos: IVec3,
    lod: LOD,
    chunk_manager: &ChunkManager,
    query: &FieldQuery<T>,
) -> Vec<u8>
where
    T: Component + ApronSample + ExtractGate,
{
    if !field.should_extract() {
        return Vec::new();
    }
    let size = lod.size();
    let padded_size = size.wrapping_add(PADDING);
    let mut vol = vec![T::Elem::default(); usize_from(padded_size).saturating_pow(3)];
    // Copy row by row: each source row of x lands at the start of the
    // same (y, z) row in the padded volume.
    let row_len = usize_from(size);
    let mut src_rows = field.data_slice().chunks_exact(row_len);
    for z in 0..size {
        for y in 0..size {
            let dst = flatten_cube(0, y, z, padded_size);
            let dst_row = vol.get_mut(dst..dst.wrapping_add(row_len));
            if let (Some(src), Some(dst)) = (src_rows.next(), dst_row) {
                dst.copy_from_slice(src);
            }
        }
    }
    fill_apron::<T>(&mut vol, pos, chunk_manager, query);
    T::gpu_bytes(&vol)
}

/// Re-extracts the chunks whose padded `T` volume may have changed.
pub fn extract_voxel_chunks<T>(
    mut commands: Commands,
    chunk_manager: Extract<Res<ChunkManager>>,
    query: Extract<FieldQuery<T>>,
    filter: ExtractFilter,
    mut changes: FieldChanges<T>,
    mut state: Local<ExtractState>,
    frame: Extract<Res<FrameCount>>,
) where
    T: Send + Sync + 'static + Component + Versionable + ApronSample + ExtractGate,
{
    state.collect(&mut changes);
    if cfg!(debug_assertions) && frame.0.is_multiple_of(AUDIT_INTERVAL) {
        audit_extraction(&mut state, &chunk_manager, &query, &filter);
    }

    let ExtractState { last, dirty, .. } = &mut *state;
    for pos in dirty.iter() {
        let Some(entity) = chunk_manager.get_chunk(pos) else {
            continue;
        };
        let Ok((_, _, field, lod)) = query.get(entity) else {
            last.remove(&entity);
            continue;
        };
        if filter.skips(entity, *pos) {
            // so gaining a surface, or leasing/un-hiding, re-extracts
            last.remove(&entity);
            continue;
        }
        if field.data_slice().len() != lod.volume() {
            continue; // can't happen (LOD and fields install together); defensive
        }

        // Keyed by entity, not position: a chunk that despawns and respawns at
        // the same spot comes back at version 0 with unchanged neighbours, so a
        // position key matched and the new entity never got an arena slot.
        let key = extract_key(field, *pos, *lod, &chunk_manager, &query);
        if last.get(&entity) == Some(&key) {
            continue;
        }
        last.insert(entity, key);

        commands.spawn(ExtractedChunkField::<T> {
            main_entity: entity.into(),
            chunk_pos: *pos,
            padded_data: padded_bytes(field, *pos, *lod, &chunk_manager, &query),
            size: lod.size().wrapping_add(PADDING),
            lod: *lod,
            seq: frame.0,
            _type: PhantomData,
        });
    }
}

/// Debug check: every chunk whose extraction is out of date must be dirty.
/// Anything missed is logged and made dirty, so it is also repaired.
fn audit_extraction<T>(
    state: &mut ExtractState,
    chunk_manager: &ChunkManager,
    query: &FieldQuery<T>,
    filter: &ExtractFilter,
) where
    T: Component + Versionable + ApronSample,
{
    let mut missed = Vec::new();
    for (entity, pos, field, lod) in query.iter() {
        if state.dirty.contains(&pos.0) || state.positions.get(&entity) != Some(&pos.0) {
            if !state.dirty.contains(&pos.0) {
                missed.push(pos.0); // never recorded
            }
            continue;
        }
        let out_of_date = if filter.skips(entity, pos.0) {
            state.last.contains_key(&entity)
        } else {
            field.data_slice().len() == lod.volume()
                && state.last.get(&entity)
                    != Some(&extract_key(field, pos.0, *lod, chunk_manager, query))
        };
        if out_of_date {
            missed.push(pos.0);
        }
    }
    if missed.is_empty() {
        return;
    }
    warn!(
        "[extract {}] change tracking missed {} chunk(s), e.g. {:?}; re-extracting",
        std::any::type_name::<T>(),
        missed.len(),
        missed.first()
    );
    state.dirty.extend(missed);
}

// ---------------------------------------------------------------------------
// release_despawned_chunks
// ---------------------------------------------------------------------------

type LiveChunkData = (
    Entity,
    &'static ChunkPosition,
    Option<&'static VisibilityField>,
);
type LiveChunkFilter = (With<Chunk>, With<LOD>, With<HasSurface>);

/// What can change the set of chunks that should hold an arena slot.
#[derive(SystemParam)]
pub struct LiveChanges<'w, 's> {
    residency: ResidencyChanges<'w, 's>,
    despawned: Extract<'w, 's, RemovedComponents<'static, 'static, Chunk>>,
    lod_added: Extract<'w, 's, Query<'static, 'static, &'static ChunkPosition, Added<LOD>>>,
    chunk_manager: Extract<'w, 's, Res<'static, ChunkManager>>,
    live: Extract<'w, 's, Query<'static, 'static, LiveChunkData, LiveChunkFilter>>,
    forced: Extract<'w, 's, Res<'static, ForceResident>>,
    frame: Extract<'w, 's, Res<'static, FrameCount>>,
}

impl LiveChanges<'_, '_> {
    fn wanted(&self, entity: Entity) -> bool {
        self.live.get(entity).is_ok_and(|(_, pos, v)| {
            !v.is_some_and(VisibilityField::is_all_empty) || self.forced.contains(&pos.0)
        })
    }
}

/// Keeps `arena_set.live` current and releases the slots of chunks that
/// despawned, lost their surface or became hidden.
pub fn release_despawned_chunks(
    mut changes: LiveChanges,
    mut arena_set: ResMut<VoxelChunkArenaSet>,
    mut dirty: Local<HashSet<IVec3>>,
) {
    let despawned: Vec<Entity> = changes.despawned.read().collect();
    for e in despawned {
        drop_chunk(&mut arena_set, e.into());
    }

    dirty.clear();
    changes.residency.collect(&mut dirty);
    dirty.extend(changes.lod_added.iter().map(|p| p.0));
    for pos in &*dirty {
        let Some(e) = changes.chunk_manager.get_chunk(pos) else {
            continue;
        };
        if changes.wanted(e) {
            arena_set.live.insert(e.into());
        } else {
            drop_chunk(&mut arena_set, e.into());
        }
    }

    if cfg!(debug_assertions) && changes.frame.0.is_multiple_of(AUDIT_INTERVAL) {
        audit_live(&changes, &mut arena_set);
    }
}

fn drop_chunk(arena_set: &mut VoxelChunkArenaSet, e: MainEntity) {
    arena_set.live.remove(&e);
    arena_set.empty.remove(&e);
    arena_set.release(e);
}

/// Debug check: `live` must match a full recount. Repairs and logs misses.
fn audit_live(changes: &LiveChanges, arena_set: &mut VoxelChunkArenaSet) {
    let wanted: std::collections::HashSet<MainEntity> = changes
        .live
        .iter()
        .map(|(e, _, _)| e)
        .filter(|&e| changes.wanted(e))
        .map(MainEntity::from)
        .collect();
    let held = arena_set
        .iter()
        .flat_map(|a| a.slot_of_main_entity.keys().copied());
    let extra: Vec<MainEntity> = arena_set
        .live
        .iter()
        .copied()
        .chain(held)
        .filter(|e| !wanted.contains(e))
        .collect();
    let missing = wanted
        .iter()
        .filter(|e| !arena_set.live.contains(*e))
        .count();
    if extra.is_empty() && missing == 0 {
        return;
    }
    warn!(
        "[release_despawned_chunks] change tracking missed {} stale and {missing} new chunk(s); repairing",
        extra.len()
    );
    for e in extra {
        drop_chunk(arena_set, e);
    }
    arena_set.live = wanted;
}
