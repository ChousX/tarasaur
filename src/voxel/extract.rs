//! Change-driven hand-off of chunk fields to the render world.
//!
//! The main world, in `Last`, works out which chunks' inputs changed this
//! frame (their own field, a neighbour their apron reads, or their residency:
//! surface, visibility, lease) and builds those chunks' padded volumes in
//! parallel. `ExtractSchedule`, which stalls both worlds, only forwards the
//! results. Building the volumes inside extract, one chunk at a time, cost up
//! to ~50 ms in a frame while flying.
//!
//! Debug builds re-check every chunk every [`AUDIT_INTERVAL`] frames and warn
//! about (and repair) anything the change tracking missed.

use std::{marker::PhantomData, sync::Arc};

use bevy::{
    diagnostic::FrameCount,
    ecs::{entity::EntityHashMap, system::SystemParam},
    platform::collections::HashSet,
    prelude::*,
    render::{Extract, ExtractSchedule, RenderApp, sync_world::MainEntity},
    tasks::ComputeTaskPool,
};

use crate::{
    chunk::{Chunk, ChunkManager, ChunkPosition},
    field::{ApronSample, ExtractGate, LOD, Versionable, VisibilityField},
    field::{
        apron::{fill_apron, seam_lod_ranks},
        surface::HasSurface,
    },
    grid::{NEIGHBORS_MASK, PADDING, flatten_cube},
    num::usize_from,
    voxel::{arena::VoxelChunkArenaSet, resident::ForceResident},
};

/// Frames between full consistency checks in debug builds.
const AUDIT_INTERVAL: u32 = 120;

/// What a chunk's upload depends on: its LOD, its own version, each
/// `NEIGHBORS_MASK` neighbour's (border) version with that neighbour's LOD
/// folded in, and its `seam_lod_ranks` (which add the -face neighbours'
/// LODs). Unchanged key = unchanged upload.
type ExtractKey = (LOD, [u64; 8], u32);

type FieldQuery<'w, 's, T> =
    Query<'w, 's, (Entity, &'static ChunkPosition, &'static T, &'static LOD)>;

/// Marks `pos`, every chunk whose apron reads `pos` (`pos - offset`), and
/// the `+x/+y/+z` neighbours, whose seam stitching depends on its LOD.
fn dirty_with_dependents(dirty: &mut HashSet<IVec3>, pos: IVec3) {
    dirty.insert(pos);
    dirty.extend(NEIGHBORS_MASK.iter().map(|off| pos.wrapping_sub(*off)));
    dirty.extend([IVec3::X, IVec3::Y, IVec3::Z].map(|axis| pos.wrapping_add(axis)));
}

/// Registers the hand-off of field `T`: the main-world padding system and
/// the render-world extract system that forwards its output.
pub fn add_field_extraction<T>(app: &mut App)
where
    T: Send + Sync + 'static + Component + Versionable + ApronSample + ExtractGate,
{
    app.init_resource::<PaddedOutbox<T>>()
        .add_systems(Last, pad_dirty_chunks::<T>);
    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app.add_systems(ExtractSchedule, extract_voxel_chunks::<T>);
    }
}

// ---------------------------------------------------------------------------
// Residency: which chunks are extracted at all
// ---------------------------------------------------------------------------

/// Which chunks are not extracted: no surface, or fully hidden (unless
/// forced resident).
#[derive(SystemParam)]
pub struct ExtractFilter<'w, 's> {
    surface: Query<'w, 's, (), With<HasSurface>>,
    visibility: Query<'w, 's, Option<&'static VisibilityField>>,
    forced: Res<'w, ForceResident>,
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
    positions: Query<'w, 's, &'static ChunkPosition>,
    surface_added: Query<'w, 's, &'static ChunkPosition, Added<HasSurface>>,
    surface_removed: RemovedComponents<'w, 's, HasSurface>,
    visibility_changed: Query<'w, 's, &'static ChunkPosition, Changed<VisibilityField>>,
    visibility_removed: RemovedComponents<'w, 's, VisibilityField>,
    forced: Res<'w, ForceResident>,
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
// Fields: main-world padding, render-world forwarding
// ---------------------------------------------------------------------------

/// One chunk's padded volume, ready to upload.
pub struct PaddedChunk {
    pub entity: Entity,
    pub pos: IVec3,
    pub lod: LOD,
    /// See `seam_lod_ranks`.
    pub lod_ranks: u32,
    /// Shared with the render world rather than copied.
    pub bytes: Arc<[u8]>,
    /// Main-world frame it was built in.
    pub seq: u32,
}

/// This frame's padded volumes of field `T`, for `extract_voxel_chunks`.
#[derive(Resource)]
pub struct PaddedOutbox<T> {
    pub chunks: Vec<PaddedChunk>,
    _type: PhantomData<T>,
}

impl<T> Default for PaddedOutbox<T> {
    fn default() -> Self {
        Self {
            chunks: Vec::new(),
            _type: PhantomData,
        }
    }
}

/// Everything that can make chunks holding `T` need a new padded volume.
#[derive(SystemParam)]
pub struct FieldChanges<'w, 's, T: Component> {
    changed: Query<'w, 's, (Entity, &'static ChunkPosition), Changed<T>>,
    removed: RemovedComponents<'w, 's, T>,
    residency: ResidencyChanges<'w, 's>,
}

#[derive(Default)]
pub struct ExtractState {
    /// Key of the last padded volume of each chunk currently extracted.
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

fn extract_key<T: Component + Versionable + ApronSample>(
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
    (
        lod,
        versions,
        seam_lod_ranks(pos, lod, chunk_manager, query),
    )
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

/// Builds new padded volumes, in parallel, for the chunks whose inputs
/// changed this frame. Runs in `Last`, after every edit, load and surface
/// flag of the frame.
pub fn pad_dirty_chunks<T>(
    chunk_manager: Res<ChunkManager>,
    query: FieldQuery<T>,
    filter: ExtractFilter,
    mut changes: FieldChanges<T>,
    mut state: Local<ExtractState>,
    frame: Res<FrameCount>,
    mut outbox: ResMut<PaddedOutbox<T>>,
) where
    T: Send + Sync + 'static + Component + Versionable + ApronSample + ExtractGate,
{
    outbox.chunks.clear();
    state.collect(&mut changes);
    if cfg!(debug_assertions) && frame.0.is_multiple_of(AUDIT_INTERVAL) {
        audit_extraction(&mut state, &chunk_manager, &query, &filter);
    }

    // Which chunks need a new volume: cheap, sequential.
    let ExtractState { last, dirty, .. } = &mut *state;
    let mut jobs = Vec::new();
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
        let lod_ranks = key.2;
        last.insert(entity, key);
        jobs.push((entity, *pos, field, *lod, lod_ranks));
    }

    // Building the volumes: the expensive part, one task per chunk.
    let (manager, fields) = (&*chunk_manager, &query);
    let mut built: Vec<(usize, Vec<u8>)> = ComputeTaskPool::get().scope(|s| {
        for (i, &(_, pos, field, lod, _)) in jobs.iter().enumerate() {
            s.spawn(async move { (i, padded_bytes(field, pos, lod, manager, fields)) });
        }
    });
    built.sort_unstable_by_key(|b| b.0);
    outbox.chunks.extend(jobs.iter().zip(built).map(
        |(&(entity, pos, _, lod, lod_ranks), (_, bytes))| PaddedChunk {
            entity,
            pos,
            lod,
            lod_ranks,
            bytes: bytes.into(),
            seq: frame.0,
        },
    ));
}

/// A chunk's padded volume of field `T` in the render world, waiting to be
/// uploaded (see `voxel::upload`).
#[derive(Component)]
pub struct ExtractedChunkField<T: Send + Sync + 'static> {
    pub main_entity: MainEntity,
    pub chunk_pos: IVec3,
    pub padded_data: Arc<[u8]>,
    pub size: u32,
    pub lod: LOD,
    /// See [`seam_lod_ranks`]; only the SDF upload uses it.
    pub lod_ranks: u32,
    /// Main-world frame it was extracted in. An upload can wait several
    /// frames; when a newer one for the same chunk arrives, the newest wins.
    pub seq: u32,
    pub _type: PhantomData<T>,
}

/// A chunk that lost its `VisibilityField`: sends an empty mask, which
/// clears the slot's mask flag.
pub fn extract_visibility_removed(
    mut commands: Commands,
    mut removed: Extract<RemovedComponents<VisibilityField>>,
    chunks: Extract<Query<(&ChunkPosition, &LOD, Has<VisibilityField>)>>,
    frame: Extract<Res<FrameCount>>,
) {
    for entity in removed.read() {
        let Ok((pos, lod, still_has)) = chunks.get(entity) else {
            continue;
        };
        if still_has {
            continue;
        }
        commands.spawn(ExtractedChunkField::<VisibilityField> {
            main_entity: entity.into(),
            chunk_pos: pos.0,
            padded_data: Arc::from([]),
            size: lod.size().wrapping_add(PADDING),
            lod: *lod,
            lod_ranks: 0,
            seq: frame.0,
            _type: PhantomData,
        });
    }
}

/// Forwards this frame's padded volumes of `T` to the render world.
pub fn extract_voxel_chunks<T>(mut commands: Commands, outbox: Extract<Res<PaddedOutbox<T>>>)
where
    T: Send + Sync + 'static,
{
    for c in &outbox.chunks {
        commands.spawn(ExtractedChunkField::<T> {
            main_entity: c.entity.into(),
            chunk_pos: c.pos,
            padded_data: Arc::clone(&c.bytes),
            size: c.lod.size().wrapping_add(PADDING),
            lod: c.lod,
            lod_ranks: c.lod_ranks,
            seq: c.seq,
            _type: PhantomData,
        });
    }
}

/// Debug check: every chunk whose padded volume is out of date must be
/// dirty. Anything missed is logged and made dirty, so it is also repaired.
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
// Live set: which chunks hold an arena slot
// ---------------------------------------------------------------------------

/// This frame's changes to the set of chunks that should hold an arena slot.
#[derive(Resource, Default)]
pub struct LiveOutbox {
    /// Chunks that despawned.
    pub despawned: Vec<Entity>,
    /// Chunks whose residency was re-evaluated, and whether they're wanted.
    pub changed: Vec<(Entity, bool)>,
}

/// What can change the set of chunks that should hold an arena slot.
#[derive(SystemParam)]
pub struct LiveChanges<'w, 's> {
    residency: ResidencyChanges<'w, 's>,
    despawned: RemovedComponents<'w, 's, Chunk>,
    lod_added: Query<'w, 's, &'static ChunkPosition, Added<LOD>>,
    chunks: Query<'w, 's, (), (With<Chunk>, With<LOD>)>,
    chunk_manager: Res<'w, ChunkManager>,
    filter: ExtractFilter<'w, 's>,
}

/// Records which chunks gained or lost their claim to an arena slot this
/// frame, for `release_despawned_chunks`.
pub fn track_live_chunks(
    mut changes: LiveChanges,
    mut outbox: ResMut<LiveOutbox>,
    mut dirty: Local<HashSet<IVec3>>,
) {
    outbox.despawned.clear();
    outbox.changed.clear();
    let despawned: Vec<Entity> = changes.despawned.read().collect();
    outbox.despawned.extend(despawned);

    dirty.clear();
    changes.residency.collect(&mut dirty);
    dirty.extend(changes.lod_added.iter().map(|p| p.0));
    for pos in &*dirty {
        let Some(e) = changes.chunk_manager.get_chunk(pos) else {
            continue;
        };
        let wanted = changes.chunks.contains(e) && !changes.filter.skips(e, *pos);
        outbox.changed.push((e, wanted));
    }
}

type LiveChunkData = (
    Entity,
    &'static ChunkPosition,
    Option<&'static VisibilityField>,
);
type LiveChunkFilter = (With<Chunk>, With<LOD>, With<HasSurface>);

/// The full live set, for the debug audit only.
#[derive(SystemParam)]
pub struct LiveAudit<'w, 's> {
    live: Extract<'w, 's, Query<'static, 'static, LiveChunkData, LiveChunkFilter>>,
    forced: Extract<'w, 's, Res<'static, ForceResident>>,
    frame: Extract<'w, 's, Res<'static, FrameCount>>,
}

/// Applies [`LiveOutbox`]: keeps `arena_set.live` current and releases the
/// slots of chunks that despawned, lost their surface or became hidden.
pub fn release_despawned_chunks(
    outbox: Extract<Res<LiveOutbox>>,
    audit: LiveAudit,
    mut arena_set: ResMut<VoxelChunkArenaSet>,
) {
    for &e in &outbox.despawned {
        drop_chunk(&mut arena_set, e.into());
    }
    for &(e, wanted) in &outbox.changed {
        if wanted {
            arena_set.live.insert(e.into());
        } else {
            drop_chunk(&mut arena_set, e.into());
        }
    }
    if cfg!(debug_assertions) && audit.frame.0.is_multiple_of(AUDIT_INTERVAL) {
        audit_live(&audit, &mut arena_set);
    }
}

fn drop_chunk(arena_set: &mut VoxelChunkArenaSet, e: MainEntity) {
    arena_set.live.remove(&e);
    arena_set.empty.remove(&e);
    for cache in arena_set.side_data.values_mut() {
        cache.remove(&e);
    }
    arena_set.release(e);
}

/// Debug check: `live` must match a full recount. Repairs and logs misses.
fn audit_live(audit: &LiveAudit, arena_set: &mut VoxelChunkArenaSet) {
    let wanted: std::collections::HashSet<MainEntity> = audit
        .live
        .iter()
        .filter(|(_, pos, v)| {
            !v.is_some_and(VisibilityField::is_all_empty) || audit.forced.contains(&pos.0)
        })
        .map(|(e, _, _)| MainEntity::from(e))
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
