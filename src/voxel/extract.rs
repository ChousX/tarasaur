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

use std::borrow::Cow;

use crate::{
    chunk::{Chunk, ChunkManager, ChunkPosition},
    field::{GpuChannel, LOD, SDF, VisibilityField, VoxelField},
    field::{
        apron::{apron_source, fill_apron_with, seam_lod_ranks},
        surface::HasSurface,
    },
    grid::{NEIGHBORS_MASK, PADDING, flatten_cube},
    num::{f32_from_u32, usize_from},
    voxel::{
        arena::VoxelChunkArenaSet,
        layer::{LayerResidency, SubSurface, VoxelLayers},
        resident::ForceResident,
    },
};

/// The surfaces in one padded volume, each as uploaded.
pub type SurfaceBytes = Vec<(SubSurface, Vec<u8>)>;

/// A neighbour's voxels (resampled to its upload LOD if need be) and size,
/// per `NEIGHBORS_MASK` entry.
type ApronSources<'a, E> = [Option<(Cow<'a, [E]>, u32)>; 7];

/// Each surface's signed distances from one volume, in its voxels.
pub type SurfaceDistances = Vec<(SubSurface, Vec<f32>)>;

/// `SurfaceLayer::SURFACES` for a field: from a chunk's voxels at a LOD,
/// each surface's signed distances in voxels of that LOD.
pub type Surfaces<T> = fn(&[<T as VoxelField>::Elem], LOD) -> SurfaceDistances;

/// Each neighbour's surfaces and size, per `NEIGHBORS_MASK` entry.
type NeighbourSurfaces = [Option<(SurfaceDistances, u32)>; 7];

/// Frames between full consistency checks in debug builds.
const AUDIT_INTERVAL: u32 = 120;

/// What a chunk's upload depends on: its LOD, its own version, each
/// `NEIGHBORS_MASK` neighbour's (border) version with that neighbour's LOD
/// folded in, and its `seam_lod_ranks` (every neighbour's LOD). Unchanged
/// key = unchanged upload.
type ExtractKey = (LOD, [u64; 8], [u32; 2]);

type FieldQuery<'w, 's, T> =
    Query<'w, 's, (Entity, &'static ChunkPosition, &'static T, &'static LOD)>;

/// Marks `pos` and its 26 neighbours: those whose apron reads `pos`
/// (`pos - offset`), and all of them stitch seams by its LOD.
fn dirty_with_dependents(dirty: &mut HashSet<IVec3>, pos: IVec3) {
    for z in -1..=1 {
        for y in -1..=1 {
            for x in -1..=1 {
                dirty.insert(pos.wrapping_add(IVec3::new(x, y, z)));
            }
        }
    }
}

/// How a layer's field is extracted: which chunks (`residency`), at what
/// LOD at most (`upload_lod`), and what a padded volume becomes
/// (`surfaces`).
pub struct ExtractConfig<T: VoxelField> {
    pub residency: LayerResidency,
    pub upload_lod: Option<LOD>,
    /// `None`: the field is its own single surface.
    pub surfaces: Option<Surfaces<T>>,
}

// By hand: a derive would ask `T: Copy`, and nothing here holds a `T`.
impl<T: VoxelField> Clone for ExtractConfig<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: VoxelField> Copy for ExtractConfig<T> {}

/// The LOD a chunk at `lod` is uploaded at under `cap`.
fn effective(lod: LOD, cap: Option<LOD>) -> LOD {
    cap.map_or(lod, |c| lod.min(c))
}

/// Registers the hand-off of field `T`: the main-world padding system and
/// the render-world extract system that forwards its output.
pub fn add_field_extraction<T>(app: &mut App, config: ExtractConfig<T>)
where
    T: VoxelField,
{
    app.insert_resource(PaddedOutbox::<T>::new(config))
        .add_systems(Last, pad_dirty_chunks::<T>);
    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app.add_systems(ExtractSchedule, extract_voxel_chunks::<T>);
    }
}

// ---------------------------------------------------------------------------
// Residency: which chunks are extracted at all
// ---------------------------------------------------------------------------

/// Which chunks a `LayerResidency::Terrain` layer doesn't extract: no
/// terrain surface, or fully hidden (unless forced resident). A `Loaded`
/// layer extracts every chunk.
#[derive(SystemParam)]
pub struct ExtractFilter<'w, 's> {
    surface: Query<'w, 's, (), With<HasSurface>>,
    visibility: Query<'w, 's, Option<&'static VisibilityField>>,
    forced: Res<'w, ForceResident>,
}

impl ExtractFilter<'_, '_> {
    fn skips(&self, residency: LayerResidency, entity: Entity, pos: IVec3) -> bool {
        if residency == LayerResidency::Loaded {
            return false;
        }
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

/// One of a chunk's surfaces, as a padded volume ready to upload.
pub struct PaddedChunk {
    pub entity: Entity,
    pub pos: IVec3,
    pub lod: LOD,
    /// Which surface; `SubSurface::ALL` with empty `bytes` releases every
    /// surface of the chunk, and with `meta_only` restitches them all.
    pub sub: SubSurface,
    /// See `seam_lod_ranks`.
    pub seam_ranks: [u32; 2],
    /// Only `seam_ranks` changed: `bytes` is empty and the slot keeps its
    /// data (see `pad_dirty_chunks`).
    pub meta_only: bool,
    /// Shared with the render world rather than copied.
    pub bytes: Arc<[u8]>,
    /// Main-world frame it was built in.
    pub seq: u32,
}

/// This frame's padded volumes of field `T`, for `extract_voxel_chunks`,
/// and how its layer extracts them.
#[derive(Resource)]
pub struct PaddedOutbox<T: VoxelField> {
    pub chunks: Vec<PaddedChunk>,
    pub config: ExtractConfig<T>,
}

impl<T: VoxelField> PaddedOutbox<T> {
    #[must_use]
    pub const fn new(config: ExtractConfig<T>) -> Self {
        Self {
            chunks: Vec::new(),
            config,
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
    /// The surfaces each chunk's last volume had, so one that disappears
    /// can be released.
    subs: EntityHashMap<Vec<SubSurface>>,
    dirty: HashSet<IVec3>,
}

impl ExtractState {
    fn collect<T: Component>(&mut self, changes: &mut FieldChanges<T>) {
        self.dirty.clear();
        // Removals first: a field removed and re-added this frame shows up in
        // both, and must end up recorded.
        for e in changes.removed.read() {
            self.last.remove(&e);
            self.subs.remove(&e);
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

/// `lod` is the chunk's own; the key holds the LOD it is uploaded at.
fn extract_key<T: VoxelField>(
    field: &T,
    pos: IVec3,
    lod: LOD,
    cap: Option<LOD>,
    chunk_manager: &ChunkManager,
    query: &FieldQuery<T>,
) -> ExtractKey {
    let eff = effective(lod, cap);
    let mut versions = [0u64; 8];
    let (own, neighbor_versions) = versions.split_at_mut(1);
    if let Some(own) = own.first_mut() {
        *own = field.version();
    }
    // Per axis, whether the face neighbour across it is coarser (as
    // uploaded): then the apron steps by its voxel along that axis (see
    // `fill_apron_with`).
    let coarse_face = [IVec3::X, IVec3::Y, IVec3::Z].map(|axis| {
        chunk_manager
            .get_chunk(&pos.wrapping_add(axis))
            .and_then(|e| query.get(e).ok())
            .filter(|(_, _, f, l)| f.data_slice().len() == l.volume())
            .is_some_and(|(_, _, _, l)| effective(*l, cap).size() < eff.size())
    });
    for ((i, offset), slot) in NEIGHBORS_MASK.iter().enumerate().zip(neighbor_versions) {
        if let Some(n_entity) = chunk_manager.get_chunk(&pos.wrapping_add(*offset))
            && let Ok((_, _, n_field, n_lod)) = query.get(n_entity)
        {
            // At equal LOD with unit steps the apron reads only the
            // neighbour's facing slab; otherwise (including a neighbour
            // resampled to the cap) it reads more of it, so use the full
            // version.
            let unit_steps = offset
                .to_array()
                .iter()
                .zip(coarse_face)
                .all(|(d, coarse)| *d == 0 || !coarse);
            let v = if *n_lod == eff && unit_steps {
                n_field.border_version(i)
            } else {
                n_field.version()
            };
            // fold neighbour LOD in: a reload resets version to 0
            *slot = v ^ (u64::from(n_lod.rank()).wrapping_add(1) << 56);
        }
    }
    (
        eff,
        versions,
        seam_lod_ranks(pos, lod, cap, chunk_manager, query),
    )
}

/// `data`, a `from` grid, resampled to a `to` grid (coarser or equal):
/// each voxel of `to` reads the voxel of `from` it lands on (interpolated
/// or nearest, as the field samples its apron) and is rescaled to `to`'s
/// voxel units.
pub fn resample<T: VoxelField>(data: &[T::Elem], from: LOD, to: LOD) -> Vec<T::Elem> {
    let (n, m) = (from.size(), to.size());
    if n == m {
        return data.to_vec();
    }
    let ratio = f32_from_u32(n) / f32_from_u32(m);
    let mut out = Vec::with_capacity(to.volume());
    for z in 0..m {
        for y in 0..m {
            for x in 0..m {
                let at = UVec3::new(x, y, z).as_vec3() * ratio;
                let v = T::sample_apron(data, n, at.x, at.y, at.z);
                out.push(T::rescale(v, ratio));
            }
        }
    }
    out
}

/// The chunk's field plus its `+x/+y/+z` apron, at the LOD it is
/// uploaded at, as its surfaces to upload. None when the field asks not to
/// be extracted.
fn padded_surfaces<T>(
    field: &T,
    pos: IVec3,
    lod: LOD,
    chunk_manager: &ChunkManager,
    query: &FieldQuery<T>,
    config: ExtractConfig<T>,
) -> SurfaceBytes
where
    T: VoxelField,
{
    if !field.should_extract() {
        return Vec::new();
    }
    let eff = effective(lod, config.upload_lod);
    let own: Cow<[T::Elem]> = if eff == lod {
        Cow::Borrowed(field.data_slice())
    } else {
        Cow::Owned(resample::<T>(field.data_slice(), lod, eff))
    };
    // The neighbours, each at the LOD it is uploaded at.
    let sources: ApronSources<T::Elem> = NEIGHBORS_MASK.map(|offset| {
        let (f, l) = apron_source(pos.wrapping_add(offset), chunk_manager, query)?;
        let n_eff = effective(l, config.upload_lod);
        Some(if n_eff == l {
            (Cow::Borrowed(f.data_slice()), l.size())
        } else {
            (
                Cow::Owned(resample::<T>(f.data_slice(), l, n_eff)),
                n_eff.size(),
            )
        })
    });
    let Some(surfaces) = config.surfaces else {
        // The field is its own surface: pad it as it is.
        let mut vol = padded::<T>(&own, eff.size());
        fill_apron_with::<T>(&mut vol, eff.size(), |offset| {
            let i = NEIGHBORS_MASK.iter().position(|o| *o == offset)?;
            sources.get(i)?.as_ref().map(|(d, n)| (d.as_ref(), *n))
        });
        return vec![(SubSurface(0), T::gpu_bytes(&vol))];
    };
    // Each surface's distances, from the chunk and from every neighbour in
    // its own LOD; then each is padded as an SDF.
    let own_surfaces = surfaces(&own, eff);
    let neighbour_surfaces: NeighbourSurfaces = std::array::from_fn(|i| {
        let (d, n) = sources.get(i)?.as_ref()?;
        let n_lod = LOD::ALL.into_iter().find(|l| l.size() == *n)?;
        Some((surfaces(d, n_lod), *n))
    });
    own_surfaces
        .into_iter()
        .map(|(sub, distances)| {
            let mut vol = padded::<SDF>(&distances, eff.size());
            fill_apron_with::<SDF>(&mut vol, eff.size(), |offset| {
                let i = NEIGHBORS_MASK.iter().position(|o| *o == offset)?;
                let (subs, n) = neighbour_surfaces.get(i)?.as_ref()?;
                let (_, d) = subs.iter().find(|(s, _)| *s == sub)?;
                Some((d.as_slice(), *n))
            });
            (sub, SDF::gpu_bytes(&vol))
        })
        .collect()
}

/// `data` (`size`^3, x fastest) in a padded volume, its apron still
/// default.
fn padded<T: VoxelField>(data: &[T::Elem], size: u32) -> Vec<T::Elem> {
    let padded_size = size.wrapping_add(PADDING);
    let mut vol = vec![T::Elem::default(); usize_from(padded_size).saturating_pow(3)];
    // Copy row by row: each source row of x lands at the start of the
    // same (y, z) row in the padded volume.
    let row_len = usize_from(size);
    let mut src_rows = data.chunks_exact(row_len);
    for z in 0..size {
        for y in 0..size {
            let dst = flatten_cube(0, y, z, padded_size);
            let dst_row = vol.get_mut(dst..dst.wrapping_add(row_len));
            if let (Some(src), Some(dst)) = (src_rows.next(), dst_row) {
                dst.copy_from_slice(src);
            }
        }
    }
    vol
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
    T: VoxelField,
{
    outbox.chunks.clear();
    let config = outbox.config;
    let residency = config.residency;
    state.collect(&mut changes);
    if cfg!(debug_assertions) && frame.0.is_multiple_of(AUDIT_INTERVAL) {
        audit_extraction(&mut state, &chunk_manager, &query, &filter, config);
    }

    // Which chunks need a new volume: cheap, sequential.
    let ExtractState {
        last, dirty, subs, ..
    } = &mut *state;
    let mut jobs = Vec::new();
    let mut meta_only = Vec::new();
    for pos in dirty.iter() {
        let Some(entity) = chunk_manager.get_chunk(pos) else {
            continue;
        };
        let Ok((_, _, field, lod)) = query.get(entity) else {
            last.remove(&entity);
            continue;
        };
        if filter.skips(residency, entity, *pos) {
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
        let key = extract_key(field, *pos, *lod, config.upload_lod, &chunk_manager, &query);
        let Some(old) = last.insert(entity, key) else {
            jobs.push((entity, *pos, field, *lod, key.2));
            continue;
        };
        if old == key {
            continue;
        }
        // Same LOD and versions: the padded volume would come out the same
        // (it depends on the neighbours' LODs only through the versions, which
        // fold them in). Only the seam ranks moved, as they do for every chunk
        // around one that changed LOD, and only the mesher reads those.
        if (old.0, old.1) == (key.0, key.1) {
            if T::GPU == GpuChannel::Shape {
                meta_only.push((entity, *pos, key.0, key.2));
            }
            continue;
        }
        jobs.push((entity, *pos, field, *lod, key.2));
    }

    // Building the volumes: the expensive part, one task per chunk.
    let (manager, fields) = (&*chunk_manager, &query);
    let mut built: Vec<(usize, SurfaceBytes)> = ComputeTaskPool::get().scope(|s| {
        for (i, &(_, pos, field, lod, _)) in jobs.iter().enumerate() {
            s.spawn(async move {
                let out = padded_surfaces(field, pos, lod, manager, fields, config);
                (i, out)
            });
        }
    });
    built.sort_unstable_by_key(|b| b.0);
    for (&(entity, pos, _, lod, seam_ranks), (_, now)) in jobs.iter().zip(built) {
        let lod = effective(lod, config.upload_lod);
        let chunk = |sub: SubSurface, bytes: Vec<u8>| PaddedChunk {
            entity,
            pos,
            lod,
            sub,
            seam_ranks,
            meta_only: false,
            bytes: bytes.into(),
            seq: frame.0,
        };
        let before = subs.remove(&entity).unwrap_or_default();
        if now.is_empty() {
            // Nothing to show: release whatever the chunk had.
            if !before.is_empty() {
                outbox.chunks.push(chunk(SubSurface::ALL, Vec::new()));
            }
            continue;
        }
        let ids: Vec<SubSurface> = now.iter().map(|(sub, _)| *sub).collect();
        for (sub, bytes) in now {
            outbox.chunks.push(chunk(sub, bytes));
        }
        // A surface the chunk no longer has releases its slot.
        for &gone in before.iter().filter(|s| !ids.contains(s)) {
            outbox.chunks.push(chunk(gone, Vec::new()));
        }
        subs.insert(entity, ids);
    }
    outbox.chunks.extend(
        meta_only
            .into_iter()
            .map(|(entity, pos, lod, seam_ranks)| PaddedChunk {
                entity,
                pos,
                lod,
                sub: SubSurface::ALL,
                seam_ranks,
                meta_only: true,
                bytes: Arc::from([]),
                seq: frame.0,
            }),
    );
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
    /// See [`PaddedChunk::sub`].
    pub sub: SubSurface,
    /// See [`seam_lod_ranks`]; only the SDF upload uses it.
    pub seam_ranks: [u32; 2],
    /// See [`PaddedChunk::meta_only`].
    pub meta_only: bool,
    /// Main-world frame it was extracted in. An upload can wait several
    /// frames; when a newer one for the same chunk arrives, the newest wins.
    pub seq: u32,
    pub _type: PhantomData<T>,
}

/// A chunk that lost an optional field `T` (it's still loaded, the field is
/// gone): sends an empty volume, which for the mask clears the slot's flag.
pub fn extract_removed<T: VoxelField>(
    mut commands: Commands,
    mut removed: Extract<RemovedComponents<T>>,
    chunks: Extract<Query<(&ChunkPosition, &LOD, Has<T>)>>,
    frame: Extract<Res<FrameCount>>,
) {
    for entity in removed.read() {
        let Ok((pos, lod, still_has)) = chunks.get(entity) else {
            continue;
        };
        if still_has {
            continue;
        }
        commands.spawn(ExtractedChunkField::<T> {
            main_entity: entity.into(),
            chunk_pos: pos.0,
            padded_data: Arc::from([]),
            size: lod.size().wrapping_add(PADDING),
            lod: *lod,
            sub: SubSurface::ALL,
            seam_ranks: [0; 2],
            meta_only: false,
            seq: frame.0,
            _type: PhantomData,
        });
    }
}

/// Forwards this frame's padded volumes of `T` to the render world.
pub fn extract_voxel_chunks<T>(mut commands: Commands, outbox: Extract<Res<PaddedOutbox<T>>>)
where
    T: VoxelField,
{
    for c in &outbox.chunks {
        commands.spawn(ExtractedChunkField::<T> {
            main_entity: c.entity.into(),
            chunk_pos: c.pos,
            padded_data: Arc::clone(&c.bytes),
            size: c.lod.size().wrapping_add(PADDING),
            lod: c.lod,
            sub: c.sub,
            seam_ranks: c.seam_ranks,
            meta_only: c.meta_only,
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
    config: ExtractConfig<T>,
) where
    T: VoxelField,
{
    let residency = config.residency;
    let mut missed = Vec::new();
    for (entity, pos, field, lod) in query.iter() {
        if state.dirty.contains(&pos.0) || state.positions.get(&entity) != Some(&pos.0) {
            if !state.dirty.contains(&pos.0) {
                missed.push(pos.0); // never recorded
            }
            continue;
        }
        let out_of_date = if filter.skips(residency, entity, pos.0) {
            state.last.contains_key(&entity)
        } else {
            field.data_slice().len() == lod.volume()
                && state.last.get(&entity)
                    != Some(&extract_key(
                        field,
                        pos.0,
                        *lod,
                        config.upload_lod,
                        chunk_manager,
                        query,
                    ))
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

/// A chunk whose residency was re-evaluated this frame.
#[derive(Clone, Copy)]
pub struct LiveChange {
    pub entity: Entity,
    /// Loaded (`Chunk` with an `LOD`): what a `LayerResidency::Loaded`
    /// layer wants.
    pub loaded: bool,
    /// Loaded, with a terrain surface and not hidden: what a
    /// `LayerResidency::Terrain` layer wants.
    pub terrain: bool,
}

impl LiveChange {
    const fn wanted(self, residency: LayerResidency) -> bool {
        match residency {
            LayerResidency::Loaded => self.loaded,
            LayerResidency::Terrain => self.terrain,
        }
    }
}

/// This frame's changes to the set of chunks that should hold an arena slot.
#[derive(Resource, Default)]
pub struct LiveOutbox {
    /// Chunks that despawned.
    pub despawned: Vec<Entity>,
    /// Chunks whose residency was re-evaluated.
    pub changed: Vec<LiveChange>,
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
        let loaded = changes.chunks.contains(e);
        let terrain = loaded && !changes.filter.skips(LayerResidency::Terrain, e, *pos);
        outbox.changed.push(LiveChange {
            entity: e,
            loaded,
            terrain,
        });
    }
}

type LiveChunkData = (
    Entity,
    &'static ChunkPosition,
    Option<&'static VisibilityField>,
);
type LiveChunkFilter = (With<Chunk>, With<LOD>, With<HasSurface>);
type LoadedChunks = Query<'static, 'static, Entity, (With<Chunk>, With<LOD>)>;

/// The full live set, for the debug audit only.
#[derive(SystemParam)]
pub struct LiveAudit<'w, 's> {
    live: Extract<'w, 's, Query<'static, 'static, LiveChunkData, LiveChunkFilter>>,
    loaded: Extract<'w, 's, LoadedChunks>,
    forced: Extract<'w, 's, Res<'static, ForceResident>>,
    frame: Extract<'w, 's, Res<'static, FrameCount>>,
}

/// Applies [`LiveOutbox`] to every layer: keeps its `live` set current and
/// releases the slots of chunks that despawned or, by the layer's
/// residency, lost their surface or became hidden.
pub fn release_despawned_chunks(
    outbox: Extract<Res<LiveOutbox>>,
    audit: LiveAudit,
    mut layers: ResMut<VoxelLayers>,
) {
    for (_, layer) in layers.iter_mut() {
        let (residency, arena_set) = (layer.residency, &mut layer.set);
        for &e in &outbox.despawned {
            drop_chunk(arena_set, e.into());
        }
        for change in &outbox.changed {
            if change.wanted(residency) {
                arena_set.live.insert(change.entity.into());
            } else {
                drop_chunk(arena_set, change.entity.into());
            }
        }
        if cfg!(debug_assertions) && audit.frame.0.is_multiple_of(AUDIT_INTERVAL) {
            audit_live(&audit, arena_set, residency);
        }
    }
}

fn drop_chunk(arena_set: &mut VoxelChunkArenaSet, e: MainEntity) {
    arena_set.live.remove(&e);
    arena_set.empty.retain(|k| k.entity != e);
    for cache in arena_set.side_data.values_mut() {
        cache.remove(&e);
    }
    arena_set.release_entity(e);
}

/// Debug check: `live` must match a full recount. Repairs and logs misses.
fn audit_live(audit: &LiveAudit, arena_set: &mut VoxelChunkArenaSet, residency: LayerResidency) {
    let wanted: std::collections::HashSet<MainEntity> = match residency {
        LayerResidency::Loaded => audit.loaded.iter().map(MainEntity::from).collect(),
        LayerResidency::Terrain => audit
            .live
            .iter()
            .filter(|(_, pos, v)| {
                !v.is_some_and(VisibilityField::is_all_empty) || audit.forced.contains(&pos.0)
            })
            .map(|(e, _, _)| MainEntity::from(e))
            .collect(),
    };
    let held = arena_set
        .iter()
        .flat_map(|a| a.subs_of_entity.keys().copied());
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

#[cfg(test)]
mod tests {
    use bevy::prelude::*;

    use super::{ExtractConfig, PaddedOutbox, add_field_extraction, resample};
    use crate::{
        chunk::{Chunk, ChunkManager, ChunkPosition},
        field::{Field, GpuChannel, LOD, VoxelField, surface::HasSurface},
        grid::flatten_cube,
        voxel::{
            layer::{LayerResidency, SubSurface},
            resident::ForceResident,
        },
    };

    /// A field whose surfaces are the distinct nonzero ids it holds.
    #[derive(Component, Clone)]
    struct Owners {
        lod: LOD,
        data: Vec<u8>,
        version: u64,
    }

    impl Field<u8> for Owners {
        fn size(&self) -> UVec3 {
            UVec3::splat(self.lod.size())
        }
        fn get(&self, x: u32, y: u32, z: u32) -> u8 {
            self.data
                .get(flatten_cube(x, y, z, self.lod.size()))
                .copied()
                .unwrap_or(0)
        }
        fn set(&mut self, x: u32, y: u32, z: u32, value: u8) {
            if let Some(v) = self.data.get_mut(flatten_cube(x, y, z, self.lod.size())) {
                *v = value;
                self.version = self.version.wrapping_add(1);
            }
        }
    }

    impl VoxelField for Owners {
        type Value = u8;
        type Elem = u8;
        const GPU: GpuChannel = GpuChannel::Shape;
        fn new_for_lod(lod: LOD) -> Self {
            Self::from_raw(lod, vec![0; lod.volume()])
        }
        fn from_raw(lod: LOD, data: Vec<u8>) -> Self {
            Self {
                lod,
                data,
                version: 0,
            }
        }
        fn lod(&self) -> LOD {
            self.lod
        }
        fn data_slice(&self) -> &[u8] {
            &self.data
        }
        fn version(&self) -> u64 {
            self.version
        }
    }

    /// One surface per owner present: inside its voxels, outside the rest.
    fn owners(vol: &[u8], _: LOD) -> Vec<(SubSurface, Vec<f32>)> {
        let mut ids: Vec<u8> = vol.iter().copied().filter(|&v| v != 0).collect();
        ids.sort_unstable();
        ids.dedup();
        ids.into_iter()
            .map(|id| {
                let d = vol
                    .iter()
                    .map(|&v| if v == id { -1.0 } else { 1.0 })
                    .collect();
                (SubSurface(id), d)
            })
            .collect()
    }

    fn outbox(app: &App) -> Vec<(SubSurface, bool)> {
        let mut out: Vec<(SubSurface, bool)> = app
            .world()
            .resource::<PaddedOutbox<Owners>>()
            .chunks
            .iter()
            .map(|c| (c.sub, c.bytes.is_empty()))
            .collect();
        out.sort_unstable();
        out
    }

    /// Each owner in a chunk is extracted as its own surface; an owner that
    /// leaves the chunk is released, and the last one leaving releases all.
    #[test]
    fn surfaces_come_and_go_with_their_owners() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .init_resource::<ChunkManager>()
            .init_resource::<ForceResident>();
        add_field_extraction::<Owners>(
            &mut app,
            ExtractConfig {
                residency: LayerResidency::Loaded,
                upload_lod: None,
                surfaces: Some(owners),
            },
        );
        let chunk = app
            .world_mut()
            .spawn((
                Chunk,
                ChunkPosition(IVec3::ZERO),
                LOD::Lowest,
                HasSurface,
                Owners::new_for_lod(LOD::Lowest),
            ))
            .id();
        let set = |app: &mut App, cells: &[(u32, u8)]| {
            let mut field = app.world_mut().get_mut::<Owners>(chunk).unwrap();
            for &(x, id) in cells {
                field.set(x, 0, 0, id);
            }
        };

        set(&mut app, &[(0, 1), (1, 2)]);
        app.update();
        assert_eq!(
            outbox(&app),
            vec![(SubSurface(1), false), (SubSurface(2), false)]
        );

        // Nothing changed: nothing is sent.
        app.update();
        assert_eq!(outbox(&app), vec![]);

        // Owner 2 leaves: its surface is released, owner 1's re-sent.
        set(&mut app, &[(1, 0)]);
        app.update();
        assert_eq!(
            outbox(&app),
            vec![(SubSurface(1), false), (SubSurface(2), true)]
        );

        // The last owner leaves: everything is released at once.
        set(&mut app, &[(0, 0)]);
        app.update();
        assert_eq!(outbox(&app), vec![(SubSurface::ALL, true)]);

        // And with nothing held, an unchanged empty field sends nothing.
        set(&mut app, &[(2, 0)]);
        app.update();
        assert_eq!(outbox(&app), vec![]);
    }

    /// A planar SDF downsampled 4x reads the same plane in the coarser
    /// grid's voxel units.
    #[test]
    fn resample_keeps_a_plane() {
        use crate::field::SDF;
        let (from, to) = (LOD::Low, LOD::Lowest); // 16 -> 4 voxels
        let plane = |p: Vec3| p.dot(Vec3::new(0.25, 0.5, -0.125)) - 3.0;
        let mut data = vec![0.0f32; from.volume()];
        for z in 0..from.size() {
            for y in 0..from.size() {
                for x in 0..from.size() {
                    data[flatten_cube(x, y, z, from.size())] = plane(UVec3::new(x, y, z).as_vec3());
                }
            }
        }
        let coarse = resample::<SDF>(&data, from, to);
        assert_eq!(coarse.len(), to.volume());
        for z in 0..to.size() {
            for y in 0..to.size() {
                for x in 0..to.size() {
                    // Coarse voxel (x, y, z) sits on fine voxel 4(x, y, z),
                    // and its distances are 4x fewer, coarser voxels.
                    let want = plane(UVec3::new(x, y, z).as_vec3() * 4.0) / 4.0;
                    let got = coarse[flatten_cube(x, y, z, to.size())];
                    assert!(
                        (got - want).abs() < 1e-5,
                        "({x}, {y}, {z}): {got} vs {want}"
                    );
                }
            }
        }
    }
}
