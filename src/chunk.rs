//! The chunk grid and what decides which chunks exist.
//!
//! A chunk is an entity (`Chunk` + `ChunkPosition`) covering a
//! `CHUNK_SIZE`^3 cube; `ChunkManager` maps grid positions to entities.
//! Each frame `plan_chunk_ops` compares every `ChunkLoader`'s LOD bands with
//! the live chunks: despawns happen at once, spawns and `TargetLOD` changes
//! queue nearest-first and `apply_chunk_ops` drains them under a budget.
//! The chunk under the cursor counts as near (`CursorFocusChunk`), and
//! chunks being edited are held at max LOD (`EditPin`).

use bevy::{
    camera::primitives::Aabb,
    ecs::{lifecycle::HookContext, world::DeferredWorld},
    platform::collections::HashMap,
    prelude::*,
};
use std::collections::VecDeque;

use crate::{LOD, TargetLOD, voxel::cursor::CursorHit};

pub struct ChunkPlugin;
impl Plugin for ChunkPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ChunkManager>()
            .init_resource::<CursorFocusChunk>()
            .init_resource::<ChunkOpQueue>()
            .init_resource::<MaxChunkOpsPerFrame>()
            .init_resource::<EditPinTimeout>()
            .init_resource::<ReplanRequested>()
            .add_observer(new_chunk_spawned)
            .add_systems(Startup, configure_gizmo_depth_bias)
            .add_systems(
                Update,
                (
                    chunk_loader_boundary_checker,
                    expire_edit_pins,
                    update_cursor_focus, // must run before plan_chunk_ops
                    plan_chunk_ops.run_if(loaders_dirty),
                    apply_chunk_ops,
                )
                    .chain(),
            )
            .add_systems(
                Update,
                chunk_boundary_visualizer.run_if(resource_exists::<ShowChunkBounds>),
            );
    }
}

/// Chunk edge length in world units, at every LOD.
pub const CHUNK_SIZE: f32 = 10.;
/// Chunks within this Chebyshev radius of the cursor chunk are treated as
/// being at (at most) that distance from every loader.
pub const FOCUS_RADIUS: i32 = 1;

#[derive(Resource, Default)]
pub struct ShowChunkBounds;

/// Written by `update_cursor_focus` from the resolved `CursorHit`. Only
/// assign when the chunk actually changes so `is_changed` stays meaningful.
#[derive(Resource, Default, PartialEq, Eq, Clone, Copy)]
pub struct CursorFocusChunk(pub Option<IVec3>);

/// Max spawns + LOD retargets applied per frame, nearest first.
#[derive(Resource)]
pub struct MaxChunkOpsPerFrame(pub usize);
impl Default for MaxChunkOpsPerFrame {
    fn default() -> Self {
        Self(64)
    }
}

/// Grid position -> chunk entity. Kept in sync by `Chunk`'s hooks.
#[derive(Resource, Clone, Default)]
pub struct ChunkManager {
    arena: HashMap<IVec3, Entity>,
}

impl ChunkManager {
    #[must_use]
    pub fn get_chunk(&self, position: &IVec3) -> Option<Entity> {
        self.arena.get(position).copied()
    }
    #[must_use]
    pub fn is_loaded(&self, position: &IVec3) -> bool {
        self.arena.contains_key(position)
    }
    fn add_chunk(&mut self, position: IVec3, id: Entity) {
        self.arena.insert(position, id);
    }
    fn remove_chunk(&mut self, position: &IVec3) {
        self.arena.remove(position);
    }
}

#[inline]
#[must_use]
pub fn world_pos_to_chunk_pos(world_position: &Vec3) -> IVec3 {
    (world_position / CHUNK_SIZE).floor().as_ivec3()
}

/// Every chunk position overlapping the world-space box `center +/- half_extent`.
pub fn overlapping_chunks(center: Vec3, half_extent: Vec3) -> impl Iterator<Item = IVec3> {
    let min = world_pos_to_chunk_pos(&(center - half_extent));
    let max = world_pos_to_chunk_pos(&(center + half_extent));
    (min.x..=max.x).flat_map(move |x| {
        (min.y..=max.y).flat_map(move |y| (min.z..=max.z).map(move |z| IVec3::new(x, y, z)))
    })
}

/// No `LOD` here on purpose: `LOD` + fields are inserted together by the
/// loading pipeline once data for `TargetLOD` is ready.
#[derive(Component, Default, Clone, Copy)]
#[require(
    ChunkPosition,
    Visibility,
    Aabb::from_min_max(Vec3::ZERO, Vec3::splat(CHUNK_SIZE))
)]
#[component(immutable, on_add = on_add_chunk, on_remove = on_remove_chunk)]
pub struct Chunk;

fn on_add_chunk(mut world: DeferredWorld, HookContext { entity, .. }: HookContext) {
    // ChunkPosition is a required component; ChunkManager is added by ChunkPlugin.
    let Some(&ChunkPosition(chunk_pos)) = world.get::<ChunkPosition>(entity) else {
        return;
    };
    let Some(mut chunk_manager) = world.get_resource_mut::<ChunkManager>() else {
        return;
    };
    if chunk_manager.is_loaded(&chunk_pos) {
        warn!(
            "New chunk at pos:{} was despawned, there was already a chunk there",
            chunk_pos
        );
        world.commands().entity(entity).despawn();
        return;
    }
    chunk_manager.add_chunk(chunk_pos, entity);
}

fn on_remove_chunk(mut world: DeferredWorld, HookContext { entity, .. }: HookContext) {
    let Some(&ChunkPosition(chunk_pos)) = world.get::<ChunkPosition>(entity) else {
        return;
    };
    if let Some(mut chunk_manager) = world.get_resource_mut::<ChunkManager>() {
        // A rejected duplicate never owned the position: leave its owner mapped.
        if chunk_manager.get_chunk(&chunk_pos) == Some(entity) {
            chunk_manager.remove_chunk(&chunk_pos);
        }
    }
}

/// Grid coordinate; sets the `Transform` to the chunk's min corner.
#[derive(Component, Default, Deref, DerefMut)]
#[require(Transform)]
#[component(immutable, on_add = on_add_chunk_pos)]
pub struct ChunkPosition(pub IVec3);

fn on_add_chunk_pos(mut world: DeferredWorld, HookContext { entity, .. }: HookContext) {
    let Some(chunk_pos) = world.get::<ChunkPosition>(entity) else {
        return;
    };
    let translation = chunk_pos.as_vec3() * CHUNK_SIZE;
    // Transform is a required component of ChunkPosition.
    if let Some(mut transform) = world.get_mut::<Transform>(entity) {
        transform.translation = translation;
    }
}

/// Triggered when a `Chunk` is added; field observers attach defaults here.
#[derive(Event)]
pub struct NewChunkSpawned {
    pub entity: Entity,
    pub world_position: Vec3,
    pub chunk_position: IVec3,
}

fn new_chunk_spawned(
    trigger: On<Add<Chunk>>,
    chunk_q: Query<(&Transform, &ChunkPosition), With<Chunk>>,
    mut commands: Commands,
) {
    let Ok((transform, &ChunkPosition(chunk_position))) = chunk_q.get(trigger.entity) else {
        return;
    };
    commands.trigger(NewChunkSpawned {
        entity: trigger.entity,
        world_position: transform.translation,
        chunk_position,
    });
}

// ============================================================================
// Loader
// ============================================================================

/// The chunk a loader is in; changing it triggers a replan.
#[derive(Default, Deref, DerefMut, Component)]
pub struct CurrentChunk(pub IVec3);

/// `lod_bands`: finest first; each width is ADDITIVE.
///
/// Distance is Chebyshev, in chunks. `[(High,1),(Medium,1)]` => High for
/// d<=1, Medium for d==2, nothing beyond. `hysteresis` extends both LOD downgrades and despawn: an
/// existing chunk keeps its finer LOD until d > `band_edge` + hysteresis, and
/// is despawned only when d > total + hysteresis. Upgrades and new spawns
/// use the plain edges.
#[derive(Component, Clone, Debug)]
#[require(CurrentChunk)]
#[component(on_add = on_add_chunk_loader)]
pub struct ChunkLoader {
    pub lod_bands: Vec<(LOD, u8)>,
    pub hysteresis: u8,
}

impl Default for ChunkLoader {
    fn default() -> Self {
        Self {
            lod_bands: vec![
                (LOD::High, 1),
                (LOD::Medium, 2),
                (LOD::Low, 3),
                (LOD::Lowest, 4),
            ],
            hysteresis: 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Want {
    Lod(LOD),
    /// Inside the hysteresis margin: leave the chunk exactly as it is.
    Keep,
    Drop,
}

impl ChunkLoader {
    #[must_use]
    pub fn total_range(&self) -> i32 {
        self.lod_bands.iter().map(|b| i32::from(b.1)).sum()
    }

    /// `current` is `Some` for chunks that already exist.
    pub fn want(&self, d: i32, current: Option<LOD>) -> Want {
        let h = i32::from(self.hysteresis);
        let mut acc = 0i32;
        let mut ideal = None;
        let mut current_edge = None;
        for &(lod, w) in &self.lod_bands {
            acc = acc.saturating_add(i32::from(w));
            if ideal.is_none() && d <= acc {
                ideal = Some(lod);
            }
            if current == Some(lod) && current_edge.is_none() {
                current_edge = Some(acc);
            }
        }
        let Some(&(last, _)) = self.lod_bands.last() else {
            return Want::Drop;
        };
        let outside = ideal.is_none();

        let Some(cur) = current else {
            return ideal.map_or(Want::Drop, Want::Lod);
        };
        if outside && d > acc.saturating_add(h) {
            return Want::Drop;
        }
        let ideal = ideal.unwrap_or(last);
        if ideal > cur {
            return if outside {
                Want::Keep
            } else {
                Want::Lod(ideal)
            }; // instant upgrade
        }
        if ideal == cur {
            return Want::Lod(cur);
        }
        match current_edge {
            Some(e) if d <= e.saturating_add(h) => Want::Lod(cur), // hysteresis: hold the finer LOD
            _ => Want::Lod(ideal),
        }
    }
}

fn on_add_chunk_loader(mut world: DeferredWorld, HookContext { entity, .. }: HookContext) {
    let Some(chunk_pos) = world
        .get::<GlobalTransform>(entity)
        .map(GlobalTransform::translation)
    else {
        return;
    };
    if let Some(mut current) = world.get_mut::<CurrentChunk>(entity) {
        current.0 = world_pos_to_chunk_pos(&chunk_pos);
    }
}

fn chunk_loader_boundary_checker(
    mut q: Query<(&GlobalTransform, &mut CurrentChunk), Changed<GlobalTransform>>,
) {
    for (transform, mut current) in q.iter_mut() {
        let new_pos = world_pos_to_chunk_pos(&transform.translation());
        if current.0 != new_pos {
            current.0 = new_pos; // marks Changed<CurrentChunk>, which wakes plan_chunk_ops
        }
    }
}

/// A loader that changed chunk or config.
type LoaderMoved = Or<(Changed<CurrentChunk>, Changed<ChunkLoader>)>;

fn loaders_dirty(
    loaders: Query<(), LoaderMoved>,
    focus: Res<CursorFocusChunk>,
    replan: Res<ReplanRequested>,
) -> bool {
    !loaders.is_empty() || focus.is_changed() || replan.0
}

/// Chebyshev distance in chunks, saturating at `i32::MAX`.
pub(crate) fn chebyshev(a: IVec3, b: IVec3) -> i32 {
    let d = a.saturating_sub(b);
    let max =
        d.x.unsigned_abs()
            .max(d.y.unsigned_abs())
            .max(d.z.unsigned_abs());
    i32::try_from(max).unwrap_or(i32::MAX)
}

fn effective_distance(
    pos: IVec3,
    center: IVec3,
    loader: &ChunkLoader,
    focus: Option<IVec3>,
) -> i32 {
    let d = chebyshev(pos, center);
    if let Some(f) = focus {
        let fd = chebyshev(pos, f);
        // Focus only promotes chunks that are already in this loader's range.
        let range = loader
            .total_range()
            .saturating_add(i32::from(loader.hysteresis));
        if fd <= FOCUS_RADIUS && d <= range {
            return d.min(fd);
        }
    }
    d
}

#[derive(Clone, Copy)]
pub struct ChunkOp {
    dist: i32,
    pos: IVec3,
    lod: LOD,
}

#[derive(Resource, Default)]
pub struct ChunkOpQueue(VecDeque<ChunkOp>);

/// Recomputes the desired state. Despawns are immediate; spawns and LOD
/// changes go into a nearest-first queue drained by `apply_chunk_ops`.
/// Multiple loaders: finest wanted LOD wins; a chunk despawns only when
/// every loader wants it gone.
fn plan_chunk_ops(
    loaders: Query<(&ChunkLoader, &CurrentChunk)>,
    chunks: Query<(Entity, &ChunkPosition, &TargetLOD, Has<EditPin>), With<Chunk>>,
    mut replan: ResMut<ReplanRequested>,
    focus: Res<CursorFocusChunk>,
    chunk_manager: Res<ChunkManager>,
    mut queue: ResMut<ChunkOpQueue>,
    mut commands: Commands,
) {
    replan.0 = false;
    queue.0.clear();
    if loaders.is_empty() {
        return; // no loader => don't wipe the world
    }
    let mut ops: Vec<ChunkOp> = Vec::new();

    // Existing chunks: retarget or despawn.
    for (entity, pos, target, pinned) in &chunks {
        if pinned {
            continue;
        }
        let (mut best, mut keep, mut dist) = (None::<LOD>, false, i32::MAX);
        for (loader, center) in &loaders {
            let d = effective_distance(pos.0, center.0, loader, focus.0);
            dist = dist.min(d);
            match loader.want(d, Some(target.0)) {
                Want::Lod(l) => best = Some(best.map_or(l, |b| b.max(l))),
                Want::Keep => keep = true,
                Want::Drop => {}
            }
        }
        match best {
            Some(l) if l != target.0 => ops.push(ChunkOp {
                dist,
                pos: pos.0,
                lod: l,
            }),
            Some(_) => {}
            None if keep => {}
            None => {
                commands.queue(move |world: &mut World| {
                    crate::persistence::flush_and_despawn(world, entity);
                });
            }
        }
    }

    // Missing chunks: spawn candidates inside each loader's box.
    let mut spawns: HashMap<IVec3, (i32, LOD)> = HashMap::default();
    for (loader, center) in &loaders {
        let r = loader.total_range();
        for dz in r.saturating_neg()..=r {
            for dy in r.saturating_neg()..=r {
                for dx in r.saturating_neg()..=r {
                    let p = center.0.saturating_add(ivec3(dx, dy, dz));
                    if chunk_manager.is_loaded(&p) {
                        continue;
                    }
                    let d = effective_distance(p, center.0, loader, focus.0);
                    if let Want::Lod(l) = loader.want(d, None) {
                        spawns
                            .entry(p)
                            .and_modify(|e| {
                                e.0 = e.0.min(d);
                                e.1 = e.1.max(l);
                            })
                            .or_insert((d, l));
                    }
                }
            }
        }
    }
    ops.extend(
        spawns
            .into_iter()
            .map(|(pos, (dist, lod))| ChunkOp { dist, pos, lod }),
    );
    ops.sort_unstable_by_key(|o| o.dist);
    queue.0 = ops.into();
}

fn apply_chunk_ops(
    mut queue: ResMut<ChunkOpQueue>,
    budget: Res<MaxChunkOpsPerFrame>,
    chunk_manager: Res<ChunkManager>,
    targets: Query<(&TargetLOD, Has<EditPin>)>,
    mut commands: Commands,
) {
    for _ in 0..budget.0 {
        let Some(op) = queue.0.pop_front() else { break };
        match chunk_manager.get_chunk(&op.pos) {
            None => {
                commands.spawn((Chunk, ChunkPosition(op.pos), TargetLOD(op.lod)));
            }
            Some(e) => {
                if targets
                    .get(e)
                    .is_ok_and(|(t, pinned)| !pinned && t.0 != op.lod)
                {
                    commands.entity(e).insert(TargetLOD(op.lod));
                }
            }
        }
    }
}

// ============================================================================
// Visualizer
// ============================================================================

fn configure_gizmo_depth_bias(mut config_store: ResMut<GizmoConfigStore>) {
    let (config, _) = config_store.config_mut::<DefaultGizmoConfigGroup>();
    config.depth_bias = -1.0;
}

fn chunk_boundary_visualizer(
    chunk_q: Query<(&Transform, &TargetLOD), With<Chunk>>,
    mut gizmos: Gizmos,
) {
    let half_size = Vec3::splat(CHUNK_SIZE * 0.5);
    for (transform, lod) in chunk_q.iter() {
        let color = match lod.0 {
            LOD::High => Color::srgb(0.0, 1.0, 0.0),
            LOD::Medium => Color::srgb(1.0, 1.0, 0.0),
            LOD::Low => Color::srgb(1.0, 0.5, 0.0),
            LOD::Lowest => Color::srgb(1.0, 0.0, 0.0),
        };
        gizmos.cube(
            Transform::from_translation(transform.translation + half_size)
                .with_scale(Vec3::splat(CHUNK_SIZE)),
            color,
        );
    }
}

/// Keeps a chunk at max LOD while edits are landing on it. Inserted by
/// `EditableChunks`; removed by `expire_edit_pins` once idle.
#[derive(Component, Default)]
pub struct EditPin {
    pub idle: f32,
}

#[derive(Resource)]
pub struct EditPinTimeout(pub f32);
impl Default for EditPinTimeout {
    fn default() -> Self {
        Self(2.0)
    }
}

/// Set by anything that needs `plan_chunk_ops` to run even though no loader moved.
#[derive(Resource, Default)]
pub struct ReplanRequested(pub bool);

pub fn expire_edit_pins(
    time: Res<Time>,
    timeout: Res<EditPinTimeout>,
    mut pins: Query<(Entity, &mut EditPin)>,
    mut replan: ResMut<ReplanRequested>,
    mut commands: Commands,
) {
    for (e, mut pin) in &mut pins {
        pin.idle += time.delta_secs();
        if pin.idle >= timeout.0 {
            commands.entity(e).remove::<EditPin>();
            replan.0 = true; // loader decides what LOD this chunk goes back to
        }
    }
}

/// Keeps `CursorFocusChunk` pointing at the chunk under the cursor.
///
/// Backend-agnostic: reads the resolved `CursorHit` (CPU, GPU or Auto, see
/// `CursorQueryMode`). Sticky: the focus only moves once the hit leaves the
/// current `FOCUS_RADIUS` window, and a miss keeps the last focus. Otherwise a
/// LOD upgrade that nudges the hit into the next chunk (or a grazing ray
/// flickering between hit and miss) replans and reloads every frame.
pub fn update_cursor_focus(hit: Res<CursorHit>, mut focus: ResMut<CursorFocusChunk>) {
    let Some(h) = hit.0.as_ref() else {
        return;
    };
    // Nudge inside the solid so a hit on a chunk boundary doesn't flicker.
    let new = world_pos_to_chunk_pos(&(h.pos - h.normal * 0.05));
    let inside_window = focus.0.is_some_and(|f| chebyshev(new, f) <= FOCUS_RADIUS);
    if !inside_window {
        focus.0 = Some(new);
    } // assign only on change, so is_changed stays meaningful
}

#[cfg(test)]
mod tests {
    use super::{Chunk, ChunkManager, ChunkPosition};
    use bevy::prelude::*;

    #[test]
    fn duplicate_chunk_is_despawned_and_keeps_the_owner_mapped() {
        let mut world = World::new();
        world.init_resource::<ChunkManager>();
        let pos = IVec3::new(1, -2, 3);
        let owner = world.spawn((Chunk, ChunkPosition(pos))).id();
        let duplicate = world.spawn((Chunk, ChunkPosition(pos))).id();
        world.flush();

        assert!(world.get_entity(duplicate).is_err());
        assert_eq!(
            world.resource::<ChunkManager>().get_chunk(&pos),
            Some(owner)
        );

        world.despawn(owner);
        assert_eq!(world.resource::<ChunkManager>().get_chunk(&pos), None);
    }
}
