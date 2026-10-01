use bevy::{
    camera::primitives::Aabb,
    ecs::{lifecycle::HookContext, world::DeferredWorld},
    platform::collections::HashMap,
    prelude::*,
};
use std::collections::VecDeque;

use crate::{LOD, TargetLOD};

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
                    chunk_loader_boundry_checker,
                    expire_edit_pins,
                    plan_chunk_ops.run_if(loaders_dirty),
                    apply_chunk_ops,
                )
                    .chain(),
            )
            .add_systems(
                Update,
                chunk_boundry_visualizer.run_if(resource_exists::<ShowChunkBounds>),
            );
    }
}

pub const CHUNK_SIZE: f32 = 10.;
/// Chunks within this Chebyshev radius of the cursor chunk are treated as
/// being at (at most) that distance from every loader.
pub const FOCUS_RADIUS: i32 = 1;

#[derive(Resource, Default)]
pub struct ShowChunkBounds;

/// Written by whatever reads cursor query hits. Only assign when the chunk
/// actually changes so `is_changed` stays meaningful.
#[derive(Resource, Default, PartialEq, Clone, Copy)]
pub struct CursorFocusChunk(pub Option<IVec3>);

/// Max spawns + LOD retargets applied per frame, nearest first.
#[derive(Resource)]
pub struct MaxChunkOpsPerFrame(pub usize);
impl Default for MaxChunkOpsPerFrame {
    fn default() -> Self {
        Self(64)
    }
}

#[derive(Resource, Clone, Default)]
pub struct ChunkManager {
    arena: HashMap<IVec3, Entity>,
}

impl ChunkManager {
    pub fn get_chunk(&self, position: &IVec3) -> Option<Entity> {
        self.arena.get(position).copied()
    }
    pub fn is_loaded(&self, position: &IVec3) -> bool {
        self.arena.contains_key(position)
    }
}

#[inline]
pub fn world_pos_to_chunk_pos(world_position: &Vec3) -> IVec3 {
    (world_position / CHUNK_SIZE).floor().as_ivec3()
}

impl ChunkManager {
    fn add_chunk(&mut self, position: IVec3, id: Entity) {
        self.arena.insert(position, id);
    }
    fn remove_chunk(&mut self, position: &IVec3) {
        self.arena.remove(position);
    }
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
    let chunk_pos = world.get::<ChunkPosition>(entity).unwrap().0;
    let mut chunk_manager = world.get_resource_mut::<ChunkManager>().unwrap();
    if chunk_manager.is_loaded(&chunk_pos) {
        warn!(
            "New chunk at pos:{} was not spawned, there was already a chunk there",
            chunk_pos
        );
        return;
    }
    chunk_manager.add_chunk(chunk_pos, entity);
}

fn on_remove_chunk(mut world: DeferredWorld, HookContext { entity, .. }: HookContext) {
    let chunk_pos = world.get::<ChunkPosition>(entity).unwrap().0;
    world
        .get_resource_mut::<ChunkManager>()
        .unwrap()
        .remove_chunk(&chunk_pos);
}

#[derive(Component, Default, Deref, DerefMut)]
#[require(Transform)]
#[component(immutable, on_add = on_add_chunk_pos)]
pub struct ChunkPosition(pub IVec3);

fn on_add_chunk_pos(mut world: DeferredWorld, HookContext { entity, .. }: HookContext) {
    let chunk_pos = world.get::<ChunkPosition>(entity).unwrap();
    let translation = chunk_pos.as_vec3() * CHUNK_SIZE;
    world.get_mut::<Transform>(entity).unwrap().translation = translation;
}

#[derive(Event)]
pub struct NewChunkSpawned {
    pub entity: Entity,
    pub world_position: Vec3,
    pub chunk_position: IVec3,
}

fn new_chunk_spawned(
    trigger: On<Add, Chunk>,
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

#[derive(Default, Deref, DerefMut, Component)]
pub struct CurrentChunk(pub IVec3);

/// `lod_bands`: finest first; each width is ADDITIVE. Distance is Chebyshev,
/// in chunks. `[(High,1),(Medium,1)]` => High for d<=1, Medium for d==2,
/// nothing beyond. `hysteresis` extends both LOD downgrades and despawn: an
/// existing chunk keeps its finer LOD until d > band_edge + hysteresis, and
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
    pub fn total_range(&self) -> i32 {
        self.lod_bands.iter().map(|b| b.1 as i32).sum()
    }

    /// `current` is `Some` for chunks that already exist.
    pub fn want(&self, d: i32, current: Option<LOD>) -> Want {
        let h = self.hysteresis as i32;
        let mut acc = 0i32;
        let mut ideal = None;
        let mut current_edge = None;
        for &(lod, w) in &self.lod_bands {
            acc += w as i32;
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
        if outside && d > acc + h {
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
            Some(e) if d <= e + h => Want::Lod(cur), // hysteresis: hold the finer LOD
            _ => Want::Lod(ideal),
        }
    }
}

fn on_add_chunk_loader(mut world: DeferredWorld, HookContext { entity, .. }: HookContext) {
    let chunk_pos = world.get::<GlobalTransform>(entity).unwrap().translation();
    world.get_mut::<CurrentChunk>(entity).unwrap().0 = world_pos_to_chunk_pos(&chunk_pos);
}

fn chunk_loader_boundry_checker(
    mut q: Query<(&GlobalTransform, &mut CurrentChunk), Changed<GlobalTransform>>,
) {
    for (transform, mut current) in q.iter_mut() {
        let new_pos = world_pos_to_chunk_pos(&transform.translation());
        if current.0 != new_pos {
            current.0 = new_pos; // marks Changed<CurrentChunk>, which wakes plan_chunk_ops
        }
    }
}

fn loaders_dirty(
    loaders: Query<(), Or<(Changed<CurrentChunk>, Changed<ChunkLoader>)>>,
    focus: Res<CursorFocusChunk>,
    replan: Res<ReplanRequested>,
) -> bool {
    !loaders.is_empty() || focus.is_changed() || replan.0
}

fn effective_distance(
    pos: IVec3,
    center: IVec3,
    loader: &ChunkLoader,
    focus: Option<IVec3>,
) -> i32 {
    let d = (pos - center).abs().max_element();
    if let Some(f) = focus {
        let fd = (pos - f).abs().max_element();
        // Focus only promotes chunks that are already in this loader's range.
        if fd <= FOCUS_RADIUS && d <= loader.total_range() + loader.hysteresis as i32 {
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
                    crate::persistence::flush_and_despawn(world, entity)
                });
            }
        }
    }

    // Missing chunks: spawn candidates inside each loader's box.
    let mut spawns: HashMap<IVec3, (i32, LOD)> = HashMap::default();
    for (loader, center) in &loaders {
        let r = loader.total_range();
        for dz in -r..=r {
            for dy in -r..=r {
                for dx in -r..=r {
                    let p = center.0 + ivec3(dx, dy, dz);
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

fn chunk_boundry_visualizer(
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

pub fn update_cursor_focus(
    results: Res<crate::voxel::query::VoxelQueryResults>,
    mut focus: ResMut<CursorFocusChunk>,
) {
    // Assumes the cursor ray is query 0; match on `user_id` if you issue several.
    let Some(hit) = results.0.first() else { return };
    let new = (hit.did_hit != 0).then(|| {
        // Nudge inside the solid so a hit on a chunk boundary doesn't flicker.
        let p = Vec3::from(hit.hit_pos_world) - Vec3::from(hit.hit_normal) * 0.05;
        world_pos_to_chunk_pos(&p)
    });
    if focus.0 != new {
        focus.0 = new;
    } // assign only on change, so is_changed stays meaningful
}
