use bevy::{
    camera::primitives::Aabb,
    ecs::{lifecycle::HookContext, world::DeferredWorld},
    platform::collections::HashMap,
    prelude::*,
};

use crate::LOD;

pub struct ChunkPlugin;
impl Plugin for ChunkPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ChunkManager>()
            //.init_resource::<ShowChunkBounds>()
            .add_observer(new_chunk_spawned)
            .add_systems(Update, chunk_loader_boundry_checker)
            .add_observer(update_chunk_loaded)
            .add_systems(Startup, configure_gizmo_depth_bias)
            .add_systems(
                Update,
                chunk_boundry_visualizer.run_if(resource_exists::<ShowChunkBounds>),
            );
    }
}

pub const CHUNK_SIZE: f32 = 10.;

#[derive(Resource, Default)]
pub struct ShowChunkBounds;

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

// Managed by Hooks
impl ChunkManager {
    fn add_chunk(&mut self, position: IVec3, id: Entity) {
        self.arena.insert(position, id);
    }
    fn remove_chunk(&mut self, position: &IVec3) {
        self.arena.remove(position);
    }
}

#[derive(Component, Default, Clone, Copy)]
#[require(
    ChunkPosition,
    Visibility,
    LOD,
    Aabb::from_min_max(Vec3::ZERO, Vec3::splat(CHUNK_SIZE))
)]
#[component(
    immutable,
    on_add = on_add_chunk,
    on_remove = on_remove_chunk
)]
pub struct Chunk;

/// Registers the chunk with [`ChunkManager`] when added.
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

/// Unregisters the chunk from [`ChunkManager`] when removed.
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

/// Sets the entity's [`Transform`] translation based on chunk position and size.
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
    let world_position = transform.translation;
    commands.trigger(NewChunkSpawned {
        entity: trigger.entity,
        world_position,
        chunk_position,
    });
}

#[derive(Default, Deref, DerefMut, Component)]
pub struct CurrentChunk(pub IVec3);

#[derive(Component, Clone, Debug)]
#[require(CurrentChunk)]
#[component(on_add = on_add_chunk_loader)]
pub struct ChunkLoader {
    pub high_distance: i32,
    pub medium_distance: i32,
    pub low_distance: i32,
    pub lowest_distance: i32,
    /// Distance buffer in chunks before a chunk drops to a lower LOD level
    pub hysteresis: i32,
}

impl Default for ChunkLoader {
    fn default() -> Self {
        Self {
            high_distance: 1,
            medium_distance: 3,
            low_distance: 6,
            lowest_distance: 10,
            hysteresis: 1,
        }
    }
}

impl ChunkLoader {
    /// Determines target LOD level taking hysteresis into account to prevent thrashing.
    pub fn calculate_lod(&self, distance: i32, current_lod: Option<LOD>) -> Option<LOD> {
        let ideal_lod = if distance <= self.high_distance {
            LOD::High
        } else if distance <= self.medium_distance {
            LOD::Medium
        } else if distance <= self.low_distance {
            LOD::Low
        } else if distance <= self.lowest_distance {
            LOD::Lowest
        } else {
            return None;
        };

        let Some(current) = current_lod else {
            return Some(ideal_lod);
        };

        // Instant upgrade when getting closer
        if (ideal_lod as u32) > (current as u32) {
            return Some(ideal_lod);
        }

        // Apply hysteresis buffer on downgrades
        let h = self.hysteresis;
        match current {
            LOD::High if distance <= self.high_distance + h => Some(LOD::High),
            LOD::Medium if distance <= self.medium_distance + h => Some(LOD::Medium),
            LOD::Low if distance <= self.low_distance + h => Some(LOD::Low),
            LOD::Lowest if distance <= self.lowest_distance + h => Some(LOD::Lowest),
            _ => Some(ideal_lod),
        }
    }
}

fn on_add_chunk_loader(mut world: DeferredWorld, HookContext { entity, .. }: HookContext) {
    let chunk_pos = world.get::<GlobalTransform>(entity).unwrap();
    world.get_mut::<CurrentChunk>(entity).unwrap().0 =
        world_pos_to_chunk_pos(&chunk_pos.translation());
    world.commands().trigger(ChunkLoaderChunkChange { entity });
}

#[derive(Event)]
pub struct ChunkLoaderChunkChange {
    pub entity: Entity,
}

fn chunk_loader_boundry_checker(
    mut chunk_loader_q: Query<
        (&GlobalTransform, &mut CurrentChunk, Entity),
        Changed<GlobalTransform>,
    >,
    mut commands: Commands,
) {
    for (transform, mut old_chunk, entity) in chunk_loader_q.iter_mut() {
        let old_pos = old_chunk.0;
        let new_pos = world_pos_to_chunk_pos(&transform.translation());
        if old_pos != new_pos {
            old_chunk.0 = new_pos;
            commands.trigger(ChunkLoaderChunkChange { entity });
        }
    }
}

fn update_chunk_loaded(
    trigger: On<ChunkLoaderChunkChange>,
    chunk_loader_q: Query<(&ChunkLoader, &CurrentChunk)>,
    chunk_manager: Res<ChunkManager>,
    lod_q: Query<&LOD>,
    mut commands: Commands,
) {
    let Ok((loader, &CurrentChunk(center_pos))) = chunk_loader_q.get(trigger.entity) else {
        return;
    };

    let max_r = loader.lowest_distance + loader.hysteresis;
    let min_bounds = center_pos - IVec3::splat(max_r);
    let max_bounds = center_pos + IVec3::splat(max_r);

    for x in min_bounds.x..=max_bounds.x {
        for y in min_bounds.y..=max_bounds.y {
            for z in min_bounds.z..=max_bounds.z {
                let chunk_pos = ivec3(x, y, z);
                let distance = (chunk_pos - center_pos).abs().max_element();

                let existing_entity = chunk_manager.get_chunk(&chunk_pos);
                let current_lod = existing_entity.and_then(|e| lod_q.get(e).ok().copied());

                let target_lod = loader.calculate_lod(distance, current_lod);

                match (existing_entity, target_lod, current_lod) {
                    (Some(entity), Some(new_lod), Some(old_lod)) => {
                        if new_lod != old_lod {
                            commands.entity(entity).insert(new_lod);
                        }
                    }
                    (None, Some(new_lod), _) => {
                        commands.spawn((Chunk, ChunkPosition(chunk_pos), new_lod));
                    }
                    _ => {}
                }
            }
        }
    }
}

// Visualizer Systems

fn configure_gizmo_depth_bias(mut config_store: ResMut<GizmoConfigStore>) {
    let (config, _) = config_store.config_mut::<DefaultGizmoConfigGroup>();
    config.depth_bias = -1.0;
}

fn chunk_boundry_visualizer(chunk_q: Query<(&Transform, &LOD), With<Chunk>>, mut gizmos: Gizmos) {
    let half_size = Vec3::splat(CHUNK_SIZE * 0.5);
    for (transform, lod) in chunk_q.iter() {
        let color = match lod {
            LOD::High => Color::srgb(0.0, 1.0, 0.0),   // Green
            LOD::Medium => Color::srgb(1.0, 1.0, 0.0), // Yellow
            LOD::Low => Color::srgb(1.0, 0.5, 0.0),    // Orange
            LOD::Lowest => Color::srgb(1.0, 0.0, 0.0), // Red
        };

        let center = transform.translation + half_size;
        gizmos.cube(
            Transform::from_translation(center).with_scale(Vec3::splat(CHUNK_SIZE)),
            color,
        );
    }
}
