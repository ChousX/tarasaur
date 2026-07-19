use std::marker::PhantomData;

use bevy::{
    ecs::{lifecycle::HookContext, world::DeferredWorld},
    platform::collections::HashMap,
    prelude::*,
};

use crate::{Field, chunk};

pub struct ChunkPlugin;
impl Plugin for ChunkPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ChunkManager>()
            .add_observer(new_chunk_spawned)
            .add_systems(Update, chunk_loader_boundry_checker)
            .add_observer(update_chunk_loaded);
    }
}

pub const CHUNK_SIZE: Vec3 = vec3(10., 10., 10.);

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

//Managed by Hooks
impl ChunkManager {
    fn add_chunk(&mut self, position: IVec3, id: Entity) {
        self.arena.insert(position, id);
    }
    fn remove_chunk(&mut self, position: &IVec3) {
        self.arena.remove(position);
    }
}

#[derive(Component, Default, Clone, Copy)]
#[require(ChunkPosition, Visibility)]
#[component(
    immutable,
    on_add= on_add_chunk,
    on_remove = on_remove_chunk
)]
pub struct Chunk;
/// Registers the chunk with [`ChunkManager`] when added.
fn on_add_chunk(mut world: DeferredWorld, HookContext { entity, .. }: HookContext) {
    let chunk_pos = world.get::<ChunkPosition>(entity).unwrap().0;
    let mut chunk_manager = world.get_resource_mut::<ChunkManager>().unwrap();
    if chunk_manager.is_loaded(&chunk_pos) {
        warn!(
            "New chunk at pos:{} was not spawned there was already a chunk there",
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
#[component(
    immutable,
    on_add= on_add_chunk_pos,
)]
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

#[derive(Default, Deref, DerefMut, Component)]
#[require(CurrentChunk)]
#[component(
    on_add= on_add_chunk_loader,
)]
///ChunkLoader(val) val = 0 means only the chunk the chunkloader is in gets loaded
///val = 1 means the chunk the chunkloader is in and its nabaros.
///val = 2 is the nabars nabaros as well
pub struct ChunkLoader(pub u8);
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
    chunk_q: Query<(&ChunkLoader, &CurrentChunk)>,
    chunk_manager: Res<ChunkManager>,
    mut commands: Commands,
) {
    let Ok((&ChunkLoader(range), &CurrentChunk(pos))) = chunk_q.get(trigger.entity) else {
        return;
    };
    if range == 0 {
        if !chunk_manager.is_loaded(&pos) {
            commands.spawn((Chunk, ChunkPosition(pos)));
        }
        return;
    }
    let range = range as i32;
    let min = pos - range;
    let max = pos + range;
    for x in min.x..max.x {
        for y in min.y..max.y {
            for z in min.z..max.z {
                let pos = ivec3(x, y, z);
                if !chunk_manager.is_loaded(&pos) {
                    commands.spawn((Chunk, ChunkPosition(pos)));
                }
            }
        }
    }
}
