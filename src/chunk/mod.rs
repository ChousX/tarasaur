use bevy::{
    ecs::{lifecycle::HookContext, world::DeferredWorld},
    platform::collections::HashMap,
    prelude::*,
};

#[derive(Resource, Clone)]
pub struct ChunkManager {
    chunk_size: Vec3,
    areana: HashMap<IVec3, Entity>,
}

impl ChunkManager {
    pub fn get_chunk(&self, position: &IVec3) -> Option<Entity> {
        self.areana.get(position).copied()
    }
    pub fn is_loaded(&self, position: &IVec3) -> bool {
        self.areana.contains_key(position)
    }
    pub fn world_pos_to_chunk_pos(&self, world_position: Vec3) -> IVec3 {
        (world_position / self.chunk_size).floor().as_ivec3()
    }
}

//Managed by Hooks
impl ChunkManager {
    fn add_chunk(&mut self, position: IVec3, id: Entity) {
        self.areana.insert(position, id);
    }

    fn remove_chunk(&mut self, position: &IVec3) {
        self.areana.remove(position);
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
    let chunk_size = world.get_resource::<ChunkManager>().unwrap().chunk_size;
    let translation = chunk_pos.as_vec3() * chunk_size;
    world.get_mut::<Transform>(entity).unwrap().translation = translation;
}
