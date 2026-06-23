use bevy::{platform::collections::HashMap, prelude::*};
pub struct ChunkManger {
    areana: HashMap<IVec3, Entity>,
}
impl ChunkManger {
    pub fn add_chunk(position: IVec3, id: Entity, size: u8, default_size: Vec3) {}
    pub fn get_chunk(position: IVec3, id: Entity) {}
    pub fn remove_chunk(position: IVec3, id: Entity, size: u8, default_size: Vec3) {}
}
#[derive(Resource, Clone, Copy)]
pub struct ChunkBaseSize(pub Vec3);
#[derive(Resource, Clone, Copy)]
pub struct ChunkBaseVoxals(pub UVec3);

#[derive(Component, Default, Clone, Copy)]
pub struct Chunk;
#[derive(Component, Default, Clone, Copy)]
pub struct ChunkSize(pub u8);
#[derive(Component, Default, Clone, Copy)]
pub struct ChunkPosition(pub IVec3);
#[derive(Component, Default, Clone, Copy)]
pub struct LevelOfDetail(pub u8);
