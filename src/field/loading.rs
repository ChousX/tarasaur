// field/loading.rs
use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task};
use std::path::PathBuf;

use super::generator::ChunkGeneratorRegistry;
use crate::LOD;
use crate::chunk::NewChunkSpawned;
use crate::persistence::{ChunkPersistenceRegistry, ChunkSaveData, FieldSavePayload};

#[derive(Component)]
pub struct ChunkDataTask(Task<Vec<FieldSavePayload>>);

pub fn spawn_chunk_data_task(
    trigger: On<NewChunkSpawned>,
    lod_q: Query<&LOD>,
    generators: Res<ChunkGeneratorRegistry>,
    mut commands: Commands,
) {
    let NewChunkSpawned {
        entity,
        chunk_position,
        ..
    } = *trigger.event();
    let Ok(&lod) = lod_q.get(entity) else { return };
    let generators = generators.clone(); // Arc<dyn Fn> clones are cheap
    let path = PathBuf::from(format!(
        "./saves/chunks/chunk_{}_{}_{}.bin",
        chunk_position.x, chunk_position.y, chunk_position.z
    ));

    let task = AsyncComputeTaskPool::get().spawn(async move {
        if let Ok(bytes) = std::fs::read(&path) {
            if let Ok(save_data) = postcard::from_bytes::<ChunkSaveData>(&bytes) {
                return save_data.fields;
            }
        }
        generators.generate_all(chunk_position, lod)
    });

    commands.entity(entity).insert(ChunkDataTask(task));
}

/// Runs every frame; non-blocking poll of every in-flight task. On
/// completion, applies the payload through the *same* loader closures
/// ChunkPersistenceRegistry already uses for save-file restores — load
/// and generate are indistinguishable from this point on.
pub fn resolve_chunk_data_tasks(
    mut commands: Commands,
    mut tasks: Query<(Entity, &mut ChunkDataTask)>,
    registry: Res<ChunkPersistenceRegistry>,
) {
    for (entity, mut task) in tasks.iter_mut() {
        let Some(payloads) = bevy::tasks::block_on(bevy::tasks::poll_once(&mut task.0)) else {
            continue;
        };
        for payload in payloads {
            if let Some(loader) = registry.loader_for(&payload.field_type_id) {
                commands.queue(move |world: &mut World| loader(world, entity, &payload.bytes));
            }
        }
        commands.entity(entity).remove::<ChunkDataTask>();
    }
}
