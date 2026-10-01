// field/loading.rs
use crate::{DirtyField, SDF, VisibilityField, field::systems::SdfReinitTask};
use bevy::{
    prelude::*,
    tasks::{AsyncComputeTaskPool, Task, block_on, poll_once},
};

use super::generator::ChunkGeneratorRegistry;
use crate::chunk::{Chunk, ChunkPosition};
use crate::persistence::{
    ChunkPersistenceRegistry, ChunkSaveData, ChunkSaveState, FieldSavePayload, SaveCache,
    build_if_dirty, commit_save, read_save,
};
use crate::{LOD, MaxEditLod, TargetLOD};

/// In flight while data for `lod` is being produced off-thread. Its
/// presence also blocks edits (see `EditableChunks`).
#[derive(Component)]
pub struct ChunkDataTask {
    task: Task<Vec<FieldSavePayload>>,
    lod: LOD,
}

#[derive(Resource)]
pub struct MaxInFlightLoads(pub usize);
impl Default for MaxInFlightLoads {
    fn default() -> Self {
        Self(32)
    }
}

pub struct ChunkLoadingPlugin;
impl Plugin for ChunkLoadingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MaxInFlightLoads>().add_systems(
            Update,
            (start_chunk_loads, resolve_chunk_data_tasks).chain(),
        );
    }
}

/// Save (if one usable exists) -> exact downsample to `lod`; otherwise generate at `lod`.
/// A save is usable when its LOD >= target. We never upsample. Saves are
/// only ever written at max LOD, so in practice this is "use the high-res
/// save first, generate only if there isn't one".
fn make_task(
    pos: IVec3,
    lod: LOD,
    generators: ChunkGeneratorRegistry,
    registry: ChunkPersistenceRegistry,
    cache: SaveCache,
) -> Task<Vec<FieldSavePayload>> {
    AsyncComputeTaskPool::get().spawn(async move {
        if let Some(save) = read_save(&cache, pos) {
            if save.header.lod >= lod {
                return save
                    .fields
                    .iter()
                    .filter_map(|p| {
                        let vt = registry
                            .fields
                            .iter()
                            .find(|f| f.type_id == p.field_type_id)?;
                        Some(FieldSavePayload {
                            field_type_id: p.field_type_id.clone(),
                            bytes: (vt.downsample)(&p.bytes, save.header.lod, lod),
                        })
                    })
                    .collect();
            }
        }
        generators.generate_all(pos, lod)
    })
}

pub fn start_chunk_loads(
    world: &World,
    chunks: Query<
        (
            Entity,
            &ChunkPosition,
            &TargetLOD,
            Option<&LOD>,
            Has<SdfReinitTask>,
            Has<DirtyField<SDF, f32>>,
        ),
        (With<Chunk>, Without<ChunkDataTask>),
    >,
    in_flight: Query<(), With<ChunkDataTask>>,
    cap: Res<MaxInFlightLoads>,
    max: Res<MaxEditLod>,
    generators: Res<ChunkGeneratorRegistry>,
    registry: Res<ChunkPersistenceRegistry>,
    cache: Res<SaveCache>,
    mut commands: Commands,
) {
    let mut slots = cap.0.saturating_sub(in_flight.iter().count());
    for (entity, pos, target, lod, reinit, dirty) in &chunks {
        if slots == 0 {
            break;
        }
        if lod == Some(&target.0) {
            continue;
        }
        if lod == Some(&max.0) {
            if reinit || dirty {
                continue;
            } // let JFA finish so the save has real distances
            if let Some((data, _)) = build_if_dirty(world, entity) {
                commit_save(&cache, data);
            }
        }
        commands.entity(entity).insert(ChunkDataTask {
            task: make_task(
                pos.0,
                target.0,
                (*generators).clone(),
                (*registry).clone(),
                (*cache).clone(),
            ),
            lod: target.0,
        });
        slots -= 1;
    }
}

pub fn resolve_chunk_data_tasks(
    mut commands: Commands,
    mut tasks: Query<(Entity, &mut ChunkDataTask, &TargetLOD)>,
) {
    for (entity, mut t, target) in tasks.iter_mut() {
        let Some(payloads) = block_on(poll_once(&mut t.task)) else {
            continue;
        };
        let lod = t.lod;
        if lod != target.0 {
            // Target moved while loading: discard; start_chunk_loads re-issues.
            commands.entity(entity).remove::<ChunkDataTask>();
            continue;
        }
        commands.queue(move |world: &mut World| install_chunk(world, entity, lod, payloads));
    }
}

/// Replaces every field and `LOD` in one command, so they never disagree.
fn install_chunk(world: &mut World, entity: Entity, lod: LOD, payloads: Vec<FieldSavePayload>) {
    if !world.get::<TargetLOD>(entity).is_some_and(|t| t.0 == lod) {
        if let Ok(mut em) = world.get_entity_mut(entity) {
            em.remove::<ChunkDataTask>();
        }
        return;
    }
    world.resource_scope(|world, registry: Mut<ChunkPersistenceRegistry>| {
        let Ok(mut em) = world.get_entity_mut(entity) else {
            return;
        };
        for f in &registry.fields {
            let bytes = payloads
                .iter()
                .find(|p| p.field_type_id == f.type_id)
                .map(|p| p.bytes.as_slice());
            (f.install)(&mut em, lod, bytes);
        }
        em.remove::<SdfReinitTask>();
        em.remove::<DirtyField<SDF, f32>>();
        em.remove::<DirtyField<VisibilityField, bool>>();
        em.insert((lod, ChunkSaveState(0))); // fresh fields all start at version 0
        em.remove::<ChunkDataTask>();
    });
}
