//! Closes the gap between a chunk's `TargetLOD` and its installed `LOD`.
//!
//! `start_chunk_loads` (nearest and edit-pinned first, capped by
//! `MaxInFlightLoads`) spawns a `ChunkDataTask` that reads a save or runs the
//! generators. `resolve_chunk_data_tasks` builds the fields on the compute
//! pool and `install_chunk` swaps them and the `LOD` in atomically.

use crate::{DirtyField, SDF, VisibilityField, field::systems::SdfReinitTask};
use bevy::{
    prelude::*,
    tasks::{AsyncComputeTaskPool, ComputeTaskPool, Task, block_on, poll_once},
};
use std::sync::{Mutex, PoisonError};

use super::generator::ChunkGeneratorRegistry;
use crate::chunk::{Chunk, ChunkLoader, ChunkPosition, CurrentChunk, EditPin, chebyshev};
use crate::persistence::{
    ChunkPersistenceRegistry, ChunkSaveState, FieldSavePayload, PreparedField, SaveCache,
    build_if_dirty, commit_save, read_save,
};
use crate::{LOD, MaxEditLod, TargetLOD};
use bevy::ecs::system::SystemParam;

/// In flight while data for `lod` is being produced off-thread. Its
/// presence also blocks edits (see `EditableChunks`).
#[derive(Component)]
pub struct ChunkDataTask {
    task: Task<Vec<FieldSavePayload>>,
    lod: LOD,
}

/// Loads allowed in flight at once.
///
/// Most loads are low-detail chunks that
/// finish within a frame, so this caps loads started per frame; 32 left
/// streaming unable to keep up while moving (64 and 128 benched the same).
#[derive(Resource)]
pub struct MaxInFlightLoads(pub usize);
impl Default for MaxInFlightLoads {
    fn default() -> Self {
        Self(64)
    }
}

pub struct ChunkLoadingPlugin;
impl Plugin for ChunkLoadingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MaxInFlightLoads>().add_systems(
            Update,
            // Resolve first: a finished load holds its slot until resolved,
            // so starting first left those slots unusable for a frame.
            (resolve_chunk_data_tasks, start_chunk_loads).chain(),
        );
    }
}

/// Save (if one usable exists) -> exact downsample to `lod`; otherwise generate at `lod`.
/// A save is usable when its LOD >= target. We never upsample. Saves are
/// only ever written at max LOD, so in practice this is "use the high-res
/// save first, generate only if there isn't one".
///
/// Only produces bytes: the async pool is small (a quarter of the cores by
/// default), so building the fields here would slow generation. They are
/// built in `resolve_chunk_data_tasks`, on the compute pool.
fn make_task(
    pos: IVec3,
    lod: LOD,
    generators: ChunkGeneratorRegistry,
    registry: ChunkPersistenceRegistry,
    cache: SaveCache,
) -> Task<Vec<FieldSavePayload>> {
    AsyncComputeTaskPool::get().spawn(async move {
        if let Some(save) = read_save(&cache, pos)
            && save.header.lod >= lod
        {
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
        generators.generate_all(pos, lod)
    })
}

/// Builds every registered field from a load's payloads. A field with no
/// payload gets its default (or is removed, if optional).
fn prepare_fields(
    registry: &ChunkPersistenceRegistry,
    lod: LOD,
    payloads: &[FieldSavePayload],
) -> Vec<PreparedField> {
    registry
        .fields
        .iter()
        .map(|f| {
            let bytes = payloads
                .iter()
                .find(|p| p.field_type_id == f.type_id)
                .map(|p| p.bytes.as_slice());
            (f.prepare)(lod, bytes)
        })
        .collect()
}

/// A chunk that may need a load: (entity, position, target, current LOD,
/// reinit in flight, SDF dirty, edit-pinned).
type LoadCandidate = (
    Entity,
    &'static ChunkPosition,
    &'static TargetLOD,
    Option<&'static LOD>,
    Has<SdfReinitTask>,
    Has<DirtyField<SDF, f32>>,
    Has<EditPin>,
);

/// Where a load gets its data: saves first, then the generators.
#[derive(SystemParam)]
pub struct LoadSources<'w> {
    generators: Res<'w, ChunkGeneratorRegistry>,
    registry: Res<'w, ChunkPersistenceRegistry>,
    cache: Res<'w, SaveCache>,
    max: Res<'w, MaxEditLod>,
}

/// How many more loads may start this frame.
#[derive(SystemParam)]
pub struct LoadSlots<'w, 's> {
    in_flight: Query<'w, 's, (), With<ChunkDataTask>>,
    cap: Res<'w, MaxInFlightLoads>,
}

impl LoadSlots<'_, '_> {
    fn free(&self) -> usize {
        self.cap.0.saturating_sub(self.in_flight.iter().count())
    }
}

pub fn start_chunk_loads(
    world: &World,
    chunks: Query<LoadCandidate, (With<Chunk>, Without<ChunkDataTask>)>,
    loaders: Query<&CurrentChunk, With<ChunkLoader>>,
    slots: LoadSlots,
    sources: LoadSources,
    mut commands: Commands,
) {
    let mut slots = slots.free();
    if slots == 0 {
        return;
    }
    // Nearest-first (Chebyshev, to the closest loader); edit-pinned chunks
    // first of all, since an edit is waiting on them.
    let mut wanted: Vec<_> = chunks
        .iter()
        .filter(|(_, _, target, lod, ..)| *lod != Some(&target.0))
        .map(|item| {
            let pos = item.1.0;
            let pinned = item.6;
            let dist = loaders
                .iter()
                .map(|c| chebyshev(pos, c.0))
                .min()
                .unwrap_or(0);
            (if pinned { -1 } else { dist }, item)
        })
        .collect();
    wanted.sort_unstable_by_key(|(d, _)| *d);

    for (_, (entity, pos, target, lod, reinit, dirty, _)) in wanted {
        if slots == 0 {
            break;
        }
        if lod == Some(&sources.max.0) {
            if reinit || dirty {
                continue;
            } // let JFA finish so the save has real distances
            if let Some((data, _)) = build_if_dirty(world, entity) {
                commit_save(&sources.cache, data);
            }
        }
        commands.entity(entity).insert(ChunkDataTask {
            task: make_task(
                pos.0,
                target.0,
                (*sources.generators).clone(),
                (*sources.registry).clone(),
                (*sources.cache).clone(),
            ),
            lod: target.0,
        });
        slots = slots.saturating_sub(1);
    }
}

pub fn resolve_chunk_data_tasks(
    mut commands: Commands,
    mut tasks: Query<(Entity, &mut ChunkDataTask, &TargetLOD)>,
    registry: Res<ChunkPersistenceRegistry>,
) {
    let mut done = Vec::new();
    for (entity, mut t, target) in &mut tasks {
        let Some(payloads) = block_on(poll_once(&mut t.task)) else {
            continue;
        };
        if t.lod != target.0 {
            // Target moved while loading: discard; start_chunk_loads re-issues.
            commands.entity(entity).remove::<ChunkDataTask>();
            continue;
        }
        done.push((entity, t.lod, payloads));
    }
    if done.is_empty() {
        return;
    }
    // Results go through a mutex rather than as the scope's output: a scope
    // returning boxed trait objects makes rustc demand `'static` borrows.
    let registry = &*registry;
    let built: Mutex<Vec<(usize, Vec<PreparedField>)>> = Mutex::new(Vec::new());
    ComputeTaskPool::get().scope(|s| {
        for (i, (_, lod, payloads)) in done.iter().enumerate() {
            let built = &built;
            s.spawn(async move {
                let fields = prepare_fields(registry, *lod, payloads);
                if let Ok(mut b) = built.lock() {
                    b.push((i, fields));
                }
            });
        }
    });
    let built = built.into_inner().unwrap_or_else(PoisonError::into_inner);
    for (i, fields) in built {
        let Some(&(entity, lod, _)) = done.get(i) else {
            continue;
        };
        commands.queue(move |world: &mut World| install_chunk(world, entity, lod, fields));
    }
}

/// Replaces every field and `LOD` in one command, so they never disagree.
fn install_chunk(world: &mut World, entity: Entity, lod: LOD, fields: Vec<PreparedField>) {
    let Ok(mut em) = world.get_entity_mut(entity) else {
        return;
    };
    if em.get::<TargetLOD>().is_none_or(|t| t.0 != lod) {
        em.remove::<ChunkDataTask>();
        return;
    }
    for apply in fields {
        apply.apply(&mut em);
    }
    em.remove::<SdfReinitTask>();
    em.remove::<DirtyField<SDF, f32>>();
    em.remove::<DirtyField<VisibilityField, bool>>();
    em.insert((lod, ChunkSaveState(0))); // fresh fields all start at version 0
    em.remove::<ChunkDataTask>();
}
