use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use crate::chunk::{Chunk, ChunkManager, ChunkPosition};
use crate::field::generator::ChunkGeneratorRegistry;
use crate::field::loading::ChunkDataTask;
use crate::field::{FieldFromRaw, FieldNew, Versionable, VoxelDataSlice};
use crate::{DirtyField, field::systems::SdfReinitTask};
use crate::{LOD, MaxEditLod};
use crate::{SDF, chunk::ReplanRequested};
use bevy::ecs::component::Mutable;
use bevy::ecs::message::MessageReader;
use bevy::prelude::*;
use bevy::tasks::AsyncComputeTaskPool;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkHeader {
    pub chunk_pos: IVec3,
    pub lod: LOD,
    pub version: u32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FieldSavePayload {
    pub field_type_id: String,
    pub bytes: Vec<u8>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ChunkSaveData {
    pub header: ChunkHeader,
    pub fields: Vec<FieldSavePayload>,
}

pub fn chunk_path(p: IVec3) -> PathBuf {
    PathBuf::from(format!("./saves/chunks/chunk_{}_{}_{}.bin", p.x, p.y, p.z))
}

// ------------------------------------------------------------ save cache ----

/// Writes land here first (synchronously), then hit disk asynchronously.
/// Load tasks consult it before disk, so a downgrade -> re-upgrade can never
/// read a stale file, and a despawn can't lose a pending write.
#[derive(Resource, Clone, Default)]
pub struct SaveCache(Arc<SaveCacheInner>);

#[derive(Default)]
struct SaveCacheInner {
    map: Mutex<HashMap<IVec3, Arc<ChunkSaveData>>>,
    write_lock: Mutex<()>,
}

impl SaveCache {
    pub fn get(&self, pos: IVec3) -> Option<Arc<ChunkSaveData>> {
        self.0.map.lock().unwrap().get(&pos).cloned()
    }
}

pub fn read_save(cache: &SaveCache, pos: IVec3) -> Option<Arc<ChunkSaveData>> {
    if let Some(hit) = cache.get(pos) {
        return Some(hit);
    }
    let bytes = std::fs::read(chunk_path(pos)).ok()?;
    postcard::from_bytes::<ChunkSaveData>(&bytes)
        .ok()
        .map(Arc::new)
}

pub fn commit_save(cache: &SaveCache, data: ChunkSaveData) {
    let pos = data.header.chunk_pos;
    let data = Arc::new(data);
    cache.0.map.lock().unwrap().insert(pos, data.clone());
    let cache = cache.clone();
    AsyncComputeTaskPool::get()
        .spawn(async move {
            let _guard = cache.0.write_lock.lock().unwrap();
            // A newer save for this chunk superseded us while we waited: skip.
            let current = cache.0.map.lock().unwrap().get(&pos).cloned();
            if !current.is_some_and(|c| Arc::ptr_eq(&c, &data)) {
                return;
            }
            let path = chunk_path(pos);
            if let Some(dir) = path.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            match postcard::to_allocvec(&*data) {
                Ok(bytes) => {
                    let tmp = path.with_extension("tmp");
                    if let Err(e) =
                        std::fs::write(&tmp, bytes).and_then(|_| std::fs::rename(&tmp, &path))
                    {
                        eprintln!("Failed to write chunk save {pos:?}: {e}");
                        return; // keep it cached so loads still see it
                    }
                }
                Err(e) => {
                    eprintln!("Failed to serialize chunk {pos:?}: {e}");
                    return;
                }
            }
            let mut map = cache.0.map.lock().unwrap();
            if map.get(&pos).is_some_and(|c| Arc::ptr_eq(c, &data)) {
                map.remove(&pos);
            }
        })
        .detach();
}

// -------------------------------------------------------------- registry ----

/// Sum of field versions at the last save (or install). Differs from the
/// current sum => unsaved edits.
#[derive(Component, Clone, Copy, Debug)]
pub struct ChunkSaveState(pub u64);

#[derive(Clone, Copy)]
pub struct FieldVTable {
    pub type_id: &'static str,
    pub save: fn(&World, Entity) -> Option<Vec<u8>>,
    pub version: fn(&World, Entity) -> u64,
    /// Exact downsample of a serialized payload (`from >= to`).
    pub downsample: fn(&[u8], LOD, LOD) -> Vec<u8>,
    /// Builds the field for `lod` from `bytes` and inserts it. `None` (or a
    /// size mismatch) inserts the default field, or removes it if OPTIONAL.
    pub install: fn(&mut EntityWorldMut<'_>, LOD, Option<&[u8]>),
}

#[derive(Resource, Default, Clone)]
pub struct ChunkPersistenceRegistry {
    pub(crate) fields: Vec<FieldVTable>,
}

/// Power-of-two LOD sizes and voxel i sitting at i*CHUNK_SIZE/size make
/// this an exact stride sample; no interpolation, no JFA.
pub fn downsample_bytes<F: VoxelDataSlice>(bytes: &[u8], from: LOD, to: LOD) -> Vec<u8> {
    let src: Vec<F::Elem> = bytemuck::pod_collect_to_vec(bytes);
    let (fs, ts) = (from.size() as usize, to.size() as usize);
    if src.len() != fs * fs * fs || ts == 0 || fs < ts {
        return Vec::new(); // install falls back to a default field
    }
    if fs == ts {
        return bytes.to_vec();
    }
    let ratio = fs / ts;
    let mut out: Vec<F::Elem> = Vec::with_capacity(ts * ts * ts);
    for z in 0..ts {
        for y in 0..ts {
            for x in 0..ts {
                let i = (z * ratio * fs + y * ratio) * fs + x * ratio;
                out.push(F::rescale(src[i], ratio as f32));
            }
        }
    }
    bytemuck::cast_slice(&out).to_vec()
}

impl ChunkPersistenceRegistry {
    pub fn register_field<F>(&mut self)
    where
        F: Component<Mutability = Mutable> + VoxelDataSlice + FieldFromRaw + FieldNew + Versionable,
    {
        self.fields.push(FieldVTable {
            type_id: std::any::type_name::<F>(),
            save: |w, e| {
                w.get::<F>(e)
                    .map(|f| bytemuck::cast_slice(f.data_slice()).to_vec())
            },
            version: |w, e| w.get::<F>(e).map_or(0, |f| Versionable::version(f)),
            downsample: |b, from, to| downsample_bytes::<F>(b, from, to),
            install: |em, lod, bytes| {
                if let Some(b) = bytes {
                    let elems: Vec<F::Elem> = bytemuck::pod_collect_to_vec(b);
                    if elems.len() == lod.volume() {
                        em.insert(F::from_raw(lod, elems));
                        return;
                    }
                }
                if F::OPTIONAL {
                    em.remove::<F>();
                } else {
                    em.insert(F::new_for_lod(lod));
                }
            },
        });
    }

    pub fn fingerprint(&self, world: &World, e: Entity) -> u64 {
        self.fields
            .iter()
            .fold(0u64, |a, f| a.wrapping_add((f.version)(world, e)))
    }
}

pub trait RegisterSaveableFieldExt {
    fn register_saveable_field<F>(&mut self) -> &mut Self
    where
        F: Component<Mutability = Mutable> + VoxelDataSlice + FieldFromRaw + FieldNew + Versionable;
}

impl RegisterSaveableFieldExt for App {
    fn register_saveable_field<F>(&mut self) -> &mut Self
    where
        F: Component<Mutability = Mutable> + VoxelDataSlice + FieldFromRaw + FieldNew + Versionable,
    {
        self.init_resource::<ChunkPersistenceRegistry>();
        self.world_mut()
            .resource_mut::<ChunkPersistenceRegistry>()
            .register_field::<F>();
        self
    }
}

// ---------------------------------------------------------------- saving ----

/// Some((data, fingerprint)) if the chunk is at max LOD, has no load in
/// flight, and has unsaved changes.
pub fn build_if_dirty(world: &World, e: Entity) -> Option<(ChunkSaveData, u64)> {
    let lod = *world.get::<LOD>(e)?;
    if lod != world.resource::<MaxEditLod>().0 || world.get::<ChunkDataTask>(e).is_some() {
        return None;
    }
    if is_settling(world, e) {
        return None;
    }
    let pos = world.get::<ChunkPosition>(e)?.0;
    let saved = world.get::<ChunkSaveState>(e)?.0;
    let reg = world.resource::<ChunkPersistenceRegistry>();
    let fp = reg.fingerprint(world, e);
    if fp == saved {
        return None;
    }
    let fields = reg
        .fields
        .iter()
        .filter_map(|f| {
            (f.save)(world, e).map(|bytes| FieldSavePayload {
                field_type_id: f.type_id.to_string(),
                bytes,
            })
        })
        .collect();
    Some((
        ChunkSaveData {
            header: ChunkHeader {
                chunk_pos: pos,
                lod,
                version: 1,
            },
            fields,
        },
        fp,
    ))
}

/// Used by the loader when a chunk leaves range.
pub fn flush_and_despawn(world: &mut World, e: Entity) {
    if is_settling(world, e) {
        world.resource_mut::<ReplanRequested>().0 = true; // try again next frame
        return;
    }
    if let Some((data, _)) = build_if_dirty(world, e) {
        let cache = world.resource::<SaveCache>().clone();
        commit_save(&cache, data);
    }
    if let Ok(em) = world.get_entity_mut(e) {
        em.despawn();
    }
}

#[derive(Resource)]
pub struct AutosaveIntervalSecs(pub f32);
impl Default for AutosaveIntervalSecs {
    fn default() -> Self {
        Self(2.0)
    }
}

pub fn autosave_dirty_chunks(
    world: &World,
    time: Res<Time>,
    interval: Res<AutosaveIntervalSecs>,
    cache: Res<SaveCache>,
    chunks: Query<Entity, (With<Chunk>, With<ChunkSaveState>)>,
    mut acc: Local<f32>,
    mut commands: Commands,
) {
    *acc += time.delta_secs();
    if *acc < interval.0 {
        return;
    }
    *acc = 0.0;
    for e in &chunks {
        if let Some((data, fp)) = build_if_dirty(world, e) {
            commit_save(&cache, data);
            commands.entity(e).insert(ChunkSaveState(fp));
        }
    }
}

/// Manual "save now". Only chunks at max LOD with unsaved changes are written.
#[derive(Message)]
pub struct SaveChunkMessage(pub IVec3);

pub fn generic_save_chunk_system(
    mut events: MessageReader<SaveChunkMessage>,
    world: &World,
    chunk_manager: Res<ChunkManager>,
    cache: Res<SaveCache>,
    mut commands: Commands,
) {
    for SaveChunkMessage(pos) in events.read() {
        let Some(e) = chunk_manager.get_chunk(pos) else {
            warn!("Cannot save chunk at {:?}: not loaded", pos);
            continue;
        };
        if let Some((data, fp)) = build_if_dirty(world, e) {
            commit_save(&cache, data);
            commands.entity(e).insert(ChunkSaveState(fp));
        }
    }
}

pub struct ChunkPersistencePlugin;
impl Plugin for ChunkPersistencePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ChunkPersistenceRegistry>()
            .init_resource::<ChunkGeneratorRegistry>()
            .init_resource::<SaveCache>()
            .init_resource::<MaxEditLod>()
            .init_resource::<AutosaveIntervalSecs>()
            .add_message::<SaveChunkMessage>()
            .add_systems(Update, (generic_save_chunk_system, autosave_dirty_chunks));
    }
}

/// Edits are still being turned into real distances. Saving now would
/// persist pre-JFA data, and replacing the fields now would discard the edits.
pub fn is_settling(world: &World, e: Entity) -> bool {
    world.get::<SdfReinitTask>(e).is_some() || world.get::<DirtyField<SDF, f32>>(e).is_some()
}
