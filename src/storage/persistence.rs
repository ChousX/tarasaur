//! Chunk saves: one postcard file per chunk in `SaveDirectory`
//! (`./saves/chunks/` by default).
//!
//! Only max-LOD chunks with unsaved edits are written (autosave, manual
//! `SaveChunkMessage`, or on despawn). Each saveable field registers a
//! `FieldVTable`; saves go through `SaveCache` so loads never see a stale
//! file. Lower LODs are exact stride-downsamples of the saved data.
//!
//! A save's bytes (`ChunkSaveData::encode`) are also its network form.
//! Field data travels raw, so encoding is about a memcpy, and
//! `ChunkSaveView::decode` borrows the fields without copying them.

use std::{
    collections::HashMap,
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, MutexGuard, PoisonError},
};

use crate::chunk::{Chunk, ChunkManager, ChunkPosition};
use crate::field::VoxelField;
use crate::field::{LOD, MaxEditLod};
use crate::grid::flatten_cube;
use crate::num::f32_from_u32;
use crate::storage::generator::ChunkGeneratorRegistry;
use crate::storage::loading::ChunkDataTask;
use crate::{chunk::ReplanRequested, field::SDF};
use crate::{
    field::{DirtyField, VisibilityField},
    field::systems::{ReinitTask, SdfReinitTask},
};
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
    /// One byte string (length, then raw bytes) rather than a sequence of
    /// u8s, so encoding is a copy and decoding can borrow it in place. The
    /// wire format is the same either way.
    #[serde(with = "raw_bytes")]
    pub bytes: Vec<u8>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ChunkSaveData {
    pub header: ChunkHeader,
    pub fields: Vec<FieldSavePayload>,
}

/// A `ChunkSaveData` whose field bytes borrow from the buffer it was
/// decoded from: decoding copies nothing. Same wire format, so either type
/// decodes what the other encodes.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ChunkSaveView<'a> {
    pub header: ChunkHeader,
    #[serde(borrow)]
    pub fields: Vec<FieldSaveView<'a>>,
}

/// One field of a `ChunkSaveView`.
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct FieldSaveView<'a> {
    pub field_type_id: &'a str,
    #[serde(serialize_with = "raw_bytes::serialize")]
    pub bytes: &'a [u8],
}

impl ChunkSaveData {
    /// The save file's bytes; also what to send over the network. Field
    /// data is copied in raw, so this costs about a memcpy.
    ///
    /// # Errors
    /// Only if postcard fails to serialize, which these types never make it do.
    pub fn encode(&self) -> postcard::Result<Vec<u8>> {
        self.view().encode()
    }

    /// Decodes into owned data. To read fields without copying them, use
    /// `ChunkSaveView::decode`.
    ///
    /// # Errors
    /// If `bytes` isn't an encoded chunk.
    pub fn decode(bytes: &[u8]) -> postcard::Result<Self> {
        postcard::from_bytes(bytes)
    }

    #[must_use]
    pub fn view(&self) -> ChunkSaveView<'_> {
        ChunkSaveView {
            header: self.header,
            fields: self
                .fields
                .iter()
                .map(|f| FieldSaveView {
                    field_type_id: &f.field_type_id,
                    bytes: &f.bytes,
                })
                .collect(),
        }
    }
}

impl<'a> ChunkSaveView<'a> {
    /// Decodes without copying any field data.
    ///
    /// # Errors
    /// If `bytes` isn't an encoded chunk.
    pub fn decode(bytes: &'a [u8]) -> postcard::Result<Self> {
        postcard::from_bytes(bytes)
    }

    /// See `ChunkSaveData::encode`.
    ///
    /// # Errors
    /// Only if postcard fails to serialize, which these types never make it do.
    pub fn encode(&self) -> postcard::Result<Vec<u8>> {
        // Header and names fit in the slack; field data is sized exactly,
        // so the buffer is allocated once.
        let len = self.fields.iter().fold(64usize, |n, f| {
            n.saturating_add(f.bytes.len())
                .saturating_add(f.field_type_id.len())
                .saturating_add(10)
        });
        postcard::to_extend(self, Vec::with_capacity(len))
    }

    #[must_use]
    pub fn to_owned_data(&self) -> ChunkSaveData {
        ChunkSaveData {
            header: self.header,
            fields: self
                .fields
                .iter()
                .map(|f| FieldSavePayload {
                    field_type_id: f.field_type_id.to_owned(),
                    bytes: f.bytes.to_vec(),
                })
                .collect(),
        }
    }
}

/// Serde helpers that write a byte slice as one byte string. Plain serde
/// writes `Vec<u8>` as a sequence, which postcard walks a byte at a time.
mod raw_bytes {
    use serde::{Deserializer, Serializer, de};
    use std::fmt;

    pub fn serialize<S: Serializer>(v: &[u8], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_bytes(v)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        d.deserialize_byte_buf(Visitor)
    }

    struct Visitor;

    impl<'de> de::Visitor<'de> for Visitor {
        type Value = Vec<u8>;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("bytes")
        }

        fn visit_bytes<E: de::Error>(self, v: &[u8]) -> Result<Vec<u8>, E> {
            Ok(v.to_vec())
        }

        fn visit_byte_buf<E: de::Error>(self, v: Vec<u8>) -> Result<Vec<u8>, E> {
            Ok(v)
        }

        // Self-describing formats (JSON, ...) may hand back a sequence.
        fn visit_seq<A: de::SeqAccess<'de>>(self, mut seq: A) -> Result<Vec<u8>, A::Error> {
            let mut out = Vec::with_capacity(seq.size_hint().unwrap_or(0));
            while let Some(b) = seq.next_element()? {
                out.push(b);
            }
            Ok(out)
        }
    }
}

/// Where chunk saves are read and written. Set it before chunks load (e.g.
/// insert it when building the app); relative paths resolve against the
/// working directory.
#[derive(Resource, Clone, Debug)]
pub struct SaveDirectory(pub PathBuf);

impl Default for SaveDirectory {
    fn default() -> Self {
        Self(PathBuf::from("./saves/chunks"))
    }
}

/// The save file of the chunk at `p` in `dir`.
#[must_use]
pub fn chunk_path(dir: &Path, p: IVec3) -> PathBuf {
    dir.join(format!("chunk_{}_{}_{}.bin", p.x, p.y, p.z))
}

// ------------------------------------------------------------ save cache ----

/// Writes land here first (synchronously), then hit disk asynchronously.
///
/// Load tasks consult it before disk, so a downgrade -> re-upgrade can never
/// read a stale file, and a despawn can't lose a pending write.
#[derive(Resource, Clone, Default)]
pub(crate) struct SaveCache(Arc<SaveCacheInner>);

/// Write locks, striped by chunk position: one chunk's writes stay in
/// order, while different chunks write in parallel instead of parking
/// async-pool threads (which loads need) behind one global lock.
const WRITE_STRIPES: usize = 32; // the largest array with a `Default`

#[derive(Default)]
struct SaveCacheInner {
    map: Mutex<HashMap<IVec3, Arc<ChunkSaveData>>>,
    write_locks: [Mutex<()>; WRITE_STRIPES],
}

impl SaveCacheInner {
    fn write_lock(&self, pos: IVec3) -> Option<&Mutex<()>> {
        let h = pos.x.wrapping_mul(73_856_093)
            ^ pos.y.wrapping_mul(19_349_663)
            ^ pos.z.wrapping_mul(83_492_791);
        let i = usize::try_from(h.unsigned_abs()).unwrap_or(0) % WRITE_STRIPES;
        self.write_locks.get(i)
    }
}

/// A save read for loading: borrowed from the cache, or a file's bytes.
pub(crate) enum LoadedSave {
    Cached(Arc<ChunkSaveData>),
    File(Vec<u8>),
}

impl LoadedSave {
    /// The save's fields, borrowed (decoding a file copies nothing). `None`
    /// if the file is corrupt.
    pub fn view(&self) -> Option<ChunkSaveView<'_>> {
        match self {
            Self::Cached(data) => Some(data.view()),
            Self::File(bytes) => ChunkSaveView::decode(bytes).ok(),
        }
    }
}

/// Locks a cache mutex, recovering it if a panic elsewhere poisoned it: the
/// map and the write token stay valid whatever a panicking holder was doing.
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

impl SaveCache {
    #[must_use]
    pub fn get(&self, pos: IVec3) -> Option<Arc<ChunkSaveData>> {
        lock(&self.0.map).get(&pos).cloned()
    }
}

/// The chunk's pending save, else its file's bytes. Decoding is left to
/// `LoadedSave::view`, which borrows rather than copies.
pub(crate) fn read_save(cache: &SaveCache, dir: &Path, pos: IVec3) -> Option<LoadedSave> {
    if let Some(hit) = cache.get(pos) {
        return Some(LoadedSave::Cached(hit));
    }
    std::fs::read(chunk_path(dir, pos))
        .ok()
        .map(LoadedSave::File)
}

/// Writes `bytes` to `path` via a temp file and rename, so a reader never
/// sees half a save. Creates the directory only when it's missing.
fn write_atomically(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    if let Err(e) = std::fs::write(&tmp, bytes) {
        if e.kind() != std::io::ErrorKind::NotFound {
            return Err(e);
        }
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(&tmp, bytes)?;
    }
    std::fs::rename(&tmp, path)
}

pub(crate) fn commit_save(cache: &SaveCache, dir: &Path, data: ChunkSaveData) {
    let pos = data.header.chunk_pos;
    let data = Arc::new(data);
    lock(&cache.0.map).insert(pos, data.clone());
    let cache = cache.clone();
    let path = chunk_path(dir, pos);
    AsyncComputeTaskPool::get()
        .spawn(async move {
            let _guard = cache.0.write_lock(pos).map(lock);
            // A newer save for this chunk superseded us while we waited: skip.
            let current = lock(&cache.0.map).get(&pos).cloned();
            if !current.is_some_and(|c| Arc::ptr_eq(&c, &data)) {
                return;
            }
            match data.encode() {
                Ok(bytes) => {
                    if let Err(e) = write_atomically(&path, &bytes) {
                        eprintln!("Failed to write chunk save {pos:?}: {e}");
                        return; // keep it cached so loads still see it
                    }
                }
                Err(e) => {
                    eprintln!("Failed to serialize chunk {pos:?}: {e}");
                    return;
                }
            }
            let mut map = lock(&cache.0.map);
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
pub(crate) struct ChunkSaveState(pub u64);

#[derive(Clone, Copy)]
pub(crate) struct FieldVTable {
    pub type_id: &'static str,
    /// `VoxelField::OPTIONAL`: removed, not defaulted, when a load has no data.
    pub optional: bool,
    pub save: fn(&World, Entity) -> Option<Vec<u8>>,
    pub version: fn(&World, Entity) -> u64,
    /// Exact downsample of a serialized payload (`from >= to`).
    pub downsample: fn(&[u8], LOD, LOD) -> Vec<u8>,
    /// Builds the field for `lod` from `bytes`. `None` (or a size mismatch)
    /// gives the default field, or its removal if OPTIONAL. Pure, so it runs
    /// off the main thread; the result only inserts on it.
    pub prepare: fn(LOD, Option<&[u8]>) -> PreparedField,
}

/// A field built off-thread, applied to its chunk on the main thread.
///
/// A trait rather than a boxed `FnOnce(&mut EntityWorldMut)`: that closure
/// type is higher-ranked, and returning it from a `TaskPool::scope` task
/// makes the compiler demand the scope's borrows be `'static`.
pub(crate) trait ApplyField: Send {
    fn apply(self: Box<Self>, em: &mut EntityWorldMut<'_>);
}

pub(crate) type PreparedField = Box<dyn ApplyField>;

/// Inserts a built field.
struct InsertField<F>(F);

impl<F: Component> ApplyField for InsertField<F> {
    fn apply(self: Box<Self>, em: &mut EntityWorldMut<'_>) {
        em.insert(self.0);
    }
}

/// Removes an optional field that has no data.
struct RemoveField<F>(PhantomData<fn() -> F>);

impl<F: Component> ApplyField for RemoveField<F> {
    fn apply(self: Box<Self>, em: &mut EntityWorldMut<'_>) {
        em.remove::<F>();
    }
}

#[derive(Resource, Default, Clone)]
pub(crate) struct ChunkPersistenceRegistry {
    pub(crate) fields: Vec<FieldVTable>,
}

/// Power-of-two LOD sizes and voxel i sitting at i*`CHUNK_SIZE/size` make
/// this an exact stride sample; no interpolation, no JFA.
#[must_use]
pub(crate) fn downsample_bytes<F: VoxelField>(bytes: &[u8], from: LOD, to: LOD) -> Vec<u8> {
    let elem = size_of::<F::Elem>();
    let (fs, ts) = (from.size(), to.size());
    // Bad input: install falls back to a default field.
    if Some(bytes.len()) != from.volume().checked_mul(elem) || fs < ts {
        return Vec::new();
    }
    if fs == ts {
        return bytes.to_vec();
    }
    let Some(ratio) = fs.checked_div(ts) else {
        return Vec::new();
    };
    // Reads the sampled elements straight from the (possibly unaligned)
    // bytes: copying the whole source first cost more than the sampling.
    let read = |i: usize| -> Option<F::Elem> {
        let start = i.checked_mul(elem)?;
        bytes
            .get(start..start.checked_add(elem)?)
            .map(bytemuck::pod_read_unaligned)
    };
    let at = |c: u32| c.wrapping_mul(ratio);
    let mut out: Vec<u8> = Vec::with_capacity(to.volume().saturating_mul(elem));
    for z in 0..ts {
        for y in 0..ts {
            for x in 0..ts {
                let Some(v) = read(flatten_cube(at(x), at(y), at(z), fs)) else {
                    return Vec::new();
                };
                out.extend_from_slice(bytemuck::bytes_of(&F::rescale(v, f32_from_u32(ratio))));
            }
        }
    }
    out
}

impl ChunkPersistenceRegistry {
    pub fn register_field<F>(&mut self)
    where
        F: VoxelField,
    {
        self.fields.push(FieldVTable {
            type_id: std::any::type_name::<F>(),
            optional: F::OPTIONAL,
            save: |w, e| {
                w.get::<F>(e)
                    .map(|f| bytemuck::cast_slice(f.data_slice()).to_vec())
            },
            version: |w, e| w.get::<F>(e).map_or(0, VoxelField::version),
            downsample: |b, from, to| downsample_bytes::<F>(b, from, to),
            prepare: |lod, bytes| {
                if let Some(b) = bytes {
                    let elems: Vec<F::Elem> = bytemuck::pod_collect_to_vec(b);
                    if elems.len() == lod.volume() {
                        return Box::new(InsertField(F::from_raw(lod, elems)));
                    }
                }
                if F::OPTIONAL {
                    Box::new(RemoveField::<F>(PhantomData))
                } else {
                    Box::new(InsertField(F::new_for_lod(lod)))
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

/// Saves and loads `F` with every chunk (see `AppVoxelFieldExt`).
pub(crate) fn register_saveable<F: VoxelField>(app: &mut App) {
    app.init_resource::<ChunkPersistenceRegistry>();
    app.world_mut()
        .resource_mut::<ChunkPersistenceRegistry>()
        .register_field::<F>();
}

// ---------------------------------------------------------------- saving ----

/// Some((data, fingerprint)) if the chunk is at max LOD, has no load in
/// flight, and has unsaved changes.
pub(crate) fn build_if_dirty(world: &World, e: Entity) -> Option<(ChunkSaveData, u64)> {
    let lod = *world.get::<LOD>(e)?;
    if lod != world.resource::<MaxEditLod>().0
        || world.get::<ChunkDataTask>(e).is_some()
        || is_settling(world, e)
    {
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
pub(crate) fn flush_and_despawn(world: &mut World, e: Entity) {
    if is_settling(world, e) {
        world.resource_mut::<ReplanRequested>().0 = true; // try again next frame
        return;
    }
    if let Some((data, _)) = build_if_dirty(world, e) {
        let cache = world.resource::<SaveCache>();
        commit_save(cache, &world.resource::<SaveDirectory>().0, data);
    }
    if let Ok(em) = world.get_entity_mut(e) {
        em.despawn();
    }
}

/// Saves `e` if it has unsaved changes and records the new save point.
fn save_if_dirty(world: &World, commands: &mut Commands, e: Entity) {
    if let Some((data, fp)) = build_if_dirty(world, e) {
        commit_save(
            world.resource::<SaveCache>(),
            &world.resource::<SaveDirectory>().0,
            data,
        );
        commands.entity(e).insert(ChunkSaveState(fp));
    }
}

#[derive(Resource)]
pub struct AutosaveIntervalSecs(pub f32);
impl Default for AutosaveIntervalSecs {
    fn default() -> Self {
        Self(2.0)
    }
}

pub(crate) fn autosave_dirty_chunks(
    world: &World,
    time: Res<Time>,
    interval: Res<AutosaveIntervalSecs>,
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
        save_if_dirty(world, &mut commands, e);
    }
}

/// Manual "save now". Only chunks at max LOD with unsaved changes are written.
#[derive(Message)]
pub struct SaveChunkMessage(pub IVec3);

pub(crate) fn generic_save_chunk_system(
    mut events: MessageReader<SaveChunkMessage>,
    world: &World,
    chunk_manager: Res<ChunkManager>,
    mut commands: Commands,
) {
    for SaveChunkMessage(pos) in events.read() {
        let Some(e) = chunk_manager.get_chunk(pos) else {
            warn!("Cannot save chunk at {:?}: not loaded", pos);
            continue;
        };
        save_if_dirty(world, &mut commands, e);
    }
}

pub struct ChunkPersistencePlugin;
impl Plugin for ChunkPersistencePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ChunkPersistenceRegistry>()
            .init_resource::<ChunkGeneratorRegistry>()
            .init_resource::<SaveCache>()
            .init_resource::<SaveDirectory>()
            .init_resource::<MaxEditLod>()
            .init_resource::<AutosaveIntervalSecs>()
            .add_message::<SaveChunkMessage>()
            .add_systems(Update, (generic_save_chunk_system, autosave_dirty_chunks));
    }
}

/// Edits are still being turned into real distances. Saving now would
/// persist pre-JFA data, and replacing the fields now would discard the edits.
pub(crate) fn is_settling(world: &World, e: Entity) -> bool {
    world.get::<SdfReinitTask>(e).is_some()
        || world.get::<DirtyField<SDF, f32>>(e).is_some()
        || world.get::<ReinitTask<VisibilityField>>(e).is_some()
        || world.get::<DirtyField<VisibilityField, f32>>(e).is_some()
}

#[cfg(test)]
mod tests {
    use super::{ChunkHeader, ChunkSaveData, ChunkSaveView, FieldSavePayload, downsample_bytes};
    use crate::field::{LOD, SDF, VisibilityField};
    use bevy::math::IVec3;

    fn sample() -> ChunkSaveData {
        let sdf: Vec<f32> = (0..LOD::Medium.volume())
            .map(crate::num::f32_from_usize)
            .collect();
        ChunkSaveData {
            header: ChunkHeader {
                chunk_pos: IVec3::new(-3, 7, 1),
                lod: LOD::Medium,
                version: 1,
            },
            fields: vec![
                FieldSavePayload {
                    field_type_id: "sdf".into(),
                    bytes: bytemuck::cast_slice(&sdf).to_vec(),
                },
                FieldSavePayload {
                    field_type_id: "mat".into(),
                    bytes: (0..LOD::Medium.volume())
                        .map(|i| u8::try_from(i % 251).unwrap())
                        .collect(),
                },
            ],
        }
    }

    /// The old encoding: field bytes as a plain serde sequence of u8s.
    #[derive(serde::Serialize)]
    struct LegacyField<'a> {
        field_type_id: &'a str,
        bytes: &'a Vec<u8>,
    }
    #[derive(serde::Serialize)]
    struct Legacy<'a> {
        header: ChunkHeader,
        fields: Vec<LegacyField<'a>>,
    }

    #[test]
    fn wire_format_matches_the_old_saves() {
        let data = sample();
        let legacy = Legacy {
            header: data.header,
            fields: data
                .fields
                .iter()
                .map(|f| LegacyField {
                    field_type_id: &f.field_type_id,
                    bytes: &f.bytes,
                })
                .collect(),
        };
        let old = postcard::to_allocvec(&legacy).unwrap();
        let new = data.encode().unwrap();
        assert_eq!(old, new);
    }

    #[test]
    fn round_trips_owned_and_borrowed() {
        let data = sample();
        let bytes = data.encode().unwrap();
        let view = ChunkSaveView::decode(&bytes).unwrap();
        assert_eq!(view.header, data.header);
        for (v, f) in view.fields.iter().zip(&data.fields) {
            assert_eq!(v.field_type_id, f.field_type_id);
            assert_eq!(v.bytes, f.bytes.as_slice());
        }
        assert_eq!(view.encode().unwrap(), bytes);
        let owned = ChunkSaveData::decode(&bytes).unwrap();
        assert_eq!(owned.encode().unwrap(), bytes);
        assert!(ChunkSaveView::decode(&bytes[..bytes.len() - 1]).is_err());
    }

    #[test]
    fn downsamples_unaligned_bytes() {
        let data = sample();
        let sdf = &data.fields[0].bytes;
        let mut shifted = vec![0u8];
        shifted.extend_from_slice(sdf);
        let unaligned = &shifted[1..];
        for to in [LOD::Lowest, LOD::Low, LOD::Medium] {
            assert_eq!(
                downsample_bytes::<SDF>(unaligned, LOD::Medium, to),
                downsample_bytes::<SDF>(sdf, LOD::Medium, to),
            );
        }
        let down = downsample_bytes::<VisibilityField>(sdf, LOD::Medium, LOD::Low);
        assert_eq!(down.len(), LOD::Low.volume() * 4);
        // wrong size: rejected
        assert_eq!(
            downsample_bytes::<SDF>(&sdf[1..], LOD::Medium, LOD::Low),
            Vec::<u8>::new()
        );
    }

    #[test]
    #[ignore = "timing; needs saves/chunks/chunk_0_0_0.bin"]
    fn timing() {
        use std::time::Instant;
        let bytes = std::fs::read("saves/chunks/chunk_0_0_0.bin").unwrap();
        let n = 200u32;
        let t = Instant::now();
        for _ in 0..n {
            std::hint::black_box(ChunkSaveView::decode(&bytes).unwrap());
        }
        eprintln!("view decode: {:?}", t.elapsed() / n);
        let t = Instant::now();
        let mut d = None;
        for _ in 0..n {
            d = Some(ChunkSaveData::decode(&bytes).unwrap());
        }
        eprintln!("owned decode: {:?}", t.elapsed() / n);
        let d = d.unwrap();
        let t = Instant::now();
        for _ in 0..n {
            std::hint::black_box(d.encode().unwrap());
        }
        eprintln!("encode: {:?}", t.elapsed() / n);
        let sdf = ChunkSaveView::decode(&bytes).unwrap().fields[0].bytes;
        for to in LOD::ALL {
            let t = Instant::now();
            for _ in 0..n {
                std::hint::black_box(downsample_bytes::<SDF>(sdf, LOD::High, to));
            }
            eprintln!("downsample SDF -> {to:?}: {:?}", t.elapsed() / n);
        }
    }
}
