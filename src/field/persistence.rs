use std::collections::HashMap;
use std::path::{Path, PathBuf};

use bevy::ecs::component::Mutable;
use bevy::ecs::message::MessageReader;
use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, poll_once};
use serde::{Deserialize, Serialize};

use crate::LOD;
use crate::chunk::{ChunkManager, NewChunkSpawned};
use crate::field::generator::ChunkGeneratorRegistry;
use crate::field::material::VoxelMaterial;
use crate::field::{
    Field, FieldLOD, MaterialField, SDF, Versionable, VisibilityField, VoxelDataSlice,
};

// ============================================================================
// Data Transfer Objects & Headers
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkHeader {
    pub chunk_pos: IVec3,
    pub lod: LOD,
    pub version: u32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FieldSavePayload {
    /// Unique identifier for the field type (e.g., "SDFField", "VisibilityField", "MaterialField<Dirt>")
    pub field_type_id: String,
    /// Raw binary payload
    pub bytes: Vec<u8>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ChunkSaveData {
    pub header: ChunkHeader,
    pub fields: Vec<FieldSavePayload>,
}

// ============================================================================
// Field Persistence Traits
// ============================================================================

/// Trait allowing any field component to serialize its state.
pub trait SaveableField: Component + FieldLOD {
    fn field_type_id(&self) -> &'static str;
    fn save_payload(&self) -> Vec<u8>;
}

/// Trait allowing any field component to restore its state from serialized bytes.
pub trait RestorableField: Component + FieldLOD {
    fn field_type_id() -> &'static str;
    fn restore_from_bytes(&mut self, bytes: &[u8]);
}

// Blanket implementation of `SaveableField` for any component implementing `VoxelDataSlice`
impl<F> SaveableField for F
where
    F: Component + VoxelDataSlice + FieldLOD,
    <F as VoxelDataSlice>::Elem: bytemuck::Pod,
{
    fn field_type_id(&self) -> &'static str {
        std::any::type_name::<F>()
    }

    fn save_payload(&self) -> Vec<u8> {
        bytemuck::cast_slice(self.data_slice()).to_vec()
    }
}

// ============================================================================
// Field Restorable Implementations
// ============================================================================

impl RestorableField for SDF {
    fn field_type_id() -> &'static str {
        std::any::type_name::<SDF>()
    }

    fn restore_from_bytes(&mut self, bytes: &[u8]) {
        let lod = self.lod();
        let floats: &[f32] = bytemuck::cast_slice(bytes);

        if floats.len() == lod.volume() {
            // Ensure buffer is allocated first
            if self.data_slice().is_empty() {
                self.reinit();
            }

            let data_slice = unsafe {
                std::slice::from_raw_parts_mut(self.data_slice().as_ptr() as *mut f32, floats.len())
            };

            data_slice.copy_from_slice(floats);
            self.incorment_version();
            // DO NOT call self.reinit() here, as it re-initializes/clears the field buffer!
        }
    }
}

impl RestorableField for VisibilityField {
    fn field_type_id() -> &'static str {
        std::any::type_name::<VisibilityField>()
    }

    fn restore_from_bytes(&mut self, bytes: &[u8]) {
        let lod = self.lod();
        if bytes.len() == lod.volume() {
            for (idx, &val) in bytes.iter().enumerate() {
                let x = (idx as u32) % lod.size();
                let y = ((idx as u32) / lod.size()) % lod.size();
                let z = (idx as u32) / (lod.size() * lod.size());
                Field::set(self, x, y, z, val != 0);
            }
            self.incorment_version();
        }
    }
}

impl<M: VoxelMaterial> RestorableField for MaterialField<M> {
    fn field_type_id() -> &'static str {
        std::any::type_name::<MaterialField<M>>()
    }

    fn restore_from_bytes(&mut self, bytes: &[u8]) {
        let lod = self.lod();
        if bytes.len() == lod.volume() {
            let data_slice = unsafe {
                std::slice::from_raw_parts_mut(self.data_slice().as_ptr() as *mut u8, bytes.len())
            };
            data_slice.copy_from_slice(bytes);
            self.incorment_version();
        }
    }
}

// ============================================================================
// Registry Architecture
// ============================================================================

type FieldSaverFn = fn(&World, Entity) -> Option<FieldSavePayload>;
pub(crate) type FieldLoaderFn = fn(&mut World, Entity, &[u8]);

#[derive(Resource, Default)]
pub struct ChunkPersistenceRegistry {
    savers: Vec<FieldSaverFn>,
    loaders: HashMap<&'static str, FieldLoaderFn>,
}

impl ChunkPersistenceRegistry {
    pub fn register_field<F>(&mut self)
    where
        F: SaveableField + RestorableField + Component<Mutability = Mutable>,
    {
        self.savers.push(|world, entity| {
            world.get::<F>(entity).map(|field| FieldSavePayload {
                field_type_id: SaveableField::field_type_id(field).to_string(),
                bytes: field.save_payload(),
            })
        });

        self.loaders.insert(
            <F as RestorableField>::field_type_id(),
            |world, entity, bytes| {
                if let Some(mut field) = world.get_mut::<F>(entity) {
                    field.restore_from_bytes(bytes);
                }
            },
        );
    }

    /// Looks up the loader closure for a given field type id — used by
    /// `resolve_chunk_data_tasks` to apply both loaded and generated
    /// payloads through the same path.
    pub fn loader_for(&self, field_type_id: &str) -> Option<FieldLoaderFn> {
        self.loaders.get(field_type_id).copied()
    }
}

// ============================================================================
// LOD Save Guard Logic
// ============================================================================

/// Evaluates whether an existing save on disk contains a higher detail level (higher LOD value)
/// than the candidate save. Returns `true` if safe to overwrite.
fn should_overwrite_save(file_path: &Path, candidate_lod: LOD) -> bool {
    if !file_path.exists() {
        return true;
    }

    if let Ok(bytes) = std::fs::read(file_path) {
        if let Ok(existing_data) = postcard::from_bytes::<ChunkSaveData>(&bytes) {
            if existing_data.header.lod.size() > candidate_lod.size() {
                return false;
            }
        }
    }

    true
}

// ============================================================================
// Systems & Observers — saving
// ============================================================================

#[derive(Message)]
pub struct SaveChunkMessage(pub IVec3);

pub fn generic_save_chunk_system(
    mut events: MessageReader<SaveChunkMessage>,
    world: &World,
    chunk_manager: Res<ChunkManager>,
    registry: Res<ChunkPersistenceRegistry>,
) {
    let save_dir = PathBuf::from("./saves/chunks");

    for event in events.read() {
        let chunk_pos = event.0;

        let Some(entity) = chunk_manager.get_chunk(&chunk_pos) else {
            warn!("Cannot save chunk at {:?}: Not loaded", chunk_pos);
            continue;
        };

        let lod = world.get::<LOD>(entity).copied().unwrap_or_default();
        let file_path = save_dir.join(format!(
            "chunk_{}_{}_{}.bin",
            chunk_pos.x, chunk_pos.y, chunk_pos.z
        ));

        // Guard against clobbering higher resolution LOD files on disk
        if !should_overwrite_save(&file_path, lod) {
            info!(
                "Skipping save for chunk {:?}: File on disk has higher LOD resolution.",
                chunk_pos
            );
            continue;
        }

        let mut fields = Vec::new();
        for saver in &registry.savers {
            if let Some(payload) = saver(world, entity) {
                fields.push(payload);
            }
        }

        let save_data = ChunkSaveData {
            header: ChunkHeader {
                chunk_pos,
                lod,
                version: 1,
            },
            fields,
        };

        let dir = save_dir.clone();
        AsyncComputeTaskPool::get()
            .spawn(async move {
                if let Err(e) = std::fs::create_dir_all(&dir) {
                    eprintln!("Failed to create save path: {e}");
                    return;
                }

                if let Ok(bytes) = postcard::to_allocvec(&save_data) {
                    if let Err(e) = std::fs::write(file_path, bytes) {
                        eprintln!("Failed to write chunk save: {e}");
                    }
                }
            })
            .detach();
    }
}

// ============================================================================
// Systems & Observers — loading / generating
// ============================================================================

/// Attached to a chunk entity while its initial data (loaded from disk, or
/// produced by `ChunkGeneratorRegistry` when no save exists) is being
/// resolved off the main thread. Removed by `resolve_chunk_data_tasks`
/// once the task completes and its payloads have been applied.
#[derive(Component)]
pub struct ChunkDataTask(Task<Vec<FieldSavePayload>>);

/// Fires on every new chunk. Spawns a background task that checks disk
/// first and falls back to generation — both branches are pure I/O/CPU
/// work with no ECS access, so both run off the main thread identically.
/// Nothing here blocks: the observer just kicks the task off and returns.
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

    let Ok(&lod) = lod_q.get(entity) else {
        return;
    };

    let generators = generators.clone(); // Arc<dyn Fn> clones are cheap
    let save_path = PathBuf::from(format!(
        "./saves/chunks/chunk_{}_{}_{}.bin",
        chunk_position.x, chunk_position.y, chunk_position.z
    ));

    let task = AsyncComputeTaskPool::get().spawn(async move {
        if let Ok(bytes) = std::fs::read(&save_path) {
            if let Ok(save_data) = postcard::from_bytes::<ChunkSaveData>(&bytes) {
                return save_data.fields;
            }
        }
        generators.generate_all(chunk_position, lod)
    });

    commands.entity(entity).insert(ChunkDataTask(task));
}

/// Runs every frame. Non-blocking poll of every in-flight
/// `ChunkDataTask`; on completion, applies each payload through the same
/// `ChunkPersistenceRegistry` loader closures a save-file restore uses —
/// loaded and generated data are indistinguishable from this point on.
pub fn resolve_chunk_data_tasks(
    mut commands: Commands,
    mut tasks: Query<(Entity, &mut ChunkDataTask)>,
    registry: Res<ChunkPersistenceRegistry>,
) {
    for (entity, mut task) in tasks.iter_mut() {
        let Some(payloads) = block_on(poll_once(&mut task.0)) else {
            continue;
        };

        for payload in payloads {
            if let Some(loader) = registry.loader_for(&payload.field_type_id) {
                commands.queue(move |world: &mut World| loader(world, entity, &payload.bytes));
            } else {
                warn!(
                    "[resolve_chunk_data_tasks] no loader registered for field type {:?}, dropping payload",
                    payload.field_type_id
                );
            }
        }

        commands.entity(entity).remove::<ChunkDataTask>();
    }
}

// ============================================================================
// App Builder Extensions & Plugin
// ============================================================================

pub trait RegisterSaveableFieldExt {
    fn register_saveable_field<F>(&mut self) -> &mut Self
    where
        F: SaveableField + RestorableField + Component<Mutability = Mutable>;
}

impl RegisterSaveableFieldExt for App {
    fn register_saveable_field<F>(&mut self) -> &mut Self
    where
        F: SaveableField + RestorableField + Component<Mutability = Mutable>,
    {
        if !self.world().contains_resource::<ChunkPersistenceRegistry>() {
            self.init_resource::<ChunkPersistenceRegistry>();
        }

        self.world_mut()
            .resource_mut::<ChunkPersistenceRegistry>()
            .register_field::<F>();

        self
    }
}

pub struct ChunkPersistencePlugin;

impl Plugin for ChunkPersistencePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ChunkPersistenceRegistry>()
            .init_resource::<ChunkGeneratorRegistry>()
            .add_message::<SaveChunkMessage>()
            .add_systems(
                Update,
                (generic_save_chunk_system, resolve_chunk_data_tasks),
            )
            .add_observer(spawn_chunk_data_task);
    }
}
