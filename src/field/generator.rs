// field/generator.rs
use std::sync::Arc;

use bevy::prelude::*;

use crate::LOD;
use crate::field::SDF;
use crate::persistence::FieldSavePayload;

/// A registered generator: pure function of chunk position + LOD, no ECS
/// access. That purity is what makes it safe to clone into and run inside
/// an `AsyncComputeTaskPool` task — see `ChunkPersistencePlugin`'s
/// `spawn_chunk_data_task` observer.
type GenerateFn = Arc<dyn Fn(IVec3, LOD) -> FieldSavePayload + Send + Sync>;

#[derive(Resource, Default, Clone)]
pub struct ChunkGeneratorRegistry {
    generators: Vec<GenerateFn>,
}

impl ChunkGeneratorRegistry {
    /// Registers a raw-buffer generator for field type `field_type_id`.
    /// `generate` must produce exactly `lod.volume()` elements in
    /// x/y/z-flattened order — the same layout `RestorableField` impls
    /// expect when they later deserialize this payload's bytes.
    pub fn register<V: bytemuck::Pod>(
        &mut self,
        field_type_id: &'static str,
        generate: impl Fn(IVec3, LOD) -> Vec<V> + Send + Sync + 'static,
    ) {
        self.generators
            .push(Arc::new(move |pos, lod| FieldSavePayload {
                field_type_id: field_type_id.to_string(),
                bytes: bytemuck::cast_slice(&generate(pos, lod)).to_vec(),
            }));
    }

    /// For generators that already return an approximate signed distance in
    /// WORLD units (negative inside). Converts to voxel units of the chunk's own
    /// LOD and keeps the values as-is, so the surface keeps its sub-voxel
    /// information. No jump flood runs on this path.
    pub fn register_sdf_distance(
        &mut self,
        generate: impl Fn(IVec3, LOD) -> Vec<f32> + Send + Sync + 'static,
    ) {
        self.register::<f32>(std::any::type_name::<SDF>(), move |pos, lod| {
            let inv_voxel = 1.0 / lod.voxel_size();
            let mut data = generate(pos, lod);
            debug_assert_eq!(data.len(), lod.volume());
            for v in data.iter_mut() {
                *v *= inv_voxel;
            }
            data
        });
    }

    /// Runs every registered generator for this chunk and collects their
    /// output — same shape as a loaded save file's `fields` list, so both
    /// paths converge on the same apply step in `persistence.rs`.
    pub(crate) fn generate_all(&self, chunk_pos: IVec3, lod: LOD) -> Vec<FieldSavePayload> {
        self.generators
            .iter()
            .map(|generate| generate(chunk_pos, lod))
            .collect()
    }
}
