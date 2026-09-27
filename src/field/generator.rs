// field/generator.rs
use std::sync::Arc;

use bevy::prelude::*;

use crate::LOD;
use crate::field::SDF;
use crate::field::sdf::{PackedCoord, jump_flood_distance_field};
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

    /// Convenience wrapper for `SDFField`: caller only supplies the raw
    /// sign volume (negative = inside), and this runs the same Jump Flood
    /// pass `SDFField::reinit` uses to turn signs into real distances —
    /// so a generated chunk arrives with the same data shape a live edit
    /// would produce, not a naive step function.
    pub fn register_sdf(
        &mut self,
        generate_signs: impl Fn(IVec3, LOD) -> Vec<f32> + Send + Sync + 'static,
    ) {
        self.register::<f32>(std::any::type_name::<SDF>(), move |pos, lod| {
            let mut data = generate_signs(pos, lod);
            let volume = data.len();
            debug_assert_eq!(
                volume,
                lod.volume(),
                "register_sdf generator must return lod.volume() elements"
            );
            let mut seeds = vec![PackedCoord::EMPTY; volume];
            let mut scratch = vec![PackedCoord::EMPTY; volume];
            jump_flood_distance_field(&mut data, &mut seeds, &mut scratch, lod.size());
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
