use std::fs;
use std::path::PathBuf;
use std::thread::sleep;
use std::time::Duration;

use bevy::prelude::*;
use tarasaur::Chunk;
use tarasaur::chunk::{ChunkManager, ChunkPosition, NewChunkSpawned};
use tarasaur::field::persistence::{
    ChunkPersistencePlugin, RegisterSaveableFieldExt, SaveChunkMessage,
};
use tarasaur::field::{Field, FieldLOD, SDF};

fn main() {
    let mut app = App::new();

    // 1. Add MinimalPlugins, the Persistence Plugin, and register SDFField
    app.add_plugins(MinimalPlugins)
        .add_plugins(ChunkPersistencePlugin)
        .register_saveable_field::<SDF>();

    // 2. Setup test chunk state
    let chunk_pos = IVec3::new(0, 0, 0);
    let mut sdf_field = SDF::default();

    // Initialize underlying buffer before setting voxels
    sdf_field.reinit();

    let lod = sdf_field.lod();
    let size = lod.size();

    // Populate test values safely using the Field trait
    let mut expected_floats = Vec::with_capacity(lod.volume());
    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let val = (x + y * size + z * size * size) as f32 * 0.1;
                Field::set(&mut sdf_field, x, y, z, val);
                expected_floats.push(val);
            }
        }
    }

    app.init_resource::<ChunkManager>();
    // Spawn chunk entity with position and SDF field
    let entity = app
        .world_mut()
        .spawn((Chunk, ChunkPosition(chunk_pos), lod, sdf_field))
        .id();

    // Initialize ChunkManager resource

    println!("--- Step 1: Saving Chunk at {:?} ---", chunk_pos);

    // 3. Send SaveChunkMessage and step the app schedule once
    app.world_mut().write_message(SaveChunkMessage(chunk_pos));
    app.update();

    // Wait briefly for AsyncComputeTaskPool to complete file write
    sleep(Duration::from_millis(1000));

    let save_path = PathBuf::from("./saves/chunks/chunk_0_0_0.bin");
    assert!(
        save_path.exists(),
        "Expected save file to exist at {:?}",
        save_path
    );
    println!("Successfully saved chunk binary to disk!");

    // 4. Despawn original entity to simulate unloading
    println!("\n--- Step 2: Despawning original chunk entity ---");
    app.world_mut().despawn(entity);

    // 5. Spawn new empty entity with initialized SDFField and trigger `NewChunkSpawned`
    println!("\n--- Step 3: Triggering reload on new chunk spawn ---");
    let mut new_sdf_field = SDF::default();
    new_sdf_field.reinit();
    let new_lod = new_sdf_field.lod();

    let new_entity = app
        .world_mut()
        .spawn((Chunk, ChunkPosition(chunk_pos), new_lod, new_sdf_field))
        .id();

    app.world_mut().trigger(NewChunkSpawned {
        entity: new_entity,
        chunk_position: chunk_pos,
        world_position: chunk_pos.as_vec3(),
    });

    // Step app schedule to process commands queued by `load_chunk_on_spawn`
    app.update();

    // 6. Verify restored data
    let restored_field = app
        .world()
        .get::<SDF>(new_entity)
        .expect("SDFField should exist on restored entity");

    let restored_floats: &[f32] = bytemuck::cast_slice(restored_field.data_slice());

    assert_eq!(
        restored_floats,
        expected_floats.as_slice(),
        "Restored payload does not match original binary state!"
    );

    println!("\nSUCCESS: Restored SDFField payload matches target data!");

    // Cleanup generated file and directory
    let _ = fs::remove_file(&save_path);
    let _ = fs::remove_dir("./saves/chunks");
}
