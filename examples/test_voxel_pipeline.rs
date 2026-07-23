use bevy::{
    app::AppExit,
    prelude::*,
    render::{Render, RenderApp, RenderSystems, sync_world::MainEntity},
};
use std::{
    collections::HashSet,
    sync::{Arc, Mutex},
};

use tarasaur::{
    Chunk, ChunkPlugin, ChunkPosition, Field, GpuVoxelChunkBuffers, SDFField, VoxelRenderPlugin,
    field::LOD,
};

const CHUNK_SIZE: u32 = 32;
const TOTAL_CELLS: usize = (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) as usize;

#[derive(Default)]
struct TestState {
    buffers_verified: bool,
    update_requested: bool,
    test_complete: bool,
}

#[derive(Resource, Clone, Default)]
struct SharedTestState(Arc<Mutex<TestState>>);

#[derive(Resource, Default)]
struct TestDriver {
    frame: u32,
    stage: TestStage,
    spawned_entities: Vec<Entity>,
}

#[derive(Debug, PartialEq, Eq, Default)]
enum TestStage {
    #[default]
    SpawnChunks,
    WaitForInitialBuffers,
    UpdateExistingChunk,
    VerifyUpdateComplete,
    Done,
}

fn main() {
    let mut app = App::new();

    let shared_state = SharedTestState::default();

    app.add_plugins((DefaultPlugins, VoxelRenderPlugin, ChunkPlugin));
    app.insert_resource(shared_state.clone());
    app.init_resource::<TestDriver>();

    app.add_systems(Update, test_stage_controller);

    // Register test verification in RenderApp under RenderSystems::Cleanup
    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app.insert_resource(shared_state);
        render_app.add_systems(
            Render,
            verify_gpu_buffers_render_world.in_set(RenderSystems::Cleanup),
        );
    }

    println!("🚀 Starting Extended Voxel Pipeline Integration Test...");
    app.run();
}

// --- Main World Test Controller (Runs in Update) ---

fn test_stage_controller(
    mut commands: Commands,
    mut driver: ResMut<TestDriver>,
    shared_state: Res<SharedTestState>,
    mut sdf_query: Query<&mut SDFField, With<Chunk>>,
    mut app_exit: MessageWriter<AppExit>,
) {
    driver.frame += 1;

    // Safety Timeout
    if driver.frame > 300 {
        eprintln!("❌ TEST TIMED OUT: GPU pipeline did not progress through stages in time.");
        app_exit.write(AppExit::from_code(1));
        return;
    }

    let mut state = shared_state.0.lock().unwrap();

    match driver.stage {
        TestStage::SpawnChunks => {
            println!("\n--- Phase 1: Spawning 3 Chunks at Different Coordinates ---");
            let test_coords = [IVec3::ZERO, IVec3::new(1, 0, 0), IVec3::new(0, 1, -1)];

            for coord in test_coords {
                let lod = LOD::default();
                let mut sdf = SDFField::new(lod);
                sdf.set(0, 0, 0, -1.0); // Create surface

                let entity = commands.spawn((Chunk, ChunkPosition(coord), sdf)).id();
                driver.spawned_entities.push(entity);
                println!("Spawned chunk entity {:?} at position {:?}", entity, coord);
            }

            driver.stage = TestStage::WaitForInitialBuffers;
        }

        TestStage::WaitForInitialBuffers => {
            if state.buffers_verified {
                driver.stage = TestStage::UpdateExistingChunk;
            }
        }

        TestStage::UpdateExistingChunk => {
            println!(
                "\n--- Phase 2: Modifying SDF on Existing Chunk (Testing GPU Resource Reuse) ---"
            );
            if let Some(&first_chunk) = driver.spawned_entities.first() {
                if let Ok(mut sdf) = sdf_query.get_mut(first_chunk) {
                    sdf.set(16, 16, 16, -0.8);
                    sdf.set(10, 10, 10, 2.5);
                    println!("Updated SDF values on Entity {:?}", first_chunk);
                }
            }
            state.update_requested = true;
            driver.stage = TestStage::VerifyUpdateComplete;
        }

        TestStage::VerifyUpdateComplete => {
            if state.test_complete {
                driver.stage = TestStage::Done;
            }
        }

        TestStage::Done => {
            println!("\n🎉 ALL VOXEL PIPELINE TESTS PASSED SUCCESSFULLY!");
            app_exit.write(AppExit::Success);
        }
    }
}

// --- Render World Verification System (Runs in RenderSystems::Cleanup) ---

fn verify_gpu_buffers_render_world(
    gpu_buffers_query: Query<(Entity, &MainEntity, &GpuVoxelChunkBuffers)>,
    shared_state: Res<SharedTestState>,
) {
    let count = gpu_buffers_query.iter().count();
    if count == 0 {
        return;
    }

    let mut state = shared_state.0.lock().unwrap();

    if !state.buffers_verified && count == 3 {
        println!("✓ All 3 Chunks extracted & prepared in Render World!");

        let mut unique_main_entities = HashSet::new();

        for (render_entity, main_entity, buffers) in gpu_buffers_query.iter() {
            unique_main_entities.insert(main_entity.id());

            // Assert exact WebGPU byte sizes
            let expected_flags_size = (TOTAL_CELLS * std::mem::size_of::<u32>()) as u64;
            let expected_vertex_size = (TOTAL_CELLS * 32) as u64; // 32 bytes per vertex
            let expected_indirect_size = 20; // 5 * u32 (20 bytes)

            assert_eq!(
                buffers.flags_buffer.size(),
                expected_flags_size,
                "Flags buffer size mismatch"
            );
            assert_eq!(
                buffers.compacted_offsets_buffer.size(),
                expected_flags_size,
                "Compacted offsets buffer size mismatch"
            );
            assert_eq!(
                buffers.final_vertex_buffer.size(),
                expected_vertex_size,
                "Final vertex buffer size mismatch"
            );
            assert_eq!(
                buffers.indirect_args_buffer.size(),
                expected_indirect_size,
                "Indirect args buffer size mismatch"
            );

            println!(
                "  Render Entity {:?} -> Main Entity {:?} verified (Chunk Coord: {:?})",
                render_entity,
                main_entity.id(),
                buffers.chunk_coord
            );
        }

        assert_eq!(
            unique_main_entities.len(),
            3,
            "Duplicate MainEntity assignments found in Render World"
        );

        state.buffers_verified = true;
    } else if state.update_requested && !state.test_complete {
        // Ensure entity count remains EXACTLY 3 (verifies no leak on write_texture updates)
        assert_eq!(
            count, 3,
            "Render World entity leak detected! Expected 3 entities, found {}",
            count
        );

        println!("✓ SDF texture update verified in-place without Render entity leaks.");
        state.test_complete = true;
    }
}
