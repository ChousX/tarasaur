use bevy::{
    app::AppExit,
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        render_resource::{BufferDescriptor, BufferUsages, MapMode, PollType},
        renderer::{RenderDevice, RenderQueue},
        sync_world::MainEntity,
    },
};
use std::sync::{Arc, Mutex};

use tarasaur::{
    Chunk, ChunkPlugin, ChunkPosition, Field, GpuVoxelChunkBuffers, SDFField, VoxelRenderPlugin,
    field::LOD,
};

const CHUNK_SIZE: u32 = 32;

#[derive(Default)]
struct Pass1TestState {
    dispatch_verified: bool,
    active_cell_count: u32,
    test_complete: bool,
}

#[derive(Resource, Clone, Default)]
struct SharedPass1State(Arc<Mutex<Pass1TestState>>);

fn main() {
    let mut app = App::new();
    let shared_state = SharedPass1State::default();

    app.add_plugins((DefaultPlugins, VoxelRenderPlugin, ChunkPlugin));
    app.insert_resource(shared_state.clone());

    app.add_systems(Startup, (setup_sphere_chunk, setup_camera));
    app.add_systems(Update, pass1_test_controller);

    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app.insert_resource(shared_state);
        render_app.add_systems(
            Render,
            verify_pass1_compute_results.in_set(RenderSystems::Cleanup),
        );
    }

    println!("🚀 Starting Pass 1 Compute Shader Integration Test...");
    app.run();
}
fn setup_camera(mut commands: Commands) {
    commands.spawn(Camera3d::default());
}
/// Spawns a chunk populated with a sphere SDF so surface intersections exist
fn setup_sphere_chunk(mut commands: Commands) {
    let lod = LOD::default();
    let mut sdf = SDFField::new(lod);

    let center = Vec3::splat(CHUNK_SIZE as f32 / 2.0);
    let radius = 10.0f32;

    for z in 0..CHUNK_SIZE {
        for y in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                let pos = Vec3::new(x as f32, y as f32, z as f32);
                let dist = pos.distance(center) - radius;
                sdf.set(x, y, z, dist);
            }
        }
    }

    commands.spawn((Chunk, ChunkPosition(IVec3::ZERO), sdf));
    println!("Spawned sphere SDF chunk at (0, 0, 0)");
}

// --- Main World Controller ---

fn pass1_test_controller(
    shared_state: Res<SharedPass1State>,
    mut app_exit: MessageWriter<AppExit>,
) {
    let state = shared_state.0.lock().unwrap();

    if state.test_complete {
        println!("\n🎉 PASS 1 COMPUTE TEST PASSED SUCCESSFULLY!");
        println!(
            "Active surface cells flagged by GPU: {}",
            state.active_cell_count
        );
        app_exit.write(AppExit::Success);
    }
}

// --- Render World Pass 1 Verification System ---

fn verify_pass1_compute_results(
    gpu_buffers_query: Query<(&MainEntity, &GpuVoxelChunkBuffers)>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    shared_state: Res<SharedPass1State>,
) {
    let mut state = shared_state.0.lock().unwrap();
    if state.test_complete {
        return;
    }

    for (_main_entity, buffers) in gpu_buffers_query.iter() {
        let buffer_size = buffers.flags_buffer.size();

        let staging_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("flags_readback_staging"),
            size: buffer_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = render_device.create_command_encoder(&default());
        encoder.copy_buffer_to_buffer(&buffers.flags_buffer, 0, &staging_buffer, 0, buffer_size);
        render_queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();

        buffer_slice.map_async(MapMode::Read, move |res| {
            sender.send(res).unwrap();
        });

        render_device
            .wgpu_device()
            .poll(PollType::wait_indefinitely());

        if receiver.recv().unwrap().is_ok() {
            let data = buffer_slice.get_mapped_range();
            let flags: &[u32] = bytemuck::cast_slice(&data);

            let active_count = flags.iter().filter(|&&flag| flag > 0).count() as u32;

            // Wait until the compute shader actually writes to the buffer
            if active_count > 0 {
                state.active_cell_count = active_count;
                state.test_complete = true;
            } else {
                println!("⏳ Compute pass not ready yet on this frame, retrying next frame...");
            }
        }

        staging_buffer.unmap();
        if state.test_complete {
            break;
        }
    }
}
