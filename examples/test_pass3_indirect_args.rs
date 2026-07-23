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
struct Pass3TestState {
    test_complete: bool,
    index_count: u32,
}

#[derive(Resource, Clone, Default)]
struct SharedPass3State(Arc<Mutex<Pass3TestState>>);

fn main() {
    let mut app = App::new();
    let shared_state = SharedPass3State::default();

    app.add_plugins((DefaultPlugins, VoxelRenderPlugin, ChunkPlugin));
    app.insert_resource(shared_state.clone());

    app.add_systems(Startup, (setup_sphere_chunk, setup_camera));
    app.add_systems(Update, pass3_test_controller);

    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app.insert_resource(shared_state);
        render_app.add_systems(
            Render,
            verify_pass3_indirect_results.in_set(RenderSystems::Cleanup),
        );
    }

    println!("🚀 Starting Pass 3 Index Generation & Indirect Args Test...");
    app.run();
}

fn setup_camera(mut commands: Commands) {
    commands.spawn(Camera3d::default());
}

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
    println!("Spawned sphere SDF chunk for Pass 3 test");
}

fn pass3_test_controller(
    shared_state: Res<SharedPass3State>,
    mut app_exit: MessageWriter<AppExit>,
) {
    let state = shared_state.0.lock().unwrap();

    if state.test_complete {
        println!("\n🎉 PASS 3 INDIRECT ARGS TEST PASSED SUCCESSFULLY!");
        println!("Dynamic Index Count written by GPU: {}", state.index_count);
        app_exit.write(AppExit::Success);
    }
}

fn verify_pass3_indirect_results(
    gpu_buffers_query: Query<(&MainEntity, &GpuVoxelChunkBuffers)>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    shared_state: Res<SharedPass3State>,
) {
    let mut state = shared_state.0.lock().unwrap();
    if state.test_complete {
        return;
    }

    for (_main_entity, buffers) in gpu_buffers_query.iter() {
        let indirect_size = buffers.indirect_args_buffer.size();

        let staging_indirect = render_device.create_buffer(&BufferDescriptor {
            label: Some("indirect_args_readback_staging"),
            size: indirect_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = render_device.create_command_encoder(&default());
        encoder.copy_buffer_to_buffer(
            &buffers.indirect_args_buffer,
            0,
            &staging_indirect,
            0,
            indirect_size,
        );
        render_queue.submit(Some(encoder.finish()));

        let slice = staging_indirect.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();

        slice.map_async(MapMode::Read, move |res| {
            sender.send(res).unwrap();
        });

        render_device
            .wgpu_device()
            .poll(PollType::wait_indefinitely());

        if receiver.recv().unwrap().is_ok() {
            let data = slice.get_mapped_range();
            let args: &[u32] = bytemuck::cast_slice(&data);

            // WebGPU DrawIndexedIndirect layout:
            // [0]: index_count, [1]: instance_count, [2]: first_index, [3]: base_vertex, [4]: first_instance
            let index_count = args[0];
            let instance_count = args[1];

            println!(
                "DEBUG Indirect Args -> index_count: {}, instance_count: {}",
                index_count, instance_count
            );

            if index_count > 0 {
                state.index_count = index_count;
                state.test_complete = true;
            } else {
                println!("⏳ Pass 3 compute execution pending, retrying...");
            }
        }

        staging_indirect.unmap();
        if state.test_complete {
            break;
        }
    }
}
