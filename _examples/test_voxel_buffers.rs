use bevy::{
    app::{AppExit, ScheduleRunnerPlugin},
    prelude::*,
    render::{RenderApp, RenderPlugin},
};
use std::time::Duration;

use tarasaur::{
    Chunk, ChunkPlugin, ChunkPosition, Field, GpuVoxelChunkBuffers, SDFField, VoxelRenderPlugin,
    field::LOD,
};

fn main() {
    let mut app = App::new();

    app.add_plugins((DefaultPlugins, VoxelRenderPlugin, ChunkPlugin));

    app.add_systems(Startup, setup_test_chunk);

    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app.add_systems(Update, verify_gpu_buffers);
    }

    println!("Starting Voxel Pipeline Test...");
    app.run();
}

fn setup_test_chunk(mut commands: Commands) {
    let lod = LOD::default();
    let mut sdf_field = SDFField::new(lod);

    // Assign raw scalar grid buffer
    sdf_field.set(0, 0, 0, -1.0);

    commands.spawn((Chunk, ChunkPosition(IVec3::ZERO), sdf_field));

    println!("Spawned test chunk entity at (0, 0, 0)");
}

fn verify_gpu_buffers(
    query: Query<(Entity, &GpuVoxelChunkBuffers)>,
    mut app_exit: MessageWriter<AppExit>,
) {
    for (entity, buffers) in query.iter() {
        println!("=== GPU Voxel Buffer Verification Success ===");
        println!("Render Entity: {:?}", entity);
        println!("Chunk Position: {:?}", buffers.chunk_coord);
        println!("Flags Buffer Size: {} bytes", buffers.flags_buffer.size());
        println!(
            "Compacted Offsets Buffer Size: {} bytes",
            buffers.compacted_offsets_buffer.size()
        );
        println!(
            "Final Vertex Buffer Size: {} bytes",
            buffers.final_vertex_buffer.size()
        );
        println!(
            "Indirect Args Buffer Size: {} bytes",
            buffers.indirect_args_buffer.size()
        );
        println!("============================================");

        app_exit.write(AppExit::Success);
    }
}
