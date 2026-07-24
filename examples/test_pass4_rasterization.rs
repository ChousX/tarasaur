use bevy::{
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
    Field, SDFField, TarasaurPlugin,
    chunk::ChunkPosition,
    field::LOD,
    index_generation::VoxelIndexGenerationPlugin,
    indirect_draw::{VoxelIndirectDrawPlugin, VoxelMaterialAsset},
    voxel_pipeline::{GpuVoxelChunkBuffers, VoxelRenderPlugin},
};

#[derive(Default)]
struct RasterizationTestState {
    checked: bool,
}

#[derive(Resource, Clone, Default)]
struct SharedRasterizationState(Arc<Mutex<RasterizationTestState>>);

fn main() {
    let mut app = App::new();
    let shared_state = SharedRasterizationState::default();

    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "Voxel Triplanar Rasterization Test".into(),
            resolution: (1280, 720).into(),
            ..default()
        }),
        ..default()
    }))
    .add_plugins(TarasaurPlugin)
    .insert_resource(shared_state.clone())
    .add_systems(Startup, setup_scene);

    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app.insert_resource(shared_state);
        render_app.add_systems(
            Render,
            verify_indirect_args_debug.in_set(RenderSystems::Cleanup),
        );
    }

    app.run();
}

fn setup_scene(mut commands: Commands, asset_server: Res<AssetServer>) {
    let chunk_world_size = 10.0;
    let chunk_world_center = Vec3::splat(chunk_world_size / 2.0); // (5.0, 5.0, 5.0)

    // 1. Setup Camera positioned slightly outside the 10x10x10 chunk, looking at its center
    commands.spawn((
        Camera3d::default(),
        Msaa::Off,
        Transform::from_xyz(12.0, 12.0, 15.0).looking_at(chunk_world_center, Vec3::Y),
    ));

    // 2. Spawn a single sphere SDF chunk at origin (0, 0, 0) inside a 10x10x10 world bounds
    let chunk_resolution = 32; // Number of voxels across one dimension for this LOD
    let lod = LOD::default();
    let mut sdf_field = SDFField::new(lod);

    // Give the sphere a 4.0 unit radius in world space (fits well within the 10.0 chunk)
    let radius_world = 4.0;

    for z in 0..chunk_resolution {
        for y in 0..chunk_resolution {
            for x in 0..chunk_resolution {
                // Map the voxel index (0..32) into world space (0.0..10.0)
                let pos_world = Vec3::new(
                    (x as f32 / chunk_resolution as f32) * chunk_world_size,
                    (y as f32 / chunk_resolution as f32) * chunk_world_size,
                    (z as f32 / chunk_resolution as f32) * chunk_world_size,
                );

                // Calculate the SDF distance using world-space units
                let dist = pos_world.distance(chunk_world_center) - radius_world;
                sdf_field.set(x, y, z, dist);
            }
        }
    }

    commands.spawn((ChunkPosition(IVec3::ZERO), sdf_field));

    // 3. Bind fallback material or asset texture
    let texture_handle: Handle<Image> = asset_server.load("textures/terrain_albedo.png");
    commands.insert_resource(VoxelMaterialAsset { texture_handle });
}

fn verify_indirect_args_debug(
    gpu_buffers_query: Query<(&MainEntity, &GpuVoxelChunkBuffers)>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    shared_state: Res<SharedRasterizationState>,
) {
    let mut state = shared_state.0.lock().unwrap();
    if state.checked {
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

            let index_count = args[0];
            let instance_count = args[1];

            // Wait until the compute passes have successfully populated the indirect buffer
            if index_count > 0 {
                println!(
                    "🔍 [GPU Readback] Indirect Draw Args -> index_count: {}, instance_count: {}",
                    index_count, instance_count
                );

                state.checked = true;
            }
        }

        staging_indirect.unmap();
        break;
    }
}
