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
    Field, SDFField,
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
    .add_plugins((
        VoxelRenderPlugin,
        VoxelIndexGenerationPlugin,
        VoxelIndirectDrawPlugin,
    ))
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
    // 1. Setup Camera positioned at (30.0, 30.0, 30.0) looking at (16.0, 16.0, 16.0)
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(15.0, 15.0, 15.0).looking_at(Vec3::new(5.0, 5.0, 5.0), Vec3::Y),
    ));

    // 2. Spawn a single sphere SDF chunk at origin (0, 0, 0) with radius 12.0 inside a 32^3 grid
    let chunk_size = 32;
    let lod = LOD::default();
    let mut sdf_field = SDFField::new(lod);

    let center = Vec3::new(16.0, 16.0, 16.0);
    let radius = 12.0;

    for z in 0..chunk_size {
        for y in 0..chunk_size {
            for x in 0..chunk_size {
                let pos = Vec3::new(x as f32, y as f32, z as f32);
                let dist = pos.distance(center) - radius;
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

            println!(
                "🔍 [GPU Readback] Indirect Draw Args -> index_count: {}, instance_count: {}",
                index_count, instance_count
            );

            state.checked = true;
        }

        staging_indirect.unmap();
        break;
    }
}
