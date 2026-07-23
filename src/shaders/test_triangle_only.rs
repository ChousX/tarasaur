use bevy::{
    prelude::*,
    render::{Render, RenderApp, RenderSystems, render_resource::*, renderer::RenderDevice},
};
use tarasaur::{
    indirect_draw::{
        ExtractedVoxelChunks, GpuChunkDrawData, VoxelIndirectDrawPlugin, VoxelMaterialAsset,
    },
    voxel_pipeline::DrawIndexedIndirectArgs,
};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TestVertex {
    position: [f32; 4], // matches attribute 0 (Float32x4, offset 0)
    extra: [f32; 4], // matches attribute 1 (Float32x4, offset 16) - color/normal/whatever your shader expects
}

fn main() {
    let mut app = App::new();

    app.add_plugins(DefaultPlugins)
        .insert_resource(Msaa::Off) // rule out the MSAA/sample-count mismatch
        .add_plugins(VoxelIndirectDrawPlugin)
        .add_systems(Startup, setup_scene);

    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app
            .init_resource::<TestTriangleBuffers>()
            .add_systems(Render, seed_test_triangle.in_set(RenderSystems::Prepare));
    }

    app.run();
}

fn setup_scene(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 0.0, 6.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // VoxelIndirectDrawPlugin needs a material bind group before it will draw anything -
    // point it at any existing image so extract_voxel_material has something to bind.
    let texture_handle: Handle<Image> = asset_server.load("textures/terrain_albedo.png");
    commands.insert_resource(VoxelMaterialAsset { texture_handle });
}

#[derive(Resource)]
struct TestTriangleBuffers {
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    indirect_args_buffer: Buffer,
}

impl FromWorld for TestTriangleBuffers {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let vertices = [
            TestVertex {
                position: [-1.5, -1.0, 0.0, 1.0],
                extra: [1.0, 0.0, 0.0, 1.0],
            },
            TestVertex {
                position: [1.5, -1.0, 0.0, 1.0],
                extra: [0.0, 1.0, 0.0, 1.0],
            },
            TestVertex {
                position: [0.0, 1.5, 0.0, 1.0],
                extra: [0.0, 0.0, 1.0, 1.0],
            },
        ];

        let vertex_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("test_triangle_vertex_buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: BufferUsages::VERTEX,
        });

        let indices: [u32; 3] = [0, 1, 2];
        let index_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("test_triangle_index_buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: BufferUsages::INDEX,
        });

        let args = DrawIndexedIndirectArgs {
            index_count: 3, // <- the important part: non-zero, unlike your real pipeline right now
            instance_count: 1,
            first_index: 0,
            base_vertex: 0,
            first_instance: 0,
        };
        let indirect_args_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("test_triangle_indirect_args_buffer"),
            contents: bytemuck::bytes_of(&args),
            usage: BufferUsages::INDIRECT,
        });

        Self {
            vertex_buffer,
            index_buffer,
            indirect_args_buffer,
        }
    }
}

fn seed_test_triangle(
    test_buffers: Res<TestTriangleBuffers>,
    mut extracted: ResMut<ExtractedVoxelChunks>,
) {
    extracted.chunks.push(GpuChunkDrawData {
        vertex_buffer: test_buffers.vertex_buffer.clone(),
        index_buffer: test_buffers.index_buffer.clone(),
        indirect_args_buffer: test_buffers.indirect_args_buffer.clone(),
    });
}
