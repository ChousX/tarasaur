use bevy::{
    asset::RenderAssetUsages,
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
    position: [f32; 4],
    normal: [f32; 4],
}

fn main() {
    let mut app = App::new();

    app.add_plugins(DefaultPlugins)
        .add_plugins((VoxelIndirectDrawPlugin, TestTrianglePlugin))
        .add_systems(Startup, setup_scene);

    app.run();
}

fn setup_scene(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    commands.spawn((
        Camera3d::default(),
        Msaa::Off,
        Transform::from_xyz(0.0, 0.0, 6.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    let image = Image::new_fill(
        Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[255, 255, 255, 255],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    let texture_handle = images.add(image);

    commands.insert_resource(VoxelMaterialAsset { texture_handle });
}

// --- The fix lives here: build() vs finish() ---

struct TestTrianglePlugin;

impl Plugin for TestTrianglePlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        // Safe to register the system here - it doesn't touch RenderDevice
        // until it actually runs, well after finish() has completed.
        render_app.add_systems(Render, seed_test_triangle.in_set(RenderSystems::Prepare));
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        // RenderDevice now exists - safe to construct buffers that need it.
        render_app.init_resource::<TestTriangleBuffers>();
    }
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

        let n = [0.0, 0.0, 1.0, 0.0];
        let vertices = [
            TestVertex {
                position: [-1.5, -1.0, 0.0, 1.0],
                normal: n,
            },
            TestVertex {
                position: [1.5, -1.0, 0.0, 1.0],
                normal: n,
            },
            TestVertex {
                position: [0.0, 1.5, 0.0, 1.0],
                normal: n,
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
            index_count: 3,
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
