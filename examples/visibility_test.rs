// examples/visibility_test.rs
use bevy::{
    input::mouse::MouseMotion,
    prelude::*,
    render::{
        RenderPlugin,
        settings::{RenderCreation, WgpuLimits, WgpuSettings},
    },
};
use tarasaur::{
    Field, LOD, SDFField, TarasaurPlugin, VisibilityField, VoxelDataSlice,
    chunk::{CHUNK_SIZE, Chunk, ChunkPosition},
};

#[derive(Resource, Default)]
struct TestReady(bool);

#[derive(Component)]
struct FlyCam {
    move_speed: f32,
    sensitivity: f32,
    pitch: f32,
    yaw: f32,
}

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins.set(RenderPlugin {
                render_creation: RenderCreation::Automatic(Box::new(WgpuSettings {
                    limits: WgpuLimits {
                        max_buffer_size: 1024 * 1024 * 1024,
                        max_storage_buffer_binding_size: 512 * 1024 * 1024,
                        ..default()
                    }
                    .into(),
                    ..default()
                })),
                ..default()
            }),
        )
        // Deliberately no MaterialFieldPlugin/PalettePlugin — this test is
        // scoped to visibility only. Rendering falls back to
        // VoxelDummyMaterial's flat dummy texture, which is exactly what
        // we want here: if geometry shows up wrong, it's not a materials
        // question.
        .add_plugins(TarasaurPlugin)
        .init_resource::<TestReady>()
        .add_systems(Startup, (spawn_camera_and_light, spawn_test_chunks))
        .add_systems(
            Update,
            (generate_test_chunks, log_visibility_state, fly_cam_system).chain(),
        )
        .run();
}

fn spawn_camera_and_light(mut commands: Commands) {
    // Chunks are spaced 2 grid-units apart along x (see spawn_test_chunks) —
    // centered roughly on the middle chunk, far enough back to see all three.
    let mid_x = 2.0 * CHUNK_SIZE;
    let initial_transform = Transform::from_xyz(mid_x, 18.0, 35.0)
        .looking_at(Vec3::new(mid_x, CHUNK_SIZE * 0.5, 0.0), Vec3::Y);
    let (yaw, pitch, _) = initial_transform.rotation.to_euler(EulerRot::YXZ);

    commands.spawn((
        Camera3d::default(),
        initial_transform,
        FlyCam {
            move_speed: 15.0,
            sensitivity: 0.002,
            pitch,
            yaw,
        },
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 8000.0,
            ..default()
        },
        Transform::from_xyz(4.0, 10.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

/// Which visibility case this chunk is testing — tags the entity so
/// generate_test_chunks and the logger both know what to do with it.
#[derive(Component, Clone, Copy, Debug)]
enum VisibilityCase {
    AllVisible,
    AllHidden,
    HalfAndHalf,
}

fn spawn_test_chunks(mut commands: Commands) {
    // Spaced 2 chunk-units apart — guarantees a full empty gap between
    // each, so no chunk's apron ever samples a neighbor with a different
    // visibility case. Each chunk is a fully self-contained test.
    let cases = [
        (IVec3::new(0, 0, 0), VisibilityCase::AllVisible),
        (IVec3::new(2, 0, 0), VisibilityCase::AllHidden),
        (IVec3::new(4, 0, 0), VisibilityCase::HalfAndHalf),
    ];
    for (pos, case) in cases {
        commands.spawn((Chunk, ChunkPosition(pos), LOD::Medium, case));
    }
    info!("[visibility_test] spawned 3 chunks: AllVisible, AllHidden, HalfAndHalf");
}

fn generate_test_chunks(
    mut generated: Local<bool>,
    mut ready: ResMut<TestReady>,
    mut query: Query<(&VisibilityCase, &mut SDFField, &mut VisibilityField)>,
) {
    if *generated || query.iter().count() < 3 {
        return;
    }
    *generated = true;

    for (case, mut sdf, mut visibility) in query.iter_mut() {
        let size = sdf.size();
        let voxel_size = CHUNK_SIZE / size.x as f32;
        let center = Vec3::splat(CHUNK_SIZE * 0.5);
        let radius = CHUNK_SIZE * 0.35;

        for z in 0..size.z {
            for y in 0..size.y {
                for x in 0..size.x {
                    // Solid sphere centered in the chunk — guarantees a
                    // real zero-crossing surface fully inside the chunk,
                    // independent of neighbor apron data (which doesn't
                    // exist here — these chunks have no neighbors at all).
                    let local = (Vec3::new(x as f32, y as f32, z as f32) + 0.5) * voxel_size;
                    let dist = local.distance(center) - radius;
                    let value = if dist < 0.0 { -1.0 } else { 1.0 };
                    sdf.set(x, y, z, value);

                    // Direct Field::set — deliberately bypassing WorldEditor/
                    // EditFieldMessage entirely, so this test can't be
                    // affected by anything in process_box_edits/
                    // process_sphere_edits.
                    let visible = match case {
                        VisibilityCase::AllVisible => true,
                        VisibilityCase::AllHidden => false,
                        VisibilityCase::HalfAndHalf => x < size.x / 2,
                    };
                    visibility.set(x, y, z, visible);
                }
            }
        }

        sdf.reinit();
    }

    ready.0 = true;
    info!("[visibility_test] SDF + visibility data written for all 3 chunks");
}

fn log_visibility_state(
    mut frame: Local<u32>,
    mut logged: Local<bool>,
    ready: Res<TestReady>,
    query: Query<(&ChunkPosition, &VisibilityCase, &VisibilityField)>,
) {
    if *logged || !ready.0 {
        return;
    }
    *frame += 1;
    if *frame < 5 {
        return;
    }
    *logged = true;

    for (pos, case, vis) in query.iter() {
        let data = vis.data_slice();
        let visible_count = data.iter().filter(|&&b| b != 0).count();
        info!(
            "[visibility_test] chunk {:?} ({:?}): is_uniform={:?}, visible_voxels={}/{}",
            pos.0,
            case,
            vis.is_uniform(),
            visible_count,
            data.len()
        );
    }
    info!(
        "[visibility_test] EXPECTED: AllVisible is_uniform=Some(true); AllHidden is_uniform=Some(false); HalfAndHalf is_uniform=None with visible_voxels near half of total"
    );
    info!(
        "[visibility_test] EXPECTED RENDER: full sphere at x=0, nothing at x={}, half-sphere (flat cut) at x={}",
        2.0 * CHUNK_SIZE,
        4.0 * CHUNK_SIZE
    );
}

fn fly_cam_system(
    time: Res<Time>,
    mouse_button: Res<ButtonInput<MouseButton>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut mouse_motion: MessageReader<MouseMotion>,
    mut query: Query<(&mut Transform, &mut FlyCam)>,
) {
    let delta_time = time.delta_secs();
    for (mut transform, mut fly_cam) in query.iter_mut() {
        if mouse_button.pressed(MouseButton::Right) {
            for ev in mouse_motion.read() {
                fly_cam.yaw -= ev.delta.x * fly_cam.sensitivity;
                fly_cam.pitch -= ev.delta.y * fly_cam.sensitivity;
                fly_cam.pitch = fly_cam.pitch.clamp(-1.54, 1.54);
            }
            transform.rotation = Quat::from_euler(EulerRot::YXZ, fly_cam.yaw, fly_cam.pitch, 0.0);
