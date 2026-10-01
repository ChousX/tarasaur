// examples/lod_streaming.rs
use bevy::input::mouse::MouseMotion;
use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy::render::settings::{RenderCreation, WgpuLimits, WgpuSettings};
use bevy::window::PrimaryWindow;

use tarasaur::loading::ChunkDataTask;
use tarasaur::{Chunk, SDF};
use tarasaur::{
    LOD, TarasaurPlugin,
    chunk::{CHUNK_SIZE, ChunkLoader, ShowChunkBounds},
    field::generator::ChunkGeneratorRegistry,
};

fn main() {
    let mut app = App::new();

    app.add_plugins(
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
    .add_plugins(TarasaurPlugin);

    app.add_systems(Update, debug_stages);
    // Registered once, up front. Every chunk the ChunkLoader spawns or
    // retargets gets its terrain produced off-thread by the same task
    // machinery that handles save-file loading (see start_chunk_loads in
    // field/loading.rs). A saved max-LOD file is used first; the generator
    // only runs when no usable save exists.
    //
    // No VisibilityField generator: a chunk without one is fully visible.
    {
        let mut generators = app.world_mut().resource_mut::<ChunkGeneratorRegistry>();
        generators.register_sdf(terrain_sdf);
    }

    app.insert_resource(ShowChunkBounds) // color-codes each chunk's LOD in gizmos — remove once you trust the streaming
        .add_systems(Startup, spawn_player_and_cursor_marker)
        .add_systems(Update, (fly_cam_system, update_cursor_marker).chain())
        .run();
}

// ============================================================================
// Terrain generator
// ============================================================================

/// Self-contained low-octave sine "noise" — no external noise crate
/// dependency. Swap this out for a real fbm/simplex implementation if you
/// have one; the generator signature (world x,z -> height) is all that
/// matters to the rest of this example.
fn terrain_height(x: f32, z: f32) -> f32 {
    let mut h = 0.0;
    let mut amp = 6.0;
    let mut freq = 0.03;
    for _ in 0..4 {
        h += (x * freq).sin() * (z * freq * 1.3).cos() * amp;
        amp *= 0.5;
        freq *= 2.1;
    }
    h
}

/// Runs off the main thread inside a load task — no ECS access, pure
/// function of chunk_pos + lod, exactly what ChunkGeneratorRegistry
/// requires. Produces a sign volume: negative below the terrain surface,
/// positive above, in x/y/z-flattened order. `register_sdf` turns it into
/// real distances with a jump-flood pass.
///
/// Resolution-independent: voxel `i` samples world position
/// `origin + i * voxel_size`, so the same chunk generated at two LODs
/// describes the same surface.
fn terrain_sdf(chunk_pos: IVec3, lod: LOD) -> Vec<f32> {
    let size = lod.size();
    let voxel_size = CHUNK_SIZE / size as f32;
    let origin = chunk_pos.as_vec3() * CHUNK_SIZE;

    (0..lod.volume())
        .map(|i| {
            let i = i as u32;
            let x = i % size;
            let y = (i / size) % size;
            let z = i / (size * size);
            let world = origin + Vec3::new(x as f32, y as f32, z as f32) * voxel_size;
            world.y - terrain_height(world.x, world.z)
        })
        .collect()
}

// ============================================================================
// Player (flying camera, drives the main LOD falloff)
// ============================================================================

#[derive(Component)]
struct FlyCam {
    move_speed: f32,
    sensitivity: f32,
    pitch: f32,
    yaw: f32,
}

/// The cursor's world-space focus point — a separate ChunkLoader riding
/// along on its own entity, see spawn_player_and_cursor_marker.
#[derive(Component)]
struct CursorFocus;

fn spawn_player_and_cursor_marker(mut commands: Commands) {
    let start = Vec3::new(0.0, 20.0, 30.0);
    let initial_transform = Transform::from_translation(start).looking_at(Vec3::ZERO, Vec3::Y);
    let (yaw, pitch, _) = initial_transform.rotation.to_euler(EulerRot::YXZ);

    commands.spawn((
        Camera3d::default(),
        initial_transform,
        FlyCam {
            move_speed: 20.0,
            sensitivity: 0.002,
            pitch,
            yaw,
        },
        // Drives the main streaming radius: High LOD close to the player,
        // falling off to lower LODs further out. Band widths are ADDITIVE
        // and listed finest first (Chebyshev distance, in chunks):
        //   d <= 1 -> High, d <= 3 -> Medium, d <= 5 -> Low, d <= 8 -> Lowest.
        // `hysteresis` delays both LOD downgrades and despawn by that many
        // chunks (see ChunkLoader::want). Kept smaller than
        // ChunkLoader::default(), whose total range of 10 spawns thousands
        // of chunks on first load.
        ChunkLoader {
            lod_bands: vec![
                (LOD::High, 1),   // d <= 1
                (LOD::Medium, 2), // d <= 3
                (LOD::Low, 2),    // d <= 5
                (LOD::Lowest, 3), // d <= 8
            ],
            hysteresis: 1,
        },
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 8000.0,
            ..default()
        },
        Transform::from_xyz(4.0, 10.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // The cursor's high-detail marker: no mesh, just Transform + ChunkLoader.
    // A single High band of width 1 means this loader only ever wants High
    // within one chunk of the marker and nothing beyond. When several loaders
    // overlap, the finest wanted LOD wins and a chunk is despawned only when
    // every loader wants it gone, so this loader can only upgrade chunks near
    // the cursor and can't fight the player's loader for chunks outside that
    // ring.
    commands.spawn((
        CursorFocus,
        Transform::default(),
        ChunkLoader {
            lod_bands: vec![(LOD::High, 1)],
            hysteresis: 0,
        },
    ));
}

fn fly_cam_system(
    time: Res<Time>,
    mouse_button: Res<ButtonInput<MouseButton>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut mouse_motion: MessageReader<MouseMotion>,
    mut query: Query<(&mut Transform, &mut FlyCam)>,
) {
    let dt = time.delta_secs();
    for (mut transform, mut cam) in query.iter_mut() {
        if mouse_button.pressed(MouseButton::Right) {
            for ev in mouse_motion.read() {
                cam.yaw -= ev.delta.x * cam.sensitivity;
                cam.pitch -= ev.delta.y * cam.sensitivity;
                cam.pitch = cam.pitch.clamp(-1.54, 1.54);
            }
            transform.rotation = Quat::from_euler(EulerRot::YXZ, cam.yaw, cam.pitch, 0.0);
        }

        let mut velocity = Vec3::ZERO;
        let forward = transform.forward();
        let right = transform.right();
        if keyboard.pressed(KeyCode::KeyW) {
            velocity += *forward;
        }
        if keyboard.pressed(KeyCode::KeyS) {
            velocity -= *forward;
        }
        if keyboard.pressed(KeyCode::KeyA) {
            velocity -= *right;
        }
        if keyboard.pressed(KeyCode::KeyD) {
            velocity += *right;
        }
        if keyboard.pressed(KeyCode::Space) {
            velocity += Vec3::Y;
        }
        if keyboard.pressed(KeyCode::ShiftLeft) {
            velocity -= Vec3::Y;
        }

        if velocity != Vec3::ZERO {
            transform.translation += velocity.normalize() * cam.move_speed * dt;
        }
    }
}

/// Moves the cursor's ChunkLoader marker to wherever the mouse ray crosses
/// a fixed y=0 plane. This is a deliberate simplification — a real "focus
/// the actual terrain under the cursor" would use the GPU collision query
/// pipeline (see cursor_collision_test.rs's submit_cursor_ray/
/// VoxelQueryResults), which tests against the real generated SDF instead
/// of a flat plane. Good enough here since we only need an approximate
/// world position to bias LOD toward, not a precise hit point.
fn update_cursor_marker(
    windows: Query<&Window, With<PrimaryWindow>>,
    camera_q: Query<(&Camera, &GlobalTransform), With<FlyCam>>,
    mut marker_q: Query<&mut Transform, (With<CursorFocus>, Without<FlyCam>)>,
) {
    let Ok(window) = windows.single() else { return };
    let Some(cursor_pos) = window.cursor_position() else {
        return;
    };
    let Ok((camera, camera_transform)) = camera_q.single() else {
        return;
    };
    let Ok(ray) = camera.viewport_to_world(camera_transform, cursor_pos) else {
        return;
    };

    if ray.direction.y.abs() < 1e-4 {
        return; // looking parallel to the plane — no sane intersection
    }
    let t = (0.0 - ray.origin.y) / ray.direction.y;
    if t < 0.0 {
        return; // plane is behind the camera
    }
    let hit = ray.origin + ray.direction * t;

    for mut transform in marker_q.iter_mut() {
        transform.translation = hit;
    }
}

fn debug_stages(
    chunks: Query<(Has<LOD>, Has<SDF>, Has<ChunkDataTask>), With<Chunk>>,
    mut t: Local<f32>,
    time: Res<Time>,
) {
    *t += time.delta_secs();
    if *t < 2.0 {
        return;
    }
    *t = 0.0;
    let (mut total, mut installed, mut loading) = (0, 0, 0);
    for (l, s, d) in &chunks {
        total += 1;
        if l && s {
            installed += 1;
        }
        if d {
            loading += 1;
        }
    }
    info!("[stages] chunks={total} installed={installed} loading={loading}");
}
