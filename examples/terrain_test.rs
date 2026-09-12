// examples/terrain_test.rs

use bevy::{
    input::mouse::MouseMotion,
    prelude::*,
    render::{
        RenderPlugin,
        settings::{RenderCreation, WgpuLimits, WgpuSettings},
    },
};
use tarasaur::{
    Field, LOD, SDFField, TarasaurPlugin,
    chunk::{CHUNK_SIZE, Chunk, ChunkPosition},
    field::editor::WorldEditor,
};

const GRID_RADIUS: i32 = 5; // chunks from -1..=1 on x and z -> 3x3 = 9 chunks
const LOD_USED: LOD = LOD::Medium;

fn chunk_count() -> i32 {
    (2 * GRID_RADIUS + 1).pow(2)
}

#[derive(Resource, Default)]
struct TerrainReady(bool);

#[derive(Component)]
struct FlyCam {
    pub move_speed: f32,
    pub sensitivity: f32,
    pub pitch: f32,
    pub yaw: f32,
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
        .add_plugins(TarasaurPlugin)
        .init_resource::<TerrainReady>()
        .add_systems(Startup, (spawn_camera_and_light, spawn_terrain_chunks))
        .add_systems(
            Update,
            (
                fly_cam_system,
                generate_terrain,
                verify_seam,
                carve_valley_across_seam,
                log_final_stats,
            )
                .chain(),
        )
        .run();
}

fn spawn_camera_and_light(mut commands: Commands) {
    let initial_transform =
        Transform::from_xyz(35.0, 25.0, 35.0).looking_at(Vec3::new(5.0, 3.0, 5.0), Vec3::Y);

    // Extract initial orientation so rotation doesn't snap on first interaction
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

fn fly_cam_system(
    time: Res<Time>,
    mouse_button: Res<ButtonInput<MouseButton>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut mouse_motion: MessageReader<MouseMotion>,
    mut query: Query<(&mut Transform, &mut FlyCam)>,
) {
    let delta_time = time.delta_secs();

    for (mut transform, mut fly_cam) in query.iter_mut() {
        // Rotate camera when Right Mouse Button is held
        if mouse_button.pressed(MouseButton::Right) {
            for ev in mouse_motion.read() {
                fly_cam.yaw -= ev.delta.x * fly_cam.sensitivity;
                fly_cam.pitch -= ev.delta.y * fly_cam.sensitivity;
                fly_cam.pitch = fly_cam.pitch.clamp(-1.54, 1.54);
            }
            transform.rotation = Quat::from_euler(EulerRot::YXZ, fly_cam.yaw, fly_cam.pitch, 0.0);
        } else {
            mouse_motion.clear();
        }

        // Translation logic
        let mut velocity = Vec3::ZERO;
        let forward = *transform.forward();
        let right = *transform.right();

        if keyboard.pressed(KeyCode::KeyW) {
            velocity += forward;
        }
        if keyboard.pressed(KeyCode::KeyS) {
            velocity -= forward;
        }
        if keyboard.pressed(KeyCode::KeyD) {
            velocity += right;
        }
        if keyboard.pressed(KeyCode::KeyA) {
            velocity -= right;
        }
        if keyboard.pressed(KeyCode::Space) {
            velocity += Vec3::Y;
        }
        if keyboard.pressed(KeyCode::ShiftLeft) {
            velocity -= Vec3::Y;
        }

        if velocity != Vec3::ZERO {
            let speed_multiplier = if keyboard.pressed(KeyCode::ControlLeft) {
                2.5
            } else {
                1.0
            };
            transform.translation +=
                velocity.normalize() * fly_cam.move_speed * speed_multiplier * delta_time;
        }
    }
}

fn spawn_terrain_chunks(mut commands: Commands) {
    for x in -GRID_RADIUS..=GRID_RADIUS {
        for z in -GRID_RADIUS..=GRID_RADIUS {
            commands.spawn((Chunk, ChunkPosition(IVec3::new(x, 0, z)), LOD_USED));
        }
    }
    info!("[terrain_test] spawned {} chunks", chunk_count());
}

// --- Self-contained value noise (deterministic, no external deps) ---

fn hash2(x: i32, z: i32, seed: u32) -> f32 {
    let mut h = (x as u32).wrapping_mul(374761393)
        ^ (z as u32).wrapping_mul(668265263)
        ^ seed.wrapping_mul(2246822519);
    h ^= h >> 15;
    h = h.wrapping_mul(2246822519);
    h ^= h >> 13;
    h = h.wrapping_mul(3266489917);
    h ^= h >> 16;
    (h as f32) / (u32::MAX as f32)
}

#[inline]
fn smoothstep(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

fn value_noise(x: f32, z: f32, seed: u32) -> f32 {
    let x0 = x.floor() as i32;
    let z0 = z.floor() as i32;
    let tx = smoothstep(x - x0 as f32);
    let tz = smoothstep(z - z0 as f32);

    let v00 = hash2(x0, z0, seed);
    let v10 = hash2(x0 + 1, z0, seed);
    let v01 = hash2(x0, z0 + 1, seed);
    let v11 = hash2(x0 + 1, z0 + 1, seed);

    let a = v00 + (v10 - v00) * tx;
    let b = v01 + (v11 - v01) * tx;
    a + (b - a) * tz
}

fn fbm(x: f32, z: f32, octaves: u32, seed: u32) -> f32 {
    let mut amplitude = 0.5;
    let mut frequency = 1.0;
    let mut sum = 0.0;
    let mut max = 0.0;
    for i in 0..octaves {
        sum += value_noise(x * frequency, z * frequency, seed.wrapping_add(i)) * amplitude;
        max += amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    sum / max
}

fn terrain_height(world_x: f32, world_z: f32) -> f32 {
    const BASE_HEIGHT: f32 = 4.0;
    const AMPLITUDE: f32 = 2.5;
    const NOISE_SCALE: f32 = 0.08;

    let n = fbm(world_x * NOISE_SCALE, world_z * NOISE_SCALE, 4, 1337);
    BASE_HEIGHT + (n - 0.5) * 2.0 * AMPLITUDE
}

// --- Generation ---

fn generate_terrain(
    mut generated: Local<bool>,
    mut ready: ResMut<TerrainReady>,
    mut query: Query<(&ChunkPosition, &mut SDFField)>,
) {
    if *generated || query.iter().count() as i32 != chunk_count() {
        return;
    }
    *generated = true;

    for (ChunkPosition(chunk_pos), mut sdf) in query.iter_mut() {
        let size = sdf.size();
        let voxel_size = CHUNK_SIZE / size.x as f32;
        let chunk_origin = chunk_pos.as_vec3() * CHUNK_SIZE;

        for z in 0..size.z {
            for y in 0..size.y {
                for x in 0..size.x {
                    let world = chunk_origin + Vec3::new(x as f32, y as f32, z as f32) * voxel_size;
                    let height = terrain_height(world.x, world.z);
                    let value = if world.y < height { -1.0 } else { 1.0 };
                    sdf.set(x, y, z, value);
                }
            }
        }

        sdf.reinit();
    }

    ready.0 = true;
    info!(
        "[terrain_test] terrain generated across all {} chunks",
        chunk_count()
    );
}

fn verify_seam(
    mut checked: Local<bool>,
    ready: Res<TerrainReady>,
    query: Query<(&ChunkPosition, &SDFField)>,
) {
    if *checked || !ready.0 {
        return;
    }

    let mut left = None;
    let mut right = None;
    for (ChunkPosition(pos), sdf) in query.iter() {
        if *pos == IVec3::new(0, 0, 0) {
            left = Some(sdf);
        } else if *pos == IVec3::new(1, 0, 0) {
            right = Some(sdf);
        }
    }
    let (Some(left), Some(right)) = (left, right) else {
        return;
    };
    *checked = true;

    let size = left.size();
    let mut max_diff: f32 = 0.0;
    for z in 0..size.z {
        for y in 0..size.y {
            let l = left.get(size.x - 1, y, z);
            let r = right.get(0, y, z);
            max_diff = max_diff.max((l - r).abs());
        }
    }

    info!(
        "[terrain_test] seam check (0,0,0)|(1,0,0): max |Δsdf| across one voxel = {:.4}",
        max_diff
    );
}

fn carve_valley_across_seam(
    mut fired: Local<bool>,
    ready: Res<TerrainReady>,
    mut editor: WorldEditor<SDFField, f32>,
) {
    if *fired || !ready.0 {
        return;
    }
    *fired = true;

    let seam_x = CHUNK_SIZE;
    let seam_z = CHUNK_SIZE * 0.5;
    let ground = terrain_height(seam_x, seam_z);

    editor.fill_sphere(Vec3::new(seam_x, ground, seam_z), 3.5, 1.0);

    info!("[terrain_test] carved a valley across the chunk seam at x={seam_x}");
}

fn log_final_stats(
    mut frame: Local<u32>,
    mut logged: Local<bool>,
    ready: Res<TerrainReady>,
    query: Query<(&ChunkPosition, &SDFField)>,
) {
    if *logged || !ready.0 {
        return;
    }
    *frame += 1;
    if *frame < 5 {
        return;
    }
    *logged = true;

    for (pos, sdf) in query.iter() {
        let data = sdf.data_slice();
        let (mut min, mut max) = (f32::MAX, f32::MIN);
        for &v in data {
            min = min.min(v);
            max = max.max(v);
        }
        info!(
            "[terrain_test] chunk {:?}: version={} sdf=[{:.3}, {:.3}]",
            pos.0, sdf.version, min, max
        );
    }
}
