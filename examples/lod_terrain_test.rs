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
};

const GRID_RADIUS: i32 = 10; // chunks from -GRID_RADIUS..=GRID_RADIUS on x and z

/// Distance bands (in chunk units, Chebyshev distance from the grid center)
/// mapping to LOD. Ordered nearest-first; the first band whose radius the
/// chunk falls within wins. Anything beyond the last band's radius gets the
/// final entry's LOD.
///
/// "Closest / under camera -> highest LOD, falling off to lowest" per the
/// project's stated end goal — here "camera" is approximated by the grid
/// center at spawn time, since there's no live streaming yet.
const LOD_BANDS: &[(i32, LOD)] = &[
    (2, LOD::High),          // chunks within 2 of center
    (5, LOD::Medium),        // within 5
    (8, LOD::Low),           // within 8
    (i32::MAX, LOD::Lowest), // everything else
];

fn lod_for_distance(chebyshev_dist: i32) -> LOD {
    for &(radius, lod) in LOD_BANDS {
        if chebyshev_dist <= radius {
            return lod;
        }
    }
    unreachable!("LOD_BANDS must end with an i32::MAX catch-all");
}

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
            (fly_cam_system, generate_terrain, log_lod_distribution).chain(),
        )
        .run();
}

fn spawn_camera_and_light(mut commands: Commands) {
    let initial_transform =
        Transform::from_xyz(35.0, 25.0, 35.0).looking_at(Vec3::new(5.0, 3.0, 5.0), Vec3::Y);
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
    let mut counts: std::collections::HashMap<LOD, u32> = std::collections::HashMap::new();

    for x in -GRID_RADIUS..=GRID_RADIUS {
        for z in -GRID_RADIUS..=GRID_RADIUS {
            let dist = x.abs().max(z.abs()); // Chebyshev distance from center chunk (0,0)
            let lod = lod_for_distance(dist);
            *counts.entry(lod).or_insert(0) += 1;
            commands.spawn((Chunk, ChunkPosition(IVec3::new(x, 0, z)), lod));
        }
    }

    info!(
        "[lod_terrain_test] spawned {} chunks across bands {:?} -> counts: {:?}",
        chunk_count(),
        LOD_BANDS,
        counts
    );
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
//
// Each chunk samples terrain_height at its own LOD's voxel resolution —
// SDFField::size() already reflects the component's LOD, so a High-LOD
// chunk naturally gets more samples across the same CHUNK_SIZE world extent
// than a Lowest-LOD chunk sitting right next to it.

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
        "[lod_terrain_test] terrain generated across all {} chunks",
        chunk_count()
    );
}

fn log_lod_distribution(
    mut frame: Local<u32>,
    mut logged: Local<bool>,
    ready: Res<TerrainReady>,
    query: Query<(&ChunkPosition, &LOD, &SDFField)>,
) {
    if *logged || !ready.0 {
        return;
    }
    *frame += 1;
    if *frame < 5 {
        return;
    }
    *logged = true;

    // Confirms every band actually produced valid SDF data, per-LOD — a
    // quick sanity check that each concurrent arena got real geometry
    // rather than one LOD silently failing while others render fine.
    let mut per_lod: std::collections::HashMap<LOD, (u32, f32, f32)> =
        std::collections::HashMap::new();

    for (_pos, lod, sdf) in query.iter() {
        let data = sdf.data_slice();
        let (mut min, mut max) = (f32::MAX, f32::MIN);
        for &v in data {
            min = min.min(v);
            max = max.max(v);
        }
        let entry = per_lod.entry(*lod).or_insert((0, f32::MAX, f32::MIN));
        entry.0 += 1;
        entry.1 = entry.1.min(min);
        entry.2 = entry.2.max(max);
    }

    for (lod, (count, min, max)) in per_lod.iter() {
        info!(
            "[lod_terrain_test] LOD {:?}: {} chunks, sdf range=[{:.3}, {:.3}]",
            lod, count, min, max
        );
    }
}
