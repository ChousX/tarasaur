//! Minimal streaming setup: a `ChunkLoader` on the camera and a generator
//! registered in `ChunkGeneratorRegistry`.

use bevy::input::mouse::MouseMotion;
use bevy::prelude::*;

use tarasaur::storage::loading::ChunkDataTask;
use tarasaur::{
    TarasaurPlugin,
    chunk::{CHUNK_SIZE, ChunkLoader, ShowChunkBounds},
    field::LOD,
    raycast::cursor::{CursorHit, resolve_cursor_hit},
    storage::generator::ChunkGeneratorRegistry,
};
use tarasaur::{chunk::Chunk, field::SDF};

fn main() {
    let mut app = App::new();

    app.add_plugins(DefaultPlugins).add_plugins(TarasaurPlugin);

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
        generators.register_sdf_distance(terrain_sdf);
    }

    app.insert_resource(ShowChunkBounds) // color-codes each chunk's LOD in gizmos — remove once you trust the streaming
        .add_systems(Startup, spawn_player_and_cursor_marker)
        .add_systems(
            Update,
            (
                fly_cam_system.before(resolve_cursor_hit),
                update_cursor_marker.after(resolve_cursor_hit),
            ),
        )
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
        h = ((x * freq).sin() * (z * freq * 1.3).cos()).mul_add(amp, h);
        amp *= 0.5;
        freq *= 2.1;
    }
    h
}

/// Runs off the main thread inside a load task — no ECS access, pure
/// function of `chunk_pos` + lod, exactly what `ChunkGeneratorRegistry`
/// requires. Produces world-space signed distances (height above the
/// terrain): negative below the surface, positive above, in x/y/z-flattened
/// order. `register_sdf_distance` converts them to voxel units; no jump
/// flood runs.
///
/// Resolution-independent: voxel `i` samples world position
/// `origin + i * voxel_size`, so the same chunk generated at two LODs
/// describes the same surface.
fn terrain_sdf(chunk_pos: IVec3, lod: LOD) -> Vec<f32> {
    let size = lod.size();
    let voxel_size = CHUNK_SIZE / tarasaur::num::f32_from_u32(size);
    let origin = chunk_pos.as_vec3() * CHUNK_SIZE;

    let mut out = Vec::with_capacity(lod.volume());
    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let world = origin + UVec3::new(x, y, z).as_vec3() * voxel_size;
                out.push(world.y - terrain_height(world.x, world.z));
            }
        }
    }
    out
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

/// The cursor's world-space focus point — a separate `ChunkLoader` riding
/// along on its own entity, see `spawn_player_and_cursor_marker`.
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
                cam.yaw = f32::mul_add(ev.delta.x, -cam.sensitivity, cam.yaw);
                cam.pitch = f32::mul_add(ev.delta.y, -cam.sensitivity, cam.pitch);
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

/// Moves the cursor's `ChunkLoader` marker to the terrain under the mouse,
/// using the library's resolved `CursorHit` (F8 cycles the backend). On a
/// miss the marker stays where it last hit.
fn update_cursor_marker(
    hit: Res<CursorHit>,
    mut marker_q: Query<&mut Transform, (With<CursorFocus>, Without<FlyCam>)>,
) {
    let Some(hit) = hit.0 else { return };
    for mut transform in marker_q.iter_mut() {
        transform.translation = hit.pos;
    }
}

/// (has a LOD, has an SDF, load in flight).
type ChunkStage = (Has<LOD>, Has<SDF>, Has<ChunkDataTask>);

fn debug_stages(chunks: Query<ChunkStage, With<Chunk>>, mut t: Local<f32>, time: Res<Time>) {
    *t += time.delta_secs();
    if *t < 2.0 {
        return;
    }
    *t = 0.0;
    let total = chunks.iter().count();
    let installed = chunks.iter().filter(|&(lod, sdf, _)| lod && sdf).count();
    let loading = chunks.iter().filter(|&(_, _, task)| task).count();
    info!("[stages] chunks={total} installed={installed} loading={loading}");
}
