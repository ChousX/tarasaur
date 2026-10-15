//! A radius-7 SDF sphere across 8 chunks, with the cursor pinned to the GPU
//! ray-query pipeline (`CursorQueryMode::Gpu`). A small red gizmo sphere
//! follows the (2-3 frame late) hit under the cursor.

use bevy::prelude::*;
use tarasaur::field::DirtyField;
use tarasaur::{
    TarasaurPlugin,
    chunk::{CHUNK_SIZE, Chunk, ChunkPosition},
    field::{Field, LOD, SDF},
    raycast::cursor::{CursorBrush, CursorHit, CursorQueryMode, resolve_cursor_hit},
};

const SPHERE_CENTER: Vec3 = Vec3::ZERO;
const SPHERE_RADIUS: f32 = 7.0;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(TarasaurPlugin::default())
        // Test the GPU path only; the built-in brush would carve the test sphere.
        .insert_resource(CursorQueryMode::Gpu)
        .insert_resource(CursorBrush { enabled: false })
        .add_systems(Startup, (spawn_camera_and_light, spawn_sphere_chunks))
        .add_systems(
            Update,
            (
                orbit_camera.before(resolve_cursor_hit),
                draw_hit_marker.after(resolve_cursor_hit),
            ),
        )
        .run();
}

#[derive(Component)]
struct OrbitCamera {
    radius: f32,
    speed: f32,
}

fn spawn_camera_and_light(mut commands: Commands) {
    let initial_pos = Vec3::new(30.0, 25.0, 30.0);
    let radius = Vec3::new(initial_pos.x, 0.0, initial_pos.z).length();

    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(initial_pos).looking_at(Vec3::ZERO, Vec3::Y),
        OrbitCamera { radius, speed: 0.2 },
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 8000.0,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    info!("[cursor_collision_test] camera + light spawned");
}

fn orbit_camera(time: Res<Time>, mut query: Query<(&mut Transform, &OrbitCamera)>) {
    for (mut transform, orbit) in query.iter_mut() {
        let angle = time.elapsed_secs() * orbit.speed;
        let x = angle.cos() * orbit.radius;
        let z = angle.sin() * orbit.radius;
        let y = transform.translation.y;

        transform.translation = Vec3::new(x, y, z);
        transform.look_at(Vec3::ZERO, Vec3::Y);
    }
}

fn spawn_sphere_chunks(mut commands: Commands) {
    let mut spawned = Vec::new();
    for x in -1..=0 {
        for y in -1..=0 {
            for z in -1..=0 {
                let chunk_pos = IVec3::new(x, y, z);
                let lod = LOD::Medium;
                let mut sdf = SDF::new(lod);
                fill_sphere_sdf(&mut sdf, chunk_pos, SPHERE_CENTER, SPHERE_RADIUS);

                commands.spawn((
                    Chunk,
                    ChunkPosition(chunk_pos),
                    lod,
                    sdf,
                    DirtyField::<SDF, f32>::default(),
                ));
                spawned.push(chunk_pos);
            }
        }
    }

    info!(
        "[cursor_collision_test] spawned {} chunks: {:?}",
        spawned.len(),
        spawned
    );
    assert_eq!(spawned.len(), 8);
}

fn fill_sphere_sdf(field: &mut SDF, chunk_pos: IVec3, center: Vec3, radius: f32) {
    let dims = field.size();
    let voxel_size = CHUNK_SIZE / tarasaur::num::f32_from_u32(dims.x);
    let chunk_origin = chunk_pos.as_vec3() * CHUNK_SIZE;

    for z in 0..dims.z {
        for y in 0..dims.y {
            for x in 0..dims.x {
                let world = chunk_origin + UVec3::new(x, y, z).as_vec3() * voxel_size;
                let d = world.distance(center) - radius;
                field.set(x, y, z, d);
            }
        }
    }
}

/// Draws a small red sphere at the resolved `CursorHit`. The mode is pinned
/// to `Gpu`, so this lags the cursor by the 2-3 frame readback. Draws
/// nothing when the most recent query missed.
fn draw_hit_marker(hit: Res<CursorHit>, mut gizmos: Gizmos) {
    let Some(hit) = hit.0 else { return };

    gizmos.sphere(hit.pos, 0.15, Color::srgb(1.0, 0.05, 0.05));
    // Short normal tick, mostly to sanity-check compute_normal() isn't
    // returning garbage — safe to delete once you trust the pipeline.
    gizmos.line(
        hit.pos,
        hit.pos + hit.normal * 0.6,
        Color::srgb(1.0, 0.6, 0.0),
    );
}
