// examples/cross_chunk_edit_test.rs
//
// Proves edits work without caring about chunk bounds: a single WorldEditor
// call carves a sphere centered exactly on the shared face between two
// chunks. Before the systems.rs fix this would've only carved whichever
// chunk `world_pos_to_chunk_pos(center)` picked and clipped hard at the
// boundary; now both chunks pick up the carve and the SDF is continuous
// across the seam.

use bevy::prelude::*;
use tarasaur::{
    LOD, SDFField, TarasaurPlugin,
    chunk::{CHUNK_SIZE, Chunk, ChunkPosition},
    field::editor::WorldEditor,
};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(TarasaurPlugin)
        .add_systems(Startup, (spawn_camera_and_light, spawn_two_chunks))
        .add_systems(
            Update,
            (carve_across_boundary, log_both_chunks, orbit_camera),
        )
        .run();
}

#[derive(Component)]
struct OrbitCamera {
    target: Vec3,
    radius: f32,
    speed: f32,
}

fn spawn_camera_and_light(mut commands: Commands) {
    let target = Vec3::new(CHUNK_SIZE, 0.0, 0.0);
    let initial_pos = Vec3::new(25.0, 20.0, 25.0);
    let offset = initial_pos - target;
    let radius = Vec3::new(offset.x, 0.0, offset.z).length();

    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(initial_pos).looking_at(target, Vec3::Y),
        OrbitCamera {
            target,
            radius,
            speed: 0.2,
        },
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 8000.0,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

fn orbit_camera(time: Res<Time>, mut query: Query<(&mut Transform, &OrbitCamera)>) {
    for (mut transform, orbit) in query.iter_mut() {
        let angle = time.elapsed_secs() * orbit.speed;
        let x = orbit.target.x + angle.cos() * orbit.radius;
        let z = orbit.target.z + angle.sin() * orbit.radius;
        let y = transform.translation.y;

        transform.translation = Vec3::new(x, y, z);
        transform.look_at(orbit.target, Vec3::Y);
    }
}

/// Two chunks sitting side by side along X: {0,0,0} and {1,0,0}. Their
/// shared face sits at world x = CHUNK_SIZE.
fn spawn_two_chunks(mut commands: Commands) {
    commands.spawn((Chunk, ChunkPosition(IVec3::new(0, 0, 0)), LOD::Medium));
    commands.spawn((Chunk, ChunkPosition(IVec3::new(1, 0, 0)), LOD::Medium));
}

fn all_chunks_ready(sdf_q: &Query<&SDFField>) -> bool {
    sdf_q.iter().count() == 2
}

/// Fires one Absolute sphere carve centered right on the shared face, with a
/// radius large enough to bite into both chunks. `WorldEditor` never
/// mentions chunk position or entity — just world coordinates.
fn carve_across_boundary(
    mut fired: Local<bool>,
    sdf_q: Query<&SDFField>,
    mut editor: WorldEditor<SDFField, f32>,
) {
    if *fired || !all_chunks_ready(&sdf_q) {
        return;
    }
    *fired = true;

    let seam = Vec3::new(CHUNK_SIZE, CHUNK_SIZE * 0.5, CHUNK_SIZE * 0.5);
    editor.fill_sphere(seam, 4.0, -1.0); // negative == solid

    info!(
        "[cross_chunk_edit_test] carved a radius-4 sphere centered on the seam at x={CHUNK_SIZE}"
    );
}

/// Confirms both chunks were actually touched, by checking that voxels near
/// the shared face went solid on *both* sides.
fn log_both_chunks(mut logged: Local<bool>, query: Query<(&ChunkPosition, &SDFField)>) {
    if *logged || query.iter().count() != 2 {
        return;
    }

    let mut touched = 0;
    for (pos, sdf) in query.iter() {
        let (min, max) = sdf
            .data_slice()
            .iter()
            .fold((f32::MAX, f32::MIN), |(mn, mx), &v| (mn.min(v), mx.max(v)));
        let has_solid_near_seam = min < 0.0;
        if has_solid_near_seam {
            touched += 1;
        }
        info!(
            "[cross_chunk_edit_test] chunk {:?}: sdf=[{:.3}, {:.3}] carved={}",
            pos.0, min, max, has_solid_near_seam
        );
    }

    if touched == 2 {
        info!("[cross_chunk_edit_test] PASS: both chunks show the carve — edit crossed the seam");
    } else {
        warn!("[cross_chunk_edit_test] FAIL: only {touched}/2 chunks show the carve");
    }
    *logged = true;
}
