use bevy::prelude::*;
use tarasaur::{
    Field, LOD, SDF, VisibilityField,
    chunk::{CHUNK_SIZE, Chunk, ChunkPosition},
    prelude::*,
};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(TarasaurPlugin)
        .add_systems(Startup, (setup_scene, spawn_chunk_grid))
        .add_systems(
            Update,
            (
                mark_chunks_visible,
                spawn_initial_sphere_when_ready,
                handle_sphere_editing,
                rotate_camera,
            ),
        )
        .run();
}

#[derive(Component)]
struct EditableSphere {
    radius: f32,
    center: Vec3,
}

#[derive(Resource, Default)]
struct SphereSpawned(bool);

fn spawn_chunk_grid(mut commands: Commands) {
    for x in -1..=1 {
        for y in -1..=1 {
            for z in -1..=1 {
                commands.spawn((Chunk, ChunkPosition(IVec3::new(x, y, z)), LOD::Medium));
            }
        }
    }
}

/// Same pattern as cursor_collision_test.rs's mark_chunks_visible — sets
/// every voxel in every chunk to visible once, directly via Field::set,
/// bypassing WorldEditor entirely. VisibilityField defaults to all-hidden
/// on spawn, so without this extract_voxel_chunks would skip every chunk
/// forever regardless of what the SDF contains. SDF alone defines the
/// actual surface shape from here on — visibility just needs to not be
/// in the way.
fn mark_chunks_visible(mut marked: Local<bool>, mut query: Query<&mut VisibilityField>) {
    if *marked || query.iter().count() < 27 {
        return;
    }
    *marked = true;

    for mut vis in query.iter_mut() {
        let size = LOD::Medium.size();
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    vis.set(x, y, z, true);
                }
            }
        }
    }
    info!("[edit_sphere] marked all 27 chunks fully visible");
}

fn spawn_initial_sphere_when_ready(
    mut commands: Commands,
    mut spawned: ResMut<SphereSpawned>,
    mut sdf: WorldEditor<SDF, f32>,
    ready_check: Query<&SDF>,
) {
    if spawned.0 || ready_check.iter().count() < 27 {
        return;
    }
    spawned.0 = true;

    let radius = 2.5;
    let center = Vec3::splat(CHUNK_SIZE * 0.5);

    sdf.fill_sphere(center, radius, -1.0);
    info!(
        "[edit_sphere] wrote sphere edit messages, center={:?} radius={}",
        center, radius
    );

    commands.spawn((EditableSphere { radius, center }, Transform::default()));
}

fn handle_sphere_editing(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut stamps: MessageWriter<SdfSphereStamp>,
    mut spheres: Query<&mut EditableSphere>,
) {
    let mut delta_radius = 0.0;
    if keyboard_input.pressed(KeyCode::KeyW) || keyboard_input.pressed(KeyCode::ArrowUp) {
        delta_radius += 1.5 * time.delta_secs();
    }
    if keyboard_input.pressed(KeyCode::KeyS) || keyboard_input.pressed(KeyCode::ArrowDown) {
        delta_radius -= 1.5 * time.delta_secs();
    }
    if delta_radius == 0.0 {
        return;
    }

    for mut sphere in spheres.iter_mut() {
        let old_radius = sphere.radius;
        sphere.radius = (sphere.radius + delta_radius).max(0.1);

        stamps.write(SdfSphereStamp {
            center: sphere.center,
            bound_radius: old_radius.max(sphere.radius),
            sdf_radius: sphere.radius,
        });
    }
}
#[derive(Component)]
struct OrbitCamera {
    focus: Vec3,
}

fn setup_scene(mut commands: Commands) {
    commands.spawn((
        DirectionalLight {
            illuminance: 10000.0,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    let focus = Vec3::splat(CHUNK_SIZE * 0.5);
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(focus.x, focus.y + 2.0, focus.z + 8.0).looking_at(focus, Vec3::Y),
        OrbitCamera { focus },
    ));
    commands.init_resource::<SphereSpawned>();
}

fn rotate_camera(time: Res<Time>, mut query: Query<(&mut Transform, &OrbitCamera)>) {
    for (mut transform, cam) in query.iter_mut() {
        let radius = 8.0;
        let angle = time.elapsed_secs() * 0.2;
        transform.translation =
            cam.focus + Vec3::new(angle.cos() * radius, 2.0, angle.sin() * radius);
        transform.look_at(cam.focus, Vec3::Y);
    }
}
