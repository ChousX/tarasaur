// Spawns the same radius-7.0 SDF sphere as sphere_test.rs, then unprojects
// the mouse cursor into a world-space ray every frame and feeds it through
// the GPU-native collision query pipeline. A small red gizmo sphere tracks
// wherever the (1-frame-delayed) result says the ray hit the terrain.

use bevy::prelude::*;
use bevy::render::settings::{RenderCreation, WgpuLimits, WgpuSettings};
use bevy::render::{RenderApp, RenderPlugin};
use bevy::window::PrimaryWindow;
use tarasaur::{DirtyField, VisibilityField};
use tarasaur::{
    Field, LOD, SDF, TarasaurPlugin,
    chunk::{CHUNK_SIZE, Chunk, ChunkPosition},
    voxel::query::{CollisionLOD, PendingVoxelQueries, VoxelQueryResults},
    voxel::types::RayQuery,
};

const SPHERE_CENTER: Vec3 = Vec3::ZERO;
const SPHERE_RADIUS: f32 = 7.0;
const QUERY_MAX_DISTANCE: f32 = 200.0;

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
        .add_plugins(CursorCollisionSetupPlugin) // pins CollisionLOD to Medium
        .add_systems(Startup, (spawn_camera_and_light, spawn_sphere_chunks))
        .add_systems(
            Update,
            (
                orbit_camera,
                mark_chunks_visible,
                submit_cursor_ray,
                draw_hit_marker,
            ),
        )
        .run();
}

#[derive(Component)]
struct OrbitCamera {
    radius: f32,
    speed: f32,
}

/// Auto-inserted VisibilityField defaults to fully-hidden (all-zero bits),
/// which makes extract_voxel_chunks skip SDF extraction entirely for every
/// chunk (is_uniform() == Some(false) short-circuits it). Waits for the
/// FieldsPlugin-populated component to actually exist on all 8 chunks
/// before writing into it — same pattern as visibility_test.rs.
fn mark_chunks_visible(mut marked: Local<bool>, mut query: Query<&mut VisibilityField>) {
    if *marked || query.iter().count() < 8 {
        return;
    }
    *marked = true;

    for mut vis in query.iter_mut() {
        let size = LOD::Medium.size(); // must match the LOD used in spawn_sphere_chunks
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    vis.set(x, y, z, true);
                }
            }
        }
    }
    info!("[cursor_collision_test] marked all 8 chunks fully visible");
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
                // Must match CursorCollisionSetupPlugin's CollisionLOD below —
                // the query pass only tests one arena, so a mismatch here
                // means every cursor query silently misses.
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
    let voxel_size = CHUNK_SIZE / dims.x as f32;
    let chunk_origin = chunk_pos.as_vec3() * CHUNK_SIZE;

    for z in 0..dims.z {
        for y in 0..dims.y {
            for x in 0..dims.x {
                let world = chunk_origin + Vec3::new(x as f32, y as f32, z as f32) * voxel_size;
                let d = world.distance(center) - radius;
                field.set(x, y, z, d);
            }
        }
    }
}

/// Pins the collision arena to LOD::Medium, matching the chunks this
/// example spawns. Overrides TarasaurPlugin's CollisionLOD default
/// (LOD::High) regardless of plugin registration order — insert_resource
/// always wins over init_resource, whichever runs second.
struct CursorCollisionSetupPlugin;

impl Plugin for CursorCollisionSetupPlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app.insert_resource(CollisionLOD(LOD::Medium));
    }
}

/// Unprojects the cursor into a world-space ray each frame and pushes it
/// onto PendingVoxelQueries. PendingVoxelQueries is cleared automatically
/// in the First schedule by the plugin's own clear_pending_voxel_queries
/// system, so this just appends — no manual lifetime management needed.
fn submit_cursor_ray(
    windows: Query<&Window, With<PrimaryWindow>>,
    camera_query: Query<(&Camera, &GlobalTransform)>,
    mut pending: ResMut<PendingVoxelQueries>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let Some(cursor_pos) = window.cursor_position() else {
        return; // cursor outside the window
    };
    let Ok((camera, camera_transform)) = camera_query.single() else {
        return;
    };

    // ASSUMPTION: viewport_to_world returns Result<Ray3d, _> (current Bevy
    // camera API). If your fork's signature differs, this is the line to
    // adjust — everything downstream just needs `origin`/`direction`.
    let Ok(ray) = camera.viewport_to_world(camera_transform, cursor_pos) else {
        return;
    };

    pending.0.push(RayQuery {
        origin: ray.origin.into(),
        max_distance: QUERY_MAX_DISTANCE,
        direction: ray.direction.as_vec3().into(),
        user_id: 0,
    });
}

/// Draws a small red sphere at the latest hit position, one frame behind
/// the cursor by design (see FrameParity in voxel/query.rs). Draws
/// nothing when the most recent query missed.
fn draw_hit_marker(results: Res<VoxelQueryResults>, mut gizmos: Gizmos) {
    let Some(hit) = results.0.first() else {
        return;
    };
    if hit.did_hit == 0 {
        return;
    }

    let pos = Vec3::from(hit.hit_pos_world);
    let normal = Vec3::from(hit.hit_normal);

    gizmos.sphere(pos, 0.15, Color::srgb(1.0, 0.05, 0.05));
    // Short normal tick, mostly to sanity-check compute_normal() isn't
    // returning garbage — safe to delete once you trust the pipeline.
    gizmos.line(pos, pos + normal * 0.6, Color::srgb(1.0, 0.6, 0.0));
}
