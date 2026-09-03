// Spawns a radius-7.0 SDF sphere at the origin, straddling the 8 chunks
// {-1,0}^3, and logs diagnostics at every stage: CPU sampling stats,
// chunk spawn confirmation, and GPU buffer readiness.

use bevy::prelude::*;
use bevy::render::{Render, RenderApp, RenderSystems};
use tarasaur::DirtyField;
use tarasaur::{
    Field, LOD, SDFField, TarasaurPlugin,
    chunk::{CHUNK_SIZE, Chunk, ChunkPosition},
    voxel::buffers::GpuVoxelChunkBuffers,
};

const SPHERE_CENTER: Vec3 = Vec3::ZERO;
const SPHERE_RADIUS: f32 = 7.0;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(TarasaurPlugin)
        .add_plugins(SphereTestDiagnosticsPlugin)
        .add_systems(Startup, (spawn_camera_and_light, spawn_sphere_chunks))
        .add_systems(Update, orbit_camera)
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
        OrbitCamera {
            radius,
            speed: 0.2, // Radians per second for a slow orbit
        },
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 8000.0,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    info!("[sphere_test] camera + light spawned");
}

fn orbit_camera(time: Res<Time>, mut query: Query<(&mut Transform, &OrbitCamera)>) {
    for (mut transform, orbit) in query.iter_mut() {
        let angle = time.elapsed_secs() * orbit.speed;
        let x = angle.cos() * orbit.radius;
        let z = angle.sin() * orbit.radius;
        let y = transform.translation.y; // Maintain initial height

        transform.translation = Vec3::new(x, y, z);
        transform.look_at(Vec3::ZERO, Vec3::Y);
    }
}

fn spawn_sphere_chunks(mut commands: Commands) {
    let mut spawned = Vec::new();
    let mut i = 0;
    const LODS: [LOD; 3] = [LOD::Low, LOD::Medium, LOD::High];
    for x in -1..=0 {
        for y in -1..=0 {
            for z in -1..=0 {
                let chunk_pos = IVec3::new(x, y, z);
                let lod = LOD::Medium; //LODS[i];
                i = (i + 1) % 3;
                let mut sdf = SDFField::new(lod);
                let stats = fill_sphere_sdf(&mut sdf, chunk_pos, SPHERE_CENTER, SPHERE_RADIUS);

                info!(
                    "[sphere_test] chunk {:>2?}: dims={:?} sdf=[{:.3}, {:.3}] surface_voxels={} ({:.1}%)",
                    chunk_pos,
                    stats.dims,
                    stats.min,
                    stats.max,
                    stats.surface_voxels,
                    100.0 * stats.surface_voxels as f32 / stats.total_voxels as f32,
                );

                commands.spawn((
                    Chunk,
                    ChunkPosition(chunk_pos),
                    lod,
                    sdf,
                    DirtyField::<SDFField, f32>::default(),
                ));
                spawned.push(chunk_pos);
            }
        }
    }

    info!(
        "[sphere_test] spawned {} chunks: {:?}",
        spawned.len(),
        spawned
    );
    assert_eq!(
        spawned.len(),
        8,
        "expected exactly 8 chunks for radius-7 sphere at origin"
    );
}

struct SphereFillStats {
    dims: UVec3,
    total_voxels: u32,
    min: f32,
    max: f32,
    surface_voxels: u32,
}

fn fill_sphere_sdf(
    field: &mut SDFField,
    chunk_pos: IVec3,
    center: Vec3,
    radius: f32,
) -> SphereFillStats {
    let dims = field.size();
    let voxel_size = CHUNK_SIZE / dims.x as f32;
    let chunk_origin = chunk_pos.as_vec3() * CHUNK_SIZE;

    let mut values = vec![0.0f32; (dims.x * dims.y * dims.z) as usize];
    let (mut min, mut max) = (f32::MAX, f32::MIN);
    let idx = |x: u32, y: u32, z: u32| ((z * dims.y + y) * dims.x + x) as usize;

    for z in 0..dims.z {
        for y in 0..dims.y {
            for x in 0..dims.x {
                let world = chunk_origin + Vec3::new(x as f32, y as f32, z as f32) * voxel_size;
                let d = sphere_sdf(world, center, radius);
                field.set(x, y, z, d);
                values[idx(x, y, z)] = d;
                min = min.min(d);
                max = max.max(d);
            }
        }
    }

    let mut surface_voxels = 0;
    for z in 0..dims.z.saturating_sub(1) {
        for y in 0..dims.y.saturating_sub(1) {
            for x in 0..dims.x.saturating_sub(1) {
                let v0 = values[idx(x, y, z)];
                let neighbors = [
                    values[idx(x + 1, y, z)],
                    values[idx(x, y + 1, z)],
                    values[idx(x, y, z + 1)],
                ];
                if neighbors.iter().any(|&n| n.signum() != v0.signum()) {
                    surface_voxels += 1;
                }
            }
        }
    }

    SphereFillStats {
        dims,
        total_voxels: dims.x * dims.y * dims.z,
        min,
        max,
        surface_voxels,
    }
}

#[inline]
fn sphere_sdf(p: Vec3, center: Vec3, radius: f32) -> f32 {
    p.distance(center) - radius
}

struct SphereTestDiagnosticsPlugin;

impl Plugin for SphereTestDiagnosticsPlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app.add_systems(
            Render,
            log_gpu_chunk_buffers.after(RenderSystems::Queue).run_if(
                bevy::time::common_conditions::on_timer(std::time::Duration::from_secs(1)),
            ),
        );
    }
}

fn log_gpu_chunk_buffers(chunks: Query<&GpuVoxelChunkBuffers>) {
    if chunks.is_empty() {
        return;
    }
    for chunk in chunks.iter() {
        info!(
            "[sphere_test][render] chunk {:?} lod={} has GPU buffers allocated",
            chunk.chunk_coord, chunk.lod
        );
    }
}
