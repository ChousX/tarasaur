// examples/sphere_test.rs
//
// Spawns a radius-7.0 SDF sphere at the origin, straddling the 8 chunks
// {-1,0}^3, and logs diagnostics at every stage: CPU sampling stats,
// chunk spawn confirmation, and GPU buffer readiness.

use bevy::prelude::*;
use bevy::render::{Render, RenderApp, RenderSystems};
use tarasaur::DirtyField;
use tarasaur::{
    Field, LOD, SDFField, TarasaurPlugin,
    chunk::{CHUNK_SIZE, Chunk, ChunkPosition},
    voxel::{VoxelRenderPlugin, buffers::GpuVoxelChunkBuffers},
};

const SPHERE_CENTER: Vec3 = Vec3::ZERO;
const SPHERE_RADIUS: f32 = 7.0;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(TarasaurPlugin)
        .add_plugins(VoxelRenderPlugin)
        .add_plugins(SphereTestDiagnosticsPlugin)
        .add_systems(Startup, (spawn_camera_and_light, spawn_sphere_chunks))
        .run();
}

fn spawn_camera_and_light(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(30.0, 25.0, 30.0).looking_at(Vec3::ZERO, Vec3::Y),
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

fn spawn_sphere_chunks(mut commands: Commands) {
    let mut spawned = Vec::new();

    for x in -1..=0 {
        for y in -1..=0 {
            for z in -1..=0 {
                let chunk_pos = IVec3::new(x, y, z);

                // ASSUMPTION (need sdf.rs/lod.rs to confirm):
                //   - LOD::default() gives a usable voxel resolution
                //   - SDFField::new(lod) -> Self allocates dims.x*dims.y*dims.z f32s
                //   - SDFField implements Field<f32>
                let lod = LOD::default();
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

/// Samples the sphere SDF at every voxel and reports how many voxels
/// straddle the zero level-set — i.e. how many cells Pass 1 should
/// classify as active and Pass 3 should emit geometry from.
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

/// Bolts a system onto the render sub-app so you can confirm the compute
/// pipeline actually picked the chunks up (extraction -> prepare succeeded).
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
