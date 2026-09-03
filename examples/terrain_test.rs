// examples/terrain_test.rs
//
// Generates rolling-hills terrain across a 3x3 grid of chunks using a
// self-contained value-noise heightmap (no external noise crate). This is
// deliberately NOT done through EditFieldMessage/WorldEditor — generation
// writes SDF voxels directly via Field::set, then calls SDFField::reinit()
// once per chunk to compute real JFA distances from the raw sign data.
// WorldEditor is reserved for the follow-up carve, to show the two
// mechanisms composing.
//
// Continuity across chunk boundaries falls out for free: `terrain_height`
// takes world (x, z) only, so two chunks never disagree about the height at
// a shared point — there's no chunk-local seed or origin baked into the
// noise call.

use bevy::prelude::*;
use tarasaur::{
    Field, LOD, SDFField, TarasaurPlugin,
    chunk::{CHUNK_SIZE, Chunk, ChunkPosition},
    field::editor::WorldEditor,
};

const GRID_RADIUS: i32 = 1; // chunks from -1..=1 on x and z -> 3x3 = 9 chunks
const LOD_USED: LOD = LOD::Medium;

fn chunk_count() -> i32 {
    (2 * GRID_RADIUS + 1).pow(2)
}

#[derive(Resource, Default)]
struct TerrainReady(bool);

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(TarasaurPlugin)
        .init_resource::<TerrainReady>()
        .add_systems(Startup, (spawn_camera_and_light, spawn_terrain_chunks))
        .add_systems(
            Update,
            (
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
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(35.0, 25.0, 35.0).looking_at(Vec3::new(5.0, 3.0, 5.0), Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 8000.0,
            ..default()
        },
        Transform::from_xyz(4.0, 10.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
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

/// Integer-coordinate hash -> [0, 1), via bit-mixing. Deterministic per
/// (x, z, seed) so terrain is reproducible across runs.
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

/// Bilinear-interpolated value noise at a continuous (x, z).
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

/// Fractal Brownian Motion: layered octaves of `value_noise`, normalized to
/// [0, 1), for natural-looking rolling terrain.
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

/// World-space terrain height at (world_x, world_z). Depends only on world
/// coordinates — never on which chunk is asking — which is the entire
/// reason adjacent chunks agree at their shared boundary.
fn terrain_height(world_x: f32, world_z: f32) -> f32 {
    const BASE_HEIGHT: f32 = 4.0; // mid-chunk, since our grid is 1 chunk tall
    const AMPLITUDE: f32 = 2.5;
    const NOISE_SCALE: f32 = 0.08; // smaller = broader, more rolling hills

    let n = fbm(world_x * NOISE_SCALE, world_z * NOISE_SCALE, 4, 1337);
    BASE_HEIGHT + (n - 0.5) * 2.0 * AMPLITUDE
}

// --- Generation ---

/// Fills every chunk's SDF field directly from the heightmap, then
/// `reinit()`s each chunk once to turn the raw sign data into real
/// distances. This bypasses `EditFieldMessage` entirely — it's bulk world
/// init, not a sculpting operation.
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
                    // reinit()'s boundary pass only reads the sign; the
                    // magnitude here is a placeholder that gets replaced by
                    // real JFA distances below.
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

/// Spot-checks continuity across the boundary between chunks (0,0,0) and
/// (1,0,0): the last voxel column of the left chunk and the first voxel
/// column of the right chunk are one voxel-width apart in world space, so
/// their SDF values should be close (smooth terrain), never a hard jump.
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

/// Once terrain exists, carves a valley centered exactly on the world-space
/// seam between chunks (0,0,0) and (1,0,0) — the same WorldEditor call used
/// in the earlier cross-chunk-edit example, now applied on top of generated
/// terrain instead of an empty field.
fn carve_valley_across_seam(
    mut fired: Local<bool>,
    ready: Res<TerrainReady>,
    mut editor: WorldEditor<SDFField, f32>,
) {
    if *fired || !ready.0 {
        return;
    }
    *fired = true;

    let seam_x = CHUNK_SIZE; // world x = 10, the shared face
    let seam_z = CHUNK_SIZE * 0.5;
    let ground = terrain_height(seam_x, seam_z);

    editor.fill_sphere(Vec3::new(seam_x, ground, seam_z), 3.5, 1.0); // 1.0 == air

    info!("[terrain_test] carved a valley across the chunk seam at x={seam_x}");
}

/// Waits a few frames for `reinit_dirty_sdf` (in `FieldSet::Reinit`) to pick
/// up the carve's `DirtyField` markers and recompute distances, then logs
/// final per-chunk stats.
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
        return; // give the carve's edit + reinit systems time to land
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
