//! Compares surface-nets vertex normals, headless.
//!
//! - current: pass3's `compute_normal`, a central difference (h = 0.1 voxel)
//!   of the trilinear SDF, 6 trilinear samples (48 reads).
//! - analytic: the exact gradient of the vertex's own cell's trilinear
//!   interpolant, from the 8 corners pass3 already loaded (no reads).
//!
//! Inside a cell the trilinear SDF is linear along each axis, so the two only
//! differ for vertices within h of a cell face, where the central difference
//! also samples the neighbouring cell. Runs both, ported from the shader, on
//! full-detail chunks of the sandbox terrain, plain and with dug and built
//! spheres, using the GPU's quantized SDF:
//!
//! ```text
//! cargo run --release --example normal_check
//! ```

#[allow(dead_code)] // shared with the sandbox
mod common;

use bevy::math::{IVec3, UVec3, Vec3};
use common::terrain_height;
use tarasaur::{
    chunk::CHUNK_SIZE,
    field::{SDF_BAND, quantize_sdf},
    num::{f32_from_i32, f32_from_u32, f32_from_usize, usize_from},
};

/// Full-detail chunk: 64 voxels, so 65 cells and a 66-voxel padded volume.
const SIZE: u32 = 64;
const TEXTURE: u32 = SIZE + 2;
const CELLS: u32 = SIZE + 1;
const H: f32 = 0.1; // pass3's compute_normal step, in voxels

const CORNERS: [UVec3; 8] = [
    UVec3::new(0, 0, 0),
    UVec3::new(1, 0, 0),
    UVec3::new(0, 1, 0),
    UVec3::new(1, 1, 0),
    UVec3::new(0, 0, 1),
    UVec3::new(1, 0, 1),
    UVec3::new(0, 1, 1),
    UVec3::new(1, 1, 1),
];
const EDGES: [(usize, usize); 12] = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
];

#[derive(Clone, Copy)]
enum Scene {
    Plain,
    /// Spheres subtracted from the terrain.
    Dug,
    /// Spheres added to the terrain.
    Built,
}

/// Signed distance in world units, as the sandbox generator writes it
/// (height difference scaled by the slope), with the scene's spheres.
fn world_sdf(p: Vec3, scene: Scene, spheres: &[Vec3]) -> f32 {
    let e = 0.5;
    let gx = (terrain_height(p.x + e, p.z) - terrain_height(p.x - e, p.z)) / (2.0 * e);
    let gz = (terrain_height(p.x, p.z + e) - terrain_height(p.x, p.z - e)) / (2.0 * e);
    let terrain = (p.y - terrain_height(p.x, p.z)) / gz.mul_add(gz, gx.mul_add(gx, 1.0)).sqrt();
    let radius = 3.0;
    spheres.iter().fold(terrain, |d, c| match scene {
        Scene::Plain => d,
        Scene::Dug => d.max(radius - p.distance(*c)),
        Scene::Built => d.min(p.distance(*c) - radius),
    })
}

/// The padded volume as the GPU sees it: voxel units, clamped, snorm16.
fn chunk_volume(chunk: IVec3, scene: Scene, spheres: &[Vec3]) -> Vec<f32> {
    let voxel = CHUNK_SIZE / f32_from_u32(SIZE);
    let origin = chunk.as_vec3() * CHUNK_SIZE;
    let mut vol = Vec::with_capacity(usize_from(TEXTURE).pow(3));
    for z in 0..TEXTURE {
        for y in 0..TEXTURE {
            for x in 0..TEXTURE {
                let p = origin + UVec3::new(x, y, z).as_vec3() * voxel;
                let v = (world_sdf(p, scene, spheres) / voxel).clamp(-SDF_BAND, SDF_BAND);
                let q = f32::from(quantize_sdf(v));
                vol.push((q / f32::from(i16::MAX)).max(-1.0) * SDF_BAND);
            }
        }
    }
    vol
}

/// pass3's `sample_sdf`: clamped to the volume.
fn sample(vol: &[f32], c: IVec3) -> f32 {
    let max = i32::try_from(TEXTURE).unwrap_or(i32::MAX).wrapping_sub(1);
    let c = c.clamp(IVec3::ZERO, IVec3::splat(max)).as_uvec3();
    let i =
        c.z.wrapping_mul(TEXTURE)
            .wrapping_add(c.y)
            .wrapping_mul(TEXTURE)
            .wrapping_add(c.x);
    vol.get(usize_from(i)).copied().unwrap_or(0.0)
}

fn mix(a: f32, b: f32, t: f32) -> f32 {
    b.mul_add(t, a * (1.0 - t))
}

/// pass3's `sample_sdf_trilinear`.
fn trilinear(vol: &[f32], p: Vec3) -> f32 {
    let ip = p.floor().as_ivec3();
    let f = p.fract_gl();
    let c = |o: UVec3| sample(vol, ip.wrapping_add(o.as_ivec3()));
    let [c000, c100, c010, c110, c001, c101, c011, c111] = CORNERS.map(c);
    let x0 = mix(mix(c000, c100, f.x), mix(c010, c110, f.x), f.y);
    let x1 = mix(mix(c001, c101, f.x), mix(c011, c111, f.x), f.y);
    mix(x0, x1, f.z)
}

fn normalize_or_up(v: Vec3) -> Vec3 {
    if v.length() > 0.00001 {
        v.normalize()
    } else {
        Vec3::Y
    }
}

/// pass3's `compute_normal`.
fn central_difference_normal(vol: &[f32], p: Vec3) -> Vec3 {
    let d = |axis: Vec3| trilinear(vol, p + axis * H) - trilinear(vol, p - axis * H);
    normalize_or_up(Vec3::new(d(Vec3::X), d(Vec3::Y), d(Vec3::Z)))
}

/// Exact gradient of the cell's trilinear interpolant at `f` (0..1 in cell).
fn analytic_normal(c: &[f32; 8], f: Vec3) -> Vec3 {
    let [c0, c1, c2, c3, c4, c5, c6, c7] = *c;
    let dx = mix(mix(c1 - c0, c3 - c2, f.y), mix(c5 - c4, c7 - c6, f.y), f.z);
    let dy = mix(mix(c2 - c0, c3 - c1, f.x), mix(c6 - c4, c7 - c5, f.x), f.z);
    let dz = mix(mix(c4 - c0, c5 - c1, f.x), mix(c6 - c2, c7 - c3, f.x), f.y);
    normalize_or_up(Vec3::new(dx, dy, dz))
}

/// Angle between the two normals (degrees) for every vertex of the chunk.
fn chunk_angles(vol: &[f32], out: &mut Vec<f32>) {
    for z in 0..CELLS {
        for y in 0..CELLS {
            for x in 0..CELLS {
                let id = UVec3::new(x, y, z);
                let c = CORNERS.map(|o| sample(vol, id.wrapping_add(o).as_ivec3()));
                let inside = c.iter().filter(|&&v| v <= 0.0).count();
                if inside == 0 || inside == 8 {
                    continue; // not an active cell: no vertex
                }
                // pass3's vertex placement: mean of the edge crossings.
                let (mut sum, mut n) = (Vec3::ZERO, 0.0);
                for (a, b) in EDGES {
                    let (Some(&v0), Some(&v1), Some(pa), Some(pb)) =
                        (c.get(a), c.get(b), CORNERS.get(a), CORNERS.get(b))
                    else {
                        continue;
                    };
                    if (v0 <= 0.0) != (v1 <= 0.0) {
                        let diff = v1 - v0;
                        let t = if diff.abs() > 0.00001 {
                            -v0 / diff
                        } else {
                            0.5
                        };

                        sum += pa.as_vec3().lerp(pb.as_vec3(), t.clamp(0.0, 1.0));
                        n += 1.0;
                    }
                }
                let frac = if n > 0.0 { sum / n } else { Vec3::splat(0.5) };
                let current = central_difference_normal(vol, id.as_vec3() + frac);
                let analytic = analytic_normal(&c, frac);
                out.push(current.dot(analytic).clamp(-1.0, 1.0).acos().to_degrees());
            }
        }
    }
}

fn report(name: &str, mut angles: Vec<f32>) {
    angles.sort_by(f32::total_cmp);
    let n = angles.len();
    let at = |p: usize| {
        let i = n
            .saturating_sub(1)
            .saturating_mul(p)
            .checked_div(1000)
            .unwrap_or(0);
        angles.get(i).copied().unwrap_or(f32::NAN)
    };
    let over = |deg: f32| {
        100.0 * f32_from_usize(angles.iter().filter(|&&a| a > deg).count())
            / f32_from_usize(n.max(1))
    };
    let mean = angles.iter().sum::<f32>() / f32_from_usize(n.max(1));
    println!(
        "{name:<6} {n:>8} vertices | mean {mean:.3}°  p50 {:.3}°  p99 {:.3}°  p99.9 {:.3}°  max {:.2}° | >0.1° {:.1}%  >1° {:.2}%  >5° {:.3}%",
        at(500),
        at(990),
        at(999),
        angles.last().copied().unwrap_or(f32::NAN),
        over(0.1),
        over(1.0),
        over(5.0),
    );
}

fn main() {
    let scenes = [
        ("plain", Scene::Plain),
        ("dug", Scene::Dug),
        ("built", Scene::Built),
    ];
    for (name, scene) in scenes {
        let mut angles = Vec::new();
        for cz in -2..=2 {
            for cx in -2..=2 {
                // Spheres on the surface: at the chunk centre and a corner,
                // so some straddle chunk borders.
                let spheres: Vec<Vec3> = [(5.0, 5.0), (0.5, 9.5)]
                    .iter()
                    .map(|&(ox, oz)| {
                        let x = f32_from_i32(cx).mul_add(CHUNK_SIZE, ox);
                        let z = f32_from_i32(cz).mul_add(CHUNK_SIZE, oz);
                        Vec3::new(x, terrain_height(x, z), z)
                    })
                    .collect();
                for cy in -2..=1 {
                    let vol = chunk_volume(IVec3::new(cx, cy, cz), scene, &spheres);
                    chunk_angles(&vol, &mut angles);
                }
            }
        }
        report(name, angles);
    }
}
