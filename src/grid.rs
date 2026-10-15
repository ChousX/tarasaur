//! Voxel grid layout shared by every layer: how a cube of voxels is
//! flattened, the padded volume's apron, and sampling between grids.

use bevy::prelude::*;

use crate::num::{f32_from_u32, u32_from_f32, usize_from};

/// Index of (x, y, z) in a `size` grid: x fastest, then y, then z.
#[inline]
#[must_use]
pub const fn flatten_with_size(x: u32, y: u32, z: u32, size: UVec3) -> usize {
    // Index = (z * height + y) * width + x. Wrapping can't change a valid
    // index; an out-of-range one just fails the caller's `get`.
    usize_from(z)
        .wrapping_mul(usize_from(size.y))
        .wrapping_add(usize_from(y))
        .wrapping_mul(usize_from(size.x))
        .wrapping_add(usize_from(x))
}

/// [`flatten_with_size`] for a cube `size` voxels on each side.
#[inline]
#[must_use]
pub const fn flatten_cube(x: u32, y: u32, z: u32, size: u32) -> usize {
    flatten_with_size(x, y, z, UVec3::splat(size))
}

/// Apron width in voxels on the +x/+y/+z sides of a padded volume.
pub(crate) const PADDING: u32 = 2;

/// The +x/+y/+z neighbours (faces, edges, corner) whose voxels fill a
/// chunk's apron. Indexes `border_versions` too.
pub(crate) const NEIGHBORS_MASK: [IVec3; 7] = [
    ivec3(1, 0, 0),
    ivec3(0, 1, 0),
    ivec3(0, 0, 1),
    ivec3(1, 1, 0),
    ivec3(1, 0, 1),
    ivec3(0, 1, 1),
    ivec3(1, 1, 1),
];

/// Trilinear neighbor sampling, for continuous data like SDF distances.
/// Categorical fields (material ids, visibility) must never blend; they use
/// `nearest_neighbor_sample` instead.
///
/// Reads 0 if `data` is shorter than `size^3`.
#[must_use]
pub fn sample_neighbor(data: &[f32], size: u32, x: f32, y: f32, z: f32) -> f32 {
    let edge = UVec3::splat(size.saturating_sub(1));
    let pos = Vec3::new(x, y, z).clamp(Vec3::ZERO, edge.as_vec3());
    let p0 = pos.floor().as_uvec3();
    let p1 = p0.saturating_add(UVec3::ONE).min(edge);
    let frac = pos - p0.as_vec3();
    let at = |ix: u32, iy: u32, iz: u32| {
        data.get(flatten_cube(ix, iy, iz, size))
            .copied()
            .unwrap_or_default()
    };
    let lerp = |from: f32, to: f32, w: f32| to.mul_add(w, from * (1.0 - w));
    let plane = |iz: u32| {
        let y0 = lerp(at(p0.x, p0.y, iz), at(p1.x, p0.y, iz), frac.x);
        let y1 = lerp(at(p0.x, p1.y, iz), at(p1.x, p1.y, iz), frac.x);
        lerp(y0, y1, frac.y)
    };
    lerp(plane(p0.z), plane(p1.z), frac.z)
}

/// Nearest-neighbor apron sampling for categorical byte data — material
/// ids, visibility bits.
///
/// The default `VoxelField::sample_apron`, used by materials and visibility
/// rather than duplicated: unlike SDF distances, these values must never be
/// blended across a chunk boundary. Reads 0 if `data` is shorter than `size^3`.
#[must_use]
pub fn nearest_neighbor_sample<E: Copy + Default>(
    data: &[E],
    size: u32,
    x: f32,
    y: f32,
    z: f32,
) -> E {
    let max_coord = f32_from_u32(size.saturating_sub(1));
    let axis = |v: f32| u32_from_f32(v.round().clamp(0.0, max_coord));
    data.get(flatten_cube(axis(x), axis(y), axis(z), size))
        .copied()
        .unwrap_or_default()
}
