//! The signed distance field that defines terrain shape, plus the CPU jump
//! flood that turns sign volumes into distances.
//!
//! Values are in voxels of the chunk's own LOD, negative inside, clamped to
//! `±SDF_BAND`. The GPU copy is snorm16 (`quantize_sdf`).

use crate::field::{GpuChannel, VoxelField};
use crate::num::{f32_from_i32, f32_from_u32, i16_from_f32};

use super::{Field, LOD};
use crate::grid::{NEIGHBORS_MASK, PADDING, flatten_cube, sample_neighbor};
use bevy::prelude::*;

/// Packed representation of a 3D grid coordinate for JFA seeds.
/// Packs 10 bits per axis (max index 1023) cleanly into a single u32.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PackedCoord(u32);

impl PackedCoord {
    pub const EMPTY: Self = Self(u32::MAX);

    #[inline]
    #[must_use]
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self((x & 0x3FF) | ((y & 0x3FF) << 10) | ((z & 0x3FF) << 20))
    }

    #[inline]
    #[must_use]
    pub const fn unpack(self) -> (u32, u32, u32) {
        let x = self.0 & 0x3FF;
        let y = (self.0 >> 10) & 0x3FF;
        let z = (self.0 >> 20) & 0x3FF;
        (x, y, z)
    }

    #[inline]
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == u32::MAX
    }
}

/// SDF values are clamped to +/- this many voxels (a narrow-band SDF).
///
/// Only the sign and values near zero shape the mesh; beyond that, large
/// values just make sculpting slow. Unclamped, rock deep under the generated
/// surface held ~-100 and a brush needed seconds to flip a voxel there.
pub const SDF_BAND: f32 = 8.0;

/// A chunk's signed distances. Tracks its inside-voxel count (for O(1)
/// `uniform_sign`) and per-neighbour border versions (to limit re-meshing).
#[allow(clippy::upper_case_acronyms)]
#[derive(Component, Clone)]
pub struct SDF {
    pub lod: LOD,
    data: Box<[f32]>,
    seeds: Box<[PackedCoord]>,
    scratch: Vec<PackedCoord>,
    inside_count: usize, // voxels with value <= 0.0, same test as pass1
    pub version: u64,
    /// Per `NEIGHBORS_MASK` entry: bumped only when the low slab that
    /// neighbour's apron reads changes. See `VoxelField::border_version`.
    border_versions: [u64; 7],
}

impl Default for SDF {
    fn default() -> Self {
        Self::new(LOD::default())
    }
}

impl SDF {
    #[must_use]
    pub fn new(lod: LOD) -> Self {
        Self::with_data(lod, vec![f32::MAX; lod.volume()].into_boxed_slice())
    }

    fn with_data(lod: LOD, data: Box<[f32]>) -> Self {
        let mut sdf = Self {
            lod,
            data,
            seeds: Box::default(),
            scratch: Vec::new(),
            inside_count: 0,
            version: 0,
            border_versions: [0; 7],
        };
        sdf.recount();
        sdf
    }

    fn recount(&mut self) {
        self.inside_count = self.data.iter().filter(|v| **v <= 0.0).count();
    }

    /// `Some(true)` = every voxel inside, `Some(false)` = every voxel outside,
    /// `None` = mixed. O(1).
    #[inline]
    #[must_use]
    pub fn uniform_sign(&self) -> Option<bool> {
        if self.inside_count == 0 {
            Some(false)
        } else if self.inside_count == self.data.len() {
            Some(true)
        } else {
            None
        }
    }

    #[inline]
    const fn flatten(&self, x: u32, y: u32, z: u32) -> usize {
        flatten_cube(x, y, z, self.lod.size())
    }

    /// Exposes a read-only slice of the underlying SDF float data for GPU uploading
    #[inline]
    #[must_use]
    pub fn data_slice(&self) -> &[f32] {
        &self.data
    }

    pub fn reinit(&mut self) {
        self.version = self.version.wrapping_add(1);
        self.touch_all_borders();
        let volume = self.lod.volume();
        // Guarantee buffers match the expected volume before sampling
        if self.data.len() != volume {
            self.data = vec![f32::MAX; volume].into_boxed_slice();
        }
        if let Some(neg) = uniform_sign_volume(&self.data) {
            fill_uniform(&mut self.data, self.lod.size(), neg);
            self.recount();
            return; // no seeds/scratch allocation
        }
        if self.seeds.len() != volume {
            self.seeds = vec![PackedCoord::EMPTY; volume].into_boxed_slice();
        }
        // Ensure the scratchpad buffer matches current LOD dimensions
        if self.scratch.len() != volume {
            self.scratch.resize(volume, PackedCoord::EMPTY);
        }

        // Seed the zero-crossing boundary, flood, and resolve distances
        jump_flood_distance_field(
            &mut self.data,
            &mut self.seeds,
            &mut self.scratch,
            self.lod.size(),
        );
    }

    /// Snapshot for an async reinit: a clone of the current raw data plus
    /// the version it was cloned at.
    ///
    /// Cloning here (not moving) so the SDF stays fully readable — by
    /// extraction, by further edits — while the JFA runs off-thread.
    #[must_use]
    pub fn reinit_input(&self) -> (Box<[f32]>, u32, u64) {
        (self.data.clone(), self.lod.size(), self.version)
    }

    /// Applies a completed async reinit result.
    ///
    /// Caller is responsible for checking the result isn't stale (see
    /// `resolve_sdf_reinit_tasks`) — this just does the write and bumps
    /// version like any other mutation.
    pub fn apply_reinit_result(&mut self, data: Box<[f32]>) {
        let old = std::mem::replace(&mut self.data, data);
        self.recount();
        self.version = self.version.wrapping_add(1);
        // Only bump the borders the JFA actually changed, so neighbours whose
        // apron is untouched don't re-mesh after every stroke.
        let n = self.lod.size();
        let pad = PADDING;
        for z in 0..n {
            for y in 0..n {
                let x_end = if z < pad || y < pad { n } else { pad.min(n) };
                for x in 0..x_end {
                    let i = self.flatten(x, y, z);
                    if old.get(i) != self.data.get(i) {
                        self.touch_border(x, y, z);
                    }
                }
            }
        }
    }

    /// Bumps the border versions of every neighbour whose apron reads (x, y, z).
    fn touch_border(&mut self, x: u32, y: u32, z: u32) {
        let pad = PADDING;
        let near = [x < pad, y < pad, z < pad];
        if !near.contains(&true) {
            return;
        }
        let masks = NEIGHBORS_MASK.iter();
        for (version, off) in self.border_versions.iter_mut().zip(masks) {
            if (off.x == 0 || near[0]) && (off.y == 0 || near[1]) && (off.z == 0 || near[2]) {
                *version = version.wrapping_add(1);
            }
        }
    }

    fn touch_all_borders(&mut self) {
        for v in &mut self.border_versions {
            *v = v.wrapping_add(1);
        }
    }
}

const CARDINAL_NEIGHBORS: [(i32, i32, i32); 6] = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
];

/// The 26 neighbours of a voxel, x fastest, then y, then z.
const JFA_NEIGHBOR_OFFSETS: [(i32, i32, i32); 26] = {
    let mut out = [(0, 0, 0); 26];
    let mut rest: &mut [(i32, i32, i32)] = &mut out;
    let mut i = 0;
    while let [slot, tail @ ..] = rest {
        if i == 13 {
            i += 1; // skip the voxel itself
        }
        *slot = (i % 3 - 1, i / 3 % 3 - 1, i / 9 - 1);
        i += 1;
        rest = tail;
    }
    out
};

impl Field<f32> for SDF {
    fn size(&self) -> UVec3 {
        UVec3::splat(self.lod.size())
    }

    /// Out-of-range coordinates read as outside (`SDF_BAND`).
    fn get(&self, x: u32, y: u32, z: u32) -> f32 {
        let i = self.flatten(x, y, z);
        self.data.get(i).copied().unwrap_or(SDF_BAND)
    }

    /// Out-of-range coordinates are ignored.
    fn set(&mut self, x: u32, y: u32, z: u32, value: f32) {
        let i = self.flatten(x, y, z);
        let value = value.clamp(-SDF_BAND, SDF_BAND);
        let Some(slot) = self.data.get_mut(i) else {
            return;
        };
        let old = *slot;
        if old.to_bits() != value.to_bits() {
            *slot = value;
            match (old <= 0.0, value <= 0.0) {
                (false, true) => self.inside_count = self.inside_count.saturating_add(1),
                (true, false) => self.inside_count = self.inside_count.saturating_sub(1),
                _ => {}
            }
            self.version = self.version.wrapping_add(1);
            self.touch_border(x, y, z);
        }
    }
}

impl VoxelField for SDF {
    type Value = f32;
    type Elem = f32;
    const GPU: GpuChannel = GpuChannel::Shape;
    const CLEAR_DIRTY_EACH_CYCLE: bool = false;

    fn new_for_lod(lod: LOD) -> Self {
        Self::new(lod)
    }

    fn from_raw(lod: LOD, mut data: Vec<f32>) -> Self {
        debug_assert_eq!(data.len(), lod.volume());
        for v in &mut data {
            *v = v.clamp(-SDF_BAND, SDF_BAND);
        }
        Self::with_data(lod, data.into_boxed_slice())
    }

    fn lod(&self) -> LOD {
        self.lod
    }

    #[inline]
    fn data_slice(&self) -> &[f32] {
        &self.data
    }

    #[inline]
    fn version(&self) -> u64 {
        self.version
    }

    fn border_version(&self, neighbor: usize) -> u64 {
        self.border_versions
            .get(neighbor)
            .copied()
            .unwrap_or(self.version)
    }

    fn gpu_bytes(vol: &[f32]) -> Vec<u8> {
        let packed: Vec<i16> = vol.iter().map(|&v| quantize_sdf(v)).collect();
        bytemuck::cast_slice(&packed).to_vec()
    }

    /// Distances are in voxels of their own LOD: a grid `ratio`x finer has
    /// voxels `ratio`x smaller, so the same distance is `ratio`x fewer of ours.
    #[inline]
    fn rescale(v: f32, ratio: f32) -> f32 {
        v / ratio
    }

    fn sample_apron(data: &[f32], size: u32, x: f32, y: f32, z: f32) -> f32 {
        sample_neighbor(data, size, x, y, z)
    }

    /// Linear: exact for a planar surface. Clamping instead flattened the
    /// gradient across the edge, which bent normals and folded triangles
    /// along the outermost loaded chunks.
    fn extend_edge(edge: f32, inward: [f32; 3], past: UVec3) -> f32 {
        inward
            .iter()
            .zip(past.to_array())
            .fold(edge, |v, (&i, k)| (edge - i).mul_add(f32_from_u32(k), v))
    }
}

impl super::DistanceField for SDF {
    fn reinit_input(&self) -> (Box<[f32]>, u32, u64) {
        Self::reinit_input(self)
    }

    fn apply_reinit_result(&mut self, data: Box<[f32]>) {
        Self::apply_reinit_result(self, data);
    }
}

/// Quantizes an SDF value to snorm16 over +/-`SDF_BAND`, as the GPU copy
/// stores it (the shaders decode with `unpack2x16snorm`).
///
/// Sign-exact: a positive value never rounds to 0, which the GPU would read
/// as inside, so the CPU cell count and the mesh always agree on which
/// voxels are solid.
#[inline]
#[must_use]
pub fn quantize_sdf(v: f32) -> i16 {
    let q = (v / SDF_BAND).clamp(-1.0, 1.0) * f32::from(i16::MAX);
    let q = i16_from_f32(q.round());
    if v > 0.0 { q.max(1) } else { q }
}

type Coord = (u32, u32, u32);

/// `c + d * step` if it lands inside a `size^3` grid.
#[inline]
fn offset(c: Coord, d: (i32, i32, i32), step: i32, size: u32) -> Option<Coord> {
    let axis = |c: u32, d: i32| {
        c.checked_add_signed(d.checked_mul(step)?)
            .filter(|&v| v < size)
    };
    Some((axis(c.0, d.0)?, axis(c.1, d.1)?, axis(c.2, d.2)?))
}

#[inline]
fn dist_sq(a: Coord, b: Coord) -> f32 {
    let dx = f32_from_u32(a.0) - f32_from_u32(b.0);
    let dy = f32_from_u32(a.1) - f32_from_u32(b.1);
    let dz = f32_from_u32(a.2) - f32_from_u32(b.2);
    dz.mul_add(dz, dy.mul_add(dy, dx * dx))
}

/// Whether the voxel at `c` is inside (`<= 0.0`); `None` off the grid.
#[inline]
fn inside_at(data: &[f32], size: u32, c: Coord) -> Option<bool> {
    data.get(flatten_cube(c.0, c.1, c.2, size))
        .map(|&v| v <= 0.0)
}

/// The Jump Flood Algorithm.
///
/// Three stages (seed the boundary, flood outward at halving strides,
/// compute final signed distances from each voxel's nearest seed), operating
/// on borrowed buffers instead of `&mut self` so it can run
/// inside an async task with no live ECS component in scope.
///
/// `data` holds signs on entry (negative = inside) and signed distances on
/// exit. `seeds`/`scratch` are working buffers, each `size^3` long —
/// caller owns their allocation and initial `PackedCoord::EMPTY` fill.
pub fn jump_flood_distance_field(
    data: &mut [f32],
    seeds: &mut [PackedCoord],
    scratch: &mut [PackedCoord],
    size: u32,
) {
    if let Some(neg) = uniform_sign_volume(data) {
        fill_uniform(data, size, neg);
        return;
    }
    seed_boundary(data, seeds, size);
    flood(seeds, scratch, size);
    resolve_distances(data, seeds, size);
}

/// Stage 1: seed boundary voxels (same `<= 0.0` test as pass1 / `inside_count`).
fn seed_boundary(data: &[f32], seeds: &mut [PackedCoord], size: u32) {
    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let Some(seed) = seeds.get_mut(flatten_cube(x, y, z, size)) else {
                    continue;
                };
                let Some(inside) = inside_at(data, size, (x, y, z)) else {
                    continue;
                };
                let is_boundary = CARDINAL_NEIGHBORS.iter().any(|&d| {
                    offset((x, y, z), d, 1, size)
                        .and_then(|n| inside_at(data, size, n))
                        .is_some_and(|n_inside| n_inside != inside)
                });
                *seed = if is_boundary {
                    PackedCoord::new(x, y, z)
                } else {
                    PackedCoord::EMPTY
                };
            }
        }
    }
}

/// Stage 2, flood: size/2 .. 1, then two refinement passes (2, 1). Leaves the
/// nearest boundary seed of every voxel in `seeds`.
fn flood(seeds: &mut [PackedCoord], scratch: &mut [PackedCoord], size: u32) {
    let mut steps: Vec<i32> = Vec::new();
    let mut s = i32::try_from(size / 2).unwrap_or(i32::MAX);
    while s > 0 {
        steps.push(s);
        s /= 2;
    }
    steps.push(2);
    steps.push(1);

    let mut use_scratch_as_dest = true;
    for step in steps {
        let (src, dst): (&[PackedCoord], &mut [PackedCoord]) = if use_scratch_as_dest {
            (&*seeds, &mut *scratch)
        } else {
            (&*scratch, &mut *seeds)
        };
        flood_step(src, dst, size, step);
        use_scratch_as_dest = !use_scratch_as_dest;
    }
    if !use_scratch_as_dest {
        seeds.copy_from_slice(scratch);
    }
}

fn flood_step(src: &[PackedCoord], dst: &mut [PackedCoord], size: u32, step: i32) {
    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let idx = flatten_cube(x, y, z, size);
                let (Some(&seed), Some(out)) = (src.get(idx), dst.get_mut(idx)) else {
                    continue;
                };
                let mut best_seed = seed;
                let mut min_dist_sq = if seed.is_empty() {
                    f32::MAX
                } else {
                    dist_sq((x, y, z), seed.unpack())
                };

                for d in JFA_NEIGHBOR_OFFSETS {
                    let Some((nx, ny, nz)) = offset((x, y, z), d, step, size) else {
                        continue;
                    };
                    let Some(&neighbor_seed) = src.get(flatten_cube(nx, ny, nz, size)) else {
                        continue;
                    };
                    if !neighbor_seed.is_empty() {
                        let d_sq = dist_sq((x, y, z), neighbor_seed.unpack());
                        if d_sq < min_dist_sq {
                            min_dist_sq = d_sq;
                            best_seed = neighbor_seed;
                        }
                    }
                }
                *out = best_seed;
            }
        }
    }
}

/// Exact distance from `p` to the interpolated zero-crossing points on the
/// sign-changing edges of voxel `s` (squared; `None` if it has none).
fn crossing_dist_sq(data: &[f32], size: u32, p: Coord, s: Coord) -> Option<f32> {
    let &vs = data.get(flatten_cube(s.0, s.1, s.2, size))?;
    let s_in = vs <= 0.0;
    CARDINAL_NEIGHBORS
        .iter()
        .filter_map(|&(dx, dy, dz)| {
            let (nx, ny, nz) = offset(s, (dx, dy, dz), 1, size)?;
            let &vn = data.get(flatten_cube(nx, ny, nz, size))?;
            if (vn <= 0.0) == s_in {
                return None;
            }
            let (abs_s, abs_n) = (vs.abs(), vn.abs());
            let t = if abs_s + abs_n > 0.0 {
                abs_s / (abs_s + abs_n)
            } else {
                0.5
            };
            let ex = f32_from_u32(p.0) - f32_from_i32(dx).mul_add(t, f32_from_u32(s.0));
            let ey = f32_from_u32(p.1) - f32_from_i32(dy).mul_add(t, f32_from_u32(s.1));
            let ez = f32_from_u32(p.2) - f32_from_i32(dz).mul_add(t, f32_from_u32(s.2));
            Some(ez.mul_add(ez, ey.mul_add(ey, ex * ex)))
        })
        .reduce(f32::min)
}

/// Stage 3: signed distances from each voxel's nearest seed.
fn resolve_distances(data: &mut [f32], seeds: &[PackedCoord], size: u32) {
    let far = f32_from_u32(size);
    let mut out = vec![0.0f32; data.len()];
    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let idx = flatten_cube(x, y, z, size);
                let (Some(&v), Some(&seed), Some(o)) =
                    (data.get(idx), seeds.get(idx), out.get_mut(idx))
                else {
                    continue;
                };
                let mag = if seed.is_empty() {
                    far
                } else {
                    // The seed and its 6 neighbours: the true nearest crossing
                    // can sit on a neighbouring boundary voxel.
                    let s = seed.unpack();
                    std::iter::once(s)
                        .chain(
                            CARDINAL_NEIGHBORS
                                .iter()
                                .filter_map(|&d| offset(s, d, 1, size)),
                        )
                        .filter_map(|c| crossing_dist_sq(data, size, (x, y, z), c))
                        .reduce(f32::min)
                        .map_or(far, f32::sqrt)
                };
                let mag = mag.clamp(1e-4, SDF_BAND);
                *o = if v <= 0.0 { -mag } else { mag };
            }
        }
    }
    data.copy_from_slice(&out);
}

/// `Some(true)` = every voxel inside (`<= 0.0`), `Some(false)` = every voxel
/// outside, `None` = mixed. Same test as `inside_count` and the GPU passes.
#[must_use]
pub fn uniform_sign_volume(data: &[f32]) -> Option<bool> {
    let first = *data.first()? <= 0.0;
    data.iter().all(|v| (*v <= 0.0) == first).then_some(first)
}

/// A uniform volume has no surface inside it, but one may sit just past any
/// face.
///
/// Each voxel gets its distance (in voxels) to the nearest voxel of a
/// neighbouring chunk. A flat +/-size made such chunks uneditable: an
/// accumulate brush had to push a voxel through a whole chunk's worth of
/// distance before its sign flipped, and every reinit in between reset it.
pub fn fill_uniform(data: &mut [f32], size: u32, neg: bool) {
    let sign = if neg { -1.0 } else { 1.0 };
    let edge = |c: u32| c.saturating_add(1).min(size.saturating_sub(c));
    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let d = edge(x).min(edge(y)).min(edge(z));
                if let Some(v) = data.get_mut(flatten_cube(x, y, z, size)) {
                    *v = sign * f32_from_u32(d).min(SDF_BAND);
                }
            }
        }
    }
}

/// Signs in `data` -> signed distances (in voxels of this LOD).
///
/// Uniform volumes have no surface, so they get `fill_uniform` and no
/// buffers are allocated. Use this instead of calling the JFA directly.
pub fn compute_sdf_distances(data: &mut [f32], size: u32) {
    if let Some(neg) = uniform_sign_volume(data) {
        fill_uniform(data, size, neg);
        return;
    }
    let volume = data.len();
    let mut seeds = vec![PackedCoord::EMPTY; volume];
    let mut scratch = vec![PackedCoord::EMPTY; volume];
    jump_flood_distance_field(data, &mut seeds, &mut scratch, size);
}

#[cfg(test)]
mod band_tests {
    use super::{SDF, SDF_BAND};
    use crate::{
        field::Field,
        field::{LOD, VoxelField},
    };

    #[test]
    fn downsample_converts_to_coarse_voxels() {
        use crate::storage::persistence::downsample_bytes;
        // A plane at world x = 5: distance in voxels of each LOD.
        let plane = |lod: LOD| -> Vec<f32> {
            let n = crate::num::usize_from(lod.size());
            (0..lod.volume())
                .map(|i| crate::num::f32_from_usize(i % n) - 5.0 / lod.voxel_size())
                .collect()
        };
        let bytes = bytemuck::cast_slice(&plane(LOD::Medium)).to_vec();
        let down: Vec<f32> =
            bytemuck::pod_collect_to_vec(&downsample_bytes::<SDF>(&bytes, LOD::Medium, LOD::Low));
        assert_eq!(down, plane(LOD::Low));
    }

    #[test]
    fn values_stay_in_the_narrow_band() {
        let n = LOD::Lowest.volume();
        let mut sdf = SDF::from_raw(LOD::Lowest, vec![-150.0; n]);
        assert_eq!(sdf.get(1, 1, 1), -SDF_BAND);
        sdf.set(1, 1, 1, 400.0);
        assert_eq!(sdf.get(1, 1, 1), SDF_BAND);
        assert_eq!(sdf.uniform_sign(), None); // sign still tracked
    }

    #[test]
    fn border_versions_only_bump_for_touched_faces() {
        let n = LOD::Lowest.volume();
        let mut sdf = SDF::from_raw(LOD::Lowest, vec![1.0; n]);
        sdf.set(3, 3, 3, -1.0); // interior of a 4^3 grid: no neighbour reads it
        assert!((0..7).all(|i| sdf.border_version(i) == 0));
        sdf.set(0, 3, 3, -1.0); // x = 0 slab: only the +x neighbour (index 0)
        assert_eq!(sdf.border_version(0), 1);
        assert!((1..7).all(|i| sdf.border_version(i) == 0));
        sdf.set(0, 0, 0, -1.0); // corner: every neighbour
        assert!((0..7).all(|i| sdf.border_version(i) >= 1));
    }

    #[test]
    fn quantize_keeps_sign_and_precision() {
        use super::quantize_sdf;
        // Decode exactly as WGSL's unpack2x16snorm does.
        let decode = |q: i16| (f32::from(q) / f32::from(i16::MAX)).max(-1.0) * SDF_BAND;
        for &v in &[1e-9, 1e-5, 0.5, 3.7, 8.0, 50.0] {
            assert!(decode(quantize_sdf(v)) > 0.0, "{v} must stay outside");
            assert!(decode(quantize_sdf(-v)) <= 0.0, "-{v} must stay inside");
        }
        assert_eq!(quantize_sdf(0.0), 0); // 0 is inside, as in f32
        let step = SDF_BAND / f32::from(i16::MAX);
        for i in -800..=800 {
            let v = crate::num::f32_from_i32(i) * 0.01;
            assert!((decode(quantize_sdf(v)) - v).abs() <= step);
        }
    }
}
