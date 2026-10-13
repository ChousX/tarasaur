use crate::{ApronSample, ExtractGate, FieldFromRaw, FieldLOD, Versionable, VoxelDataSlice};

use super::{Field, LOD};
use bevy::prelude::*;

/// Packed representation of a 3D grid coordinate for JFA seeds.
/// Packs 10 bits per axis (max index 1023) cleanly into a single u32.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PackedCoord(u32);

impl PackedCoord {
    pub const EMPTY: Self = Self(u32::MAX);

    #[inline]
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self((x & 0x3FF) | ((y & 0x3FF) << 10) | ((z & 0x3FF) << 20))
    }

    #[inline]
    pub fn unpack(self) -> (u32, u32, u32) {
        let x = self.0 & 0x3FF;
        let y = (self.0 >> 10) & 0x3FF;
        let z = (self.0 >> 20) & 0x3FF;
        (x, y, z)
    }

    #[inline]
    pub fn is_empty(self) -> bool {
        self.0 == u32::MAX
    }
}

/// SDF values are clamped to +/- this many voxels (a narrow-band SDF). Only
/// the sign and values near zero shape the mesh; beyond that, large values
/// just make sculpting slow. Unclamped, rock deep under the generated
/// surface held ~-100 and a brush needed seconds to flip a voxel there.
pub const SDF_BAND: f32 = 8.0;

#[allow(clippy::upper_case_acronyms)]
#[derive(Component, Clone)]
pub struct SDF {
    pub lod: LOD,
    data: Box<[f32]>,
    seeds: Box<[PackedCoord]>,
    scratch: Vec<PackedCoord>,
    inside_count: u32, // voxels with value <= 0.0, same test as pass1
    pub version: u64,
    /// Per `NEIGHBORS_MASK` entry: bumped only when the low slab that
    /// neighbour's apron reads changes. See `Versionable::border_version`.
    border_versions: [u64; 7],
}

impl Default for SDF {
    fn default() -> Self {
        Self::new(LOD::default())
    }
}

impl SDF {
    pub fn new(lod: LOD) -> Self {
        Self {
            lod,
            data: vec![f32::MAX; lod.volume()].into_boxed_slice(),
            seeds: Box::default(),
            scratch: Vec::new(),
            inside_count: 0,
            version: 0,
            border_versions: [0; 7],
        }
    }
    fn recount(&mut self) {
        self.inside_count = self.data.iter().filter(|v| **v <= 0.0).count() as u32;
    }

    /// `Some(true)` = every voxel inside, `Some(false)` = every voxel outside,
    /// `None` = mixed. O(1).
    #[inline]
    pub fn uniform_sign(&self) -> Option<bool> {
        if self.inside_count == 0 {
            Some(false)
        } else if self.inside_count as usize == self.data.len() {
            Some(true)
        } else {
            None
        }
    }

    #[inline]
    fn flatten(&self, x: u32, y: u32, z: u32) -> usize {
        let size = self.lod.size();
        (z * size * size + y * size + x) as usize
    }

    /// Exposes a read-only slice of the underlying SDF float data for GPU uploading
    #[inline]
    pub fn data_slice(&self) -> &[f32] {
        &self.data
    }

    /// Exposes a read-only slice of the packed JFA seeds for GPU uploading
    #[inline]
    pub fn seeds_slice(&self) -> &[PackedCoord] {
        &self.seeds
    }

    pub fn reinit(&mut self) {
        self.version += 1;
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

        // 2. Identify and seed the initial zero-crossing boundary layer
        jump_flood_distance_field(
            &mut self.data,
            &mut self.seeds,
            &mut self.scratch,
            self.lod.size(),
        );
    }

    /// Snapshot for an async reinit: a clone of the current raw data plus
    /// the version it was cloned at. Cloning here (not moving) so the SDF
    /// stays fully readable — by extraction, by further edits — while the
    /// JFA runs off-thread.
    pub fn reinit_input(&self) -> (Box<[f32]>, u32, u64) {
        (self.data.clone(), self.lod.size(), self.version)
    }

    /// Applies a completed async reinit result. Caller is responsible for
    /// checking the result isn't stale (see `resolve_sdf_reinit_tasks`) —
    /// this just does the write and bumps version like any other mutation.
    pub fn apply_reinit_result(&mut self, data: Box<[f32]>) {
        let old = std::mem::replace(&mut self.data, data);
        self.recount();
        self.version += 1;
        // Only bump the borders the JFA actually changed, so neighbours whose
        // apron is untouched don't re-mesh after every stroke.
        let n = self.lod.size();
        let pad = crate::voxel::systems::PADDING;
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
        let pad = crate::voxel::systems::PADDING;
        let near = [x < pad, y < pad, z < pad];
        if !near.contains(&true) {
            return;
        }
        for (i, off) in crate::voxel::systems::NEIGHBORS_MASK.iter().enumerate() {
            if (off.x == 0 || near[0]) && (off.y == 0 || near[1]) && (off.z == 0 || near[2]) {
                self.border_versions[i] += 1;
            }
        }
    }

    fn touch_all_borders(&mut self) {
        for v in &mut self.border_versions {
            *v += 1;
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

const JFA_NEIGHBOR_OFFSETS: [(i32, i32, i32); 26] = [
    (-1, -1, -1),
    (0, -1, -1),
    (1, -1, -1),
    (-1, 0, -1),
    (0, 0, -1),
    (1, 0, -1),
    (-1, 1, -1),
    (0, 1, -1),
    (1, 1, -1),
    (-1, -1, 0),
    (0, -1, 0),
    (1, -1, 0),
    (-1, 0, 0),
    (1, 0, 0),
    (-1, 1, 0),
    (0, 1, 0),
    (1, 1, 0),
    (-1, -1, 1),
    (0, -1, 1),
    (1, -1, 1),
    (-1, 0, 1),
    (0, 0, 1),
    (1, 0, 1),
    (-1, 1, 1),
    (0, 1, 1),
    (1, 1, 1),
];

impl Field<f32> for SDF {
    fn size(&self) -> UVec3 {
        UVec3::splat(self.lod.size())
    }

    fn get(&self, x: u32, y: u32, z: u32) -> f32 {
        let i = self.flatten(x, y, z);
        self.data[i]
    }

    fn set(&mut self, x: u32, y: u32, z: u32, value: f32) {
        let i = self.flatten(x, y, z);
        let value = value.clamp(-SDF_BAND, SDF_BAND);
        let old = self.data[i];
        if old != value {
            match (old <= 0.0, value <= 0.0) {
                (false, true) => self.inside_count += 1,
                (true, false) => self.inside_count -= 1,
                _ => {}
            }
            self.data[i] = value;
            self.version += 1;
            self.touch_border(x, y, z);
        }
    }
}

impl Versionable for SDF {
    #[inline]
    fn version(&self) -> u64 {
        self.version
    }

    fn border_version(&self, neighbor: usize) -> u64 {
        self.border_versions.get(neighbor).copied().unwrap_or(self.version)
    }

    #[inline]
    fn incorment_version(&mut self) {
        self.version += 1;
        self.touch_all_borders();
    }
}

impl VoxelDataSlice for SDF {
    type Elem = f32;
    #[inline]
    fn data_slice(&self) -> &[f32] {
        &self.data
    }

    fn gpu_bytes(vol: &[f32]) -> Vec<u8> {
        let packed: Vec<i16> = vol.iter().map(|&v| quantize_sdf(v)).collect();
        bytemuck::cast_slice(&packed).to_vec()
    }
}

/// Quantizes an SDF value to snorm16 over +/-`SDF_BAND`, as the GPU copy
/// stores it (the shaders decode with `unpack2x16snorm`). Sign-exact: a
/// positive value never rounds to 0, which the GPU would read as inside, so
/// the CPU cell count and the mesh always agree on which voxels are solid.
#[inline]
pub fn quantize_sdf(v: f32) -> i16 {
    let q = (v / SDF_BAND).clamp(-1.0, 1.0) * i16::MAX as f32;
    let q = q.round() as i16;
    if v > 0.0 { q.max(1) } else { q }
}

impl FieldFromRaw for SDF {
    fn from_raw(lod: LOD, mut data: Vec<f32>) -> Self {
        debug_assert_eq!(data.len(), lod.volume());
        for v in &mut data {
            *v = v.clamp(-SDF_BAND, SDF_BAND);
        }
        let inside_count = data.iter().filter(|v| **v <= 0.0).count() as u32;
        Self {
            lod,
            data: data.into_boxed_slice(),
            seeds: Box::default(),
            scratch: Vec::new(),
            inside_count,
            version: 0,
            border_versions: [0; 7],
        }
    }
}

impl ApronSample for SDF {
    fn sample_apron(data: &[f32], size: u32, x: f32, y: f32, z: f32) -> f32 {
        crate::voxel::systems::sample_neighbor(data, size, x, y, z)
    }
}

impl ExtractGate for SDF {}

impl FieldLOD for SDF {
    fn lod(&self) -> LOD {
        self.lod
    }
}

/// The Jump Flood Algorithm
/// four stages (seed the boundary, flood outward at halving strides,
/// resolve ping-pong buffer, compute final signed distances), just
/// operating on borrowed buffers instead of `&mut self` so it can run
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
    #[inline]
    fn flatten(x: u32, y: u32, z: u32, size: u32) -> usize {
        (z * size * size + y * size + x) as usize
    }
    #[inline]
    fn in_bounds(x: i32, y: i32, z: i32, size: u32) -> bool {
        x >= 0 && y >= 0 && z >= 0 && x < size as i32 && y < size as i32 && z < size as i32
    }
    #[inline]
    fn dist_sq(x1: u32, y1: u32, z1: u32, x2: u32, y2: u32, z2: u32) -> f32 {
        let dx = x1 as f32 - x2 as f32;
        let dy = y1 as f32 - y2 as f32;
        let dz = z1 as f32 - z2 as f32;
        dx * dx + dy * dy + dz * dz
    }

    // 1. Seed boundary voxels (same `<= 0.0` test as pass1 / inside_count).
    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let idx = flatten(x, y, z, size);
                let inside = data[idx] <= 0.0;
                let mut is_boundary = false;
                for (dx, dy, dz) in CARDINAL_NEIGHBORS {
                    let (nx, ny, nz) = (x as i32 + dx, y as i32 + dy, z as i32 + dz);
                    if in_bounds(nx, ny, nz, size)
                        && (data[flatten(nx as u32, ny as u32, nz as u32, size)] <= 0.0) != inside
                    {
                        is_boundary = true;
                        break;
                    }
                }
                seeds[idx] = if is_boundary {
                    PackedCoord::new(x, y, z)
                } else {
                    PackedCoord::EMPTY
                };
            }
        }
    }

    // 2. Flood: size/2 .. 1, then two refinement passes (2, 1).
    let mut steps: Vec<i32> = Vec::new();
    let mut s = (size / 2) as i32;
    while s > 0 {
        steps.push(s);
        s /= 2;
    }
    steps.push(2);
    steps.push(1);

    let mut use_scratch_as_dest = true;
    for step in steps {
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let current_idx = flatten(x, y, z, size);
                    let src_buffer: &[PackedCoord] =
                        if use_scratch_as_dest { seeds } else { scratch };

                    let mut best_seed = src_buffer[current_idx];
                    let mut min_dist_sq = if best_seed.is_empty() {
                        f32::MAX
                    } else {
                        let (sx, sy, sz) = best_seed.unpack();
                        dist_sq(x, y, z, sx, sy, sz)
                    };

                    for (dx, dy, dz) in JFA_NEIGHBOR_OFFSETS {
                        let (nx, ny, nz) = (
                            x as i32 + dx * step,
                            y as i32 + dy * step,
                            z as i32 + dz * step,
                        );
                        if in_bounds(nx, ny, nz, size) {
                            let neighbor_seed =
                                src_buffer[flatten(nx as u32, ny as u32, nz as u32, size)];
                            if !neighbor_seed.is_empty() {
                                let (sx, sy, sz) = neighbor_seed.unpack();
                                let d_sq = dist_sq(x, y, z, sx, sy, sz);
                                if d_sq < min_dist_sq {
                                    min_dist_sq = d_sq;
                                    best_seed = neighbor_seed;
                                }
                            }
                        }
                    }

                    if use_scratch_as_dest {
                        scratch[current_idx] = best_seed;
                    } else {
                        seeds[current_idx] = best_seed;
                    }
                }
            }
        }
        use_scratch_as_dest = !use_scratch_as_dest;
    }
    if !use_scratch_as_dest {
        seeds.copy_from_slice(scratch);
    }

    // 3. Exact distance to the interpolated zero-crossing points on the
    // sign-changing edges of voxel `s` (squared; f32::MAX if it has none).
    let crossing_dist_sq = |px: u32, py: u32, pz: u32, sx: i32, sy: i32, sz: i32| -> f32 {
        if !in_bounds(sx, sy, sz, size) {
            return f32::MAX;
        }
        let vs = data[flatten(sx as u32, sy as u32, sz as u32, size)];
        let s_in = vs <= 0.0;
        let mut best = f32::MAX;
        for (dx, dy, dz) in CARDINAL_NEIGHBORS {
            let (nx, ny, nz) = (sx + dx, sy + dy, sz + dz);
            if !in_bounds(nx, ny, nz, size) {
                continue;
            }
            let vn = data[flatten(nx as u32, ny as u32, nz as u32, size)];
            if (vn <= 0.0) != s_in {
                let (a, b) = (vs.abs(), vn.abs());
                let t = if a + b > 0.0 { a / (a + b) } else { 0.5 };
                let ex = px as f32 - (sx as f32 + dx as f32 * t);
                let ey = py as f32 - (sy as f32 + dy as f32 * t);
                let ez = pz as f32 - (sz as f32 + dz as f32 * t);
                best = best.min(ex * ex + ey * ey + ez * ez);
            }
        }
        best
    };

    let mut out = vec![0.0f32; data.len()];
    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let idx = flatten(x, y, z, size);
                let inside = data[idx] <= 0.0;
                let mag = if seeds[idx].is_empty() {
                    size as f32
                } else {
                    let (sx, sy, sz) = seeds[idx].unpack();
                    let (sx, sy, sz) = (sx as i32, sy as i32, sz as i32);
                    // The seed and its 6 neighbours: the true nearest crossing
                    // can sit on a neighbouring boundary voxel.
                    let mut best = crossing_dist_sq(x, y, z, sx, sy, sz);
                    for (dx, dy, dz) in CARDINAL_NEIGHBORS {
                        best = best.min(crossing_dist_sq(x, y, z, sx + dx, sy + dy, sz + dz));
                    }
                    if best == f32::MAX {
                        size as f32
                    } else {
                        best.sqrt()
                    }
                };
                let mag = mag.clamp(1e-4, SDF_BAND);
                out[idx] = if inside { -mag } else { mag };
            }
        }
    }
    data.copy_from_slice(&out);
}

/// `Some(true)` = every voxel inside (`<= 0.0`), `Some(false)` = every voxel
/// outside, `None` = mixed. Same test as `inside_count` and the GPU passes.
pub fn uniform_sign_volume(data: &[f32]) -> Option<bool> {
    let first = *data.first()? <= 0.0;
    data.iter().all(|v| (*v <= 0.0) == first).then_some(first)
}

/// A uniform volume has no surface inside it, but one may sit just past any
/// face. Each voxel gets its distance (in voxels) to the nearest voxel of a
/// neighbouring chunk. A flat +/-size made such chunks uneditable: an
/// accumulate brush had to push a voxel through a whole chunk's worth of
/// distance before its sign flipped, and every reinit in between reset it.
pub fn fill_uniform(data: &mut [f32], size: u32, neg: bool) {
    let sign = if neg { -1.0 } else { 1.0 };
    let edge = |c: u32| (c + 1).min(size - c);
    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let d = edge(x).min(edge(y)).min(edge(z));
                data[((z * size + y) * size + x) as usize] = sign * (d as f32).min(SDF_BAND);
            }
        }
    }
}

/// Signs in `data` -> signed distances (in voxels of this LOD). Uniform
/// volumes have no surface, so they get `fill_uniform` and no buffers are
/// allocated. Use this instead of calling the JFA directly.
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
    use crate::{FieldFromRaw, LOD, field::Field};

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
        use crate::Versionable;
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
        let decode = |q: i16| (q as f32 / i16::MAX as f32).max(-1.0) * SDF_BAND;
        for &v in &[1e-9, 1e-5, 0.5, 3.7, 8.0, 50.0] {
            assert!(decode(quantize_sdf(v)) > 0.0, "{v} must stay outside");
            assert!(decode(quantize_sdf(-v)) <= 0.0, "-{v} must stay inside");
        }
        assert_eq!(quantize_sdf(0.0), 0); // 0 is inside, as in f32
        let step = SDF_BAND / i16::MAX as f32;
        for i in -800..=800 {
            let v = i as f32 * 0.01;
            assert!((decode(quantize_sdf(v)) - v).abs() <= step);
        }
    }
}
