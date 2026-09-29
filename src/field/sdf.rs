use crate::{ApronSample, ExtractGate, FieldLOD, Versionable, VoxelDataSlice};

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

#[allow(clippy::upper_case_acronyms)]
#[derive(Component, Clone)]
pub struct SDF {
    pub lod: LOD,
    data: Box<[f32]>,
    seeds: Box<[PackedCoord]>,
    scratch: Vec<PackedCoord>,
    pub version: u64,
}

impl Default for SDF {
    fn default() -> Self {
        Self::new(LOD::default())
    }
}

impl SDF {
    pub fn new(lod: LOD) -> Self {
        let volume = lod.volume();
        Self {
            lod,
            data: vec![f32::MAX; volume].into_boxed_slice(),
            seeds: vec![PackedCoord::EMPTY; volume].into_boxed_slice(),
            scratch: vec![PackedCoord::EMPTY; volume],
            version: 0,
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
        let volume = self.lod.volume();

        // Guarantee buffers match the expected volume before sampling
        if self.data.len() != volume {
            self.data = vec![f32::MAX; volume].into_boxed_slice();
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
        self.data = data;
        self.version += 1;
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
        if self.data[i] != value {
            self.data[i] = value;
            self.version += 1;
        }
    }
}

impl Versionable for SDF {
    #[inline]
    fn version(&self) -> u64 {
        self.version
    }

    #[inline]
    fn incorment_version(&mut self) {
        self.version += 1;
    }
}

impl VoxelDataSlice for SDF {
    type Elem = f32;
    #[inline]
    fn data_slice(&self) -> &[f32] {
        &self.data
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

/// The Jump Flood Algorithm pass extracted from `SDFField::reinit` — same
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
    #[inline]
    fn flatten(x: u32, y: u32, z: u32, size: u32) -> usize {
        (z * size * size + y * size + x) as usize
    }

    #[inline]
    fn dist_sq(x1: u32, y1: u32, z1: u32, x2: u32, y2: u32, z2: u32) -> f32 {
        let dx = x1 as f32 - x2 as f32;
        let dy = y1 as f32 - y2 as f32;
        let dz = z1 as f32 - z2 as f32;
        dx * dx + dy * dy + dz * dz
    }

    // 1. Identify and seed the initial zero-crossing boundary layer.
    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let idx = flatten(x, y, z, size);
                let current_sign = data[idx].is_sign_negative();

                let mut is_boundary = false;
                for (dx, dy, dz) in CARDINAL_NEIGHBORS {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    let nz = z as i32 + dz;

                    if nx >= 0
                        && nx < size as i32
                        && ny >= 0
                        && ny < size as i32
                        && nz >= 0
                        && nz < size as i32
                    {
                        let n_idx = flatten(nx as u32, ny as u32, nz as u32, size);
                        if data[n_idx].is_sign_negative() != current_sign {
                            is_boundary = true;
                            break;
                        }
                    }
                }

                if is_boundary {
                    seeds[idx] = PackedCoord::new(x, y, z);
                } else {
                    seeds[idx] = PackedCoord::EMPTY;
                }
            }
        }
    }

    // 2. Main Jump Flood loop (e.g., 32 -> 16 -> 8 -> 4 -> 2 -> 1).
    let mut step = (size / 2) as i32;
    let mut use_scratch_as_dest = true;

    while step > 0 {
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let current_idx = flatten(x, y, z, size);

                    // Select source buffer based on current ping-pong orientation.
                    let src_buffer: &[PackedCoord] =
                        if use_scratch_as_dest { seeds } else { scratch };

                    let mut best_seed = src_buffer[current_idx];
                    let mut min_dist_sq = if best_seed.is_empty() {
                        f32::MAX
                    } else {
                        let (sx, sy, sz) = best_seed.unpack();
                        dist_sq(x, y, z, sx, sy, sz)
                    };

                    // Evaluate 26 neighbors at our active jump stride length.
                    for (dx, dy, dz) in JFA_NEIGHBOR_OFFSETS {
                        let nx = x as i32 + dx * step;
                        let ny = y as i32 + dy * step;
                        let nz = z as i32 + dz * step;

                        if nx >= 0
                            && nx < size as i32
                            && ny >= 0
                            && ny < size as i32
                            && nz >= 0
                            && nz < size as i32
                        {
                            let neighbor_idx = flatten(nx as u32, ny as u32, nz as u32, size);
                            let neighbor_seed = src_buffer[neighbor_idx];

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

                    // Commit result to opposite buffer.
                    if use_scratch_as_dest {
                        scratch[current_idx] = best_seed;
                    } else {
                        seeds[current_idx] = best_seed;
                    }
                }
            }
        }

        use_scratch_as_dest = !use_scratch_as_dest;
        step /= 2;
    }

    // If the final pass ended inside the scratch buffer, copy it back over
    // into persistent storage.
    if !use_scratch_as_dest {
        seeds.copy_from_slice(scratch);
    }

    // 3. Final distance calculation & sign normalization pass.
    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let idx = flatten(x, y, z, size);
                let final_seed = seeds[idx];

                if final_seed.is_empty() {
                    data[idx] = size as f32;
                } else {
                    let (sx, sy, sz) = final_seed.unpack();
                    let distance = dist_sq(x, y, z, sx, sy, sz).sqrt();

                    if data[idx].is_sign_negative() {
                        data[idx] = -distance;
                    } else {
                        data[idx] = distance;
                    }
                }
            }
        }
    }
}
