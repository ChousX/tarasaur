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
#[derive(Component, Clone, Default)]
pub struct SDFField {
    pub lod: LOD,
    data: Box<[f32]>,
    /// Persistent state: Tracks the exact source feature voxel commanding every coordinate.
    seeds: Box<[PackedCoord]>,
    /// Reusable scratch space to prevent runtime allocations during the JFA pass
    scratch: Vec<PackedCoord>,
    is_dirty: bool,
}

impl SDFField {
    pub fn new(lod: LOD) -> Self {
        let volume = lod.volume();
        Self {
            lod,
            data: vec![f32::MAX; volume].into_boxed_slice(),
            seeds: vec![PackedCoord::EMPTY; volume].into_boxed_slice(),
            scratch: vec![PackedCoord::EMPTY; volume],
            is_dirty: false,
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

    pub fn is_dirty(&self) -> bool {
        self.is_dirty
    }

    pub fn reinit_if_dirty(&mut self) {
        if !self.is_dirty {
            return;
        }

        let size = self.lod.size();
        let volume = self.lod.volume();

        // Ensure the scratchpad buffer matches current LOD dimensions
        if self.scratch.len() != volume {
            self.scratch.resize(volume, PackedCoord::EMPTY);
        }

        // 2. Identify and seed the initial zero-crossing boundary layer
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let idx = self.flatten(x, y, z);
                    let current_sign = self.data[idx].is_sign_negative();

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
                            let n_idx = self.flatten(nx as u32, ny as u32, nz as u32);
                            if self.data[n_idx].is_sign_negative() != current_sign {
                                is_boundary = true;
                                break;
                            }
                        }
                    }

                    if is_boundary {
                        self.seeds[idx] = PackedCoord::new(x, y, z);
                    } else {
                        self.seeds[idx] = PackedCoord::EMPTY;
                    }
                }
            }
        }

        // 3. Main Jump Flood Loops (e.g., 32 -> 16 -> 8 -> 4 -> 2 -> 1)
        let mut step = (size / 2) as i32;
        let mut use_scratch_as_dest = true;

        while step > 0 {
            for z in 0..size {
                for y in 0..size {
                    for x in 0..size {
                        let current_idx = self.flatten(x, y, z);

                        // Select source buffer based on current ping-pong orientation
                        let src_buffer = if use_scratch_as_dest {
                            &self.seeds
                        } else {
                            self.scratch.as_slice()
                        };

                        let mut best_seed = src_buffer[current_idx];
                        let mut min_dist_sq = if best_seed.is_empty() {
                            f32::MAX
                        } else {
                            let (sx, sy, sz) = best_seed.unpack();
                            self.dist_sq(x, y, z, sx, sy, sz)
                        };

                        // Evaluate 26 neighbors at our active jump stride step length
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
                                let neighbor_idx = self.flatten(nx as u32, ny as u32, nz as u32);
                                let neighbor_seed = src_buffer[neighbor_idx];

                                if !neighbor_seed.is_empty() {
                                    let (sx, sy, sz) = neighbor_seed.unpack();
                                    let d_sq = self.dist_sq(x, y, z, sx, sy, sz);
                                    if d_sq < min_dist_sq {
                                        min_dist_sq = d_sq;
                                        best_seed = neighbor_seed;
                                    }
                                }
                            }
                        }

                        // Commit result to opposite buffer
                        if use_scratch_as_dest {
                            self.scratch[current_idx] = best_seed;
                        } else {
                            self.seeds[current_idx] = best_seed;
                        }
                    }
                }
            }

            use_scratch_as_dest = !use_scratch_as_dest;
            step /= 2;
        }

        // If our final pass ended inside the scratch buffer, copy it back over into persistent storage
        if !use_scratch_as_dest {
            self.seeds.copy_from_slice(&self.scratch);
        }

        // 4. Final Distance Calculation & Sign Normalization Pass
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let idx = self.flatten(x, y, z);
                    let final_seed = self.seeds[idx];

                    if final_seed.is_empty() {
                        self.data[idx] = size as f32;
                    } else {
                        let (sx, sy, sz) = final_seed.unpack();
                        let distance = self.dist_sq(x, y, z, sx, sy, sz).sqrt();

                        if self.data[idx].is_sign_negative() {
                            self.data[idx] = -distance;
                        } else {
                            self.data[idx] = distance;
                        }
                    }
                }
            }
        }

        self.is_dirty = false;
    }

    #[inline]
    fn dist_sq(&self, x1: u32, y1: u32, z1: u32, x2: u32, y2: u32, z2: u32) -> f32 {
        let dx = x1 as f32 - x2 as f32;
        let dy = y1 as f32 - y2 as f32;
        let dz = z1 as f32 - z2 as f32;
        dx * dx + dy * dy + dz * dz
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

impl Field<f32> for SDFField {
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
            self.is_dirty = true;
        }
    }
}
