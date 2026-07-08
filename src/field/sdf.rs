use super::{Field, LOD};
use bevy::{platform::collections::HashSet, prelude::*};

#[allow(clippy::upper_case_acronyms)]
#[derive(Component, Clone)]
pub struct SDFField {
    pub size: u32,
    data: Box<[f32]>,
    dirty: HashSet<u32>,
}

impl SDFField {
    pub fn new(lod: LOD) -> Self {
        let size = lod.size();
        let volume = lod.volume();
        Self {
            size,
            data: vec![f32::MAX; volume].into_boxed_slice(),
            dirty: HashSet::default(),
        }
    }

    #[inline]
    fn flatten(&self, x: u32, y: u32, z: u32) -> u32 {
        z * self.size * self.size + y * self.size + x
    }

    #[inline]
    fn unflatten(&self, flat: u32) -> (u32, u32, u32) {
        let x = flat % self.size;
        let y = (flat / self.size) % self.size;
        let z = flat / (self.size * self.size);
        (x, y, z)
    }

    pub fn is_dirty(&self) -> bool {
        !self.dirty.is_empty()
    }

    pub fn reinit_if_dirty(&mut self) {
        if self.dirty.is_empty() {
            return;
        }
        let mut to_fix: HashSet<u32> = HashSet::default();
        for &flat in &self.dirty {
            let (x, y, z) = self.unflatten(flat);
            to_fix.insert(flat);
            for (dx, dy, dz) in NEIGHBOR_OFFSETS {
                if let Some((nx, ny, nz)) = self.offset(x, y, z, dx, dy, dz) {
                    to_fix.insert(self.flatten(nx, ny, nz));
                }
            }
        }
        for flat in to_fix {
            let (x, y, z) = self.unflatten(flat);
            self.fixup_cell(x, y, z);
        }
        self.dirty.clear();
    }

    fn fixup_cell(&mut self, _x: u32, _y: u32, _z: u32) {}

    #[inline]
    fn offset(&self, x: u32, y: u32, z: u32, dx: i32, dy: i32, dz: i32) -> Option<(u32, u32, u32)> {
        let (nx, ny, nz) = (x as i32 + dx, y as i32 + dy, z as i32 + dz);
        if nx < 0 || ny < 0 || nz < 0 {
            return None;
        }
        let (nx, ny, nz) = (nx as u32, ny as u32, nz as u32);
        (nx < self.size && ny < self.size && nz < self.size).then_some((nx, ny, nz))
    }
}

const NEIGHBOR_OFFSETS: [(i32, i32, i32); 6] = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
];

impl Field<f32> for SDFField {
    fn size(&self) -> UVec3 {
        UVec3::splat(self.size)
    }

    fn get(&self, x: u32, y: u32, z: u32) -> f32 {
        let i = self.flatten(x, y, z);
        self.data[i as usize]
    }

    fn set(&mut self, x: u32, y: u32, z: u32, value: f32) {
        let i = self.flatten(x, y, z);
        if self.data[i as usize] != value {
            self.data[i as usize] = value;
            self.dirty.insert(i);
        }
    }
}
