// fields/sdf.rs
use super::{Field, consts::*};
use bevy::{platform::collections::HashSet, prelude::*};

#[derive(Component, Clone)]
pub struct SdfField {
    data: Box<[f32; CHUNK_VOLUME]>,
    dirty: HashSet<u32>,
}

impl Default for SdfField {
    fn default() -> Self {
        Self {
            data: Box::new([f32::MAX; CHUNK_VOLUME]),
            dirty: HashSet::default(),
        }
    }
}

impl SdfField {
    pub fn is_dirty(&self) -> bool {
        !self.dirty.is_empty()
    }

    pub fn reinit_if_dirty(&mut self) {
        if self.dirty.is_empty() {
            return;
        }
        let mut to_fix: HashSet<u32> = HashSet::default();
        for &flat in &self.dirty {
            let (x, y, z) = unflatten(flat);
            to_fix.insert(flat);
            for (dx, dy, dz) in NEIGHBOR_OFFSETS {
                if let Some((nx, ny, nz)) = offset(x, y, z, dx, dy, dz) {
                    to_fix.insert(flatten(nx, ny, nz));
                }
            }
        }
        for flat in to_fix {
            let (x, y, z) = unflatten(flat);
            self.fixup_cell(x, y, z);
        }
        self.dirty.clear();
    }

    fn fixup_cell(&mut self, _x: u32, _y: u32, _z: u32) {
        // local sign/distance fixup goes here
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

#[inline]
fn offset(x: u32, y: u32, z: u32, dx: i32, dy: i32, dz: i32) -> Option<(u32, u32, u32)> {
    let (nx, ny, nz) = (x as i32 + dx, y as i32 + dy, z as i32 + dz);
    if nx < 0 || ny < 0 || nz < 0 {
        return None;
    }
    let (nx, ny, nz) = (nx as u32, ny as u32, nz as u32);
    (nx < CHUNK_SIZE && ny < CHUNK_SIZE && nz < CHUNK_SIZE).then_some((nx, ny, nz))
}

impl Field<f32> for SdfField {
    fn size(&self) -> UVec3 {
        UVec3::splat(CHUNK_SIZE)
    }

    fn get(&self, x: u32, y: u32, z: u32) -> f32 {
        self.data[flatten(x, y, z) as usize]
    }

    fn set(&mut self, x: u32, y: u32, z: u32, value: f32) {
        let i = flatten(x, y, z);
        if self.data[i as usize] != value {
            self.data[i as usize] = value;
            self.dirty.insert(i);
        }
    }
}

