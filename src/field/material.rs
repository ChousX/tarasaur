use super::{Field, Lod};
use bevy::prelude::*;

#[derive(Component, Clone)]
pub struct MaterialField {
    pub size: u32,
    data: Box<[u8]>,
}

impl MaterialField {
    pub fn new(lod: Lod) -> Self {
        let size = lod.size();
        let volume = lod.volume();
        Self {
            size,
            data: vec![0; volume].into_boxed_slice(),
        }
    }
}

impl Field<u8> for MaterialField {
    fn size(&self) -> UVec3 {
        UVec3::splat(self.size)
    }

    fn get(&self, x: u32, y: u32, z: u32) -> u8 {
        let idx = (z * self.size * self.size + y * self.size + x) as usize;
        self.data[idx]
    }

    fn set(&mut self, x: u32, y: u32, z: u32, value: u8) {
        let idx = (z * self.size * self.size + y * self.size + x) as usize;
        self.data[idx] = value;
    }
}

