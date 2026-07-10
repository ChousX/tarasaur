use super::{Field, LOD};
use bevy::prelude::*;

#[derive(Component, Clone)]
pub struct MaterialField {
    lod: LOD,
    data: Box<[u8]>,
}

impl MaterialField {
    pub fn new(lod: LOD) -> Self {
        let volume = lod.volume();
        Self {
            lod,
            data: vec![0; volume].into_boxed_slice(),
        }
    }
}

impl Field<u8> for MaterialField {
    fn size(&self) -> UVec3 {
        UVec3::splat(self.lod as u32)
    }

    fn get(&self, x: u32, y: u32, z: u32) -> u8 {
        let size = self.lod.size();
        let idx = (z * size * size + y * size + x) as usize;
        self.data[idx]
    }

    fn set(&mut self, x: u32, y: u32, z: u32, value: u8) {
        let size = self.lod.size();
        let idx = (z * size * size + y * size + x) as usize;
        self.data[idx] = value;
    }
}
