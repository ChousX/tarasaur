// fields/material.rs
use super::{Field, consts::*};
use bevy::prelude::*;

#[derive(Component, Clone)]
pub struct MaterialField {
    data: Box<[u8; CHUNK_VOLUME]>,
}

impl Default for MaterialField {
    fn default() -> Self {
        Self {
            data: Box::new([0; CHUNK_VOLUME]),
        }
    }
}

impl Field<u8> for MaterialField {
    fn size(&self) -> UVec3 {
        UVec3::splat(CHUNK_SIZE)
    }

    fn get(&self, x: u32, y: u32, z: u32) -> u8 {
        self.data[flatten(x, y, z) as usize]
    }

    fn set(&mut self, x: u32, y: u32, z: u32, value: u8) {
        self.data[flatten(x, y, z) as usize] = value;
    }
}
