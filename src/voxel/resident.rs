use bevy::prelude::*;
use std::collections::HashMap;

/// Frames a lease survives after its last touch. Hysteresis stops chunks
/// from thrashing between released and resident at the edge of reach.
const LEASE_FRAMES: u32 = 120;

#[derive(Resource, Default)]
pub struct ForceResident {
    frame: u32,
    until: HashMap<IVec3, u32>,
}

impl ForceResident {
    pub fn touch(&mut self, pos: IVec3) {
        self.until.insert(pos, self.frame + LEASE_FRAMES);
    }

    /// Lease every chunk overlapping the sphere (world space).
    pub fn touch_sphere(&mut self, center: Vec3, radius: f32) {
        for pos in crate::chunk::overlapping_chunks(center, Vec3::splat(radius)) {
            self.touch(pos);
        }
    }

    #[inline]
    pub fn contains(&self, pos: &IVec3) -> bool {
        self.until.get(pos).is_some_and(|&u| u > self.frame)
    }
}

pub fn tick_force_resident(mut r: ResMut<ForceResident>) {
    r.frame += 1;
    if r.frame % 64 == 0 {
        let f = r.frame;
        r.until.retain(|_, &mut u| u > f);
    }
}

/// How far from the camera hidden chunks stay editable/hittable (world units).
const HIDDEN_EDIT_REACH: f32 = 24.0;

/// Leases hidden chunks around the camera so the cursor ray can find them
/// before any edit happens. Without this you couldn't start the first edit.
pub fn lease_chunks_near_camera(
    cams: Query<&GlobalTransform, With<Camera3d>>,
    mut r: ResMut<ForceResident>,
) {
    for t in &cams {
        r.touch_sphere(t.translation(), HIDDEN_EDIT_REACH);
    }
}
