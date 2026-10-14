//! Leases that keep fully hidden chunks on the GPU near the camera and
//! edits, so rays can still hit (and edits can still reveal) them.

use bevy::prelude::*;
use std::collections::HashMap;

/// Frames a lease survives after its last touch. Hysteresis stops chunks
/// from thrashing between released and resident at the edge of reach.
const LEASE_FRAMES: u64 = 120;

#[derive(Resource, Default)]
pub struct ForceResident {
    frame: u64, // 64 bits: never wraps, so `until > frame` stays valid
    until: HashMap<IVec3, u64>,
    /// Positions whose lease started or ran out this frame. Extraction only
    /// revisits chunks that changed, so it needs these flips spelled out.
    changed: Vec<IVec3>,
}

impl ForceResident {
    pub fn touch(&mut self, pos: IVec3) {
        if !self.contains(&pos) {
            self.changed.push(pos);
        }
        self.until
            .insert(pos, self.frame.saturating_add(LEASE_FRAMES));
    }

    /// Lease every chunk overlapping the sphere (world space).
    pub fn touch_sphere(&mut self, center: Vec3, radius: f32) {
        for pos in crate::chunk::overlapping_chunks(center, Vec3::splat(radius)) {
            self.touch(pos);
        }
    }

    #[inline]
    #[must_use]
    pub fn contains(&self, pos: &IVec3) -> bool {
        self.until.get(pos).is_some_and(|&u| u > self.frame)
    }

    /// Positions whose lease started or ended this frame (may repeat).
    #[must_use]
    pub fn changed(&self) -> &[IVec3] {
        &self.changed
    }
}

pub fn tick_force_resident(mut r: ResMut<ForceResident>) {
    let r = &mut *r;
    r.frame = r.frame.saturating_add(1);
    let frame = r.frame;
    r.changed.clear();
    // Leases cover the few chunks near the camera and edits, so scanning
    // them every frame is cheap.
    r.changed.extend(
        r.until
            .iter()
            .filter(|&(_, &u)| u == frame)
            .map(|(&pos, _)| pos),
    );
    if frame.is_multiple_of(64) {
        r.until.retain(|_, &mut u| u > frame);
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
