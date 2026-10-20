//! The visibility field: signed distances to the edge of what is shown.
//!
//! Values are in voxels of the chunk's own LOD, negative where the terrain
//! is visible, clamped to `±SDF_BAND`, exactly like the `SDF`. The terrain
//! layer meshes `max(SDF, visibility)`: solid *and* visible. The hidden part
//! of the terrain is cut away and the cut is closed by a surface (the cap)
//! that follows this field's zero crossing, so hiding is smooth at
//! sub-voxel precision and never shows the hollow inside of the terrain.
//!
//! Edited like the `SDF`: `VisibilityCsgSphere` hides or reveals an exact
//! sphere or swept capsule, and `WorldEditor<VisibilityField, f32>` value
//! edits (any sign inside a box, say) are followed by the same jump-flood
//! reinit. A chunk without the field is fully visible.
//!
//! With the `visibility-mask` feature the terrain instead uploads it as a
//! per-voxel mask (hidden = distance > 0) and the fragment shader discards
//! hidden voxels: the older way, with hard voxel-aligned edges and no cap,
//! but no re-meshing when the visibility changes.

use super::{DistanceField, Field};
use crate::field::{GpuChannel, LOD, SDF, SDF_BAND, VoxelField, sdf::uniform_sign_volume};
use bevy::prelude::*;

/// Optional per-chunk signed distances to the visibility boundary, negative
/// where visible. Absent or all-negative = fully visible; all-positive =
/// chunk hidden; mixed = cut by the cap.
#[derive(Component, Clone)]
pub struct VisibilityField(SDF);

impl VisibilityField {
    /// Fully visible.
    #[must_use]
    pub fn new(lod: LOD) -> Self {
        Self::filled(lod, true)
    }

    /// Uniformly `visible` (or hidden), at the band's full distance so it
    /// leaves the terrain's own distances alone.
    #[must_use]
    pub fn filled(lod: LOD, visible: bool) -> Self {
        let v = if visible { -SDF_BAND } else { SDF_BAND };
        Self(SDF::from_raw(lod, vec![v; lod.volume()]))
    }

    /// `Some(true)` = every voxel visible, `Some(false)` = every voxel
    /// hidden, `None` = mixed. O(1).
    #[inline]
    #[must_use]
    pub fn uniform(&self) -> Option<bool> {
        self.0.uniform_sign()
    }

    #[inline]
    #[must_use]
    pub fn is_all_hidden(&self) -> bool {
        self.uniform() == Some(false)
    }

    #[inline]
    #[must_use]
    pub fn is_all_visible(&self) -> bool {
        self.uniform() == Some(true)
    }

    #[inline]
    #[must_use]
    pub fn is_mixed(&self) -> bool {
        self.uniform().is_none()
    }

    /// Whether the voxel is visible (its distance is `<= 0`, the same test
    /// as the mesher's inside test).
    #[inline]
    #[must_use]
    pub fn visible_at(&self, x: u32, y: u32, z: u32) -> bool {
        self.get(x, y, z) <= 0.0
    }
}

impl Default for VisibilityField {
    fn default() -> Self {
        Self::new(LOD::default())
    }
}

impl Field<f32> for VisibilityField {
    fn size(&self) -> UVec3 {
        self.0.size()
    }

    /// Out-of-range coordinates read as hidden (`SDF_BAND`).
    fn get(&self, x: u32, y: u32, z: u32) -> f32 {
        self.0.get(x, y, z)
    }

    fn set(&mut self, x: u32, y: u32, z: u32, value: f32) {
        self.0.set(x, y, z, value);
    }
}

impl VoxelField for VisibilityField {
    type Value = f32;
    type Elem = f32;
    /// Not uploaded: the terrain's extraction folds it into the shape.
    #[cfg(not(feature = "visibility-mask"))]
    const GPU: GpuChannel = GpuChannel::None;
    /// Uploaded as a mask: one byte per voxel, 0 where hidden.
    #[cfg(feature = "visibility-mask")]
    const GPU: GpuChannel = GpuChannel::Mask;
    const OPTIONAL: bool = true;
    /// Value edits are signs until the jump flood lands, as for `SDF`.
    const CLEAR_DIRTY_EACH_CYCLE: bool = false;

    fn new_for_lod(lod: LOD) -> Self {
        Self::new(lod)
    }

    fn from_raw(lod: LOD, data: Vec<f32>) -> Self {
        Self(SDF::from_raw(lod, data))
    }

    fn lod(&self) -> LOD {
        self.0.lod
    }

    fn data_slice(&self) -> &[f32] {
        self.0.data_slice()
    }

    fn version(&self) -> u64 {
        self.0.version
    }

    fn border_version(&self, neighbor: usize) -> u64 {
        self.0.border_version(neighbor)
    }

    /// A mask needs uploading only where something is hidden; an
    /// all-visible chunk sends nothing (which clears the slot's mask).
    #[cfg(feature = "visibility-mask")]
    fn should_extract(&self) -> bool {
        !self.is_all_visible()
    }

    /// The mask bytes: 1 where visible (`<= 0`), 0 where hidden.
    #[cfg(feature = "visibility-mask")]
    fn gpu_bytes(vol: &[f32]) -> Vec<u8> {
        vol.iter().map(|&v| u8::from(v <= 0.0)).collect()
    }

    fn rescale(v: f32, ratio: f32) -> f32 {
        SDF::rescale(v, ratio)
    }

    fn sample_apron(data: &[f32], size: u32, x: f32, y: f32, z: f32) -> f32 {
        SDF::sample_apron(data, size, x, y, z)
    }

    fn extend_edge(edge: f32, inward: [f32; 3], past: UVec3) -> f32 {
        SDF::extend_edge(edge, inward, past)
    }
}

impl DistanceField for VisibilityField {
    fn reinit_input(&self) -> (Box<[f32]>, u32, u64) {
        self.0.reinit_input()
    }

    /// A uniform result is pinned at the band's full distance: the jump
    /// flood gives uniform volumes their distance to the chunk's faces
    /// (so the `SDF` stays editable there), which here would pull the
    /// terrain's distances toward zero near every chunk face.
    fn apply_reinit_result(&mut self, mut data: Box<[f32]>) {
        if let Some(visible) = uniform_sign_volume(&data) {
            data.fill(if visible { -SDF_BAND } else { SDF_BAND });
        }
        self.0.apply_reinit_result(data);
    }
}

#[cfg(test)]
mod tests {
    use super::VisibilityField;
    use crate::field::{DistanceField, Field, LOD, SDF_BAND, VoxelField, ops::FieldSphereOps};
    use bevy::math::Vec3;

    #[test]
    fn starts_fully_visible_at_full_depth() {
        let f = VisibilityField::new(LOD::Lowest);
        assert!(f.is_all_visible());
        assert!(f.data_slice().iter().all(|&v| v == -SDF_BAND));
        assert!(VisibilityField::filled(LOD::Lowest, false).is_all_hidden());
    }

    /// Hiding a sphere is a CSG subtraction from the visible region: the
    /// sphere's inside goes positive with its exact distance, the rest
    /// stays at the band's depth.
    #[test]
    fn hide_sphere_is_a_subtraction() {
        let mut f = VisibilityField::new(LOD::Low);
        f.csg_sphere(Vec3::splat(8.0), 3.0, 4.0, true, None, |_, _, _| None);
        assert!(f.is_mixed());
        assert!(!f.visible_at(8, 8, 8));
        assert!((f.get(8, 8, 8) - 3.0).abs() < 1e-5); // 3 voxels inside the sphere
        assert!((f.get(8, 8, 12) - -1.0).abs() < 1e-5); // 1 voxel outside it
        assert!(f.visible_at(0, 0, 0));
        assert_eq!(f.get(0, 0, 0), -SDF_BAND);

        // Revealing it again is the union with the same sphere.
        f.csg_sphere(Vec3::splat(8.0), 3.0, 4.0, false, None, |_, _, _| None);
        assert!(f.visible_at(8, 8, 8));
    }

    /// A reinit of a uniform field lands at the band's depth, not at the
    /// distance to the chunk's faces.
    #[test]
    fn uniform_reinit_stays_at_full_depth() {
        let mut f = VisibilityField::new(LOD::Lowest);
        let (mut data, size, _) = f.reinit_input();
        crate::field::sdf::compute_sdf_distances(&mut data, size);
        assert!(data.iter().any(|&v| v > -SDF_BAND)); // what the SDF would keep
        f.apply_reinit_result(data);
        assert!(f.data_slice().iter().all(|&v| v == -SDF_BAND));
    }
}
