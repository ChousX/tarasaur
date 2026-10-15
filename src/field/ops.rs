//! Brush math in grid space: per-value-type edit semantics
//! (`AccumulateExt`, `BlendExt`) and sphere/box ops blanket-implemented for
//! every `Field`.

use super::{Field, IVec3, UVec3, Vec3};
use crate::num::u8_from_f32;

use bevy::shape::{Cuboid, Sphere};

/// Trait to handle adding small amounts or deltas to generic field values safely.
pub trait AccumulateExt {
    #[must_use]
    fn accumulate(self, delta: Self) -> Self;
}

/// Trait to handle linear blending (interpolation/falloff) between generic field values.
pub trait BlendExt {
    #[must_use]
    fn blend_towards(self, target: Self, factor: f32) -> Self;
}

// Concrete implementation for f32 (SDF fields)
impl AccumulateExt for f32 {
    #[inline]
    fn accumulate(self, delta: Self) -> Self {
        // Keeps values in a sane SDF space if needed, or simple addition
        self + delta
    }
}

impl BlendExt for f32 {
    #[inline]
    fn blend_towards(self, target: Self, factor: f32) -> Self {
        // Standard linear interpolation (lerp)
        (target - self).mul_add(factor.clamp(0.0, 1.0), self)
    }
}

// Concrete implementation for u8 (Material / Weight fields)
impl AccumulateExt for u8 {
    #[inline]
    fn accumulate(self, delta: Self) -> Self {
        self.saturating_add(delta)
    }
}

impl BlendExt for u8 {
    #[inline]
    fn blend_towards(self, target: Self, factor: f32) -> Self {
        let f = factor.clamp(0.0, 1.0);
        let start = f32::from(self);
        let end = f32::from(target);
        u8_from_f32((end - start).mul_add(f, start).round())
    }
}

// Concrete implementation for bool (Visibility fields)
impl AccumulateExt for bool {
    #[inline]
    fn accumulate(self, delta: Self) -> Self {
        self | delta // Binary addition/accumulation acts like an OR switch
    }
}

impl BlendExt for bool {
    #[inline]
    fn blend_towards(self, target: Self, factor: f32) -> Self {
        // If brush influence passes a threshold (e.g. 50%), flip the boolean state
        if factor >= 0.5 { target } else { self }
    }
}

/// Lets a field type be written directly from a signed-distance value —
/// only meaningful for actual distance fields (f32), unlike `AccumulateExt`/
/// `BlendExt` which apply to every field type.
pub trait FromSignedDistance {
    fn from_signed_distance(distance: f32) -> Self;
}

impl FromSignedDistance for f32 {
    #[inline]
    fn from_signed_distance(distance: f32) -> Self {
        distance
    }
}

/// An inclusive voxel range as unsigned coordinates, or `None` if it is
/// empty or reaches below zero on any axis.
fn grid_range(min: IVec3, max: IVec3) -> Option<(UVec3, UVec3)> {
    if min.cmplt(IVec3::ZERO).any() || max.cmplt(min).any() {
        return None;
    }
    Some((min.as_uvec3(), max.as_uvec3()))
}

/// The voxels a sphere (grid space) can touch, one voxel of margin
/// included, clamped to a grid of `size`.
fn sphere_range(center: Vec3, radius: f32, size: UVec3) -> Option<(UVec3, UVec3)> {
    let max_bound = size.as_vec3() - Vec3::ONE;
    let min = (center - Vec3::splat(radius + 1.0))
        .max(Vec3::ZERO)
        .as_ivec3();
    let max = (center + Vec3::splat(radius + 1.0))
        .min(max_bound)
        .as_ivec3();
    grid_range(min, max)
}

/// The slowest a speed-limited CSG edit moves (as a fraction of its step),
/// at its rim, so the rim still reaches the sphere.
const MIN_STEP_WEIGHT: f32 = 0.15;

/// Brush weight: 1 at the center, easing smoothly to 0 at `radius`.
#[inline]
fn brush_falloff(dist: f32, radius: f32) -> f32 {
    if radius <= 0.0 {
        return 1.0;
    }
    let t = (dist / radius).clamp(0.0, 1.0);
    let k = t.mul_add(-t, 1.0);
    k * k
}

/// Extension trait for fields that support spherical operations.
pub trait FieldSphereOps<T: Copy + Default>: Field<T> {
    /// Applies an operation to all voxels within a sphere.
    ///
    /// # Arguments
    /// * `center` - Center of the sphere in grid coordinates
    /// * `shape` - The sphere primitive (radius) to apply
    /// * `op` - Operation to apply: receives (`current_value`, `distance_from_center`) and returns new value
    fn apply_sphere<F>(&mut self, center: Vec3, shape: Sphere, mut op: F)
    where
        F: FnMut(T, f32) -> T,
    {
        self.apply_sphere_at(center, shape.radius, |current, _, dist| op(current, dist));
    }

    /// Like `apply_sphere`, but `op` also gets the voxel's grid position:
    /// (`current_value`, `position`, `distance_from_center`).
    fn apply_sphere_at<F>(&mut self, center: Vec3, radius: f32, mut op: F)
    where
        F: FnMut(T, Vec3, f32) -> T,
    {
        let Some((min, max)) = sphere_range(center, radius, self.size()) else {
            return;
        };
        for z in min.z..=max.z {
            for y in min.y..=max.y {
                for x in min.x..=max.x {
                    let pos = UVec3::new(x, y, z).as_vec3();
                    let dist = pos.distance(center);
                    if dist <= radius {
                        let current = self.get(x, y, z);
                        let new_value = op(current, pos, dist);
                        self.set(x, y, z, new_value);
                    }
                }
            }
        }
    }

    /// Fills a sphere with a constant value.
    fn fill_sphere(&mut self, center: Vec3, shape: Sphere, value: T) {
        self.apply_sphere(center, shape, |_, _| value);
    }

    /// Gradually adds an amount to voxels inside a sphere.
    fn accumulate_sphere(&mut self, center: Vec3, shape: Sphere, delta: T)
    where
        T: AccumulateExt,
    {
        self.apply_sphere(center, shape, |current, _| current.accumulate(delta));
    }

    /// Blends values based on how close they are to the center (Falloff/Soft brush).
    fn blend_sphere(&mut self, center: Vec3, shape: Sphere, target_val: T, rate: f32)
    where
        T: BlendExt,
    {
        let radius = shape.radius;
        self.apply_sphere(center, shape, |current, dist| {
            // Linear falloff: 1.0 at center, 0.0 at edge
            let falloff = if radius > 0.0 {
                1.0 - (dist / radius)
            } else {
                1.0
            };
            let factor = falloff * rate;
            current.blend_towards(target_val, factor)
        });
    }
    /// Writes the exact analytic signed distance to this sphere's surface
    /// for every voxel in range — negative inside, positive outside, a
    /// real continuous gradient everywhere. `apply_sphere` already hands
    /// us `dist` (grid-space distance from center), so the sphere SDF
    /// formula is just `dist - radius`: no flat sentinel, no seam, no
    /// waiting on JFA to fix a discontinuity that a constant stamp created
    /// in the first place.
    fn stamp_sdf_sphere(&mut self, center: Vec3, bound_radius: f32, sdf_radius: f32)
    where
        T: FromSignedDistance,
    {
        self.apply_sphere(
            center,
            Sphere {
                radius: bound_radius,
            },
            |_, dist| T::from_signed_distance(dist - sdf_radius),
        );
    }

    /// CSG against a sphere. Subtract = max(old, -s), add = min(old, s),
    /// with s = dist - radius. Keeps sub-voxel detail, needs no JFA, and is
    /// seamless across chunks because it depends only on world position.
    /// `reach` (voxels, >= 2) extends the write past the surface.
    ///
    /// `max_step` (voxels) limits how far each value moves toward that
    /// result per call: all of it at the center, fading to
    /// `MIN_STEP_WEIGHT` of it at the rim and in the reach band. `None` is
    /// the exact result at once.
    ///
    /// Returns the voxels this call turned from outside to inside.
    fn csg_sphere(
        &mut self,
        center: Vec3,
        radius: f32,
        reach: f32,
        subtract: bool,
        max_step: Option<f32>,
    ) -> Vec<UVec3>
    where
        T: SignedDistance,
    {
        let mut filled = Vec::new();
        self.apply_sphere_at(center, radius + reach, |cur, pos, dist| {
            let s = dist - radius;
            let c = cur.distance();
            let target = if subtract { c.max(-s) } else { c.min(s) };
            let new = max_step.map_or(target, |step| {
                let step = step * brush_falloff(dist, radius).max(MIN_STEP_WEIGHT);
                // Toward the target, never past it.
                c + (target - c).clamp(-step, step)
            });
            if c > 0.0 && new <= 0.0 {
                filled.push(pos.as_uvec3());
            }
            T::from_signed_distance(new)
        });
        filled
    }

    /// `voxels` plus their face neighbours that are outside (air): what a
    /// fragment on the surface around `voxels` blends its material from.
    fn with_air_neighbours(&self, voxels: &[UVec3]) -> Vec<UVec3>
    where
        T: SignedDistance,
    {
        let size = self.size();
        let mut out = Vec::with_capacity(voxels.len().saturating_mul(3));
        for &v in voxels {
            out.push(v);
            for d in [
                IVec3::X,
                IVec3::NEG_X,
                IVec3::Y,
                IVec3::NEG_Y,
                IVec3::Z,
                IVec3::NEG_Z,
            ] {
                let Some(n) = v.checked_add_signed(d).filter(|n| n.cmplt(size).all()) else {
                    continue;
                };
                if self.get(n.x, n.y, n.z).distance() > 0.0 {
                    out.push(n);
                }
            }
        }
        out.sort_unstable_by_key(|v| (v.z, v.y, v.x));
        out.dedup();
        out
    }

    /// Relaxes the distances inside the sphere toward their 6-neighbour
    /// average, which rounds off bumps and creases. `amount` (0..=1) is how
    /// far each voxel moves at the center, fading to nothing at the edge.
    ///
    /// Reads every neighbour before writing any voxel, so the result doesn't
    /// depend on iteration order. An axis whose neighbour lies outside the
    /// chunk is left as is (as if the field continued linearly), so linear
    /// fields — flat ground — never move, and chunk borders stay put.
    fn smooth_sphere(&mut self, center: Vec3, radius: f32, amount: f32)
    where
        T: SignedDistance,
    {
        let size = self.size();
        let Some((min, max)) = sphere_range(center, radius, size) else {
            return;
        };
        let read = |x: u32, y: u32, z: u32| self.get(x, y, z).distance();
        // Two neighbours along one axis, or twice the center if either is
        // missing (a linear continuation adds no curvature there).
        let pair = |c: f32, lo: Option<f32>, hi: Option<f32>| match (lo, hi) {
            (Some(a), Some(b)) => a + b,
            _ => 2.0 * c,
        };
        let below = |i: u32| i.checked_sub(1);
        let above = |i: u32, n: u32| i.checked_add(1).filter(|&j| j < n);

        let mut out = Vec::new();
        for z in min.z..=max.z {
            for y in min.y..=max.y {
                for x in min.x..=max.x {
                    let dist = UVec3::new(x, y, z).as_vec3().distance(center);
                    if dist > radius {
                        continue;
                    }
                    let c = read(x, y, z);
                    let sum = pair(
                        c,
                        below(x).map(|i| read(i, y, z)),
                        above(x, size.x).map(|i| read(i, y, z)),
                    ) + pair(
                        c,
                        below(y).map(|i| read(x, i, z)),
                        above(y, size.y).map(|i| read(x, i, z)),
                    ) + pair(
                        c,
                        below(z).map(|i| read(x, y, i)),
                        above(z, size.z).map(|i| read(x, y, i)),
                    );
                    let t = (amount * brush_falloff(dist, radius)).clamp(0.0, 1.0);
                    out.push((x, y, z, (sum / 6.0 - c).mul_add(t, c)));
                }
            }
        }
        for (x, y, z, v) in out {
            self.set(x, y, z, T::from_signed_distance(v));
        }
    }

    /// Pulls the distances inside the sphere toward those of a plane
    /// through `plane_point` with unit `plane_normal` (grid space, normal
    /// pointing out of the terrain): fills dips below it and shaves bumps
    /// above it. `amount` (0..=1) as in `smooth_sphere`. Depends only on
    /// position and the current value, so it's seamless across chunks.
    fn flatten_sphere(
        &mut self,
        center: Vec3,
        radius: f32,
        plane_point: Vec3,
        plane_normal: Vec3,
        amount: f32,
    ) where
        T: SignedDistance,
    {
        self.apply_sphere_at(center, radius, |cur, pos, dist| {
            let c = cur.distance();
            let plane = (pos - plane_point).dot(plane_normal);
            let t = (amount * brush_falloff(dist, radius)).clamp(0.0, 1.0);
            T::from_signed_distance((plane - c).mul_add(t, c))
        });
    }
}

/// Extension trait for fields that support box operations.
pub trait FieldBoxOps<T: Copy + Default>: Field<T> {
    /// Applies an operation to all voxels within an axis-aligned box.
    fn apply_box<F>(&mut self, center: Vec3, shape: Cuboid, mut op: F)
    where
        F: FnMut(T) -> T,
    {
        // Only voxels inside the box: round the min corner up, the max down.
        let min_bound = (center - shape.half_size).ceil().as_ivec3();
        let max_bound = (center + shape.half_size).floor().as_ivec3();

        let size = self.size();
        let limit = size.as_ivec3().saturating_sub(IVec3::ONE);
        let Some((min, max)) = grid_range(min_bound.max(IVec3::ZERO), max_bound.min(limit)) else {
            return;
        };

        for z in min.z..=max.z {
            for y in min.y..=max.y {
                for x in min.x..=max.x {
                    let current = self.get(x, y, z);
                    let new_value = op(current);
                    self.set(x, y, z, new_value);
                }
            }
        }
    }

    /// Fills a box with a constant value.
    fn fill_box(&mut self, center: Vec3, shape: Cuboid, value: T) {
        self.apply_box(center, shape, |_| value);
    }

    /// Gradually adds an amount to voxels inside a box.
    fn accumulate_box(&mut self, center: Vec3, shape: Cuboid, delta: T)
    where
        T: AccumulateExt,
    {
        self.apply_box(center, shape, |current| current.accumulate(delta));
    }

    /// Blends values uniformly across a box (e.g. constant rate application).
    fn blend_box(&mut self, center: Vec3, shape: Cuboid, target_val: T, rate: f32)
    where
        T: BlendExt,
    {
        self.apply_box(center, shape, |current| {
            current.blend_towards(target_val, rate)
        });
    }
}

// Blanket implementations for all Field types, accounting for dynamically sized types (?Sized)
impl<T: Copy + Default, F: Field<T> + ?Sized> FieldSphereOps<T> for F {}
impl<T: Copy + Default, F: Field<T> + ?Sized> FieldBoxOps<T> for F {}

/// Gives the world-space half-extent (bounding-box radius) of a brush shape.
///
/// Used to figure out which chunks a world-space edit could touch, without
/// the caller having to know how the shape is represented.
pub trait ShapeBounds {
    fn half_extent(&self) -> Vec3;
}

impl ShapeBounds for Sphere {
    #[inline]
    fn half_extent(&self) -> Vec3 {
        Vec3::splat(self.radius)
    }
}

impl ShapeBounds for Cuboid {
    #[inline]
    fn half_extent(&self) -> Vec3 {
        self.half_size
    }
}

/// Produces a copy of this shape scaled by `factor`.
///
/// Used to convert a world-space brush shape into voxel/grid space before it reaches
/// `FieldSphereOps`/`FieldBoxOps`, which only ever operate on raw grid
/// indices and know nothing about world units.
pub trait ShapeScale: Sized {
    #[must_use]
    fn scaled_by(&self, factor: f32) -> Self;
}

impl ShapeScale for Sphere {
    #[inline]
    fn scaled_by(&self, factor: f32) -> Self {
        Self {
            radius: self.radius * factor,
        }
    }
}

impl ShapeScale for Cuboid {
    #[inline]
    fn scaled_by(&self, factor: f32) -> Self {
        Self {
            half_size: self.half_size * factor,
        }
    }
}

pub trait ShapeEditOps<T: Copy + Default, S>: Field<T> {
    fn fill(&mut self, center: Vec3, shape: S, value: T);
    fn accumulate(&mut self, center: Vec3, shape: S, delta: T)
    where
        T: AccumulateExt;
    fn blend(&mut self, center: Vec3, shape: S, target: T, rate: f32)
    where
        T: BlendExt;
}

impl<T: Copy + Default, F: Field<T> + ?Sized> ShapeEditOps<T, Sphere> for F {
    fn fill(&mut self, c: Vec3, s: Sphere, v: T) {
        self.fill_sphere(c, s, v);
    }
    fn accumulate(&mut self, c: Vec3, s: Sphere, d: T)
    where
        T: AccumulateExt,
    {
        self.accumulate_sphere(c, s, d);
    }
    fn blend(&mut self, c: Vec3, s: Sphere, t: T, r: f32)
    where
        T: BlendExt,
    {
        self.blend_sphere(c, s, t, r);
    }
}
impl<T: Copy + Default, F: Field<T> + ?Sized> ShapeEditOps<T, Cuboid> for F {
    fn fill(&mut self, c: Vec3, s: Cuboid, v: T) {
        self.fill_box(c, s, v);
    }
    fn accumulate(&mut self, c: Vec3, s: Cuboid, d: T)
    where
        T: AccumulateExt,
    {
        self.accumulate_box(c, s, d);
    }
    fn blend(&mut self, c: Vec3, s: Cuboid, t: T, r: f32)
    where
        T: BlendExt,
    {
        self.blend_box(c, s, t, r);
    }
}

pub trait SignedDistance: FromSignedDistance {
    fn distance(self) -> f32;
}
impl SignedDistance for f32 {
    #[inline]
    fn distance(self) -> f32 {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::{Cuboid, FieldBoxOps, FieldSphereOps, UVec3, Vec3};
    use crate::{
        field::Field,
        field::{LOD, SDF, VisibilityField},
        num::f32_from_u32,
    };

    /// An SDF holding `f(x, y, z)` at every voxel.
    fn sdf_from(f: impl Fn(f32, f32, f32) -> f32) -> SDF {
        let mut sdf = SDF::new(LOD::Lowest);
        let n = sdf.size().x;
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let p = (f32_from_u32(x), f32_from_u32(y), f32_from_u32(z));
                    sdf.set(x, y, z, f(p.0, p.1, p.2));
                }
            }
        }
        sdf
    }

    #[test]
    fn smoothing_leaves_flat_ground_alone() {
        // A tilted plane, through the whole grid, borders included.
        let plane = |x: f32, y: f32, z: f32| 0.2f32.mul_add(x, 0.1f32.mul_add(-z, y) - 2.0);
        let mut sdf = sdf_from(plane);
        let n = f32_from_u32(sdf.size().x);
        sdf.smooth_sphere(Vec3::splat(n / 2.0), n, 1.0);
        for (x, y, z) in [(0, 0, 0), (1, 2, 1), (2, 2, 2), (3, 1, 0)] {
            let want = plane(f32_from_u32(x), f32_from_u32(y), f32_from_u32(z));
            assert!(
                (sdf.get(x, y, z) - want).abs() < 1e-5,
                "({x}, {y}, {z}) moved"
            );
        }
    }

    #[test]
    fn smoothing_flattens_a_spike() {
        let mut sdf = sdf_from(|_, y, _| y - 1.5);
        sdf.set(2, 2, 2, -3.0); // a lone solid voxel above ground
        sdf.smooth_sphere(Vec3::splat(2.0), 2.0, 1.0);
        assert!(sdf.get(2, 2, 2) > -1.0, "spike kept {}", sdf.get(2, 2, 2));
    }

    #[test]
    fn limited_dig_moves_gradually_then_matches_exact_csg() {
        let ground = |_: f32, y: f32, _: f32| y - 2.5;
        let center = Vec3::splat(2.0); // on voxel (2, 2, 2): full step there
        let mut exact = sdf_from(ground);
        exact.csg_sphere(center, 1.5, 2.0, true, None);

        // One small step: the center voxel moves up from -0.5 by exactly 0.25.
        let mut slow = sdf_from(ground);
        slow.csg_sphere(center, 1.5, 2.0, true, Some(0.25));
        assert!((slow.get(2, 2, 2) - -0.25).abs() < 1e-5);

        // Held long enough, it lands on the exact result and stays there.
        for _ in 0..200 {
            slow.csg_sphere(center, 1.5, 2.0, true, Some(0.25));
        }
        for (x, y, z) in [(2, 2, 2), (2, 1, 2), (1, 2, 2), (3, 3, 3), (0, 0, 0)] {
            let (got, want) = (slow.get(x, y, z), exact.get(x, y, z));
            assert!(
                (got - want).abs() < 1e-4,
                "({x}, {y}, {z}): {got} vs {want}"
            );
        }
    }

    #[test]
    fn build_reports_only_the_voxels_it_fills() {
        let ground = |_: f32, y: f32, _: f32| y - 1.5; // y <= 1 solid
        let mut sdf = sdf_from(ground);
        // A small step: only air right at the sphere's core can cross.
        let filled = sdf.csg_sphere(Vec3::new(2.0, 2.0, 2.0), 1.2, 2.0, false, Some(0.6));
        assert_eq!(filled, vec![UVec3::new(2, 2, 2)]);

        // Paint covers it and the air around it, never the ground below.
        let painted = sdf.with_air_neighbours(&filled);
        assert!(painted.contains(&UVec3::new(2, 3, 2)));
        assert!(painted.contains(&UVec3::new(1, 2, 2)));
        assert!(!painted.contains(&UVec3::new(2, 1, 2)));
        assert_eq!(painted.len(), 6);
    }

    #[test]
    fn flatten_pulls_toward_the_plane() {
        // Ground at y = 1.5; flatten toward y = 2.5 at full strength.
        let mut sdf = sdf_from(|_, y, _| y - 1.5);
        let center = Vec3::new(2.0, 2.0, 2.0);
        sdf.flatten_sphere(center, 1.5, Vec3::new(0.0, 2.5, 0.0), Vec3::Y, 1.0);
        // The center gets the plane's distance exactly...
        assert!((sdf.get(2, 2, 2) - -0.5).abs() < 1e-5);
        // ...voxels outside the sphere keep the ground's.
        assert!((sdf.get(0, 2, 0) - 0.5).abs() < 1e-5);
    }

    fn hidden_count(field: &VisibilityField) -> usize {
        let n = field.size().x;
        (0..n)
            .flat_map(|z| (0..n).flat_map(move |y| (0..n).map(move |x| (x, y, z))))
            .filter(|&(x, y, z)| !field.get(x, y, z))
            .count()
    }

    #[test]
    fn box_edits_only_voxels_inside_it() {
        let mut field = VisibilityField::new(LOD::Lowest);
        // Spans x, y, z in [1.6, 2.4]: only voxel (2, 2, 2) is inside.
        let shape = Cuboid {
            half_size: Vec3::splat(0.4),
        };
        field.fill_box(Vec3::splat(2.0), shape, false);
        assert_eq!(hidden_count(&field), 1);
        assert!(!field.get(2, 2, 2));

        // Spans [2.1, 2.9]: holds no voxel at all.
        let mut field = VisibilityField::new(LOD::Lowest);
        field.fill_box(Vec3::splat(2.5), shape, false);
        assert_eq!(hidden_count(&field), 0);

        // Straddles the grid's low edge: [-0.9, -0.1] is outside the grid.
        let mut field = VisibilityField::new(LOD::Lowest);
        field.fill_box(Vec3::splat(-0.5), shape, false);
        assert_eq!(hidden_count(&field), 0);
    }
}
