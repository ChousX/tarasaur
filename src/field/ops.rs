use super::{Field, *};
use bevy::math::primitives::{Cuboid, Sphere};

/// Trait to handle adding small amounts or deltas to generic field values safely.
pub trait AccumulateExt {
    fn accumulate(self, delta: Self) -> Self;
}

/// Trait to handle linear blending (interpolation/falloff) between generic field values.
pub trait BlendExt {
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
        self + (target - self) * factor.clamp(0.0, 1.0)
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
        let start = self as f32;
        let end = target as f32;
        (start + (end - start) * f).round() as u8
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

/// Extension trait for fields that support spherical operations.
pub trait FieldSphereOps<T: Copy + Default>: Field<T> {
    /// Applies an operation to all voxels within a sphere.
    ///
    /// # Arguments
    /// * `center` - Center of the sphere in grid coordinates
    /// * `shape` - The sphere primitive (radius) to apply
    /// * `op` - Operation to apply: receives (current_value, distance_from_center) and returns new value
    fn apply_sphere<F>(&mut self, center: Vec3, shape: Sphere, mut op: F)
    where
        F: FnMut(T, f32) -> T,
    {
        let radius = shape.radius;
        let size = self.size();
        let max_bound = size.as_vec3() - Vec3::ONE;

        let min = (center - Vec3::splat(radius + 1.0))
            .max(Vec3::ZERO)
            .as_ivec3();
        let max = (center + Vec3::splat(radius + 1.0))
            .min(max_bound)
            .as_ivec3();

        for z in min.z..=max.z {
            for y in min.y..=max.y {
                for x in min.x..=max.x {
                    let pos = Vec3::new(x as f32, y as f32, z as f32);
                    let dist = pos.distance(center);
                    if dist <= radius {
                        let current = self.get(x as u32, y as u32, z as u32);
                        let new_value = op(current, dist);
                        self.set(x as u32, y as u32, z as u32, new_value);
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
}

/// Extension trait for fields that support box operations.
pub trait FieldBoxOps<T: Copy + Default>: Field<T> {
    /// Applies an operation to all voxels within an axis-aligned box.
    fn apply_box<F>(&mut self, center: Vec3, shape: Cuboid, mut op: F)
    where
        F: FnMut(T) -> T,
    {
        let min_bound = (center - shape.half_size).as_ivec3();
        let max_bound = (center + shape.half_size).as_ivec3();

        let size = self.size();
        let limit = size.as_ivec3() - IVec3::ONE;
        let min = min_bound.max(IVec3::ZERO);
        let max = max_bound.min(limit);

        for z in min.z..=max.z {
            for y in min.y..=max.y {
                for x in min.x..=max.x {
                    let current = self.get(x as u32, y as u32, z as u32);
                    let new_value = op(current);
                    self.set(x as u32, y as u32, z as u32, new_value);
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
