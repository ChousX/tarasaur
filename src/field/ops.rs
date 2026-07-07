use super::{Field, *};

/// Extension trait for fields that support spherical operations.
pub trait FieldSphereOps<T: Copy + Default>: Field<T> {
    /// Applies an operation to all voxels within a sphere.
    ///
    /// # Arguments
    /// * `center` - Center of the sphere in grid coordinates
    /// * `radius` - Radius in grid units
    /// * `op` - Operation to apply: receives (current_value, distance_from_center) and returns new value
    fn apply_sphere<F>(&mut self, center: Vec3, radius: f32, mut op: F)
    where
        F: FnMut(T, f32) -> T,
    {
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
                    let pos = vec3(x as f32, y as f32, z as f32);
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
    fn fill_sphere(&mut self, center: Vec3, radius: f32, value: T) {
        self.apply_sphere(center, radius, |_, _| value);
    }
}

/// Extension trait for fields that support box operations.
pub trait FieldBoxOps<T: Copy + Default>: Field<T> {
    /// Applies an operation to all voxels within an axis-aligned box.
    ///
    /// # Arguments
    /// * `min_bound` - Minimum corner (inclusive)
    /// * `max_bound` - Maximum corner (inclusive)
    /// * `op` - Operation to apply: receives current value and returns new value
    fn apply_box<F>(&mut self, min_bound: IVec3, max_bound: IVec3, mut op: F)
    where
        F: FnMut(T) -> T,
    {
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
    fn fill_box(&mut self, min: IVec3, max: IVec3, value: T) {
        self.apply_box(min, max, |_| value);
    }
}

// Blanket implementations for all Field types, accounting for dynamically sized types (?Sized)
impl<T: Copy + Default, F: Field<T> + ?Sized> FieldSphereOps<T> for F {}
impl<T: Copy + Default, F: Field<T> + ?Sized> FieldBoxOps<T> for F {}

