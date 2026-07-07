use bevy::prelude::*;
mod consts;
mod editor;
mod material;
pub mod ops;
mod sdf;
mod visibility;

/// Core trait representing a 3D grid of data.
pub trait Field<T: Copy + Default> {
    /// Returns the dimensions of this specific field.
    fn size(&self) -> UVec3;
    /// Gets the value at the given grid coordinates.
    fn get(&self, x: u32, y: u32, z: u32) -> T;
    /// Sets the value at the given grid coordinates.
    fn set(&mut self, x: u32, y: u32, z: u32, value: T);
}

// =========================================================================
// Tuple Implementations for Multi-Field Operations
// =========================================================================

/// Extension implementation allowing any two fields to be driven simultaneously.
impl<'a, A, B, TA, TB> Field<(TA, TB)> for (&'a mut A, &'a mut B)
where
    A: Field<TA>,
    B: Field<TB>,
    TA: Copy + Default,
    TB: Copy + Default,
{
    #[inline]
    fn size(&self) -> UVec3 {
        // Assumes matching sizes between paired layers
        self.0.size()
    }

    #[inline]
    fn get(&self, x: u32, y: u32, z: u32) -> (TA, TB) {
        (self.0.get(x, y, z), self.1.get(x, y, z))
    }

    #[inline]
    fn set(&mut self, x: u32, y: u32, z: u32, value: (TA, TB)) {
        self.0.set(x, y, z, value.0);
        self.1.set(x, y, z, value.1);
    }
}

/// Extension implementation allowing three fields to be driven simultaneously.
impl<'a, A, B, C, TA, TB, TC> Field<(TA, TB, TC)> for (&'a mut A, &'a mut B, &'a mut C)
where
    A: Field<TA>,
    B: Field<TB>,
    C: Field<TC>,
    TA: Copy + Default,
    TB: Copy + Default,
    TC: Copy + Default,
{
    #[inline]
    fn size(&self) -> UVec3 {
        self.0.size()
    }

    #[inline]
    fn get(&self, x: u32, y: u32, z: u32) -> (TA, TB, TC) {
        (
            self.0.get(x, y, z),
            self.1.get(x, y, z),
            self.2.get(x, y, z),
        )
    }

    #[inline]
    fn set(&mut self, x: u32, y: u32, z: u32, value: (TA, TB, TC)) {
        self.0.set(x, y, z, value.0);
        self.1.set(x, y, z, value.1);
        self.2.set(x, y, z, value.2);
    }
}
