use super::Field;
use bevy::prelude::*;

/// A coordinator that wraps a primary field and handles boundary-safe mutations
/// across chunk seams automatically.
pub struct FieldEditor<'a, T: Copy + Default, F: Field<T>> {
    /// The size of a single chunk (e.g., 32x32x32)
    pub chunk_size: UVec3,
    /// The absolute chunk coordinate of the primary chunk being edited
    pub center_chunk_pos: IVec3,
    /// A 3x3x3 neighborhood of mutable chunk references.
    /// Index 13 is the center chunk itself.
    pub neighborhood: [Option<&'a mut F>; 27],
}

impl<'a, T: Copy + Default, F: Field<T>> FieldEditor<'a, T, F> {
    /// Creates an editor for a specific chunk given its 3x3x3 local neighborhood references.
    pub fn new(
        chunk_size: UVec3,
        center_chunk_pos: IVec3,
        neighborhood: [Option<&'a mut F>; 27],
    ) -> Self {
        Self {
            chunk_size,
            center_chunk_pos,
            neighborhood,
        }
    }

    /// Helper to convert a relative chunk offset (-1 to 1) to the flat 27-array index.
    #[inline]
    fn offset_to_idx(offset: IVec3) -> usize {
        ((offset.z + 1) * 9 + (offset.y + 1) * 3 + (offset.x + 1)) as usize
    }

    /// Resolves a world-relative voxel coordinate to the correct neighborhood chunk and its local coordinates.
    ///
    /// # Arguments
    /// * `local_x`, `local_y`, `local_z` - Coordinates relative to the *center* chunk's origin.
    ///   Can be negative or greater than `chunk_size` due to brush bleeding.
    #[inline]
    fn resolve_mut(&mut self, x: i32, y: i32, z: i32) -> Option<(&mut F, u32, u32, u32)> {
        let size = self.chunk_size.as_ivec3();

        // Calculate which neighbor chunk offset this coordinate lands in
        let chunk_offset = IVec3::new(
            x.div_euclid(size.x),
            y.div_euclid(size.y),
            z.div_euclid(size.z),
        );

        // Bounds check: Make sure the edit hasn't bled past our 3x3x3 neighborhood
        if chunk_offset.x.abs() > 1 || chunk_offset.y.abs() > 1 || chunk_offset.z.abs() > 1 {
            return None;
        }

        let idx = Self::offset_to_idx(chunk_offset);

        if let Some(Some(chunk)) = self.neighborhood.get_mut(idx) {
            // Remap coordinate to be positive and local to that specific neighbor chunk
            let remap_x = x.rem_euclid(size.x) as u32;
            let remap_y = y.rem_euclid(size.y) as u32;
            let remap_z = z.rem_euclid(size.z) as u32;
            Some((chunk, remap_x, remap_y, remap_z))
        } else {
            None
        }
    }
}

impl<'a, T: Copy + Default, F: Field<T>> Field<T> for FieldEditor<'a, T, F> {
    #[inline]
    fn size(&self) -> UVec3 {
        // Expose a virtual size spanning the entire 3x3x3 neighborhood
        self.chunk_size * 3
    }

    fn get(&self, x: u32, y: u32, z: u32) -> T {
        // Shift input virtual coordinates (0..size*3) so that (size, size, size) is the center chunk origin
        let size = self.chunk_size.as_ivec3();
        let rx = x as i32 - size.x;
        let ry = y as i32 - size.y;
        let rz = z as i32 - size.z;

        // Implementation shortcut: cast to mut to re-use resolution logic safely
        let mut_self = unsafe { &mut *(self as *const Self as *mut Self) };
        if let Some((chunk, lx, ly, lz)) = mut_self.resolve_mut(rx, ry, rz) {
            chunk.get(lx, ly, lz)
        } else {
            T::default()
        }
    }

    fn set(&mut self, x: u32, y: u32, z: u32, value: T) {
        let size = self.chunk_size.as_ivec3();
        let rx = x as i32 - size.x;
        let ry = y as i32 - size.y;
        let rz = z as i32 - size.z;

        if let Some((chunk, lx, ly, lz)) = self.resolve_mut(rx, ry, rz) {
            chunk.set(lx, ly, lz, value);
        }
    }
}
