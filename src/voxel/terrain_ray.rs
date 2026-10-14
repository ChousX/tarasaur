//! CPU ray casting and SDF sampling over the main world's chunks. Same
//! frame, every LOD; used by the cursor and usable as terrain collision.

use bevy::{ecs::system::SystemParam, prelude::*};

use crate::{
    SDF,
    chunk::{CHUNK_SIZE, Chunk, ChunkManager, world_pos_to_chunk_pos},
    field::Field,
    num::f32_from_u32,
};

pub struct TerrainHit {
    pub pos: Vec3,
    pub normal: Vec3,
    pub distance: f32,
}

/// System param: `sample` the world SDF or `cast` a ray at the terrain.
#[derive(SystemParam)]
pub struct TerrainRay<'w, 's> {
    chunks: Res<'w, ChunkManager>,
    sdfs: Query<'w, 's, &'static SDF, With<Chunk>>,
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    (b - a).mul_add(t, a)
}

impl TerrainRay<'_, '_> {
    /// Signed distance in WORLD units (negative = inside), trilinear.
    /// `None` if no chunk is loaded at `pos`.
    #[must_use]
    pub fn sample(&self, pos: Vec3) -> Option<f32> {
        let cp = world_pos_to_chunk_pos(&pos);
        let sdf = self.sdfs.get(self.chunks.get_chunk(&cp)?).ok()?;
        let size = sdf.size().x;
        let voxel = CHUNK_SIZE / f32_from_u32(size);

        let local = (pos - cp.as_vec3() * CHUNK_SIZE) / voxel;
        let base_f = local.floor();
        let frac = local - base_f;
        let edge = UVec3::splat(size.saturating_sub(1));
        // Negative coordinates saturate to 0, matching a clamp to the grid.
        let base = base_f.as_uvec3().min(edge);

        // Index `size` is voxel 0 of the +neighbour (same as the mesh apron).
        let at = |offset: UVec3| -> f32 {
            let q = base.saturating_add(offset);
            let over = q.cmpge(UVec3::splat(size));
            if over.any() {
                let neighbor = cp.wrapping_add(IVec3::select(over, IVec3::ONE, IVec3::ZERO));
                let in_neighbor = UVec3::select(over, q.saturating_sub(UVec3::splat(size)), q);
                if let Some(nb) = self
                    .chunks
                    .get_chunk(&neighbor)
                    .and_then(|e| self.sdfs.get(e).ok())
                    && nb.size().x == size
                {
                    return nb.get(in_neighbor.x, in_neighbor.y, in_neighbor.z);
                }
                let own = q.min(edge); // no usable neighbour: own edge
                return sdf.get(own.x, own.y, own.z);
            }
            sdf.get(q.x, q.y, q.z)
        };

        let corner = |x: u32, y: u32, z: u32| at(UVec3::new(x, y, z));
        let x00 = lerp(corner(0, 0, 0), corner(1, 0, 0), frac.x);
        let x10 = lerp(corner(0, 1, 0), corner(1, 1, 0), frac.x);
        let x01 = lerp(corner(0, 0, 1), corner(1, 0, 1), frac.x);
        let x11 = lerp(corner(0, 1, 1), corner(1, 1, 1), frac.x);
        let y0 = lerp(x00, x10, frac.y);
        let y1 = lerp(x01, x11, frac.y);
        Some(lerp(y0, y1, frac.z) * voxel)
    }

    #[must_use]
    pub fn voxel_size_at(&self, p: Vec3) -> Option<f32> {
        let cp = world_pos_to_chunk_pos(&p);
        let sdf = self.sdfs.get(self.chunks.get_chunk(&cp)?).ok()?;
        Some(CHUNK_SIZE / f32_from_u32(sdf.size().x))
    }

    /// First outside -> inside crossing along the ray. Detects the sign
    /// change itself, so it doesn't depend on distance values being accurate
    /// (uniform chunks only store distance to their own faces).
    #[must_use]
    pub fn cast(&self, origin: Vec3, dir: Vec3, max_dist: f32) -> Option<TerrainHit> {
        let dir = dir.normalize_or_zero();
        if dir == Vec3::ZERO {
            return None;
        }
        let mut vs = CHUNK_SIZE / 16.0; // fallback scale over unloaded space
        let mut t = 0.0f32;
        let mut prev_t = 0.0f32;
        // Start "inside" so a camera in rock doesn't register an instant hit.
        let mut prev_outside = false;

        loop {
            if t > max_dist {
                return None;
            }
            let p = origin + dir * t;
            let step = match self.sample(p) {
                None => {
                    prev_outside = false;
                    vs * 2.0
                }
                Some(d) => {
                    let outside = d > 0.0;
                    if prev_outside && !outside {
                        return Some(self.refine(origin, dir, prev_t, t));
                    }
                    prev_outside = outside;
                    prev_t = t;
                    vs = self.voxel_size_at(p).unwrap_or(vs);
                    d.abs().clamp(vs * 0.25, vs * 1.5)
                }
            };
            let next = t + step;
            if next <= t {
                return None; // step lost to f32 precision: no progress possible
            }
            t = next;
        }
    }

    fn refine(&self, origin: Vec3, dir: Vec3, mut lo: f32, mut hi: f32) -> TerrainHit {
        for _ in 0..10 {
            let mid = f32::midpoint(lo, hi);
            if self.sample(origin + dir * mid).unwrap_or(1.0) > 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let pos = origin + dir * hi;
        let h = self.voxel_size_at(pos).unwrap_or(0.5) * 0.5;
        let s = |q: Vec3| self.sample(q).unwrap_or(1.0);
        let g = Vec3::new(
            s(pos + Vec3::X * h) - s(pos - Vec3::X * h),
            s(pos + Vec3::Y * h) - s(pos - Vec3::Y * h),
            s(pos + Vec3::Z * h) - s(pos - Vec3::Z * h),
        );
        TerrainHit {
            pos,
            normal: g.try_normalize().unwrap_or(Vec3::Y),
            distance: hi,
        }
    }
}
