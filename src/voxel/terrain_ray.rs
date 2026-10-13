use bevy::{ecs::system::SystemParam, prelude::*};

use crate::{
    SDF,
    chunk::{CHUNK_SIZE, Chunk, ChunkManager, world_pos_to_chunk_pos},
    field::Field,
};

pub struct TerrainHit {
    pub pos: Vec3,
    pub normal: Vec3,
    pub distance: f32,
}

#[derive(SystemParam)]
pub struct TerrainRay<'w, 's> {
    chunks: Res<'w, ChunkManager>,
    sdfs: Query<'w, 's, &'static SDF, With<Chunk>>,
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

impl TerrainRay<'_, '_> {
    /// Signed distance in WORLD units (negative = inside), trilinear.
    /// `None` if no chunk is loaded at `p`.
    pub fn sample(&self, p: Vec3) -> Option<f32> {
        let cp = world_pos_to_chunk_pos(&p);
        let sdf = self.sdfs.get(self.chunks.get_chunk(&cp)?).ok()?;
        let n = sdf.size().x as i32;
        let vs = CHUNK_SIZE / n as f32;

        let l = (p - cp.as_vec3() * CHUNK_SIZE) / vs;
        let i0f = l.floor();
        let f = l - i0f;
        let i0 = i0f.as_ivec3().clamp(IVec3::ZERO, IVec3::splat(n - 1));

        // Index n is voxel 0 of the +neighbour (same as the mesh apron).
        let at = |o: IVec3| -> f32 {
            let q = i0 + o;
            let mut c = cp;
            let mut lq = q;
            for a in 0..3 {
                if q[a] >= n {
                    c[a] += 1;
                    lq[a] -= n;
                }
            }
            if c != cp {
                if let Some(nb) = self
                    .chunks
                    .get_chunk(&c)
                    .and_then(|e| self.sdfs.get(e).ok())
                {
                    if nb.size().x as i32 == n {
                        return nb.get(lq.x as u32, lq.y as u32, lq.z as u32);
                    }
                }
                lq = q.min(IVec3::splat(n - 1)); // no usable neighbour: own edge
            }
            sdf.get(lq.x as u32, lq.y as u32, lq.z as u32)
        };

        let c = |x: i32, y: i32, z: i32| at(IVec3::new(x, y, z));
        let x00 = lerp(c(0, 0, 0), c(1, 0, 0), f.x);
        let x10 = lerp(c(0, 1, 0), c(1, 1, 0), f.x);
        let x01 = lerp(c(0, 0, 1), c(1, 0, 1), f.x);
        let x11 = lerp(c(0, 1, 1), c(1, 1, 1), f.x);
        let y0 = lerp(x00, x10, f.y);
        let y1 = lerp(x01, x11, f.y);
        Some(lerp(y0, y1, f.z) * vs)
    }

    pub fn voxel_size_at(&self, p: Vec3) -> Option<f32> {
        let cp = world_pos_to_chunk_pos(&p);
        let sdf = self.sdfs.get(self.chunks.get_chunk(&cp)?).ok()?;
        Some(CHUNK_SIZE / sdf.size().x as f32)
    }

    /// First outside -> inside crossing along the ray. Detects the sign
    /// change itself, so it doesn't depend on distance values being accurate
    /// (uniform chunks store +-size).
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

        while t <= max_dist {
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
            t += step;
        }
        None
    }

    fn refine(&self, origin: Vec3, dir: Vec3, mut lo: f32, mut hi: f32) -> TerrainHit {
        for _ in 0..10 {
            let mid = 0.5 * (lo + hi);
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
