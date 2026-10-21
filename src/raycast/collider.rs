//! Ray colliders: anything a ray can march, and how they combine.
//!
//! A collider is a signed distance field in world units (`RayCollider`).
//! The march (`cast`) only needs a sample and a step scale, so a layer's
//! fields (`LayerCollider`), the union or intersection of two colliders
//! (`Union`, `Intersect`), or any shape of your own can be hit the same
//! way. A hit says what it hit (`HitTarget`), and `cast_filtered` marches
//! on past hits a predicate rejects.
//!
//! Combining is CSG, as everywhere else in the crate: a union is the `min`
//! of its members, an intersection the `max`. The terrain clipped by its
//! visibility is an intersection, so `LayerCollider::unclipped` is the
//! same terrain without it.

use bevy::{ecs::system::SystemParam, prelude::*};

use crate::{
    chunk::{CHUNK_SIZE, Chunk, ChunkManager, world_pos_to_chunk_pos},
    field::{SDF_BAND, VoxelField},
    grid::flatten_cube,
    num::f32_from_u32,
    voxel::layer::{ClipChannel, FieldLayer, LayerIndex, SubSurface, SurfaceLayer},
};

/// What a ray can hit: a signed distance field in world units, negative
/// inside, over whatever part of the world is loaded.
pub trait RayCollider {
    /// Signed world distance from `pos` to the surface. `None` where
    /// nothing is loaded, which the march steps over.
    fn sample(&self, pos: Vec3) -> Option<f32>;
    /// The march's step scale at `pos`: the local voxel size. `None` where
    /// nothing is loaded.
    fn voxel_size_at(&self, pos: Vec3) -> Option<f32>;
    /// What a hit at `pos` (on the surface) is on. A union answers with the
    /// member that is there; a plain shape has nothing to say.
    fn target_at(&self, _pos: Vec3) -> HitTarget {
        HitTarget::default()
    }
}

impl<C: RayCollider + ?Sized> RayCollider for &C {
    fn sample(&self, pos: Vec3) -> Option<f32> {
        (**self).sample(pos)
    }
    fn voxel_size_at(&self, pos: Vec3) -> Option<f32> {
        (**self).voxel_size_at(pos)
    }
    fn target_at(&self, pos: Vec3) -> HitTarget {
        (**self).target_at(pos)
    }
}

/// What a hit is on: a registered layer (if the collider is one) and which
/// of its surfaces.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HitTarget {
    /// `None` for a collider that isn't a layer, or in an app without a
    /// render world.
    pub layer: Option<LayerIndex>,
    pub sub: SubSurface,
}

/// Where a ray met a collider.
#[derive(Clone, Copy, Debug)]
pub struct RayHit {
    pub pos: Vec3,
    /// Unit, pointing out of the surface.
    pub normal: Vec3,
    /// Along the ray, from its origin.
    pub distance: f32,
    pub target: HitTarget,
}

/// `A ∪ B`: the nearer surface of the two. Hits name the member they are
/// on.
#[derive(Clone, Copy, Debug)]
pub struct Union<A, B>(pub A, pub B);

impl<A: RayCollider, B: RayCollider> RayCollider for Union<A, B> {
    /// The `min`; where only one member is loaded, that one.
    fn sample(&self, pos: Vec3) -> Option<f32> {
        match (self.0.sample(pos), self.1.sample(pos)) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }
    }
    /// The finer of the two, so the march never overshoots either.
    fn voxel_size_at(&self, pos: Vec3) -> Option<f32> {
        match (self.0.voxel_size_at(pos), self.1.voxel_size_at(pos)) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }
    }
    /// The member whose surface `pos` is on: the one reading nearer zero.
    fn target_at(&self, pos: Vec3) -> HitTarget {
        if nearer_zero(self.0.sample(pos), self.1.sample(pos)) {
            self.0.target_at(pos)
        } else {
            self.1.target_at(pos)
        }
    }
}

/// `A ∩ B`: only where both are inside. Hits name the member whose
/// surface bounds the result there.
#[derive(Clone, Copy, Debug)]
pub struct Intersect<A, B>(pub A, pub B);

impl<A: RayCollider, B: RayCollider> RayCollider for Intersect<A, B> {
    /// The `max`; unknown where either member is unloaded.
    fn sample(&self, pos: Vec3) -> Option<f32> {
        Some(self.0.sample(pos)?.max(self.1.sample(pos)?))
    }
    fn voxel_size_at(&self, pos: Vec3) -> Option<f32> {
        match (self.0.voxel_size_at(pos), self.1.voxel_size_at(pos)) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }
    }
    fn target_at(&self, pos: Vec3) -> HitTarget {
        if nearer_zero(self.0.sample(pos), self.1.sample(pos)) {
            self.0.target_at(pos)
        } else {
            self.1.target_at(pos)
        }
    }
}

/// Whether `a` is at least as close to zero as `b` (an unloaded `a` never
/// is; an unloaded `b` always loses).
fn nearer_zero(a: Option<f32>, b: Option<f32>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => a.abs() <= b.abs(),
        (Some(_), None) => true,
        (None, _) => false,
    }
}

/// Combinators on any collider: `a.union(b)`, `a.intersect(b)`.
pub trait RayColliderExt: RayCollider + Sized {
    fn union<B: RayCollider>(self, other: B) -> Union<Self, B> {
        Union(self, other)
    }
    fn intersect<B: RayCollider>(self, other: B) -> Intersect<Self, B> {
        Intersect(self, other)
    }
}

impl<C: RayCollider> RayColliderExt for C {}

/// The step over unloaded space, in world units, until a loaded chunk says
/// otherwise.
const UNLOADED_STEP: f32 = CHUNK_SIZE / 16.0;

/// First outside -> inside crossing of `collider` along the ray, within
/// `max_dist`.
///
/// Detects the sign change itself, so it doesn't depend on the distances
/// being accurate far from the surface (uniform chunks only store the
/// distance to their own faces). Starts "inside", so a ray leaving rock
/// doesn't hit at once.
#[must_use]
pub fn cast<C: RayCollider + ?Sized>(
    collider: &C,
    origin: Vec3,
    dir: Vec3,
    max_dist: f32,
) -> Option<RayHit> {
    let dir = dir.normalize_or_zero();
    if dir == Vec3::ZERO {
        return None;
    }
    march(collider, origin, dir, 0.0, max_dist)
}

/// `cast`, marching on past every hit `keep` rejects: the first it accepts,
/// or `None` within `max_dist`.
///
/// Each continuation starts just past the rejected hit and must leave that
/// surface before it can hit again, so the same surface isn't reported
/// twice.
#[must_use]
pub fn cast_filtered<C: RayCollider + ?Sized>(
    collider: &C,
    origin: Vec3,
    dir: Vec3,
    max_dist: f32,
    mut keep: impl FnMut(&RayHit) -> bool,
) -> Option<RayHit> {
    let dir = dir.normalize_or_zero();
    if dir == Vec3::ZERO {
        return None;
    }
    let mut from = 0.0;
    loop {
        let hit = march(collider, origin, dir, from, max_dist)?;
        if keep(&hit) {
            return Some(hit);
        }
        let step = collider.voxel_size_at(hit.pos).unwrap_or(UNLOADED_STEP) * 0.25;
        from = hit.distance + step;
    }
}

/// The march from `t` along a unit `dir`, as `cast` describes.
fn march<C: RayCollider + ?Sized>(
    collider: &C,
    origin: Vec3,
    dir: Vec3,
    mut t: f32,
    max_dist: f32,
) -> Option<RayHit> {
    let mut vs = UNLOADED_STEP;
    let mut prev_t = t;
    let mut prev_outside = false;
    loop {
        if t > max_dist {
            return None;
        }
        let p = origin + dir * t;
        let step = match collider.sample(p) {
            None => {
                prev_outside = false;
                vs * 2.0
            }
            Some(d) => {
                let outside = d > 0.0;
                if prev_outside && !outside {
                    return Some(refine(collider, origin, dir, prev_t, t));
                }
                prev_outside = outside;
                prev_t = t;
                vs = collider.voxel_size_at(p).unwrap_or(vs);
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

/// Bisects the crossing between `lo` (outside) and `hi` (inside) and reads
/// the normal from the gradient there.
fn refine<C: RayCollider + ?Sized>(
    collider: &C,
    origin: Vec3,
    dir: Vec3,
    mut lo: f32,
    mut hi: f32,
) -> RayHit {
    for _ in 0..10 {
        let mid = f32::midpoint(lo, hi);
        if collider.sample(origin + dir * mid).unwrap_or(1.0) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let pos = origin + dir * hi;
    let h = collider.voxel_size_at(pos).unwrap_or(0.5) * 0.5;
    let s = |q: Vec3| collider.sample(q).unwrap_or(1.0);
    let g = Vec3::new(
        s(pos + Vec3::X * h) - s(pos - Vec3::X * h),
        s(pos + Vec3::Y * h) - s(pos - Vec3::Y * h),
        s(pos + Vec3::Z * h) - s(pos - Vec3::Z * h),
    );
    RayHit {
        pos,
        normal: g.try_normalize().unwrap_or(Vec3::Y),
        distance: hi,
        target: collider.target_at(pos),
    }
}

/// System param: the surface layer `L` as a collider, over the main
/// world's chunks. Same frame, every LOD.
///
/// Sees what is drawn: the shape clipped by the layer's `Clip`
/// (`collider`), or the shape alone (`unclipped`); one of its surfaces with
/// `surface`. `sample`, `voxel_size_at` and `cast` on the param itself are
/// the clipped layer's first surface: `TerrainRay` is
/// `LayerCollider<TerrainLayer>`.
#[derive(SystemParam)]
pub struct LayerCollider<'w, 's, L: SurfaceLayer> {
    chunks: Res<'w, ChunkManager>,
    shapes: Query<'w, 's, &'static <L as SurfaceLayer>::Shape, With<Chunk>>,
    clips: Query<
        'w,
        's,
        &'static <<L as SurfaceLayer>::Clip as ClipChannel>::Field,
        With<Chunk>,
    >,
    /// Which layer `L` is; absent without a render world.
    layer: Option<Res<'w, FieldLayer<<L as SurfaceLayer>::Shape>>>,
}

impl<L> LayerCollider<'_, '_, L>
where
    L: SurfaceLayer,
    L::Shape: VoxelField<Elem = f32>,
{
    /// The layer as drawn: its first surface, clipped.
    #[must_use]
    pub fn collider(&self) -> LayerView<'_, '_, '_, L> {
        LayerView {
            layer: self,
            sub: SubSurface::default(),
            clipped: true,
        }
    }

    /// The shape alone, with the clip ignored: the terrain under its
    /// hidden regions.
    #[must_use]
    pub fn unclipped(&self) -> LayerView<'_, '_, '_, L> {
        self.collider().unclipped()
    }

    /// One of the layer's surfaces (see `SurfaceLayer::SURFACE_AT`),
    /// clipped.
    #[must_use]
    pub fn surface(&self, sub: SubSurface) -> LayerView<'_, '_, '_, L> {
        self.collider().surface(sub)
    }

    /// Signed distance in WORLD units (negative = inside) to the layer as
    /// drawn, trilinear. `None` if no chunk is loaded at `pos`.
    #[must_use]
    pub fn sample(&self, pos: Vec3) -> Option<f32> {
        self.collider().sample(pos)
    }

    #[must_use]
    pub fn voxel_size_at(&self, pos: Vec3) -> Option<f32> {
        self.collider().voxel_size_at(pos)
    }

    /// `cast` against the layer as drawn.
    #[must_use]
    pub fn cast(&self, origin: Vec3, dir: Vec3, max_dist: f32) -> Option<RayHit> {
        cast(&self.collider(), origin, dir, max_dist)
    }

    /// The field value at voxel `c` of `entity`'s shape, through
    /// `SURFACE_AT` for `sub`; outside (`SDF_BAND`) without a field.
    fn shape_at(&self, entity: Entity, c: UVec3, size: u32, sub: SubSurface) -> f32 {
        self.shapes.get(entity).map_or(SDF_BAND, |f| {
            let v = f
                .data_slice()
                .get(flatten_cube(c.x, c.y, c.z, size))
                .copied()
                .unwrap_or(SDF_BAND);
            L::SURFACE_AT.map_or(v, |at| at(v, sub, f.lod()))
        })
    }

    /// The clip value at voxel `c` of `entity`; fully kept (`-SDF_BAND`)
    /// without a field or at a different LOD.
    fn clip_at(&self, entity: Entity, c: UVec3, size: u32) -> f32 {
        self.clips
            .get(entity)
            .ok()
            .filter(|f| f.lod().size() == size)
            .map_or(-SDF_BAND, |f| {
                f.data_slice()
                    .get(flatten_cube(c.x, c.y, c.z, size))
                    .copied()
                    .unwrap_or(-SDF_BAND)
            })
    }
}

/// One way of looking at a `LayerCollider`: which surface, and whether the
/// clip applies. What the param hands out; implements `RayCollider`.
pub struct LayerView<'a, 'w, 's, L: SurfaceLayer> {
    layer: &'a LayerCollider<'w, 's, L>,
    sub: SubSurface,
    clipped: bool,
}

impl<L> LayerView<'_, '_, '_, L>
where
    L: SurfaceLayer,
    L::Shape: VoxelField<Elem = f32>,
{
    /// The same view with the clip ignored.
    #[must_use]
    pub const fn unclipped(mut self) -> Self {
        self.clipped = false;
        self
    }

    /// The same view of surface `sub`.
    #[must_use]
    pub const fn surface(mut self, sub: SubSurface) -> Self {
        self.sub = sub;
        self
    }
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    (b - a).mul_add(t, a)
}

impl<L> RayCollider for LayerView<'_, '_, '_, L>
where
    L: SurfaceLayer,
    L::Shape: VoxelField<Elem = f32>,
{
    /// Trilinear over the chunk's voxels (and its +neighbours' first
    /// voxels, as the mesh apron reads them), `max` with the clip when
    /// clipped, scaled to world units.
    fn sample(&self, pos: Vec3) -> Option<f32> {
        let layer = self.layer;
        let cp = world_pos_to_chunk_pos(&pos);
        let own = layer.chunks.get_chunk(&cp)?;
        let shape = layer.shapes.get(own).ok()?;
        let size = shape.lod().size();
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
            let (entity, c) = if over.any() {
                let neighbor = cp.wrapping_add(IVec3::select(over, IVec3::ONE, IVec3::ZERO));
                let in_neighbor = UVec3::select(over, q.saturating_sub(UVec3::splat(size)), q);
                let usable = layer.chunks.get_chunk(&neighbor).filter(|&e| {
                    layer
                        .shapes
                        .get(e)
                        .is_ok_and(|nb| nb.lod().size() == size)
                });
                // No usable neighbour: this chunk's own edge.
                usable.map_or_else(|| (own, q.min(edge)), |e| (e, in_neighbor))
            } else {
                (own, q)
            };
            let s = layer.shape_at(entity, c, size, self.sub);
            if self.clipped {
                s.max(layer.clip_at(entity, c, size))
            } else {
                s
            }
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

    fn voxel_size_at(&self, p: Vec3) -> Option<f32> {
        let cp = world_pos_to_chunk_pos(&p);
        let shape = self.layer.shapes.get(self.layer.chunks.get_chunk(&cp)?).ok()?;
        Some(shape.lod().voxel_size())
    }

    fn target_at(&self, _pos: Vec3) -> HitTarget {
        HitTarget {
            layer: self.layer.layer.as_ref().map(|l| l.layer),
            sub: self.sub,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{HitTarget, RayCollider, RayColliderExt, SubSurface, cast, cast_filtered};
    use bevy::math::Vec3;

    /// A sphere, tagged so hits can tell it apart.
    struct Ball {
        center: Vec3,
        radius: f32,
        id: u8,
    }

    impl RayCollider for Ball {
        fn sample(&self, pos: Vec3) -> Option<f32> {
            Some(pos.distance(self.center) - self.radius)
        }
        fn voxel_size_at(&self, _: Vec3) -> Option<f32> {
            Some(0.25)
        }
        fn target_at(&self, _: Vec3) -> HitTarget {
            HitTarget {
                layer: None,
                sub: SubSurface(self.id),
            }
        }
    }

    fn ball(x: f32, id: u8) -> Ball {
        Ball {
            center: Vec3::new(x, 0.0, 0.0),
            radius: 1.0,
            id,
        }
    }

    #[test]
    fn hits_the_near_side_with_an_outward_normal() {
        let hit = cast(&ball(5.0, 0), Vec3::ZERO, Vec3::X, 20.0).expect("hit");
        assert!((hit.distance - 4.0).abs() < 0.02, "{}", hit.distance);
        assert!(hit.normal.dot(Vec3::NEG_X) > 0.99);
        assert!(cast(&ball(5.0, 0), Vec3::ZERO, Vec3::X, 3.0).is_none());
    }

    #[test]
    fn a_union_hits_the_nearer_member_and_names_it() {
        let both = ball(5.0, 1).union(ball(9.0, 2));
        let hit = cast(&both, Vec3::ZERO, Vec3::X, 20.0).expect("hit");
        assert!((hit.distance - 4.0).abs() < 0.02);
        assert_eq!(hit.target.sub, SubSurface(1));
        // From the other side, the other member.
        let hit = cast(&both, Vec3::X * 14.0, Vec3::NEG_X, 20.0).expect("hit");
        assert_eq!(hit.target.sub, SubSurface(2));
    }

    #[test]
    fn an_intersection_is_only_where_both_are() {
        let a = Ball {
            center: Vec3::ZERO,
            radius: 2.0,
            id: 1,
        };
        let b = Ball {
            center: Vec3::X * 3.0,
            radius: 2.0,
            id: 2,
        };
        // The lens between x = 1 and x = 2: the ray enters it at b's surface.
        let hit = cast(&a.intersect(b), Vec3::X * -5.0, Vec3::X, 20.0).expect("hit");
        assert!((hit.distance - 6.0).abs() < 0.02, "{}", hit.distance);
        assert_eq!(hit.target.sub, SubSurface(2));
    }

    #[test]
    fn a_filter_marches_on_past_rejected_hits() {
        let both = ball(5.0, 1).union(ball(9.0, 2));
        let hit = cast_filtered(&both, Vec3::ZERO, Vec3::X, 20.0, |h| {
            h.target.sub != SubSurface(1)
        })
        .expect("hit");
        assert_eq!(hit.target.sub, SubSurface(2));
        assert!((hit.distance - 8.0).abs() < 0.05, "{}", hit.distance);
        // Rejecting everything runs out of ray.
        assert!(cast_filtered(&both, Vec3::ZERO, Vec3::X, 20.0, |_| false).is_none());
    }
}
