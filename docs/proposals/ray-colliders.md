# Ray colliders

**Status:** CPU slice built (phases 1 to 4 below); the GPU slice is open.

A ray could only hit the terrain. `TerrainRay` marched the `SDF` clipped
by the `VisibilityField`, hard-wired, and a hit said where but not what.
Other surface layers (the water example's `WaterField`) had fields a ray
could march, and nothing to march them. The cursor could not point at
hidden terrain, so the Reveal brush could only hit the cap of a cut.

## Proposal

The march only ever asked two things of the terrain: a signed distance at
a point and the step scale there. Make that the collider:

```rust
pub trait RayCollider {
    fn sample(&self, pos: Vec3) -> Option<f32>;        // world units, None if unloaded
    fn voxel_size_at(&self, pos: Vec3) -> Option<f32>;
    fn target_at(&self, pos: Vec3) -> HitTarget { .. } // what a hit here is on
}
```

`cast(&collider, origin, dir, max)` is the old march, generic. A hit
(`RayHit`) carries a `HitTarget { layer: Option<LayerIndex>, sub:
SubSurface }`.

### Colliders are distance fields, so they combine as CSG

- `Union(a, b)` samples the `min` of its members: one march hits
  whichever surface comes first. Where only one member is loaded, that
  one answers.
- `Intersect(a, b)` samples the `max`, unknown where either is unloaded.
- Both name the member a hit is on by asking which reads nearer zero
  there.
- `a.union(b)`, `a.intersect(b)` through `RayColliderExt`.

### A layer is a collider

`LayerCollider<L: SurfaceLayer>` is a system param over `L::Shape` and
`L::Clip::Field` on the main world's chunks. It hands out views:

- `collider()`: the layer as drawn, `max(shape, clip)`, first surface.
- `unclipped()`: the shape alone, seeing through hidden regions.
- `surface(sub)`: another of the layer's surfaces, through the new
  `SurfaceLayer::SURFACE_AT`, the point-wise form of `SURFACES`. The
  water example gives one, so its flood level can be hit as well as its
  table.

`TerrainRay` is `LayerCollider<TerrainLayer>`, with `sample`,
`voxel_size_at` and `cast` on the param as before, so nothing that used it
changes. The clip is folded into the view rather than written as an
`Intersect`, because a missing clip field means "fully kept" and the field
is read at the same voxel as the shape.

`add_surface_layer` now also inserts `FieldLayer<L::Shape>` into the main
world, so a hit can name the layer without the render world.

### Filtering

`cast_filtered(&c, origin, dir, max, |hit| keep)` marches on past every
hit the predicate rejects. Each continuation starts a quarter voxel past
the rejected hit and must leave that surface before it can hit again, so
the same surface is not reported twice.

### The cursor

`CursorSurfaceHit` gains `target`. A `CursorFilter { clip }` resource says
whether the cursor ray stops at the visibility clip; the brush sets it from
its tool each frame it runs (off for Reveal), and an app without the brush
owns it. The GPU backend still traces what is drawn and names the terrain
layer.

## Phases

1. `RayCollider`, `RayHit`, `HitTarget`, `cast`, `cast_filtered`, `Union`,
   `Intersect`; the march moved out of `TerrainRay` unchanged.
2. `LayerCollider<L>` and its views; `TerrainRay` as its alias;
   `SurfaceLayer::SURFACE_AT`; the layer index in the main world.
3. The cursor: `target` on the hit, `CursorFilter`, Reveal through the
   clip.
4. Unit tests on synthetic colliders: near side and normal, union
   nearest-and-named, intersection, filtered continuation.

## Open

- **GPU.** `RayQuery` has no layer: the shader traces the one queryable
  layer's arenas, once per arena. A layer mask on the query would need
  either a dispatch per layer with a closest-hit merge, or every
  queryable layer's arenas bound at once (which may exceed the Adreno's
  per-stage limits). Predicate filters do not translate; the GPU would
  honour the mask only. Best done once the CPU API has settled in use.
- **Many colliders.** `Union` is a pair; a union of N is nested pairs.
  A `Vec<Box<dyn RayCollider>>` collider would be a few lines if it is
  ever wanted.
- **Cost.** A union samples every member per step. Fine for a cursor or
  a handful of gameplay rays; bulk queries belong on the GPU.
