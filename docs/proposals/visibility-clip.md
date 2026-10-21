# Visibility as a clip surface

**Status:** in progress on the `visibility-clip` branch (off `surface-layers`).
Phases 1 to 4 below are written; the bench's `visibility` phase and a
sandbox run are the gates.

The visibility field hides parts of the terrain: fog of war, a player's
claimed volume, a dungeon's unexplored rooms. Today it is a per-voxel bit
mask applied per *pixel*: the terrain is meshed as if nothing were hidden,
and the fragment shader discards every pixel whose nearest voxel is hidden.
Two things are wrong with what that shows:

1. **The edge is blocky.** The discard follows the voxel grid, so the
   border between shown and hidden terrain is a staircase of voxel-sized
   steps, however smooth the terrain is.
2. **The cut is hollow.** Discarding the surface opens the mesh: through the
   hole you see the back faces (culled, so nothing) of the terrain beyond.
   Where the ground should continue into the hidden region there is a void,
   and the inside of the terrain shows as missing.

Both come from hiding at the pixel instead of at the shape.

## Proposal

Hide at the shape. The visibility field becomes a **signed distance field
to the visibility boundary**, negative where visible (the same convention as
the `SDF`, negative inside), and the terrain layer meshes

    max(sdf, visibility)

solid *and* visible. That is the CSG intersection of the terrain with the
visible region: the hidden part is cut away, and the cut is closed by a
surface along the visibility field's zero crossing, the **cap**, with the
terrain's material on it. Surface nets places the cap's vertices by
interpolating the distances, so the edge is smooth at sub-voxel precision,
and the crease where the cap meets the terrain surface is exact.

Nothing is discarded per pixel any more. That also removes the one
`discard` in the raster shader, so every chunk keeps early depth testing on
tiling GPUs, and the masked draw variant and its pipelines go away.

### The field

`VisibilityField` wraps the `SDF`'s representation (`f32` per voxel, in
voxels of the chunk's LOD, clamped to `±SDF_BAND`, with the inside count,
border versions and jump-flood buffers), so it edits, saves, downsamples and
pads exactly like the shape. It is still optional: a chunk without one is
fully visible. Its default is `-SDF_BAND` everywhere: a uniform field sits at
the band's full depth rather than at the distance to the chunk's faces (what
the jump flood gives a uniform `SDF`, to keep it editable), because
`max(sdf, -1)` near a face would pull the terrain's distances toward zero and
move its surface.

Both distance fields share the edit machinery through a `DistanceField`
trait:

- **CSG.** `process_csg_edits::<M>` is generic over a `CsgMessage`:
  `SdfCsgSphere` edits the `SDF`, the new `VisibilityCsgSphere { hide, .. }`
  and `VisibilityCsgBox` the visibility. A message resolves to a `CsgOp`
  whose `CsgShape` is a swept sphere or an axis-aligned box; the field op
  only asks the shape for a signed distance. Hiding a sphere is `max(field, -sphere)`, the same
  subtraction as a dig; revealing it is the union, the same as a build.
  Swept capsules and `max_step` work unchanged. A reveal also marks the
  chunk for a reinit (`CsgMessage::reinit_after`): the union is exact only
  outside the sphere and leaves the sphere's own distances where it was,
  which `max(sdf, visibility)` would read as a dent in the terrain. The
  jump flood keeps the zero crossing and rebuilds the rest; a fully
  revealed chunk lands back at the band's depth.
- **Value edits.** `WorldEditor<VisibilityField, f32>` (fill a box with
  `+SDF_BAND` to hide it) leaves signs, and the jump-flood reinit, now
  `spawn_reinit_tasks::<F>` / `resolve_reinit_tasks::<F>`, turns them into
  distances, as it does for the `SDF`.

The cursor brush gains `Hide` and `Reveal` tools (keys 8 and 9 in the
sandbox), and the CPU `TerrainRay` samples the clipped field, so the cursor
lands on the cap, as the GPU query (which reads the uploaded shape) already
does.

### Where the clip is applied

In extraction, on the CPU, before upload. `SurfaceLayer::Mask` becomes
`SurfaceLayer::Clip: ClipChannel`, a field of `f32` distances the shape's
`pad_dirty_chunks::<T, C>` reads alongside the shape:

- The clip is padded like the shape (own voxels, resampled to the upload
  LOD if capped, plus the apron from the neighbours' clips), then
  `max`ed into the shape's padded distances voxel by voxel. A loaded
  neighbour without a clip contributes a fully visible volume; an unloaded
  one continues the edge, as for the shape.
- Nothing is built when neither the chunk nor a neighbour the apron reads
  has a clip: the common case costs one query per neighbour.
- The extract key carries the clip's own version (with a presence bit, so
  removing a fresh field re-extracts) and each neighbour's, so a clip edit
  re-extracts the chunk and the neighbours whose apron reads it, and
  nothing else.
- A layer whose shape isn't `f32` distances and has no `SURFACES` gets a
  warning and no clip; with `SURFACES`, the clip applies to the distances
  they produce.

The alternative, uploading the clip as a side channel and taking the `max`
in the mesher, was rejected: it touches pass1, pass3 and the query shader,
needs a second distance buffer per arena, and the ray query, the active cell
count and the mesh would each have to agree on the combination separately.
Clipping once on the CPU keeps the GPU reading one shape.

`HasSurface` (residency) now also counts a visibility boundary through
solid: a chunk of uniform rock whose clip, or a neighbour's, is mixed or
differs from its own has a cap to mesh. A fully hidden chunk is still not
resident.

### The old way, behind a feature

The per-pixel mask path stays, as the `visibility-mask` Cargo feature: the
terrain layer then takes `VisibilityField` as its `Mask` instead of its
`Clip`, the field uploads as a `GpuChannel::Mask` (one byte per voxel,
hidden where the distance is positive) and the fragment shader discards
hidden voxels, exactly as before. The data model and the edits are the same
either way; only the terrain's look changes. Any layer can choose too:
`type Mask` for the discard, `type Clip` for the cut. Under the feature
the CPU ray and `HasSurface` ignore the visibility, as the mesh does.

What goes away in both modes: the debug `detect_topology_desync` warning,
which assumed the SDF and visibility change together, no longer means
anything. `ArenaInspector` gains `shape_slot` (the clipped SDF the mesher
read) beside `mask_slot`.

### Saves

The field's payload is now `f32`s. A save from before this change holds
one byte per voxel for it; its length no longer matches, so the loader drops
it (the field is optional) and the chunk loads fully visible. The `SDF` and
materials are unaffected.

## Phases

1. `VisibilityField` as distances, `DistanceField`, generic CSG and reinit,
   `VisibilityCsgSphere`, the `Hide`/`Reveal` brush tools.
2. The clip in extraction (`ClipChannel`, `pad_dirty_chunks::<T, C>`,
   `padded_clip`), clip-aware `HasSurface`, the CPU ray clipped.
3. The `visibility-mask` feature: the mask path driven by the distance
   field.
4. Bench: the `visibility` phase reads the shape slots back and checks them
   against `max(sdf, visibility)` on the CPU; one chunk's field removed must
   read back as the plain SDF. With the feature, the GPU masks are checked
   as before.

5. `VisibilityCsgBox`: the box CSG for fog-of-war cells, through a
   `CsgShape` the sphere path was generalized over. The bench's box is
   hidden this way (it was a value edit plus a reinit). The brush's
   `BrushShape` (B in the sandbox) makes Hide and Reveal place a cube of
   the brush radius instead of the sphere.

## Open

- The cap carries the material of the voxels it cuts through. A layer may
  want a distinct look for it (a fog wall); the mesher would have to tag
  cap vertices, which it can tell from `visibility > sdf` at the cell.
