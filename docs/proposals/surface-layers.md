# Surface layers

**Status:** draft for review. Nothing in this document is implemented.

Tarasaur meshes and draws exactly one surface per chunk: the `SDF`, textured
by the palette. This proposes generalizing that into *surface layers*, so an
app can register a second shape field with its own look (a translucent zone,
a fluid level, a player's claimed volume) and have it meshed by the same GPU
surface nets, streamed by the same LOD bands, and drawn in the same frame.

The motivating case is Domain Expance, which today meshes its domain shell on
the CPU with a 430-line surface-nets pass because the GPU path cannot take a
second surface. With layers that file becomes a trait impl and a fragment
shader.

## What stops a second surface today

Four places assume one shape, verified against `dev` at `87c06d2`:

1. **One shape field.** `register_field_upload` in `src/voxel/mod.rs` keeps a
   `ShapeFieldRegistered` resource and logs an error for a second
   `GpuChannel::Shape` field. The `GpuChannel` docs say why: two would
   overwrite each other's slots.
2. **One arena set, keyed by chunk.** `VoxelChunkArenaSet` (`src/voxel/arena.rs`)
   is a single render-world resource. Each `VoxelChunkArena` maps
   `slot_of_main_entity: HashMap<MainEntity, u32>`, so a chunk entity holds at
   most one slot per LOD. `side_data` caches the material and mask uploads per
   channel `TypeId`, again one per chunk.
3. **One raster look.** `VoxelRasterPipeline` is specialized on
   `VoxelRasterKey { view, arm, normal, pass, masked }` and always binds the
   palette (`VoxelMaterialBindGroup`, group 2) and
   `voxel_raster.wesl`. `raster_pipeline_descriptor` hardcodes
   `BlendState::REPLACE`, depth write on, and `voxel_raster_pass` is scheduled
   after `main_opaque_pass_3d` and before `main_transparent_pass_3d`.
4. **No mesh comes back.** Pass3 writes vertices and indices into arena
   buffers and the raster pass draws them with `multi_draw_indexed_indirect`.
   There is no `Mesh` asset, so Bevy's `Material` trait cannot be used for a
   different look.

Everything else is already generic over the field type: extraction
(`PaddedOutbox<T>`, `ExtractedChunkField<T>`, `pad_dirty_chunks<T>`), the
upload systems (`prepare_shape_upload<F>`, `prepare_material_upload<F>`,
`prepare_mask_upload<F>`), apron filling and LOD seam stitching. That is what
makes layers a refactor of the render side rather than a new pipeline.

## Proposal

A **layer** is a shape field plus how it is drawn. The terrain becomes layer
0, registered by `TarasaurPlugin` exactly as it is today, so existing apps
change nothing.

```rust
/// A meshed surface: a shape field, its optional side channels, and its look.
pub trait SurfaceLayer: Send + Sync + 'static {
    /// The field whose zero crossing is meshed. Its `GpuChannel` must be `Shape`.
    type Shape: VoxelField;
    /// Per-voxel ids blended into vertices and read per fragment, or `NoChannel`.
    type Material: SideChannel;
    /// Per-voxel discard mask, or `NoChannel`.
    type Mask: SideChannel;
    /// How the meshes are drawn.
    type Raster: LayerRaster;

    /// Coarsest LOD this layer is uploaded at. A chunk at a finer LOD is
    /// downsampled first (`persistence::downsample_bytes`). `None` = the
    /// chunk's own LOD. A smooth, coarse surface at `Some(LOD::Low)` costs
    /// 1/64 of the arena memory of `High`.
    const UPLOAD_LOD: Option<LOD> = None;

    /// Whether the GPU ray query and the cursor see this layer. Only one
    /// layer may be queryable; the terrain is.
    const QUERYABLE: bool = false;

    /// The surfaces in one padded volume. Each is `size^3` snorm16 distances
    /// (see `quantize_sdf`) under its own `SubSurface` id. Default: one, from
    /// `Shape::gpu_bytes`. A field that stores several owners per voxel
    /// returns one surface per owner present.
    fn surfaces(vol: &[<Self::Shape as VoxelField>::Elem]) -> Vec<(SubSurface, Vec<u8>)> {
        vec![(SubSurface(0), Self::Shape::gpu_bytes(vol))]
    }
}

/// Which of a layer's surfaces in a chunk a slot holds. 0 for most layers.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct SubSurface(pub u8);

pub trait AppSurfaceLayerExt {
    /// Registers `L`'s fields (as `add_voxel_field` does) and its arenas,
    /// uploads, meshing, draws and raster pipelines.
    fn add_surface_layer<L: SurfaceLayer>(&mut self) -> &mut Self;
}
```

```rust
/// How a layer's meshes are drawn. The vertex format is fixed: pass3's
/// `Vertex` (32 bytes: world position + packed material a, normal + packed
/// material b and slot), so a raster only supplies the fragment side.
pub trait LayerRaster: Send + Sync + 'static {
    /// Which Bevy pass to draw in. Opaque layers draw after
    /// `main_opaque_pass_3d`; transparent ones after `main_transparent_pass_3d`,
    /// back to front per chunk, without writing depth.
    const PASS: LayerPass;
    const BLEND: Option<BlendState>;
    const CULL: Option<Face>;
    /// The shader; `vs_main` is shared, `fs_main` is the layer's.
    fn shader() -> Handle<Shader>;
    /// Group 2's layout and bind group. The palette raster binds the texture
    /// arrays; a zone raster might bind a small uniform of colors.
    fn material_layout(device: &RenderDevice) -> BindGroupLayout;
    fn material_bind_group(world: &mut World, layout: &BindGroupLayout) -> Option<BindGroup>;
    /// Shader defs this raster specializes on, from its own resources
    /// (the palette's `arm` and `normal` today).
    fn specialize(world: &World) -> Vec<ShaderDefVal>;
}

pub enum LayerPass { Opaque, Transparent }
```

The terrain's raster becomes `PaletteRaster`, which is today's code moved:
`material_layout_entries`, `material_bind_group`, `update_voxel_material_bind_group`
and the `arm`/`normal` key bits. `voxel_raster.wesl` keeps `vs_main` and
becomes the palette's `fs_main`; the mask discard and the depth prepass stay
features of the opaque path.

### What an app writes

Domain Expance's shell, with the layer in place:

```rust
pub struct DomainLayer;

impl SurfaceLayer for DomainLayer {
    type Shape = DomainField;         // u16 per voxel: owner id + claim strength
    type Material = NoChannel;
    type Mask = NoChannel;
    type Raster = DomainRaster;
    const UPLOAD_LOD: Option<LOD> = Some(LOD::Low); // 16 cells per chunk, as the CPU mesher uses now

    fn surfaces(vol: &[u16]) -> Vec<(SubSurface, Vec<u8>)> {
        // One surface per owner present, positive inside that owner's domain.
        owners_in(vol)
            .map(|owner| (SubSurface(owner.0), distances_for(vol, owner)))
            .collect()
    }
}

impl LayerRaster for DomainRaster {
    const PASS: LayerPass = LayerPass::Transparent;
    const BLEND: Option<BlendState> = Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING);
    const CULL: Option<Face> = None; // the film is seen from inside too
    fn shader() -> Handle<Shader> { DOMAIN_SHADER }          // fresnel rim + pulse
    fn material_layout(..) -> BindGroupLayout { /* uniform: [owner color; 256] */ }
    ...
}

app.add_surface_layer::<DomainLayer>();
```

The owner's color reaches the fragment shader through the sub-surface id,
which pass3 already carries in the vertex (`real_slot` in `normal.w`): the
raster bind group for a transparent layer gets a `[slot] -> SubSurface` table
alongside the per-arena chunk data in group 3.

## Changes by module

### `src/voxel/arena.rs`

- `VoxelChunkArenaSet` becomes the per-layer type (call it `LayerArenas`) and
  a new `VoxelLayers { layers: Vec<LayerArenas> }` resource holds one per
  registered layer, indexed by a `LayerIndex` assigned at registration. The
  fields `arenas`, `config`, `live`, `deferred`, `empty`, `reslotted` and
  `side_data` all move down a level unchanged.
- `slot_of_main_entity: HashMap<MainEntity, u32>` becomes
  `HashMap<(MainEntity, SubSurface), u32>`; `place`, `release`,
  `existing_slot`, `find_slot` take the pair. `chunk_pos_of_slot` is
  unchanged (culling only needs the chunk box).
- `slot_of_chunk_pos` and `rebuild_merged_lookup` stay, but only the
  `QUERYABLE` layer maintains them. The lookup entry format in
  `cursor_query.wesl` does not change.
- `ArenaConfig` moves into the layer (each layer sizes its own arenas).
  `TarasaurPlugin::arenas` keeps configuring layer 0.

### `src/voxel/mod.rs`

- `register_field_upload::<F>` is replaced by `add_surface_layer::<L>`, which
  calls `add_voxel_field` for `L::Shape`, `L::Material` and `L::Mask` (skipping
  `NoChannel`), allocates the `LayerIndex`, and registers the three upload
  systems with that index. `ShapeFieldRegistered` goes away; the one-shape
  rule becomes one shape per layer, enforced by the trait.
- `TarasaurPlugin` calls `add_surface_layer::<TerrainLayer>()` where
  `TerrainLayer { Shape = SDF, Material = ?, Mask = VisibilityField, Raster = PaletteRaster }`.
  The material field is the open question below.
- The raster pass is scheduled twice: the opaque layers where
  `voxel_raster_pass` runs today, and the transparent layers after
  `main_transparent_pass_3d`.

### `src/voxel/extract.rs`

- `pad_dirty_chunks<T>` is unchanged for the apron fill. After it,
  `padded_bytes` calls `L::surfaces(vol)` instead of `T::gpu_bytes(vol)` and
  the outbox carries one `PaddedChunk` per sub-surface, each tagged. `PaddedChunk`
  and `ExtractedChunkField<T>` gain `sub: SubSurface`.
- A sub-surface that disappears (an owner loses its last voxel in a chunk)
  must release its slot: `pad_dirty_chunks` compares the set of ids it
  produced last time with this time and sends an empty volume for each one
  gone, the way `extract_removed` does for an optional field.
- `track_live_chunks` and `release_despawned_chunks` run per layer.
- `UPLOAD_LOD`: when the chunk's LOD is finer, the padded volume is built at
  the chunk's LOD (so the apron is right) and downsampled before quantizing.
  The `ExtractKey` is unchanged.

### `src/voxel/upload.rs`

- `prepare_shape_upload<F>`, `prepare_material_upload<F>` and
  `prepare_mask_upload<F>` take their `LayerIndex` (a `Local` set at
  registration, or a generic parameter `L`) and act on
  `layers.layers[index]`. `admit`, `clusters`, the `RemeshBudget` and the
  side-channel refill logic are otherwise unchanged.
- The budget is shared across layers, oldest cluster first as today. A
  per-layer share can come later if a cheap layer starves behind terrain.

### `src/voxel/mesh.rs`

- `dispatch_voxel_compute_passes_batched` loops over layers. Pass1 and pass3
  are unchanged: they read the slot's distances, material bytes and seam ranks
  and know nothing about layers. A layer without a material channel binds a
  dummy material buffer, as a chunk without one does today.

### `src/voxel/draw.rs`, `src/voxel/raster.rs`

- `ArenaDraws` gains `layer: LayerIndex`. `prepare_voxel_view_draws` culls per
  layer; transparent layers sort back to front.
- `VoxelViewDraws` keeps one list per view; `voxel_raster_pass` takes the
  `LayerPass` it is drawing and skips the others' draws.
- `DrawVariant` (`masked`, `far`) is palette-specific. It becomes part of
  `PaletteRaster`'s key narrowing; other rasters have one variant.

### `src/voxel/pipeline/mod.rs`

- `VoxelRasterKey` gains `layer: LayerIndex`, and `arm`/`normal` move into a
  raster-owned `ShaderDefVal` list (`LayerRaster::specialize`). The pipeline
  cache is keyed by the full key, so each layer compiles its own variants.
- `raster_pipeline_descriptor` reads blend, cull, depth write and the
  fragment shader from the layer's raster. The vertex state is shared.
- `VoxelMaterialBindGroup` becomes `PaletteRaster`'s bind group; a generic
  `LayerMaterialBindGroups` resource holds one per layer, refreshed by
  `LayerRaster::material_bind_group`.

### `src/raycast/gpu.rs`, `src/raycast/cursor.rs`

- Read the `QUERYABLE` layer's arenas. No change to the query shader.

### `src/voxel/{stats,inspect,telemetry}.rs`

- Per-layer totals; the HUD in the sandbox shows the terrain's as now.

### Shaders

- `voxel_raster.wesl`: split into `voxel_raster_vertex.wesl` (shared
  `vs_main`, the `Vertex` unpacking) and `palette_raster.wesl` (today's
  `fs_main`, `fs_prepass`). A layer's shader imports the vertex library.
- `surface_nets_pass1.wesl`, `surface_nets_pass3.wesl`,
  `draw_compaction.wesl`, `cursor_query.wesl`: unchanged.

## Phases

Each phase builds and passes the existing tests and examples on its own.

1. **Layer 0 without behavior change.** Introduce `LayerIndex`, `VoxelLayers`
   with one entry, and loop the upload, dispatch, draw and raster systems
   over it. `VoxelChunkArenaSet` is renamed; nothing else moves.
2. **`LayerRaster`.** Extract `PaletteRaster`; key pipelines by layer; add the
   transparent raster pass slot. Still one layer.
3. **`SurfaceLayer` and `add_surface_layer`.** `TarasaurPlugin` registers
   `TerrainLayer`. `register_field_upload` and `ShapeFieldRegistered` are
   removed. An `examples/layers.rs` registers a second opaque layer (a
   flat-colored "water table" from a trivial field) to prove the path.
4. **Sub-surfaces.** The `(MainEntity, SubSurface)` slot key, `surfaces()`,
   and the disappearing-sub-surface release. The example gets a two-owner
   zone field.
5. **`UPLOAD_LOD`.** Downsample on upload. Measure arena memory with the
   domain layer at `Low` against the terrain at `High`.
6. **Domain Expance switches over** and deletes its CPU mesher.

Phases 1 and 2 are the bulk of the diff and carry no new feature, which is
why they go first: they are reviewable as pure refactors.

## Costs

- **Memory.** A layer owns arenas. At `High`, one arena is about three times
  `ArenaConfig::max_buffer_bytes` (768 MB by default). `UPLOAD_LOD = Low`
  makes a coarse layer negligible; a second `High` layer is a real decision
  for the app, which is why `ArenaConfig` is per layer.
- **Remesh budget.** Shared, so a layer that changes every frame (a held
  claim brush) competes with terrain edits. The sub-surface split also means
  a chunk with `n` owners re-meshes `n` volumes.
- **Transparent sorting** is per chunk, back to front. Two intersecting
  translucent sub-surfaces in one chunk are not sorted against each other.
  Premultiplied or additive blending hides this; plain alpha does not.
- **No per-layer LOD bands.** A layer is meshed at every chunk the loaders
  stream, clamped by `UPLOAD_LOD`. A layer that only matters near the
  camera cannot yet say so.

## Open questions

1. **Is the material channel per layer or global?** Terrain is the only
   material user today. Per layer is cleaner (`type Material`) and costs
   nothing for `NoChannel` layers; it does mean `MaterialFieldPlugin<M>` is
   replaced by naming `MaterialField<M>` in `TerrainLayer`. Suggest a
   `TarasaurPlugin::with_material::<M>()` to keep the current ergonomics.
2. **`SubSurface` width.** `u8` allows 255 sub-surfaces per chunk per layer.
   The vertex format is untouched either way, since the raster reads the id
   from a `[slot] -> SubSurface` table rather than from the vertex. Enough
   for owners; not enough if someone wants per-entity surfaces.
3. **Should transparent layers get the depth prepass?** No in this draft:
   they do not write depth. They do test against terrain depth, so a shell
   under ground is hidden, which is what Domain Expance wants.
4. **Naming.** "Layer" is used here; "surface" reads better in some places
   (`SubSurface`, `add_surface_layer`). One of them should win.
5. **Is `UPLOAD_LOD` the right knob,** or should a layer's field simply be
   stored at a capped LOD by the loader? Storing it coarse touches
   persistence and downsampling on load; uploading it coarse touches only
   extraction. This draft takes the smaller change.

## Out of scope

- Meshing a layer from anything but a signed distance (density fields that
  are not distances will shade with flat normals; `Claim::density_for` is a
  step, so the domain layer should quantize a smoothed distance instead).
- CPU readback of a layer's mesh (collision for a layer). The queryable
  layer answers rays; nothing answers for the others.
- More than one queryable layer.
