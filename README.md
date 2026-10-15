# Tarasaur

Smooth, editable voxel terrain for [Bevy](https://bevyengine.org) 0.20,
meshed on the GPU.

Tarasaur aims to be a minimal smooth-voxel engine with everything needed to
build a continuous, persistent environment. It should run well on low-end
hardware such as a Steam Deck.

## Features

- **GPU surface nets.** Compute shaders mesh chunks in two passes. Only
  chunks whose data changed are re-meshed.
- **Streaming LOD.** `ChunkLoader` bands (64³ / 32³ / 16³ / 4³ voxels per
  chunk) follow the camera, with hysteresis. Seams between LODs are
  stitched.
- **Generic voxel fields.** A field implements `VoxelField` (how to build,
  version, save and upload it) and `Field<T>` (how to edit it), and
  `app.add_voxel_field::<F>()` wires it into editing, saving and its GPU
  channel (shape, material, mask, or CPU-only). Built-in fields:
  - `SDF`: narrow-band signed distances that define the shape
  - `VisibilityField`: an optional per-voxel mask, discarded per pixel
  - `MaterialField<M>`: per-voxel ids from your own material enum
- **Editing.** World-space sphere and box brushes (`WorldEditor`), exact
  CSG spheres (`SdfCsgSphere`), smooth and flatten sculpts
  (`SdfSculptSphere`), and material painting by palette id
  (`MaterialPaint`). Edits that span chunks apply to every chunk they
  touch.
- **Persistence.** Edited chunks autosave to `./saves/chunks/` (`SaveDirectory`). Saves load
  before generators, and lower LODs are downsampled from the save.
- **Triplanar texturing with PBR lighting.** A `TexturePalette` albedo
  array, plus an optional ARM array (ambient occlusion, roughness,
  metallic), blended between voxel materials. Lighting is Bevy's own PBR,
  so terrain gets the scene's lights, shadows from other meshes, fog, light
  probes and tonemapping.
- **Terrain ray casts.** A CPU `TerrainRay` answers in the same frame at any
  LOD. A GPU query pipeline reads results back a few frames later.
- **Batched drawing.** Chunks live in large per-LOD GPU arenas. Each view
  frustum-culls them and draws each arena with one
  `multi_draw_indexed_indirect`.

## Quick start

```rust
use bevy::prelude::*;
use tarasaur::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_plugins((DefaultPlugins, TarasaurPlugin::default()));

    // Terrain comes from generators: a pure function of (chunk, LOD) that
    // runs off-thread. This one returns world-space signed distances.
    app.world_mut()
        .resource_mut::<ChunkGeneratorRegistry>()
        .register_sdf_distance(flat_ground);

    app.add_systems(Startup, setup).run();
}

/// `lod.volume()` values, x fastest, then y, then z. Negative = solid.
fn flat_ground(chunk: IVec3, lod: LOD) -> Vec<f32> {
    let n = lod.size() as usize;
    let base_y = chunk.y as f32 * CHUNK_SIZE;
    (0..lod.volume())
        .map(|i| base_y + ((i / n) % n) as f32 * lod.voxel_size() - 4.0)
        .collect()
}

fn setup(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 20.0, 30.0).looking_at(Vec3::ZERO, Vec3::Y),
        ChunkLoader::default(), // streams chunks around the camera
    ));
    commands.spawn(DirectionalLight::default());
}
```

The built-in cursor brush is on by default: hold left mouse to apply
`CursorBrush::tool` (`BrushTool::Dig`, `Build`, `Smooth`, `Flatten` or
`Paint`), and hold Ctrl as well to swap dig and build. Set its `radius`,
`strength`, `material` (a palette id, used by Paint and by Build) and
`flatten` (`FlattenPlane::Level` or `Slope`) at runtime; the app binds
the keys. F8 cycles the ray backend (Auto / CPU / GPU). Turn the brush off
with `CursorBrush { enabled: false, ..default() }` and edit from
`CursorHit` yourself.

### Materials and textures

1. Implement `VoxelMaterial` (plus `AccumulateExt` and `BlendExt`) on your
   enum. Each id is a palette layer.
2. Add `MaterialFieldPlugin::<YourMaterial>::new()`.
3. Register a `u8` generator under
   `std::any::type_name::<MaterialField<YourMaterial>>()`.
4. Build a palette with `PaletteBuilder`, then insert
   `ActivePalette(handle)`. Add `.with_arm(handle)` for an ARM texture
   array (R = AO, G = roughness, B = metallic, linear format, same size and
   layer count as the albedo). Without one, every material is matte and
   non-metallic. `PaletteMaterial::with_roughness` / `with_metallic`
   override the texture per material.

`examples/common/mod.rs` does all four steps for a heightfield terrain,
including a generated ARM array.

The terrain doesn't cast shadows yet, and it isn't drawn in Bevy's
prepasses, so screen-space effects (SSAO, SSR) don't see it.

### Editing

```rust
use tarasaur::prelude::*; // WorldEditor, SDF

fn dig(mut sdf: WorldEditor<SDF, f32>) {
    sdf.accumulate_sphere(Vec3::new(0.0, 4.0, 0.0), 3.0, 0.5); // + = remove
}
```

You can only edit a chunk at `MaxEditLod` (High by default). An edit that
lands on a coarser chunk pins it, upgrades it, and retries until the chunk
is ready.

## How it works

```
ChunkLoader ─► plan/apply ops ─► load task (save | generator) ─► install fields + LOD
                                                                     │
   edits (WorldEditor, CSG) ─► fields ─► async JFA reinit (sign → distance)
                                                                     │
Last: changed chunks ─► padded volumes (own + 2-voxel neighbour apron), in parallel
                                                                     │
Render world: upload into per-LOD arena slot ─► pass1 (active cells) ─► pass3 (verts,
              normals, materials, seams, indices) ─► per-view cull ─► multi-draw (Bevy PBR)
```

| Module | Role |
|---|---|
| `chunk` | Chunk entities, `ChunkManager`, `ChunkLoader` LOD bands, the spawn and despawn planner |
| `field` | Field traits, `SDF`, `VisibilityField`, `MaterialField`, `EditableChunks`; `field::apron` builds padded volumes |
| `grid` | Voxel grid layout: flattening, the apron, sampling between grids |
| `storage::loading` | Off-thread load tasks; installs fields and LOD atomically |
| `field::systems`, `ops`, `editor` | Edit messages, brushes, async jump-flood reinit |
| `storage::persistence` | Save files, save cache, autosave, downsampling |
| `field::surface` | Tags the chunks that can produce geometry (`HasSurface`) |
| `voxel::extract` | Change-driven builds of padded volumes for the render world |
| `voxel::arena` | GPU slot and vertex-range allocation per LOD |
| `voxel::upload`, `mesh`, `raster` | Arena uploads, the surface-nets dispatch, the raster pass |
| `voxel::pipeline` | Meshing and query pipelines; PBR raster pipelines specialized per view; palette bind group |
| `voxel::draw` | Per-view frustum culling and draw-argument compaction |
| `raycast::gpu`, `cursor`, `terrain_ray` | GPU and CPU ray casts, `CursorHit`, the default brush |
| `texture_palette` | `TexturePalette` asset, builder, validation |
| `num` | The checked numeric casts (the crate denies `as`) |

Every module starts with a `//!` summary. `cargo doc --open` shows the public
API; add `--document-private-items` for the internal modules too. Each item has
one path in its module, and `tarasaur::prelude` gathers the common ones.

## Tuning

Engine settings are fields on `TarasaurPlugin`:

```rust
app.add_plugins(TarasaurPlugin {
    save_dir: "my_world/chunks".into(),
    depth_prepass: true,
    ..default()
});
```

| Field | Default | Effect |
|---|---|---|
| `max_edit_lod` | `High` | The only LOD that is edited and saved |
| `save_dir` | `./saves/chunks` | Where chunk saves are read and written |
| `autosave_secs` | 2.0 | Autosave period |
| `edit_pin_secs` | 2.0 | How long an edited chunk stays at max LOD after the last edit |
| `max_chunk_ops_per_frame` | 64 | Chunk spawns and LOD retargets applied per frame |
| `max_loads_in_flight` | 64 | Load tasks running at once |
| `remesh_cells_per_frame` | 4M | Cells re-meshed per frame. Past the budget, chunks keep their old mesh |
| `arenas` | see below | GPU arena sizes and how many each LOD may have |
| `depth_prepass` | `false` | Draw terrain depth before shading, so each pixel is shaded once. Slower on tile-based GPUs (+3.4 ms on Adreno); bench it on yours |
| `guard_telemetry` | debug builds | Log pass3's range guard (should never fire) |

Each becomes a resource of the same name (`MaxEditLod`, `SaveDirectory`, ...);
the main-world ones can be changed at runtime. Two more are plain resources:
`CursorQueryMode` (cursor ray backend, `Auto` uses the CPU) and
`ShowChunkBounds` (insert it to draw chunk bounds, colored by LOD).

The arenas size themselves from the device's
`max_storage_buffer_binding_size` and `max_buffer_size`, capped by
`ArenaConfig::max_buffer_bytes` (256 MiB by default; each arena takes about
3x that). Bevy requests the adapter's own limits, so `WgpuSettings::limits`
doesn't change them.

## Examples

```sh
cargo run --release --example sandbox
```

| Example | What it shows |
|---|---|
| `sandbox` | Streamed, textured terrain. WASD, Space and Shift to fly, hold right mouse to look. Left mouse applies the brush; 1-5 pick dig, build, smooth, flatten or paint, Q/E the material, scroll the radius, Ctrl + scroll the strength, R toggles level or slope flattening |
| `lod_streaming` | The smallest streaming setup |
| `terrain_test` | Seam check and a valley carved across a chunk border |
| `edit_sphere` | Hand-spawned chunks and a resizable SDF sphere |
| `cursor_collision_test` | GPU ray-query hits on a sphere |
| `normal_check` | Headless comparison of normal-estimation methods |
| `bench` | A repeatable benchmark (see below) |

## Development

`nix develop` gives you the toolchain, Vulkan, Tracy and RenderDoc. Without
Nix, use a recent Rust (edition 2024) and Bevy's
[Linux dependencies](https://github.com/bevyengine/bevy/blob/main/docs/linux_dependencies.md).

The crate denies panics, `as` casts, indexing and unchecked arithmetic
(`[lints.clippy]` in `Cargo.toml`), and runs clippy pedantic and nursery at
deny level.

```sh
cargo clippy --all-targets
cargo test -j 2      # full parallelism can run out of memory
```

## Benchmarking and profiling

The bench runs repeatable phases (load, hold, spin, fly, edit, box, query,
reslot, visibility) and writes its results to `bench_results/`. The last four also
check correctness: box brushes stay inside the box, GPU ray queries return
one result per ray and agree with the CPU ray, and chunks that lose and
regain their arena slot get their materials back, and visibility masks on
the GPU match the edited and removed `VisibilityField`s (both read back from
the GPU).
A failed check makes the bench exit with an error.

`--small-arenas` shrinks the GPU arenas so High terrain spans several, and
also checks GPU queries against terrain past a LOD's first arena. More
arenas cost frame time (~7-11% in hold, spin and fly), so only compare runs
made with the same setting.

```sh
cargo run --release --example bench -- --label baseline [--msaa 1|2|4|8] [--max-loads N] [--depth-prepass] [--small-arenas]
cargo run --release --example bench -- --compare bench_results/A.summary.csv bench_results/B.summary.csv
```

CPU profiling uses [Tracy](https://github.com/wolfpld/tracy). The dev shell
pins Tracy 0.14.1, the protocol version Bevy 0.20's client speaks
(`nix/tracy.nix`). The app only records while Tracy is connected, so
connect when the part you care about starts:

```sh
cargo run --release --features tracy --example bench   # or --example sandbox
tracy                                                  # GUI: click Connect
tracy-capture -o hold.tracy -s 8                       # or headless, 8 seconds
tracy-csvexport hold.tracy > hold.csv                  # per-zone totals
```

GPU frame captures use RenderDoc, which is also in the dev shell. The bench
captures one frame in the middle of its `hold` phase, and the summary
script replays that frame headlessly:

```sh
renderdoccmd capture -w target/release/examples/bench --renderdoc --label rdc
RDC=bench_results/rdc_frameN.rdc RDC_OUT=summary.txt qrenderdoc --python tools/renderdoc_summary.py
```

On tiling GPUs (Adreno, Mali), RenderDoc times each draw by replaying it on
its own. Use its timings for proportions, and the bench's per-pass times
for totals.
