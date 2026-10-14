# Tarasaur

Bevy Smooth Geomotry Voxel Engain

## Description

Tarasaur ames to be a minal smoth voxel Engain with every thing needed to build out a contuines percent evenrent. The enverment must be preformant and optomized to run low end hardware like a steamdeck.

## Features:
* Voxel Field Traits.
* Prebuilt Fields: 
    * SDF
    * Material
    * Visibility
- Dynmic LOD around the camera.
* GPU Driven SurfaceNets implemtation.
- TriPlanerProjection for material texture.
- TerrainCollitions

## Benchmarking & profiling

Repeatable benchmark (load / hold / spin / fly / edit phases, results in `bench_results/`):

```sh
cargo run --release --example bench -- --label baseline
cargo run --release --example bench -- --compare bench_results/A.summary.csv bench_results/B.summary.csv
```

CPU profiling with [Tracy](https://github.com/wolfpld/tracy). The dev shell provides
Tracy 0.14.1, the version Bevy 0.20's client speaks (`nix/tracy.nix`). The app only
records while Tracy is connected, so connect whenever the part you care about starts:

```sh
cargo run --release --features tracy --example bench   # or --example sandbox
tracy                                                  # GUI: click Connect
tracy-capture -o hold.tracy -s 8                       # or headless, 8 seconds
tracy-csvexport hold.tracy > hold.csv                  # per-zone totals
```

GPU frame capture with RenderDoc (also in the dev shell). The bench grabs one frame
in the middle of its `hold` phase, and the summary script replays it headlessly:

```sh
renderdoccmd capture -w target/release/examples/bench --renderdoc --label rdc
RDC=bench_results/rdc_frameN.rdc RDC_OUT=summary.txt qrenderdoc --python tools/renderdoc_summary.py
```

On tiling GPUs (Adreno, Mali) RenderDoc's per-draw timings come from replaying each
draw on its own, so trust them for proportions, and the bench's per-pass times for totals.
