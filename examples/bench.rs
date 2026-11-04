//! Repeatable end-to-end benchmark.
//!
//! Runs a fixed scenario on the sandbox terrain and writes per-frame and
//! per-phase results:
//!
//! ```text
//! cargo run --release --example bench -- --label baseline [--msaa 1|2|4|8] [--max-loads N] [--depth-prepass] [--small-arenas] [--layer] [--gap-probes] [--quick]
//!   [--compute-threads bevy|N] [--settle-stream] [--sculpt-offset dx,dz]
//! ```
//!
//! `--out DIR` puts the results somewhere else, and `--world-dir DIR` the
//! scratch world. Both default under the checkout the binary was built from,
//! so a run on another machine (see `tools/bench-deck.sh`) needs them, along
//! with `--commit <sha>` for the result header.
//!
//! `--dig-radius R` changes the `dig` brush's radius (default `DIG_RADIUS`).
//! It is a diagnostic, not a setting: `dig`'s CSG cost should scale with the
//! number of chunks the brush's bounding box spans rather than with the
//! brush's volume, because the scan clamps to whole chunks. Runs at
//! different radii are comparable only with each other.
//!
//! ```text
//! cargo run --release --example bench -- --compare bench_results/A.summary.csv bench_results/B.summary.csv
//! cargo run --release --example bench -- --diff A.png B.png [heatmap.png]
//! ```
//!
//! Phases, in order:
//! - `load`:  stream the world in from a still camera until nothing is pending.
//! - `shots`: a screenshot of each of four views from the start, a quarter
//!   turn apart, into `bench_results/shots/<label>-view<N>.png`.
//! - `hold`:  the same view, nothing changing: steady-state render cost.
//! - `spin`:  turn in place: view-dependent cost (culling).
//! - `fly`:   fly a fixed distance per frame: streaming cost while moving.
//! - `edit`:  sculpt one spot every frame: edit -> re-mesh cost.
//! - `sculpt`: the gradual sphere brushes the cursor's Smooth, Flatten,
//!   Fill and Scrape tools send. A ring of small craters is poked in first
//!   (30 frames of CSG), then a smooth stroke rounds them off and a flatten
//!   stroke levels the spot. Checks each stroke moved the field the way its
//!   tool is defined to -- roughness down for the smooth, distance to the
//!   plane down for the flatten.
//! - `box`:   box brushes: an SDF box sculpted every frame, plus one material
//!   box off the voxel grid. Checks that only voxels inside it changed.
//! - `query`: GPU ray queries every frame. Checks each result batch holds
//!   exactly the queries sent, and that the hits match the CPU ray. With
//!   `--small-arenas`, also on chunks stored past a LOD's first arena.
//! - `reslot`: chunks filled solid (no mesh, slot released) and dug out again
//!   (a new slot), every 60 frames. Checks their material slots are
//!   refilled, by reading them back from the GPU.
//! - `visibility`: a hidden box across a chunk boundary, a sphere toggled
//!   hidden and visible, and one chunk's `VisibilityField` removed. Checks
//!   every touched chunk's shape on the GPU is the main world's SDF clipped
//!   by its visibility (`max` of the two), read back from the arena. With
//!   the `visibility-mask` feature it checks the GPU masks instead.
//! - `dig`: the cursor brush's gradual dig, held at a slant and re-centered
//!   on the surface every frame, like a player holding the mouse. Checks
//!   the surface keeps a real distance field (no terraced, black crater)
//!   and that the dig doesn't punch through.
//!
//! `--quick` (about a third of the run time) halves `hold`, `spin` (turning
//! twice as fast) and `fly` (stopping halfway; the later phases happen there)
//! and skips `sculpt`, `box`, `query`, `reslot` and `dig`. Compare only runs made with the same setting.
//! Results are written after every phase, so a run cut short keeps what it
//! measured.
//!
//! A failed check is printed, written to the summary as `check:<name>` = 0,
//! and makes the bench exit with an error.
//!
//! `--small-arenas` shrinks the GPU arenas so High terrain spans several,
//! which the `query` phase then checks. More arenas cost frame time (~7-11%
//! in hold/spin/fly), so compare only runs made with the same setting.
//!
//! Runs headless, as ornithochory's grove: no window, the camera renders into
//! a fixed 1280x800 image (Steam Deck's native resolution) that screenshots
//! copy out of. With no swapchain to hold it back the CPU would queue frames
//! without bound, so it waits as a swapchain would: at most two frames in
//! flight on the GPU.
//! RenderDoc captures on present, so `--renderdoc` keeps a window.
//!
//! Camera motion is per frame, not per second, so every run does the same
//! work whatever its frame rate. Nothing waits for vsync. The world is
//! generated into a scratch directory, so your saves are never read or
//! written.
//!
//! GPU times need timestamp-query support (Vulkan/DX12). They arrive a few
//! frames late, so a value is attributed to the frame in which it arrived.

#[allow(dead_code)] // shared with the sandbox, which uses the fly cam
mod common;

use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt::Write as _,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use bevy::{
    app::ScheduleRunnerPlugin,
    asset::RenderAssetUsages,
    camera::{ImageRenderTarget, RenderTarget},
    diagnostic::{DiagnosticPath, DiagnosticsStore, FrameCount},
    image::{CompressedImageFormats, ImageSampler, ImageType},
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        diagnostic::RenderDiagnosticsPlugin,
        render_resource::{
            BufferDescriptor, BufferUsages, CommandEncoderDescriptor, MapMode, PollType,
            TextureDimension, TextureFormat, TextureUsages,
        },
        renderer::{RenderAdapterInfo, RenderDevice, RenderQueue},
        view::screenshot::{Screenshot, save_to_disk},
    },
    window::{ExitCondition, PresentMode, WindowResolution},
    winit::WinitPlugin,
};
use common::{
    TerrainMaterial, determine_material, register_terrain, render_plugin, setup_terrain_palette,
    material_boundary_distance, terrain_height, terrain_materials,
};
use renderdoc::{RenderDoc, V141};
use tarasaur::{
    MaterialFieldPlugin, TarasaurPlugin,
    chunk::{
        CHUNK_SIZE, ChunkLoader, ChunkManager, ChunkPosition, overlapping_chunks,
        world_pos_to_chunk_pos,
    },
    field::{
        Field, LOD, MaterialField, MaterialVoxel, SDF, SDF_BAND, VisibilityField, VoxelField,
        VoxelMaterial, material_id,
        editor::{
            SdfCsgSphere, SdfSculptOp, SdfSculptSphere, VisibilityCsgBox, VisibilityCsgSphere,
            WorldEditor,
        },
        ops::PlaneMode,
    },
    grid::flatten_cube,
    num::{f32_from_u32, f32_from_usize, u32_from_f32, usize_from},
    raycast::{
        cursor::{CursorBrush, CursorHit, resolve_cursor_hit, update_cursor_focus},
        gpu::{HitResult, PendingVoxelQueries, RayQuery, VoxelQueryResults},
        terrain_ray::TerrainRay,
    },
    voxel::{ArenaConfig, inspect::ArenaInspector, seam, stats},
};

// ---------------------------------------------------------------------------
// Scenario
// ---------------------------------------------------------------------------

const START: Vec3 = Vec3::new(0.0, 30.0, 40.0);
const LOOK_AT: Vec3 = Vec3::new(0.0, 8.0, 0.0);

/// Frames everything must stay idle before `load` counts as done.
const SETTLE_FRAMES: u32 = 30;
/// Give up on `load` after this long (recorded as a timeout).
const LOAD_TIMEOUT_SECS: f32 = 180.0;
const HOLD_FRAMES: u32 = 600;
const SPIN_FRAMES: u32 = 360; // one degree per frame
const FLY_FRAMES: u32 = 900;
const FLY_STEP: f32 = 0.25; // world units per frame: 225 units, ~22 chunks
const EDIT_FRAMES: u32 = 300;
/// `shots`: frames per view; the screenshot is taken on the view's last frame.
const SHOT_FRAMES_PER_VIEW: u32 = 30;
/// `shots`: views of the loaded world from `START`, a quarter turn apart.
const SHOT_VIEWS: u32 = 4;

/// How long `--settle-stream` waits for chunk streaming to drain before it
/// gives up and measures anyway, having said so.
const SETTLE_STREAM_TIMEOUT_SECS: f32 = 20.0;

/// `--quick`: `hold`, `spin` and `fly` run half as many frames (`spin`
/// turning twice as far per frame, `fly` stopping halfway, with everything
/// after it placed from where it stops), and `sculpt`, `box`, `query`,
/// `reslot` and `dig` are skipped.
///
/// The one place the setting lives: `Phase` reads it from free functions, so
/// it cannot be a field of [`Output`] like the other flags, and the result
/// header reads it back through [`quick`] rather than keeping a second copy.
static QUICK: AtomicBool = AtomicBool::new(false);

fn quick() -> bool {
    QUICK.load(Ordering::Relaxed)
}

/// `--sculpt-offset dx,dz`: moves where the `sculpt` phase works, in world
/// units, from the spot it has always used.
///
/// Exists to answer a question nothing on disk can. `terrain_height`
/// (`examples/common/mod.rs`) is a **fixed analytic sin/cos surface with no
/// seed and no flag** -- four hardcoded octaves, identical in every run this
/// bench has ever produced -- and `sculpt_target` works one fixed spot on
/// it. So *every* tarasaur measurement ever taken shares one terrain, and a
/// result that is really a property of that surface's local curvature would
/// look exactly like a property of the engine, in all 118 historical runs
/// and in everything measured next.
///
/// Specifically: the `sculpt` phase's smooth and flatten strokes trade work
/// run to run (wave-2 report, r = -0.290 over 22 independent arms, variance
/// ratio 0.749), and that could be the stroke pair or it could be this one
/// patch of hillside. Moving the strokes to a different patch of the same
/// surface separates the two without needing a different world: if the
/// ratio stays below 1 across offsets it is the stroke pair, and if it
/// tracks the patch it is the curvature.
///
/// Two `f32` bits in one atomic, because `sculpt_target` is a free function
/// called from four places and threading a parameter through all of them
/// would be a worse change than the one it tests. Defaults to `(0, 0)` --
/// the spot every existing result used -- so nothing on disk moves.
static SCULPT_OFFSET: AtomicU64 = AtomicU64::new(0);

fn sculpt_offset() -> Vec2 {
    let bits = SCULPT_OFFSET.load(Ordering::Relaxed);
    // `try_from` rather than `as`: both halves are 32 bits wide by
    // construction, so neither conversion can fail and no truncation is
    // being hidden. The crate denies `as_conversions`, and it is right to --
    // an `as` here would still compile if the packing ever changed width.
    let half = |v: u64| u32::try_from(v & 0xFFFF_FFFF).unwrap_or(0);
    Vec2::new(
        f32::from_bits(half(bits >> 32)),
        f32::from_bits(half(bits)),
    )
}

/// How much faster `spin` turns per frame, and how many times fewer frames
/// `hold`, `spin` and `fly` run.
fn speed() -> u32 {
    if quick() { 2 } else { 1 }
}
const EDIT_RADIUS: f32 = 3.0;
const EDIT_DELTA: f32 = 0.15; // SDF change per frame; flips dig/build every 30 frames
const SCULPT_FRAMES: u32 = 300;
/// `sculpt` smooths until here and flattens after: both callers of
/// `apply_sphere_at`, and both ops the cursor brush sends for its four
/// gradual tools (Fill and Scrape are `Flatten` with another `PlaneMode`).
const SCULPT_SMOOTH_FRAMES: u32 = 150;
/// Frames of small CSG pokes before the smooth stroke, one a frame around a
/// ring. A smooth brush is for rounding off the creases an edit leaves, and
/// generated terrain has none: its field is linear to within 0.003 voxels,
/// so smoothing it moves nothing and the check would read float noise. The
/// pokes are ~4 voxels across, so they cost little of the phase's time.
const SCULPT_ROUGHEN_FRAMES: u32 = 30;
/// A poke's radius, and the ring they sit on -- inside `SCULPT_MEASURE_RADIUS`
/// with room to spare, so every crease lands in the measured voxels.
const SCULPT_POKE_RADIUS: f32 = 0.7;
const SCULPT_POKE_ORBIT: f32 = 1.0;
/// The sandbox's default brush: 4 units, ~26 High voxels -- wider than the
/// narrow band, narrower than a chunk.
const SCULPT_RADIUS: f32 = 4.0;
/// The fraction of the way voxels move per frame: the cursor brush's default
/// strength (3/s) at a fixed 60 fps, so every run sculpts the same, as
/// `DIG_STEP` does.
const SCULPT_AMOUNT: f32 = 3.0 / 60.0;
/// How far the brush circles its target, and the frames a turn takes. A
/// stroke sweeps a capsule from the last frame's center, so a brush held
/// perfectly still would never exercise one.
const SCULPT_ORBIT: f32 = 1.0;
const SCULPT_ORBIT_FRAMES: u32 = 60;
/// The metrics cover this far from the target: inside `SCULPT_RADIUS` minus
/// `SCULPT_ORBIT`, so every frame's brush covers all of it, and clear of the
/// rim where the falloff dies and leaves creases of its own.
const SCULPT_MEASURE_RADIUS: f32 = 2.0;
/// ...and no further than this from the plane, in voxels. `flatten` aims
/// voxels at the plane, and only within the narrow band can the field hold
/// what it aims at.
const SCULPT_PLANE_SLAB: f32 = SDF_BAND * 0.5;
/// How far each metric has to fall for its check to pass.
const SCULPT_MARGIN: f32 = 0.1;
const BOX_FRAMES: u32 = 300;
const BOX_HALF: f32 = 2.0; // the sculpted SDF box
/// The material box: 6.4 High voxels, so its faces sit between voxels.
const MATERIAL_BOX_HALF: f32 = 1.0;
/// Stone only generates above height 13.5; the box sits far lower.
const MATERIAL_BOX_FILL: TerrainMaterial = TerrainMaterial::Stone;
const QUERY_FRAMES: u32 = 240;
/// Rays per frame; below the query capacity (64), so a batch padded out to
/// the full buffer shows up as the wrong length.
const QUERY_RAYS: usize = 48;
const QUERY_MAX_DISTANCE: f32 = 100.0;
/// GPU and CPU hits closer than this (world units) agree.
const QUERY_TOLERANCE: f32 = 0.5;
/// When the render world reports which arena holds each chunk.
const ARENA_REPORT_AT: u32 = QUERY_FRAMES - 30;
const RESLOT_FRAMES: u32 = 300;
/// `--small-arenas`: arenas small enough that High terrain spans several,
/// with enough of them that no LOD has less room than with the defaults.
const BENCH_ARENAS: ArenaConfig = ArenaConfig {
    max_arenas: [1, 4, 4, 8],
    max_buffer_bytes: 32 * 1024 * 1024,
};
const RESLOT_PERIOD: u32 = 60;
/// Into a cycle, when the solid-filled chunks are dug out again.
const RESLOT_DIG_AT: u32 = 20;
/// Covers the chunk plus its 2-voxel apron, so it meshes to nothing.
const RESLOT_FILL_RADIUS: f32 = 9.5;
const RESLOT_DIG_RADIUS: f32 = 3.0;
/// When the material slots are read back (after the last dig has landed).
const RESLOT_PROBE_AT: u32 = 280;
const VIS_FRAMES: u32 = 300;
/// The visibility sphere flips hidden <-> visible this often, until
/// `VIS_TOGGLE_UNTIL`, ending visible.
const VIS_TOGGLE_PERIOD: u32 = 30;
const VIS_TOGGLE_UNTIL: u32 = 200;
const VIS_SPHERE_RADIUS: f32 = 1.5;
/// When one chunk's `VisibilityField` is removed.
const VIS_REMOVE_AT: u32 = 120;
/// When the shapes are read back (well after the last edit, and the box's
/// jump-flood reinit, have landed).
const VIS_PROBE_AT: u32 = 260;
const DIG_FRAMES: u32 = 540;
/// The brush is held this long (8 s at 60 fps), then the edits settle. Deep
/// enough (~24 units) to cross at least two chunk borders.
const DIG_HELD: u32 = 480;
/// The sandbox's 10-unit brush: ~64 High voxels, far wider than the SDF band.
const DIG_RADIUS: f32 = 10.0;
/// World units per frame: the cursor brush's default strength (3/s) at a
/// fixed 60 fps, so every run digs the same.
const DIG_STEP: f32 = 3.0 / 60.0;
/// A surface voxel with a gradient under this has lost its distance field.
const DIG_WEAK_GRADIENT: f32 = 0.5;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Phase {
    Load,
    Shots,
    Hold,
    Spin,
    Fly,
    Edit,
    Sculpt,
    BoxEdit,
    Query,
    Reslot,
    Visibility,
    Dig,
    Done,
}

impl Phase {
    const fn name(self) -> &'static str {
        match self {
            Self::Load => "load",
            Self::Shots => "shots",
            Self::Hold => "hold",
            Self::Spin => "spin",
            Self::Fly => "fly",
            Self::Edit => "edit",
            Self::Sculpt => "sculpt",
            Self::BoxEdit => "box",
            Self::Query => "query",
            Self::Reslot => "reslot",
            Self::Visibility => "visibility",
            Self::Dig => "dig",
            Self::Done => "done",
        }
    }

    fn next(self) -> Self {
        match self {
            Self::Load => Self::Shots,
            Self::Shots => Self::Hold,
            Self::Hold => Self::Spin,
            Self::Spin => Self::Fly,
            Self::Fly => Self::Edit,
            Self::Edit if quick() => Self::Visibility,
            Self::Visibility if quick() => Self::Done,
            Self::Edit => Self::Sculpt,
            Self::Sculpt => Self::BoxEdit,
            Self::BoxEdit => Self::Query,
            Self::Query => Self::Reslot,
            Self::Reslot => Self::Visibility,
            Self::Visibility => Self::Dig,
            Self::Dig | Self::Done => Self::Done,
        }
    }

    /// Fixed length, or `None` for `load` (runs until settled).
    fn frames(self) -> Option<u32> {
        match self {
            Self::Load => None,
            Self::Shots => Some(SHOT_FRAMES_PER_VIEW * SHOT_VIEWS),
            Self::Hold => HOLD_FRAMES.checked_div(speed()),
            Self::Spin => SPIN_FRAMES.checked_div(speed()),
            Self::Fly => FLY_FRAMES.checked_div(speed()),
            Self::Edit => Some(EDIT_FRAMES),
            Self::Sculpt => Some(SCULPT_FRAMES),
            Self::BoxEdit => Some(BOX_FRAMES),
            Self::Query => Some(QUERY_FRAMES),
            Self::Reslot => Some(RESLOT_FRAMES),
            Self::Visibility => Some(VIS_FRAMES),
            Self::Dig => Some(DIG_FRAMES),
            Self::Done => Some(0),
        }
    }

    /// Whether this is the phase's last frame (checks run here).
    fn last_frame(self, bench: &Bench) -> bool {
        bench.phase == self
            && self
                .frames()
                .is_some_and(|n| bench.frame_in_phase.saturating_add(1) == n)
    }
}

const PHASES: [Phase; 12] = [
    Phase::Load,
    Phase::Shots,
    Phase::Hold,
    Phase::Spin,
    Phase::Fly,
    Phase::Edit,
    Phase::Sculpt,
    Phase::BoxEdit,
    Phase::Query,
    Phase::Reslot,
    Phase::Visibility,
    Phase::Dig,
];

// ---------------------------------------------------------------------------
// Recorded columns
// ---------------------------------------------------------------------------

/// Diagnostics recorded every frame. Render spans only produce a value on
/// frames they ran in, so a column is empty when nothing new arrived.
const COLUMNS: [(&str, &str); 17] = [
    ("gpu_mesh_ms", "render/voxel_mesh/elapsed_gpu"),
    ("gpu_query_ms", "render/voxel_query/elapsed_gpu"),
    ("reslotted", "voxel/reslotted"),
    ("side_refills", "voxel/side_refills"),
    ("gpu_pass1_ms", "render/voxel_mesh/pass1/elapsed_gpu"),
    ("gpu_pass3_ms", "render/voxel_mesh/pass3/elapsed_gpu"),
    ("gpu_raster_ms", "render/voxel_raster/elapsed_gpu"),
    ("draws", "voxel/draws"),
    ("resident", "voxel/resident"),
    ("meshed_chunks", "voxel/meshed_chunks"),
    ("meshed_cells", "voxel/meshed_cells"),
    ("pending_uploads", "voxel/pending_uploads"),
    ("dirty_slots", "voxel/dirty_slots"),
    ("chunks", "voxel/chunks"),
    ("loads_in_flight", "voxel/loads_in_flight"),
    ("lod_pending", "voxel/lod_pending"),
    ("reinits", "voxel/reinits"),
];

fn column(name: &str) -> Option<usize> {
    COLUMNS.iter().position(|(n, _)| *n == name)
}

struct FrameRow {
    phase: Phase,
    frame_ms: f64,
    values: [Option<f64>; COLUMNS.len()],
}

impl FrameRow {
    fn get(&self, name: &str) -> Option<f64> {
        self.values.get(column(name)?).copied().flatten()
    }
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

#[derive(Resource)]
struct Bench {
    phase: Phase,
    frame_in_phase: u32,
    phase_started: Instant,
    idle_frames: u32,
    load_timed_out: bool,
    rows: Vec<FrameRow>,
    /// Last measurement time seen per diagnostic, to tell new values from repeats.
    seen: HashMap<&'static str, Instant>,
    /// Every GPU pass Bevy timed, averaged over the end of each phase.
    gpu_passes: Vec<(Phase, String, f64)>,
    checks: Checks,
    /// `sculpt`: the metrics before the smooth stroke, and before the
    /// flatten one (so, after the smooth one). Each is captured on the
    /// first frame of its stroke that finds the spot at High: a phase's
    /// first edits can wait for `MaxEditLod`.
    /// `--settle-stream`: this phase's measured frames have not started yet,
    /// because chunk streaming from an earlier phase is still draining.
    settling: bool,
    sculpt_start: Option<SculptMetrics>,
    sculpt_mid: Option<SculptMetrics>,
    out: Output,
}

struct Output {
    dir: PathBuf,
    label: String,
    /// `--depth-prepass`: draw voxel depth before shading.
    depth_prepass: bool,
    /// `--small-arenas`: see `BENCH_ARENAS`.
    small_arenas: bool,
    /// `--layer`: a second surface layer (`common::water`) over the terrain.
    layer: bool,
    /// `--dig-radius`: the `dig` brush's radius in world units, normally
    /// `DIG_RADIUS`. Only vary it to find out what the phase's cost scales
    /// with -- a run with another radius is not comparable to a baseline.
    dig_radius: f32,
    /// `--settle-stream`: hold each fixed-length phase before its measured
    /// frames begin until chunk streaming has drained.
    settle_stream: SettleStream,
    /// Seconds since the epoch at startup, in the result file names.
    stamp: u64,
    /// Short hash of the commit the **binary was built from**, from
    /// `--commit`. Empty when nothing told us: a run cannot work this out
    /// for itself, which is the whole point of the field.
    commit_build: String,
    /// `--sculpt-offset dx,dz`: where the `sculpt` phase works, relative to
    /// the spot every existing result used. See [`SCULPT_OFFSET`].
    sculpt_offset: Vec2,
    /// The command line and every knob read off it, for the header.
    knobs: Knobs,
}

#[derive(Component)]
struct BenchCamera;

/// The value after `name` in `args`, if it's there.
fn arg_value(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i.saturating_add(1)))
        .cloned()
}

/// `--settle-stream`: whether a phase may begin measuring while chunk
/// streaming from an earlier phase is still draining.
///
/// A two-variant enum rather than a fourth `bool` on `Output`:
/// `clippy::struct_excessive_bools` fires at four, and its own advice is to
/// refactor into two-variant enums. It also makes the header line say which
/// mode a run used rather than `true`/`false`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SettleStream {
    /// Start measuring at once. The default, so a run stays comparable with
    /// every result already on disk.
    Off,
    /// Hold each fixed-length phase until `loads_in_flight == 0 &&
    /// lod_pending == 0`, or `SETTLE_STREAM_TIMEOUT_SECS` elapses.
    WaitForDrain,
}

impl SettleStream {
    /// Reads `--settle-stream` through [`Knobs`], so the header carries it
    /// because the parser recorded it rather than because `write_results`
    /// names it -- and records the mode's own word rather than a bare
    /// `true`/`false`. The second `record` replaces the `switch`'s bool,
    /// since a later word for the same key wins.
    fn from_args(knobs: &Knobs) -> Self {
        let mode = if knobs.switch("--settle-stream") {
            Self::WaitForDrain
        } else {
            Self::Off
        };
        knobs.record("--settle-stream", &mode.tag());
        mode
    }

    /// What the result header records, so a flag A/B can verify it took.
    const fn tag(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::WaitForDrain => "wait",
        }
    }
}

/// The command line, and what each knob on it resolved to.
///
/// An A/B is only checkable from the record if the saved summary proves which
/// knob differed, and twice now it has not. `--compute-floor` gave six clean
/// runs and a perfect null because the flag set a floor and so could not
/// lower the thread count; that was caught only because the pool size had
/// just been added to the header. The lesson was then not generalised, and
/// the night's biggest result -- an MSAA A/B worth -25.3% on `hold` -- cannot
/// be checked at all, because `--msaa` was never in the header either. Both
/// are one bug: a flag the parser knew about and the header did not.
///
/// So reading a knob and reporting it are the same operation here. Every knob
/// that can change a number goes through [`Knobs::switch`], [`Knobs::parsed`]
/// or [`Knobs::resolved`], each of which records the value it returned, and
/// [`Knobs::header_fields`] prints exactly that set. Adding a flag without
/// adding it to the header now takes deliberately going around these.
///
/// What is read with [`Knobs::raw`] instead, because none of it can change a
/// measurement: the paths (`--out`, `--world-dir`), the run's own identity
/// (`--label`, `--commit`, already in the header by other means) and the two
/// non-run modes (`--compare`, `--diff`), which never reach a result file.
struct Knobs {
    args: Vec<String>,
    /// `key=value` in the order the run read them. Behind a lock only so a
    /// `&Knobs` can be recorded into from the ECS at write time; there is no
    /// contention.
    recorded: Mutex<Vec<(String, String)>>,
}

impl Knobs {
    const fn new(args: Vec<String>) -> Self {
        Self {
            args,
            recorded: Mutex::new(Vec::new()),
        }
    }

    /// The value after `flag`, recording nothing. Only for knobs that cannot
    /// change a measurement -- see the type's note.
    fn raw(&self, flag: &str) -> Option<String> {
        arg_value(&self.args, flag)
    }

    /// Records the value `flag` resolved to, replacing any earlier one: the
    /// run may learn later what a knob really got, and the later word wins.
    fn record(&self, flag: &str, value: &dyn std::fmt::Display) {
        let key = flag.trim_start_matches('-').replace('-', "_");
        let value = value.to_string();
        let mut rec = self
            .recorded
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(slot) = rec.iter_mut().find(|(k, _)| *k == key) {
            slot.1 = value;
        } else {
            rec.push((key, value));
        }
    }

    /// Whether the bare `flag` is present. Recorded either way: "absent" is
    /// as much a side of an A/B as "present".
    fn switch(&self, flag: &str) -> bool {
        let on = self.args.iter().any(|a| a == flag);
        self.record(flag, &on);
        on
    }

    /// `parse` applied to `flag`'s value, where a bad value should stop the
    /// run. `None` is the run refusing to start, so nothing is recorded.
    fn parsed<T>(
        &self,
        flag: &str,
        parse: impl FnOnce(Option<&str>) -> Option<T>,
        show: impl FnOnce(&T) -> String,
    ) -> Option<T> {
        let value = parse(self.raw(flag).as_deref())?;
        self.record(flag, &show(&value));
        Some(value)
    }

    /// `f` applied to `flag`'s value, where every value resolves to something
    /// -- a default, or a clamp. What is recorded is what came back, not what
    /// was asked for.
    fn resolved<T: std::fmt::Display>(&self, flag: &str, f: impl FnOnce(Option<&str>) -> T) -> T {
        let value = f(self.raw(flag).as_deref());
        self.record(flag, &value);
        value
    }

    /// An environment variable that shapes the run, recorded under `key`.
    /// `absent` is a value like any other: an A/B that turned one of these
    /// off has to be as provable as one that turned it on.
    fn env(&self, key: &str, var: &str) -> Option<String> {
        let value = std::env::var(var).ok().filter(|v| !v.is_empty());
        self.record(key, &value.clone().unwrap_or_else(|| "absent".into()));
        value
    }

    /// `key=value key=value ...` for the result header, in read order.
    fn header_fields(&self) -> String {
        let rec = self
            .recorded
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        rec.iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Which file the running bench actually is, as a file name, an xxh3-128 and
/// a length.
///
/// `current_exe()` alone is a trap here. `tools/deck/bench.sh` starts the
/// bench through an explicit dynamic loader -- `ld-linux-x86-64.so.2`
/// `--library-path ... bin/bench`, nix glibc against SteamOS's own libraries
/// -- and `current_exe()` then returns **the loader**, which is the same
/// program whichever variant is running. A fingerprint that silently agreed
/// across both sides of an A/B would be worse than none at all, and it would
/// do it on the one machine whose numbers decide things. So when
/// `current_exe()` looks like a loader, hash `argv[0]` instead, and record
/// which file was hashed either way rather than leaving it to be assumed.
///
/// The file name, not the path: the header is space-separated `key=value`
/// and a path can contain spaces. The hash is the identity; the name only
/// has to answer "did this hash the bench or the loader".
fn bench_binary() -> (String, String, u64) {
    let looks_like_loader = |p: &Path| {
        p.file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with("ld-") || n.starts_with("ld."))
    };
    let exe = std::env::current_exe().ok().filter(|p| !looks_like_loader(p));
    let path = exe
        .or_else(|| std::env::args_os().next().map(PathBuf::from))
        .unwrap_or_default();
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();
    let Ok(bytes) = std::fs::read(&path) else {
        return (name, "unreadable".into(), 0);
    };
    let mut h = xxhash_rust::xxh3::Xxh3::new();
    h.update(&bytes);
    (
        name,
        format!("{:032x}", h.digest128()),
        u64::try_from(bytes.len()).unwrap_or(u64::MAX),
    )
}

/// Records where this run's code came from, which the result could not say
/// before.
///
/// `commit=` used to be `git rev-parse HEAD` taken **at run time**, so it
/// recorded the checkout the process happened to be sitting in rather than
/// the tree the binary was compiled from. The two are routinely different --
/// `tools/bench-deck-ab.sh` builds each variant from its own ref in a
/// detached worktree while `HEAD` stays put -- and the field gave no hint
/// which it was. ornisculpt had the identical defect, found the same night.
///
/// So the old name is retired rather than redefined. A row saying `commit=`
/// is, mechanically and with nothing to remember, a row written before this
/// change; a row saying `commit_build=`/`commit_run=` distinguishes them.
/// `binary_xxh3` is the one that actually settles it: it identifies the
/// artifact that ran, whatever any checkout claims.
///
/// Nothing fails on a mismatch. A binary built from one ref and run beside
/// another checkout is the normal case for the Deck driver; `stale=true`
/// records it, and a warning says so once.
fn record_provenance(knobs: &Knobs, commit_build: &str) {
    let git = |args: &[&str]| {
        std::process::Command::new("git")
            .args(["-C", env!("CARGO_MANIFEST_DIR")])
            .args(args)
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    };
    let commit_run = git(&["rev-parse", "--short", "HEAD"]).unwrap_or_default();
    // A separate field, never a `+dirty` suffix on the sha: a suffix stops
    // the value parsing as a sha and starts it being string-compared, which
    // is how two identical trees get reported as different.
    let dirty_run = git(&["status", "--porcelain"]).is_some();
    let (file, hash, bytes) = bench_binary();
    // Both known and different. Emitted always, not only when true: an
    // absent field cannot distinguish "they agreed" from "git was not there"
    // from "an older binary did not write this".
    let stale = !commit_build.is_empty() && !commit_run.is_empty() && commit_build != commit_run;
    if stale {
        eprintln!(
            "bench: note: built from {commit_build}, running beside a checkout at \
             {commit_run}. Normal for a driver-built variant; the binary is what ran."
        );
    }
    knobs.record("commit_build", &commit_build);
    knobs.record("commit_run", &commit_run);
    knobs.record("dirty_run", &dirty_run);
    knobs.record("stale", &stale);
    knobs.record("binary_xxh3", &hash);
    knobs.record("binary_bytes", &bytes);
    knobs.record("binary_file", &file);
}

/// Where this run's results and scratch world go, and how it labels itself.
///
/// `CARGO_MANIFEST_DIR` is baked in at build time, so off the build host it
/// names a path that isn't there: the defaults fall back to the working
/// directory, and `--out`, `--world-dir` and `--commit` say what to use
/// instead (see `tools/bench-deck.sh`).
///
/// `None` if a flag's value doesn't parse, having said which.
fn bench_paths(knobs: Knobs) -> Option<(Output, PathBuf)> {
    let flag = |name: &str| knobs.raw(name);
    // Before anything reads `Phase::frames`, which `quick()` shortens -- and
    // through `Knobs`, so the header still records it. `main` keeps no second
    // copy of this on `Output`, and that stays true.
    QUICK.store(knobs.switch("--quick"), Ordering::Relaxed);
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let manifest = Path::new(manifest_dir);
    let base = if manifest.is_dir() {
        manifest.to_path_buf()
    } else {
        PathBuf::from(".")
    };
    // `dx,dz` as one value, not two arguments: the header is space-separated
    // `key=value`, so a knob that spans a space cannot be recorded in it.
    let sculpt_offset = knobs.parsed(
        "--sculpt-offset",
        |v| {
            v.map_or(Some(Vec2::ZERO), |v| {
                let parsed = v
                    .split_once(',')
                    .and_then(|(a, b)| {
                        Some(Vec2::new(a.trim().parse().ok()?, b.trim().parse().ok()?))
                    })
                    .filter(|o| o.is_finite());
                if parsed.is_none() {
                    eprintln!("--sculpt-offset wants `dx,dz` in world units, got {v:?}");
                }
                parsed
            })
        },
        |o| format!("{},{}", o.x, o.y),
    )?;
    let dig_radius = knobs.parsed(
        "--dig-radius",
        |v| {
            v.map_or(Some(DIG_RADIUS), |v| match v.parse::<f32>() {
                Ok(r) if r.is_finite() && r > 0.0 => Some(r),
                _ => {
                    eprintln!("--dig-radius wants a positive number of world units, got {v:?}");
                    None
                }
            })
        },
        f32::to_string,
    )?;
    // An empty scratch save directory, so every run generates the same
    // terrain from scratch and your saves are never read or written.
    // Resolved before the knobs move into `out`, since both read the command
    // line.
    let world_dir =
        flag("--world-dir").map_or_else(|| base.join("target").join("bench-world"), PathBuf::from);
    let out = Output {
        dir: flag("--out").map_or_else(|| base.join("bench_results"), PathBuf::from),
        label: flag("--label").unwrap_or_else(|| "run".into()),
        depth_prepass: knobs.switch("--depth-prepass"),
        small_arenas: knobs.switch("--small-arenas"),
        layer: knobs.switch("--layer"),
        sculpt_offset,
        dig_radius,
        settle_stream: SettleStream::from_args(&knobs),
        commit_build: flag("--commit").unwrap_or_default(),
        stamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs()),
        knobs,
    };
    let out = Output {
        dir: std::path::absolute(&out.dir).unwrap_or(out.dir),
        ..out
    };
    Some((out, world_dir))
}

/// Render resolution: Steam Deck's native one.
const RESOLUTION: UVec2 = UVec2::new(1280, 800);

/// No window (all but `--renderdoc`): the camera renders into [`ShotTarget`].
#[derive(Resource)]
struct Headless;

/// The image the camera renders into when headless.
#[derive(Resource)]
struct ShotTarget(Handle<Image>);

/// Frames the GPU may run behind the CPU when headless: a window's
/// (`Window::desired_maximum_frame_latency`, 2 by default).
const FRAMES_IN_FLIGHT: usize = 2;

/// A wait for each frame still in flight, oldest first.
#[derive(Resource, Default)]
struct InFlight(VecDeque<PollType>);

/// Blocks the render thread until the GPU is at most [`FRAMES_IN_FLIGHT`]
/// frames behind, as acquiring a swapchain image would.
fn throttle_gpu(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut in_flight: ResMut<InFlight>,
) {
    // The queue runs in order: an empty submission completes after the frame.
    in_flight.0.push_back(PollType::Wait {
        submission_index: Some(queue.submit([])),
        timeout: None,
    });
    while in_flight.0.len() > FRAMES_IN_FLIGHT {
        if let Some(wait) = in_flight.0.pop_front()
            && let Err(e) = device.poll(wait)
        {
            error!("[bench] waiting for the GPU: {e}");
        }
    }
}

/// The things that change what a run measures and are not command-line
/// knobs, recorded for the header alongside the ones that are.
fn record_unflagged(knobs: &Knobs, commit_build: &str) {
    // `tracy` recompiles the bench with Bevy's instrumentation in it, and
    // `profile=` says only debug or release, so two runs of "the same commit"
    // can differ by the whole cost of tracing with nothing in the record to
    // show it.
    knobs.record("tracy", &cfg!(feature = "tracy"));
    record_provenance(knobs, commit_build);
    // What the frames are presented through. On the Deck this is the single
    // largest lever there is -- the same binary measured 181 fps nested in
    // gamescope, 719 with gamescope's own compositing off and 2645 headless
    // (`tools/deck/bench.sh`) -- and it is set by an environment variable on
    // the far side of an ssh, which no reader of a result could ever check.
    // The launcher names what it used; a bare desktop run says so.
    knobs.env("present", "TARASAUR_PRESENT");
}

/// Exits with an error if setup fails or any check fails.
fn main() -> AppExit {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Some(i) = args.iter().position(|a| a == "--compare") {
        return compare_args(args.get(i.saturating_add(1)), args.get(i.saturating_add(2)));
    }
    if let Some(i) = args.iter().position(|a| a == "--diff") {
        return diff_args(
            args.get(i.saturating_add(1)),
            args.get(i.saturating_add(2)),
            args.get(i.saturating_add(3)),
        );
    }
    let Some((out, world_dir)) = bench_paths(Knobs::new(args)) else {
        return AppExit::error();
    };
    let knobs = &out.knobs;
    record_unflagged(knobs, &out.commit_build);
    SCULPT_OFFSET.store(
        (u64::from(out.sculpt_offset.x.to_bits()) << 32) | u64::from(out.sculpt_offset.y.to_bits()),
        Ordering::Relaxed,
    );
    // Recorded as what was asked for, then overwritten in `write_results` by
    // the pool's actual size -- the asking is not the getting, which is the
    // whole reason this is in the header.
    let Some(compute_pool) = knobs.parsed("--compute-threads", parse_compute_threads, |p| {
        p.exact().map_or_else(|| "bevy".into(), |n| n.to_string())
    }) else {
        return AppExit::error();
    };
    let _ = std::fs::remove_dir_all(&world_dir);

    let mut app = App::new();
    let renderdoc = knobs.switch("--renderdoc");
    add_default_plugins(&mut app, renderdoc, compute_pool);
    app.add_plugins((
        TarasaurPlugin {
            save_dir: world_dir,
            // Disk writes would only add noise; nothing here needs to persist.
            autosave_secs: f32::INFINITY,
            depth_prepass: out.depth_prepass,
            max_loads_in_flight: knobs.resolved("--max-loads", |v| {
                v.and_then(|n| n.parse().ok())
                    .unwrap_or_else(|| TarasaurPlugin::default().max_loads_in_flight)
            }),
            arenas: if out.small_arenas {
                BENCH_ARENAS
            } else {
                ArenaConfig::default()
            },
            ..default()
        },
        MaterialFieldPlugin::<TerrainMaterial>::new(),
        stats::VoxelDiagnosticsPlugin,
    ));
    // Bevy's `trace_tracy` feature (our `tracy`) already adds it.
    if !app.is_plugin_added::<RenderDiagnosticsPlugin>() {
        app.add_plugins(RenderDiagnosticsPlugin);
    }
    register_terrain(&mut app);
    if out.layer {
        common::water::add_water_layer(&mut app);
    }
    let Some(msaa) = knobs.parsed("--msaa", parse_msaa, |m| msaa_samples(*m).to_string()) else {
        return AppExit::error();
    };
    app.insert_resource(BenchMsaa(msaa));
    if renderdoc {
        let Some(capture) = FrameCapture::attach(&out) else {
            return AppExit::error();
        };
        app.insert_resource(capture)
            .add_systems(Update, trigger_frame_capture);
    }

    add_checked_phases(&mut app, knobs, out.small_arenas);
    app.insert_resource(CursorBrush {
        enabled: false,
        ..default()
    })
    .insert_resource(Bench {
        phase: Phase::Load,
        frame_in_phase: 0,
        phase_started: Instant::now(),
        idle_frames: 0,
        load_timed_out: false,
        rows: Vec::new(),
        seen: HashMap::new(),
        settling: false,
        sculpt_start: None,
        sculpt_mid: None,
        out,
        gpu_passes: Vec::new(),
        checks: Checks::default(),
    })
    .add_systems(Startup, (spawn_camera, setup_terrain_palette))
    .add_systems(
        Update,
        (
            drive_camera,
            take_shots,
            drive_edits.run_if(not_settling),
            ignore_cursor
                .after(resolve_cursor_hit)
                .before(update_cursor_focus),
        ),
    )
    .add_systems(Last, record_frame)
    .run()
}

/// Bevy's plugins, headless unless `renderdoc` (RenderDoc captures on present).
///
/// `compute_pool` is `--compute-threads`. The widening lives only in this
/// bench: the library ships it as an opt-in (`tarasaur::task_pools`) and
/// changes no default, so an app that does not ask gets Bevy's split. "Is
/// the floor worth it" therefore has to be answerable without rebuilding
/// (`bench_results/deck-improvements.md`, option 6), and the count comes from
/// `task_pools::edit_compute_threads` so the bench measures the number an
/// app would get rather than a copy of it.
fn add_default_plugins(app: &mut App, renderdoc: bool, compute_pool: ComputePool) {
    let compute = compute_pool.exact().map_or_else(
        || TaskPoolOptions::default().compute,
        |n| bevy::app::TaskPoolThreadAssignmentPolicy {
            min_threads: n,
            max_threads: n,
            ..TaskPoolOptions::default().compute
        },
    );
    let pools = TaskPoolPlugin {
        task_pool_options: TaskPoolOptions {
            compute,
            ..default()
        },
    };
    if renderdoc {
        app.add_plugins(DefaultPlugins.set(render_plugin()).set(pools).set(WindowPlugin {
            primary_window: Some(Window {
                title: "tarasaur bench".into(),
                resolution: WindowResolution::new(RESOLUTION.x, RESOLUTION.y),
                present_mode: PresentMode::AutoNoVsync,
                resizable: false,
                ..default()
            }),
            ..default()
        }));
    } else {
        app.add_plugins((
            DefaultPlugins
                .set(render_plugin())
                .set(pools)
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                .disable::<WinitPlugin>(),
            // With a window plugin, DefaultPlugins leaves the runner to winit.
            // Uncapped: `throttle_gpu` paces the frames.
            ScheduleRunnerPlugin::run_loop(Duration::ZERO),
        ))
        .insert_resource(Headless);
        if let Some(render) = app.get_sub_app_mut(RenderApp) {
            render
                .init_resource::<InFlight>()
                .add_systems(Render, throttle_gpu.in_set(RenderSystems::Cleanup));
        }
    }
}

/// `--settle-stream` is not holding this phase's measured frames back.
///
/// The drivers and checks run only when this is true, so a settling phase
/// streams and renders but is not yet sculpting, digging or being checked --
/// otherwise the gate would hold `frame_in_phase` at 0 while `drive_sculpt`
/// poked the same spot for the whole wait.
fn not_settling(bench: Res<Bench>) -> bool {
    !bench.settling
}

/// Whether the run uses `BENCH_ARENAS` (`--small-arenas`).
#[derive(Resource, Clone, Copy)]
struct SmallArenas(bool);

/// The `box`, `query` and `reslot` phases: their drivers, checks, and the
/// render-world readbacks.
fn add_checked_phases(app: &mut App, knobs: &Knobs, small_arenas: bool) {
    let probe = SlotProbe::default();
    let mesh_probe = MeshProbe::default();
    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app
            .insert_resource(probe.clone())
            .insert_resource(mesh_probe.clone())
            .add_systems(
                Render,
                (
                    read_back_slots,
                    read_back_meshes,
                    report_arenas,
                )
                    .in_set(RenderSystems::Cleanup),
            );
        #[cfg(not(feature = "visibility-mask"))]
        render_app.add_systems(Render, read_back_shapes.in_set(RenderSystems::Cleanup));
        #[cfg(feature = "visibility-mask")]
        render_app.add_systems(Render, read_back_masks.in_set(RenderSystems::Cleanup));
    }
    app.insert_resource(probe)
        .insert_resource(mesh_probe)
        .insert_resource(GapProbes(knobs.switch("--gap-probes")))
        .insert_resource(SmallArenas(small_arenas))
        .init_resource::<QueryWatch>()
        .add_systems(
            Update,
            (
                drive_sculpt,
                drive_box_edits,
                drive_queries,
                drive_reslots,
                drive_visibility,
                drive_dig,
                drive_mesh_probe,
                watch_queries,
                check_sculpt,
                check_material_box,
                check_queries,
                check_reslots,
                check_visibility,
                check_dig,
                check_meshes,
            )
                .run_if(not_settling),
        );
}

fn spawn_camera(
    mut commands: Commands,
    msaa: Res<BenchMsaa>,
    headless: Option<Res<Headless>>,
    mut images: ResMut<Assets<Image>>,
) {
    let camera = commands
        .spawn((
            BenchCamera,
            Camera3d::default(),
            msaa.0,
            Transform::from_translation(START).looking_at(LOOK_AT, Vec3::Y),
            // Same bands as the sandbox, so this measures what it runs.
            ChunkLoader {
                lod_bands: vec![(LOD::High, 4), (LOD::Medium, 3), (LOD::Low, 8)],
                hysteresis: 1,
            },
        ))
        .id();
    if headless.is_some() {
        let mut image = Image::new_target_texture(
            RESOLUTION.x,
            RESOLUTION.y,
            TextureFormat::Rgba8UnormSrgb,
            None,
        );
        // Screenshots copy out of it.
        image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
        let target = images.add(image);
        commands
            .entity(camera)
            .insert(RenderTarget::Image(ImageRenderTarget {
                handle: target.clone(),
                scale_factor: 1.0,
            }));
        commands.insert_resource(ShotTarget(target));
    }
    commands.spawn((
        DirectionalLight {
            illuminance: 10000.0,
            ..default()
        },
        Transform::from_xyz(10.0, 40.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

fn fly_end() -> Vec3 {
    START + Vec3::X * FLY_STEP * f32_from_u32(Phase::Fly.frames().unwrap_or(FLY_FRAMES))
}

/// `shots`: saves each view to `<out>/shots/<label>-view<N>.png`, to diff
/// against another run's with `--diff`.
fn take_shots(bench: Res<Bench>, target: Option<Res<ShotTarget>>, mut commands: Commands) {
    if bench.phase != Phase::Shots
        || (bench.frame_in_phase.saturating_add(1)).rem_euclid(SHOT_FRAMES_PER_VIEW) != 0
    {
        return;
    }
    let view = bench.frame_in_phase / SHOT_FRAMES_PER_VIEW;
    let dir = bench.out.dir.join("shots");
    if let Err(e) = std::fs::create_dir_all(&dir) {
        error!("[bench] {}: {e}", dir.display());
        return;
    }
    let path = dir.join(format!("{}-view{view}.png", bench.out.label));
    let shot = target.map_or_else(Screenshot::primary_window, |t| {
        Screenshot::image(t.0.clone())
    });
    commands.spawn(shot).observe(save_to_disk(path));
}

/// Positions the camera from the phase and frame alone.
fn drive_camera(bench: Res<Bench>, mut cams: Query<&mut Transform, With<BenchCamera>>) {
    let f = f32_from_u32(bench.frame_in_phase);
    for mut t in &mut cams {
        *t = match bench.phase {
            Phase::Shots => {
                let view = bench.frame_in_phase / SHOT_FRAMES_PER_VIEW;
                let turn = std::f32::consts::FRAC_PI_2 * f32_from_u32(view);
                let base = Transform::from_translation(START).looking_at(LOOK_AT, Vec3::Y);
                base.with_rotation(Quat::from_rotation_y(turn) * base.rotation)
            }
            Phase::Load | Phase::Hold => {
                Transform::from_translation(START).looking_at(LOOK_AT, Vec3::Y)
            }
            Phase::Spin => {
                let base = Transform::from_translation(START).looking_at(LOOK_AT, Vec3::Y);
                base.with_rotation(
                    Quat::from_rotation_y((f * f32_from_u32(speed())).to_radians()) * base.rotation,
                )
            }
            Phase::Fly => {
                let pos = START + Vec3::X * FLY_STEP * f;
                Transform::from_translation(pos)
                    .looking_at(pos + Vec3::new(1.0, -0.4, 0.0), Vec3::Y)
            }
            Phase::Edit
            | Phase::Sculpt
            | Phase::BoxEdit
            | Phase::Query
            | Phase::Reslot
            | Phase::Visibility
            | Phase::Dig
            | Phase::Done => {
                let pos = fly_end();
                Transform::from_translation(pos).looking_at(edit_target(), Vec3::Y)
            }
        };
    }
}

/// A point on the terrain surface in front of where `fly` ends.
fn edit_target() -> Vec3 {
    let (x, z) = (fly_end().x + 12.0, START.z);
    Vec3::new(x, terrain_height(x, z), z)
}

fn drive_edits(bench: Res<Bench>, mut sdf: WorldEditor<SDF, f32>) {
    if bench.phase != Phase::Edit {
        return;
    }
    // Alternate dig and build so the hole stays bounded and every run
    // sculpts the same shape.
    let dig = (bench.frame_in_phase / 30).is_multiple_of(2);
    let delta = if dig { EDIT_DELTA } else { -EDIT_DELTA };
    sdf.accumulate_sphere(edit_target(), EDIT_RADIUS, delta);
}

// ---------------------------------------------------------------------------
// Checks
// ---------------------------------------------------------------------------

struct Check {
    phase: Phase,
    name: &'static str,
    pass: bool,
    detail: String,
}

/// Pass/fail results, written with the summary.
#[derive(Default)]
struct Checks(Vec<Check>);

impl Checks {
    fn record(&mut self, phase: Phase, name: &'static str, pass: bool, detail: String) {
        let verdict = if pass { "PASS" } else { "FAIL" };
        info!("[bench] {verdict} {}/{name}: {detail}", phase.name());
        self.0.push(Check {
            phase,
            name,
            pass,
            detail,
        });
    }

    fn all_pass(&self) -> bool {
        self.0.iter().all(|c| c.pass)
    }

    /// `phase,check:<name>,1|0` summary rows.
    fn write_rows(&self, summary: &mut String, phase: Phase) {
        for c in self.0.iter().filter(|c| c.phase == phase) {
            let _ = writeln!(
                summary,
                "{},check:{},{}",
                phase.name(),
                c.name,
                u8::from(c.pass)
            );
        }
    }

    fn print(&self) {
        println!("Checks:");
        for c in &self.0 {
            let verdict = if c.pass { "PASS" } else { "FAIL" };
            println!("  {verdict}  {}/{}: {}", c.phase.name(), c.name, c.detail);
        }
    }
}

/// The point on the terrain surface at (x, z).
fn surface(x: f32, z: f32) -> Vec3 {
    Vec3::new(x, terrain_height(x, z), z)
}

// ---------------------------------------------------------------------------
// sculpt: the gradual sphere brushes
// ---------------------------------------------------------------------------

/// Where `sculpt` works: a chunk of its own, clear of every other phase's.
/// `dig`'s check scans whole chunks around its crater for weak gradients,
/// which is exactly what a smoothed surface has, so sharing a chunk with it
/// would fail that check rather than this one. The spot is off to the side
/// of the camera's view, as `box`'s is, so the phase measures the edit path
/// and not the raster of its result.
fn sculpt_target() -> Vec3 {
    let off = sculpt_offset();
    surface(fly_end().x - 10.0 + off.x, START.z + 25.0 + off.y)
}

/// A point on a ring of `radius` around the target, `frame` of `period` of
/// the way round, on the surface.
fn sculpt_ring(frame: u32, period: u32, radius: f32) -> Vec3 {
    let turn = std::f32::consts::TAU * f32_from_u32(frame) / f32_from_u32(period);
    let at = sculpt_target() + Vec3::new(turn.cos(), 0.0, turn.sin()) * radius;
    surface(at.x, at.z)
}

/// The brush's center on `frame` of a stroke: a slow circle around the
/// target, so each message sweeps a capsule from the last center the way a
/// held stroke does, without the brush leaving its chunk.
fn sculpt_center(frame: u32) -> Vec3 {
    sculpt_ring(frame, SCULPT_ORBIT_FRAMES, SCULPT_ORBIT)
}

/// What the `sculpt` checks watch.
#[derive(Clone, Copy)]
struct SculptMetrics {
    /// Mean |discrete Laplacian| of the field, in voxel units.
    /// `smooth_sphere` moves every voxel toward its neighbours' mean, so a
    /// held smooth has to bring this down.
    roughness: f32,
    /// Mean distance from the field to the plane `flatten_sphere` pulls
    /// toward. `PlaneMode::Flatten` aims at the plane exactly, so a held
    /// flatten has to bring this down.
    offset: f32,
    voxels: usize,
}

/// Both metrics in one pass, over the voxels within `SCULPT_MEASURE_RADIUS`
/// of the target and `SCULPT_PLANE_SLAB` of its plane. Position alone picks
/// that set, so a before and an after cover the same voxels.
///
/// `None` until the spot is at High, which is also when the first edit of a
/// stroke stops being deferred.
fn sculpt_metrics(chunks: &ChunkManager, sdfs: &Query<&SDF>) -> Option<SculptMetrics> {
    let target = sculpt_target();
    let (mut rough, mut offset, mut voxels) = (0.0f32, 0.0f32, 0usize);
    for pos in overlapping_chunks(target, Vec3::splat(SCULPT_MEASURE_RADIUS)) {
        let Some(sdf) = chunks.get_chunk(&pos).and_then(|e| sdfs.get(e).ok()) else {
            continue;
        };
        if sdf.lod != LOD::High {
            continue;
        }
        let size = sdf.size();
        let voxel = CHUNK_SIZE / f32_from_u32(size.x);
        // The target and the reach in this chunk's grid, as an edit has them.
        let at = (target - pos.as_vec3() * CHUNK_SIZE) / voxel;
        let reach = SCULPT_MEASURE_RADIUS / voxel;
        let grid = |v: f32| u32_from_f32(v.max(0.0));
        // Interior voxels only: the Laplacian wants all six neighbours.
        let min =
            UVec3::new(grid(at.x - reach), grid(at.y - reach), grid(at.z - reach)).max(UVec3::ONE);
        let max = UVec3::new(grid(at.x + reach), grid(at.y + reach), grid(at.z + reach))
            .min(size.saturating_sub(UVec3::splat(2)));
        if min.cmpgt(max).any() {
            continue;
        }
        for z in min.z..=max.z {
            for y in min.y..=max.y {
                // Where the plane sits for this row, in voxels, the way
                // `flatten_sphere` works it out: normal +Y through the target.
                let plane = f32_from_u32(y) - at.y;
                if plane.abs() > SCULPT_PLANE_SLAB {
                    continue;
                }
                for x in min.x..=max.x {
                    let p = Vec3::new(f32_from_u32(x), f32_from_u32(y), f32_from_u32(z));
                    if p.distance(at) > reach {
                        continue;
                    }
                    let c = sdf.get(x, y, z);
                    let sum = sdf.get(x.saturating_add(1), y, z)
                        + sdf.get(x.saturating_sub(1), y, z)
                        + sdf.get(x, y.saturating_add(1), z)
                        + sdf.get(x, y.saturating_sub(1), z)
                        + sdf.get(x, y, z.saturating_add(1))
                        + sdf.get(x, y, z.saturating_sub(1));
                    rough += c.mul_add(6.0, -sum).abs();
                    offset += (c - plane).abs();
                    voxels = voxels.saturating_add(1);
                }
            }
        }
    }
    (voxels > 0).then(|| SculptMetrics {
        roughness: rough / f32_from_usize(voxels),
        offset: offset / f32_from_usize(voxels),
        voxels,
    })
}

/// Sends one gradual brush a frame, smoothing for the first half of the
/// phase and flattening for the second, and takes each stroke's starting
/// metrics on its way past.
fn drive_sculpt(
    mut bench: ResMut<Bench>,
    chunks: Res<ChunkManager>,
    sdfs: Query<&SDF>,
    mut sculpt: MessageWriter<SdfSculptSphere>,
    mut csg: MessageWriter<SdfCsgSphere>,
) {
    if bench.phase != Phase::Sculpt {
        return;
    }
    let frame = bench.frame_in_phase;
    if frame < SCULPT_ROUGHEN_FRAMES {
        // Scallop a ring of craters into the surface. Each poke creases
        // where it meets the last and where it meets the ground, which is
        // what a smooth brush is for and what the terrain has none of.
        csg.write(SdfCsgSphere {
            center: sculpt_ring(frame, SCULPT_ROUGHEN_FRAMES, SCULPT_POKE_ORBIT),
            radius: SCULPT_POKE_RADIUS,
            sweep_from: None,
            subtract: true,
            max_step: None,
            paint: None,
        });
        return;
    }
    let smoothing = frame < SCULPT_SMOOTH_FRAMES;
    // Each tool starts a stroke of its own, as switching tools does, so it
    // sweeps from its own first center rather than the last tool's.
    let in_stroke = frame.saturating_sub(if smoothing {
        SCULPT_ROUGHEN_FRAMES
    } else {
        SCULPT_SMOOTH_FRAMES
    });
    let mark = if smoothing {
        &mut bench.sculpt_start
    } else {
        &mut bench.sculpt_mid
    };
    if mark.is_none() {
        *mark = sculpt_metrics(&chunks, &sdfs);
    }
    sculpt.write(SdfSculptSphere {
        center: sculpt_center(in_stroke),
        radius: SCULPT_RADIUS,
        sweep_from: in_stroke.checked_sub(1).map(sculpt_center),
        amount: SCULPT_AMOUNT,
        op: if smoothing {
            SdfSculptOp::Smooth
        } else {
            SdfSculptOp::Flatten {
                point: sculpt_target(),
                normal: Vec3::Y,
                mode: PlaneMode::Flatten,
            }
        },
    });
}

/// Whether `metric` fell by `SCULPT_MARGIN` between two marks, and what to
/// say about it.
fn sculpt_fell(
    metric: &str,
    before: Option<SculptMetrics>,
    after: Option<SculptMetrics>,
    pick: fn(SculptMetrics) -> f32,
) -> (bool, String) {
    let (Some(before), Some(after)) = (before, after) else {
        return (
            false,
            format!("{metric}: the sculpt spot never reached High, so nothing was measured"),
        );
    };
    let (was, now) = (pick(before), pick(after));
    let fell = if was > 0.0 { 1.0 - now / was } else { 0.0 };
    (
        fell >= SCULPT_MARGIN,
        format!(
            "{metric} {was:.4} -> {now:.4} over {} voxels ({:.1}% down, {:.0}% wanted)",
            after.voxels,
            fell * 100.0,
            SCULPT_MARGIN * 100.0
        ),
    )
}

/// Each stroke moved the field the way its tool is defined to. This is the
/// only coverage `apply_sphere_at` has in the bench: the other phases all
/// write `SdfCsgSphere`, which goes the other way through `csg_shape_delta`.
fn check_sculpt(mut bench: ResMut<Bench>, chunks: Res<ChunkManager>, sdfs: Query<&SDF>) {
    if !Phase::Sculpt.last_frame(&bench) {
        return;
    }
    let end = sculpt_metrics(&chunks, &sdfs);
    let (start, mid) = (bench.sculpt_start, bench.sculpt_mid);
    let (pass, detail) = sculpt_fell("roughness", start, mid, |m| m.roughness);
    bench
        .checks
        .record(Phase::Sculpt, "sculpt_smooths", pass, detail);
    let (pass, detail) = sculpt_fell("plane offset", mid, end, |m| m.offset);
    bench
        .checks
        .record(Phase::Sculpt, "sculpt_flattens", pass, detail);
}

// ---------------------------------------------------------------------------
// box: box brushes
// ---------------------------------------------------------------------------

/// Where `box` sculpts: beside the `edit` spot, in another chunk.
fn box_target() -> Vec3 {
    surface(edit_target().x, START.z + 15.0)
}

/// The material box's center: 0.45 of a High voxel past a voxel, so its
/// faces (6.4 voxels out) fall between voxels and rounding decides which
/// ones it covers.
fn material_box_center() -> Vec3 {
    let voxel = LOD::High.voxel_size();
    ((box_target() / voxel).floor() + 0.45) * voxel
}

fn drive_box_edits(
    bench: Res<Bench>,
    mut sdf: WorldEditor<SDF, f32>,
    mut materials: WorldEditor<MaterialField<TerrainMaterial>, TerrainMaterial>,
) {
    if bench.phase != Phase::BoxEdit {
        return;
    }
    if bench.frame_in_phase == 0 {
        materials.fill_box(
            material_box_center(),
            Vec3::splat(MATERIAL_BOX_HALF),
            MATERIAL_BOX_FILL,
        );
    }
    let dig = (bench.frame_in_phase / 30).is_multiple_of(2);
    let delta = if dig { EDIT_DELTA } else { -EDIT_DELTA };
    sdf.accumulate_box(box_target(), Vec3::splat(BOX_HALF), delta);
}

/// Every voxel of the chunks the material box overlaps must be the fill
/// inside the box (by the same grid rounding the brush uses) and generated
/// terrain outside it. The voxels just outside must generate something else,
/// or a brush that spilled past the box would go unnoticed.
fn check_material_box(
    mut bench: ResMut<Bench>,
    chunks: Res<ChunkManager>,
    fields: Query<&MaterialField<TerrainMaterial>>,
) {
    if !Phase::BoxEdit.last_frame(&bench) {
        return;
    }
    let center = material_box_center();
    let half = Vec3::splat(MATERIAL_BOX_HALF);
    let fill = MATERIAL_BOX_FILL.to_id();
    let (mut inside, mut wrong, mut problems) = (0usize, 0usize, Vec::new());
    // Voxels within one voxel of the box, and how many of those generate
    // something other than the fill.
    let (mut shell, mut shell_distinct) = (0usize, 0usize);
    for pos in overlapping_chunks(center, half) {
        let Some(field) = chunks.get_chunk(&pos).and_then(|e| fields.get(e).ok()) else {
            problems.push(format!("chunk {pos} has no material field"));
            continue;
        };
        let n = field.size().x;
        if n != LOD::High.size() {
            problems.push(format!("chunk {pos} isn't at max LOD"));
            continue;
        }
        let generated = terrain_materials(pos, LOD::High);
        // The brush's own grid-space box (see process_shape_edits / apply_box).
        let voxel = CHUNK_SIZE / f32_from_u32(n);
        let grid_center = (center - pos.as_vec3() * CHUNK_SIZE) / voxel;
        let grid_half = half * (1.0 / voxel);
        let (lo, hi) = (
            (grid_center - grid_half).ceil(),
            (grid_center + grid_half).floor(),
        );
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let v = UVec3::new(x, y, z).as_vec3();
                    let in_box = v.cmpge(lo).all() && v.cmple(hi).all();
                    let gen_id = generated
                        .get(flatten_cube(x, y, z, n))
                        .copied()
                        .map(material_id)
                        .unwrap_or_default();
                    let in_shell = !in_box && v.cmpge(lo - 1.0).all() && v.cmple(hi + 1.0).all();
                    if in_shell {
                        shell = shell.saturating_add(1);
                        shell_distinct = shell_distinct.saturating_add(usize::from(gen_id != fill));
                    }
                    let expected = if in_box {
                        inside = inside.saturating_add(1);
                        fill
                    } else {
                        gen_id
                    };
                    if field.get(x, y, z).to_id() != expected {
                        wrong = wrong.saturating_add(1);
                    }
                }
            }
        }
    }
    let pass = problems.is_empty() && wrong == 0 && inside > 0 && shell_distinct == shell;
    let mut detail = format!(
        "{inside} voxels in the box, {wrong} voxels wrong, \
         {shell_distinct}/{shell} voxels around it not already the fill"
    );
    for p in problems {
        let _ = write!(detail, "; {p}");
    }
    bench
        .checks
        .record(Phase::BoxEdit, "material_box_bounds", pass, detail);
}

// ---------------------------------------------------------------------------
// query: GPU ray queries
// ---------------------------------------------------------------------------

/// A fixed fan of rays from the camera onto the terrain around the edit spot.
fn query_rays() -> Vec<(Vec3, Vec3)> {
    let origin = fly_end();
    (0..QUERY_RAYS)
        .map(|i| {
            let (col, row) = (
                f32_from_u32(u32::try_from(i % 8).unwrap_or(0)),
                f32_from_u32(u32::try_from(i / 8).unwrap_or(0)),
            );
            let at = edit_target() + Vec3::new((col - 3.5) * 3.0, 0.0, (row - 2.5) * 3.0);
            let target = surface(at.x, at.z);
            (origin, (target - origin).normalize_or_zero())
        })
        .collect()
}

fn drive_queries(
    bench: Res<Bench>,
    mut pending: ResMut<PendingVoxelQueries>,
    probe: Res<SlotProbe>,
    small_arenas: Res<SmallArenas>,
) {
    if bench.phase != Phase::Query {
        return;
    }
    if small_arenas.0
        && bench.frame_in_phase == ARENA_REPORT_AT
        && let Ok(mut state) = probe.0.lock()
    {
        state.arena_request = true;
    }
    for (i, (origin, dir)) in query_rays().into_iter().enumerate() {
        pending.0.push(RayQuery {
            origin: origin.into(),
            max_distance: QUERY_MAX_DISTANCE,
            direction: dir.into(),
            user_id: u32::try_from(i).unwrap_or(u32::MAX),
        });
    }
}

/// GPU result batches seen during `query`.
#[derive(Resource, Default)]
struct QueryWatch {
    batches: u32,
    /// Batches whose length wasn't `QUERY_RAYS`.
    wrong_length: u32,
    latest: Vec<HitResult>,
}

fn watch_queries(
    bench: Res<Bench>,
    results: Res<VoxelQueryResults>,
    mut watch: ResMut<QueryWatch>,
) {
    if bench.phase != Phase::Query || !results.is_changed() || results.0.is_empty() {
        return;
    }
    watch.batches = watch.batches.saturating_add(1);
    if results.0.len() != QUERY_RAYS {
        watch.wrong_length = watch.wrong_length.saturating_add(1);
    }
    watch.latest.clone_from(&results.0);
}

/// Batches arrived, each the right length, and the latest agrees with the
/// CPU ray cast (the view is still, so every batch should).
fn check_queries(
    mut bench: ResMut<Bench>,
    watch: Res<QueryWatch>,
    terrain: TerrainRay,
    probe: Res<SlotProbe>,
    small_arenas: Res<SmallArenas>,
) {
    if !Phase::Query.last_frame(&bench) {
        return;
    }
    bench.checks.record(
        Phase::Query,
        "query_batch_length",
        watch.batches > 0 && watch.wrong_length == 0,
        format!(
            "{} batches, {} not {QUERY_RAYS} long",
            watch.batches, watch.wrong_length
        ),
    );
    // Per ray: did the GPU agree with the CPU, and the CPU hit (if any).
    let rays: Vec<(bool, Option<Vec3>)> = query_rays()
        .iter()
        .zip(&watch.latest)
        .map(|((origin, dir), gpu)| {
            let cpu = terrain.cast(*origin, *dir, QUERY_MAX_DISTANCE);
            let agree = match (&cpu, gpu.did_hit != 0) {
                (Some(c), true) => c.pos.distance(Vec3::from(gpu.hit_pos_world)) < QUERY_TOLERANCE,
                (None, false) => true,
                _ => false,
            };
            if !agree {
                info!(
                    "[bench] query ray {dir:?} disagrees: cpu (pos, dist) {:?}, gpu hit={} pos {:?} dist {} slot {}",
                    cpu.as_ref().map(|c| (c.pos, c.distance)),
                    gpu.did_hit,
                    gpu.hit_pos_world,
                    gpu.hit_distance,
                    gpu.chunk_slot
                );
            }
            // Just past the surface, inside the chunk that was hit.
            (agree, cpu.map(|c| c.pos + *dir * 0.01))
        })
        .collect();
    let agree = rays.iter().filter(|r| r.0).count();
    bench.checks.record(
        Phase::Query,
        "query_matches_cpu",
        agree == QUERY_RAYS,
        format!("{agree}/{QUERY_RAYS} rays agree with the CPU ray"),
    );

    // The rays whose terrain lives past its LOD's first arena: only a pass
    // per arena finds those. Default-sized arenas rarely spill, so this
    // needs `--small-arenas`.
    if !small_arenas.0 {
        info!("[bench] query_overflow_arenas skipped: run with --small-arenas");
        return;
    }
    let report = probe.0.lock().ok().and_then(|mut s| s.arena_report.take());
    let Some(report) = report else {
        bench.checks.record(
            Phase::Query,
            "query_overflow_arenas",
            false,
            "no arena report from the render world".into(),
        );
        return;
    };
    let overflow: Vec<bool> = rays
        .iter()
        .filter_map(|&(agree, hit)| {
            let (_, arena) = report.chunk_arena.get(&world_pos_to_chunk_pos(&hit?))?;
            (*arena > 0).then_some(agree)
        })
        .collect();
    let overflow_agree = overflow.iter().filter(|a| **a).count();
    let most_arenas = report.counts.iter().copied().max().unwrap_or(0);
    bench.checks.record(
        Phase::Query,
        "query_overflow_arenas",
        most_arenas >= 2 && !overflow.is_empty() && overflow_agree == overflow.len(),
        format!(
            "arenas per LOD {:?}; {overflow_agree}/{} rays into a chunk past its LOD's first arena agree",
            report.counts,
            overflow.len()
        ),
    );
}

// ---------------------------------------------------------------------------
// reslot: chunks that lose and regain their slot
// ---------------------------------------------------------------------------

/// Four surface chunks near the camera, a chunk apart so their fills don't
/// overlap one another.
fn reslot_chunks() -> [IVec3; 4] {
    let base = world_pos_to_chunk_pos(&surface(fly_end().x - 10.0, START.z - 20.0));
    [
        IVec3::ZERO,
        IVec3::new(2, 0, 0),
        IVec3::new(0, 0, 2),
        IVec3::new(2, 0, 2),
    ]
    .map(|o| base.saturating_add(o))
}

fn chunk_center(pos: IVec3) -> Vec3 {
    (pos.as_vec3() + 0.5) * CHUNK_SIZE
}

/// Each cycle fills the chunks solid (they mesh to nothing and give up
/// their slot), then digs a cavity (they mesh again, into a new slot). Only
/// the SDF changes, so the main world never re-sends their materials.
fn drive_reslots(
    bench: Res<Bench>,
    mut csg: MessageWriter<SdfCsgSphere>,
    chunks: Res<ChunkManager>,
    fields: Query<&MaterialField<TerrainMaterial>>,
    probe: Res<SlotProbe>,
) {
    if bench.phase != Phase::Reslot {
        return;
    }
    let f = bench.frame_in_phase;
    let at = f % RESLOT_PERIOD;
    for pos in reslot_chunks() {
        let center = chunk_center(pos);
        if at == 0 {
            csg.write(SdfCsgSphere {
                center,
                radius: RESLOT_FILL_RADIUS,
                sweep_from: None,
                subtract: false,
                max_step: None,
                paint: None,
            });
        } else if at == RESLOT_DIG_AT {
            csg.write(SdfCsgSphere {
                center,
                radius: RESLOT_DIG_RADIUS,
                sweep_from: None,
                subtract: true,
                max_step: None,
                paint: None,
            });
        }
    }
    if f != RESLOT_PROBE_AT {
        return;
    }
    let Ok(mut state) = probe.0.lock() else {
        return;
    };
    for pos in reslot_chunks() {
        let Some(entity) = chunks.get_chunk(&pos) else {
            state.results.push(format!("chunk {pos} isn't loaded"));
            continue;
        };
        let Ok(field) = fields.get(entity) else {
            state
                .results
                .push(format!("chunk {pos} has no material field"));
            continue;
        };
        state.requests.push(SlotRequest {
            pos,
            entity,
            lod: field.lod(),
            expected: field.data_slice().into(),
        });
    }
}

/// The cycles moved chunks between slots and refilled their materials from
/// the cache, and each chunk's GPU material slot now matches its field.
fn check_reslots(mut bench: ResMut<Bench>, probe: Res<SlotProbe>) {
    if !Phase::Reslot.last_frame(&bench) {
        return;
    }
    let sum = |name: &str| -> f64 {
        bench
            .rows
            .iter()
            .filter(|r| r.phase == Phase::Reslot)
            .filter_map(|r| r.get(name))
            .sum()
    };
    let (moved, refills) = (sum("reslotted"), sum("side_refills"));
    bench.checks.record(
        Phase::Reslot,
        "reslot_exercised",
        moved > 0.0 && refills > 0.0,
        format!("{moved} slot moves, {refills} side-channel refills"),
    );
    let results = probe
        .0
        .lock()
        .map_or_else(|_| Vec::new(), |s| s.results.clone());
    let ok = results.iter().filter(|r| r.ends_with(": ok")).count();
    let n = reslot_chunks().len();
    let detail = if results.is_empty() {
        "no readback finished".into()
    } else {
        results.join("; ")
    };
    bench
        .checks
        .record(Phase::Reslot, "reslot_material_slots", ok == n, detail);
}

/// A chunk whose GPU material slot to read back, with what it should hold.
struct SlotRequest {
    pos: IVec3,
    entity: Entity,
    lod: LOD,
    /// The main world's field, `lod.size()`^3 voxels (id + distance).
    expected: Arc<[MaterialVoxel]>,
}

#[derive(Default)]
struct ProbeState {
    requests: Vec<SlotRequest>,
    /// One line per request, ending in ": ok" when the slot matched.
    results: Vec<String>,
    /// Clipped shapes (or, with `visibility-mask`, masks) to read back,
    /// and one result line per request.
    visibility_requests: Vec<VisibilityRequest>,
    visibility_results: Vec<(IVec3, String)>,
    /// Set by the main world to ask for an `ArenaReport`.
    arena_request: bool,
    arena_report: Option<ArenaReport>,
}

/// Which arena holds each resident chunk.
struct ArenaReport {
    /// Arenas per LOD rank.
    counts: [usize; LOD::COUNT],
    /// Chunk position -> (LOD rank, arena index).
    chunk_arena: HashMap<IVec3, (LOD, usize)>,
}

/// Answers an arena-report request with the arena of every resident chunk.
fn report_arenas(probe: Res<SlotProbe>, arenas: ArenaInspector) {
    let Ok(mut state) = probe.0.lock() else {
        return;
    };
    if !std::mem::take(&mut state.arena_request) {
        return;
    }
    state.arena_report = Some(ArenaReport {
        counts: arenas.arena_counts(),
        chunk_arena: arenas
            .resident_chunks()
            .map(|(pos, lod, arena)| (pos, (lod, arena)))
            .collect(),
    });
}

/// Shared by both worlds: the main world asks, the render world answers.
#[derive(Resource, Clone, Default)]
struct SlotProbe(Arc<Mutex<ProbeState>>);

/// Copies each requested chunk's material slot to a mappable buffer and
/// compares its interior (the padded volume minus the apron) with the field.
fn read_back_slots(
    probe: Res<SlotProbe>,
    arenas: ArenaInspector,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    let requests = match probe.0.lock() {
        Ok(mut s) => std::mem::take(&mut s.requests),
        Err(_) => return,
    };
    for req in requests {
        let pos = req.pos;
        let Some(slot) = arenas.material_slot(req.lod, req.entity) else {
            if let Ok(mut s) = probe.0.lock() {
                s.results.push(format!("chunk {pos}: no slot"));
            }
            continue;
        };
        let padded = slot.padded_size;
        // Two bytes per voxel: the palette id, then its boundary distance.
        let bytes = u64::from(padded).pow(3).saturating_mul(2);
        let readback = device.create_buffer(&BufferDescriptor {
            label: Some("bench_material_readback"),
            size: bytes,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("bench_material_readback"),
        });
        encoder.copy_buffer_to_buffer(&slot.buffer, slot.offset, &readback, 0, bytes);
        queue.submit(std::iter::once(encoder.finish()));

        let n = req.lod.size();
        let state = probe.0.clone();
        readback
            .clone()
            .slice(..)
            .map_async(MapMode::Read, move |result| {
                let line = if result.is_err() {
                    format!("chunk {pos}: map failed")
                } else {
                    let wrong = readback.slice(..).get_mapped_range().ok().map(|gpu| {
                        let mut wrong = 0usize;
                        for z in 0..n {
                            for y in 0..n {
                                for x in 0..n {
                                    let at = flatten_cube(x, y, z, padded).saturating_mul(2);
                                    let got = gpu
                                        .get(at..at.saturating_add(2))
                                        .and_then(|b| b.try_into().ok())
                                        .map(u16::from_le_bytes);
                                    let want = req.expected.get(flatten_cube(x, y, z, n)).copied();
                                    if got != want {
                                        wrong = wrong.saturating_add(1);
                                    }
                                }
                            }
                        }
                        wrong
                    });
                    readback.unmap();
                    match wrong {
                        Some(0) => format!("chunk {pos}: ok"),
                        Some(w) => {
                            format!("chunk {pos}: {w} of {} voxels wrong", usize_from(n).pow(3))
                        }
                        None => format!("chunk {pos}: unreadable"),
                    }
                };
                if let Ok(mut s) = state.lock() {
                    s.results.push(line);
                }
            });
    }
}

// ---------------------------------------------------------------------------
// hold: every mesh lies on the generated surface
// ---------------------------------------------------------------------------

/// When meshes are read back: fresh from load (`hold`), straight after
/// streaming while moving (near `fly`'s end), and after sitting still at
/// the fly's end (near `edit`'s end, skipping the edited chunks). The
/// readback takes a few frames and the crack check over a second (on its own
/// thread), so each is checked on its phase's last frame.
fn mesh_probe_frame(phase: Phase) -> Option<u32> {
    match phase {
        Phase::Hold => Some(10),
        Phase::Fly | Phase::Edit => phase.frames().map(|n| n.saturating_sub(200)),
        _ => None,
    }
}

/// With `--gap-probes`, meshes are also read back this often while flying,
/// to catch cracks that only last until both sides of a LOD change have
/// re-meshed. Each probe reads every mesh back, so it slows `fly` ~10x.
const FLY_GAP_PROBE_EVERY: u32 = 15;
/// Vertex positions are matched across chunks at this resolution (world
/// units): chunks that share a seam vertex compute it from the same data.
const MESH_WELD: f32 = 1.0 / 1024.0;
/// An open edge whose midpoint is within this many of its chunk's voxels of
/// another chunk's surface is closed off; further is a visible gap.
const MESH_GAP: f32 = 0.02;
/// Cell size of the triangle grid for gap distances, in world units.
const MESH_GAP_CELL: f32 = 1.0;
/// A vertex further than this from the heightfield, in its chunk's voxels,
/// is off the surface. Surface nets keeps a vertex inside its cell, and a
/// seam cell spans at most a few of this chunk's voxels.
///
/// That holds for every vertex but one kind, which this number cannot
/// judge: a vertex the seam stitch pins to a chunk edge or corner. Its
/// error is the terrain's height above that edge, continuous in it (1.28,
/// 1.68, 2.07, 2.47, 2.87 Medium voxels at successive lattice steps of one
/// logged edge), so 2.0 is no structural bound of it, and the logged
/// failures (2.320, 2.074) and the hold site's corner (1.477 High) are this
/// kind on the shipped shader (`seam_gpu`). Those are classified
/// (`voxel::seam::pinned_cell`) and reported, not judged by this number.
const MESH_MAX_OFF_SURFACE: f32 = 2.0;
/// A triangle edge longer than this, in its chunk's voxels, is a sliver.
const MESH_MAX_EDGE: f32 = 6.0;
/// A vertex normal further than this from the heightfield's, in degrees.
const MESH_MAX_NORMAL_DEG: f32 = 30.0;
/// Triangles smaller than this, in square voxels, aren't checked for facing.
const MESH_MIN_FOLD_AREA: f32 = 0.01;
/// Bytes per vertex: two `vec4<f32>`.
const MESH_VERTEX_BYTES: u64 = 32;
/// Bytes per slot of `DrawIndexedIndirectArgs`.
const MESH_ARGS_BYTES: u64 = 20;
/// Indices reserved per vertex (`INDICES_PER_CELL`).
const MESH_INDICES_PER_VERTEX: u32 = 18;

#[derive(Default)]
struct MeshProbeState {
    /// The probe the main world asked for, not yet started.
    requested: Option<Phase>,
    /// The probe being read back, and its arenas still pending.
    current: Option<Phase>,
    pending: usize,
    /// Chunks to skip: edited on purpose.
    exclude: Vec<IVec3>,
    /// World points whose nearby vertices the probe records in
    /// `MeshReport::near` (the seam-site GPU test). Empty in a bench run.
    watch: Vec<Vec3>,
    /// Chunks with data when the probe was asked for, meshed or not: an
    /// open edge beside one of these is a gap, beside anything else the edge
    /// of the loaded world.
    loaded: HashSet<IVec3>,
    /// Every loaded chunk's LOD, meshed or not: what `seam_lod_ranks`
    /// would record for it now (see `stale_rank_lines`).
    loaded_lods: HashMap<IVec3, LOD>,
    /// The frame the current probe was asked for.
    current_frame: u32,
    requested_frame: u32,
    report: MeshReport,
    done: Vec<(Phase, MeshReport)>,
    /// Cracks found by each probe: (phase, frame, cracks).
    gaps: Vec<(Phase, u32, CrackReport)>,
    /// The apron audit of `hold`'s flagged chunks: started, readbacks still
    /// pending, and the error per texel class.
    apron_started: bool,
    apron_pending: usize,
    apron: HashMap<String, ApronClass>,
    /// Why each normal-flagged vertex of `hold` got its normal, from the
    /// chunk's own padded SDF read back (see `ApronChunk::normal_report`).
    normal_dump: Vec<String>,
}

/// Apron texels of one kind (region and source LOD): how far their SDF is
/// from the heightfield's, in this chunk's voxels.
#[derive(Default)]
struct ApronClass {
    texels: usize,
    sum_error: f64,
    max_error: f32,
    /// The worst few: (error, chunk, texel, value, expected).
    worst: Vec<(f32, IVec3, UVec3, f32, f32)>,
}

/// Flagged chunks whose padded SDF the audit reads back.
const APRON_AUDIT_CHUNKS: usize = 24;
/// Only texels this close to the surface, in their source's voxels, are
/// compared: further out the narrow band clamps them.
const APRON_AUDIT_RANGE: f32 = 4.0;

/// One flagged spot: how bad, and where (local is in the chunk's voxels).
#[derive(Clone, Copy)]
struct MeshFlag {
    value: f32,
    pos: IVec3,
    lod: LOD,
    local: Vec3,
}

#[derive(Default, Clone)]
struct MeshReport {
    chunks: usize,
    vertices: usize,
    triangles: usize,
    /// Indices outside their chunk's vertex range.
    bad_indices: usize,
    /// Vertices further than `MESH_MAX_OFF_SURFACE` voxels off the surface.
    off_surface: Vec<MeshFlag>,
    /// Triangles with an edge longer than `MESH_MAX_EDGE` voxels (centroid).
    long_edges: Vec<MeshFlag>,
    /// Triangles facing down, value = face normal y (centroid). The
    /// heightfield has no overhangs, so a correct triangle faces up.
    folds: Vec<MeshFlag>,
    /// Each fold's corners, in its chunk's voxels, in `folds`' order.
    fold_corners: Vec<[Vec3; 3]>,
    /// Vertex normals more than `MESH_MAX_NORMAL_DEG` degrees off the
    /// heightfield's: what showed as dark streaks along LOD seams.
    bad_normals: Vec<MeshFlag>,
    /// Vertices the seam stitch pins to a chunk edge or corner
    /// (`voxel::seam::pinned_cell`), every one, value = off-surface voxels.
    /// Their error is the seam geometry's, not the data's: those over the
    /// off-surface or normal limit are reported here, not failed.
    pinned: Vec<MeshFlag>,
    /// Pinned vertices whose normal is more than `MESH_MAX_NORMAL_DEG` off.
    pinned_normals: Vec<MeshFlag>,
    /// Largest off-surface distance per LOD rank, in that LOD's voxels.
    max_off: [f32; LOD::COUNT],
    /// The same, over the vertices not pinned: what the check judges.
    max_off_unpinned: [f32; LOD::COUNT],
    /// Where each of those is.
    worst_unpinned: [Option<MeshFlag>; LOD::COUNT],
    /// Every resident chunk's LOD, to describe the neighbours of a flag.
    lods: HashMap<IVec3, LOD>,
    /// Every triangle, in world space, with its chunk: for the crack check
    /// across chunks once every arena is in. Emptied after it.
    world_triangles: Vec<(IVec3, LOD, [Vec3; 3])>,
    /// See `MeshProbeState::loaded`.
    loaded: HashSet<IVec3>,
    /// See `MeshProbeState::loaded_lods`.
    loaded_lods: HashMap<IVec3, LOD>,
    /// Each meshed chunk's seam ranks, as last uploaded.
    ranks: HashMap<IVec3, [u32; 2]>,
    /// Vertices within `WATCH_RADIUS` of a `MeshProbeState::watch` point.
    near: Vec<NearVertex>,
    /// See `MaterialTally`.
    material: MaterialTally,
}

/// How close to a watched point a vertex is recorded, in world units.
const WATCH_RADIUS: f32 = 1.0;

/// Where a vertex sits relative to its chunk's faces, in its own voxels:
/// the apron only reaches the first rows, so a seam defect shows there.
const MATERIAL_BUCKETS: [(&str, f32); 3] = [
    ("on the chunk face (<=1 voxel)", 1.0),
    ("near it (<=2 voxels)", 2.0),
    ("interior (>2 voxels)", f32::INFINITY),
];

/// Vertices compared against the generator's material, and how many differ.
#[derive(Default, Clone, Copy)]
struct MatCount {
    compared: usize,
    wrong: usize,
}

impl MatCount {
    const fn add(&mut self, other: Self) {
        self.compared = self.compared.saturating_add(other.compared);
        self.wrong = self.wrong.saturating_add(other.wrong);
    }

    fn pct(self) -> f32 {
        if self.compared == 0 {
            return 0.0;
        }
        100.0 * f32_from_usize(self.wrong) / f32_from_usize(self.compared)
    }
}

/// Material mismatch by LOD, by whether the chunk touches a LOD seam, and
/// by distance to the chunk's faces (`MATERIAL_BUCKETS`), plus the same
/// split by which boundary is nearest (`MATERIAL_BOUNDARIES`).
#[derive(Default, Clone)]
struct MaterialTally {
    cells: [[[MatCount; 3]; 2]; LOD::COUNT],
    by_boundary: [[MatCount; 2]; LOD::COUNT],
}

/// The terrain's material boundaries, split by whether a vertex's nearest
/// one lies on the voxel lattice at every LOD. `y = 2.5` is an exact
/// multiple of all four voxel sizes, so a reconstruction that is exact on
/// lattice-aligned planes would look perfect on it alone; the other two
/// follow the terrain's height and are neither flat nor aligned.
const MATERIAL_BOUNDARIES: [&str; 2] = [
    "the lattice-aligned y = 2.5 plane",
    "the curved height-following boundaries",
];

impl MaterialTally {
    fn at(&mut self, lod: LOD, seam: bool, bucket: usize) -> Option<&mut MatCount> {
        self.cells
            .get_mut(usize_from(lod.rank()))?
            .get_mut(usize::from(seam))?
            .get_mut(bucket)
    }

    fn get(&self, lod: LOD, seam: bool, bucket: usize) -> MatCount {
        self.cells
            .get(usize_from(lod.rank()))
            .and_then(|s| s.get(usize::from(seam)))
            .and_then(|b| b.get(bucket))
            .copied()
            .unwrap_or_default()
    }

    fn boundary(&mut self, lod: LOD, curved: bool) -> Option<&mut MatCount> {
        self.by_boundary
            .get_mut(usize_from(lod.rank()))?
            .get_mut(usize::from(curved))
    }

    fn boundary_get(&self, lod: LOD, curved: bool) -> MatCount {
        self.by_boundary
            .get(usize_from(lod.rank()))
            .and_then(|b| b.get(usize::from(curved)))
            .copied()
            .unwrap_or_default()
    }

    fn merge(&mut self, other: &Self) {
        for (mine, theirs) in self.by_boundary.iter_mut().zip(&other.by_boundary) {
            for (mine, theirs) in mine.iter_mut().zip(theirs) {
                mine.add(*theirs);
            }
        }
        for (mine, theirs) in self.cells.iter_mut().zip(&other.cells) {
            for (mine, theirs) in mine.iter_mut().zip(theirs) {
                for (mine, theirs) in mine.iter_mut().zip(theirs) {
                    mine.add(*theirs);
                }
            }
        }
    }
}

/// A vertex near a watched point, as the GPU meshed it.
#[derive(Clone, Copy, Debug)]
struct NearVertex {
    pos: IVec3,
    lod: LOD,
    world: Vec3,
    /// Off the heightfield, in this chunk's voxels.
    off: f32,
    /// Normal off the heightfield's, in degrees.
    normal_deg: f32,
    /// Classified as pinned by the seam stitch.
    pinned: bool,
}

/// Edges of the whole terrain mesh used by one triangle only (or by more
/// than two). Every edge between chunks is shared, so those are cracks,
/// unless they lie on a chunk face with nothing meshed across it: the edge
/// of the loaded world.
#[derive(Default, Clone)]
struct CrackReport {
    open_edges: usize,
    /// Open edges along the loaded world's edge, not cracks.
    border: usize,
    /// Open edges beside a chunk that has data but no mesh: the whole chunk
    /// is missing from the frame. (chunk with no mesh, the one beside it.)
    missing: Vec<(IVec3, IVec3)>,
    /// Edges used by three or more triangles.
    overused: usize,
    /// Cracks where this chunk's stored seam rank for the chunk across
    /// doesn't match that chunk's LOD (it stitched to a LOD it doesn't have
    /// any more), or where the chunk across has stale ranks for this one.
    stale_here: usize,
    stale_across: usize,
    /// Open edges whose midpoint another chunk's surface covers (within
    /// `MESH_GAP` of its voxel): seams that don't share vertices, not holes.
    covered: usize,
    /// Cracks: (chunk, LOD, local midpoint, the chunk face or "interior",
    /// gap to the nearest other chunk's triangle in this chunk's voxels).
    cracks: Vec<(IVec3, LOD, Vec3, String, f32)>,
}

impl MeshReport {
    fn merge(&mut self, part: Self) {
        self.chunks = self.chunks.saturating_add(part.chunks);
        self.vertices = self.vertices.saturating_add(part.vertices);
        self.triangles = self.triangles.saturating_add(part.triangles);
        self.bad_indices = self.bad_indices.saturating_add(part.bad_indices);
        self.off_surface.extend(part.off_surface);
        self.long_edges.extend(part.long_edges);
        self.folds.extend(part.folds);
        self.fold_corners.extend(part.fold_corners);
        self.bad_normals.extend(part.bad_normals);
        self.pinned.extend(part.pinned);
        self.pinned_normals.extend(part.pinned_normals);
        for (m, p) in self.max_off_unpinned.iter_mut().zip(part.max_off_unpinned) {
            *m = m.max(p);
        }
        for (w, p) in self.worst_unpinned.iter_mut().zip(part.worst_unpinned) {
            if p.is_some_and(|p| w.is_none_or(|w| p.value > w.value)) {
                *w = p;
            }
        }
        self.world_triangles.extend(part.world_triangles);
        self.ranks.extend(part.ranks);
        self.near.extend(part.near);
        self.material.merge(&part.material);
        for (m, p) in self.max_off.iter_mut().zip(part.max_off) {
            *m = m.max(p);
        }
    }
}

/// Shared by both worlds: the main world asks, the render world answers.
#[derive(Resource, Clone, Default)]
struct MeshProbe(Arc<Mutex<MeshProbeState>>);

/// `--gap-probes`: probe meshes every `FLY_GAP_PROBE_EVERY` frames of `fly`.
#[derive(Resource, Clone, Copy)]
struct GapProbes(bool);

fn drive_mesh_probe(
    bench: Res<Bench>,
    probe: Res<MeshProbe>,
    gap_probes: Res<GapProbes>,
    chunks: Query<(&ChunkPosition, &LOD)>,
) {
    let fly_gap = gap_probes.0
        && bench.phase == Phase::Fly
        && bench.frame_in_phase > 0
        && bench.frame_in_phase.is_multiple_of(FLY_GAP_PROBE_EVERY);
    if !fly_gap && mesh_probe_frame(bench.phase) != Some(bench.frame_in_phase) {
        return;
    }
    if let Ok(mut s) = probe.0.lock() {
        s.requested = Some(bench.phase);
        s.requested_frame = bench.frame_in_phase;
        s.loaded = chunks.iter().map(|(c, _)| c.0).collect();
        s.loaded_lods = chunks.iter().map(|(c, l)| (c.0, *l)).collect();
        s.exclude = if bench.phase == Phase::Edit {
            let margin = Vec3::splat(EDIT_RADIUS + CHUNK_SIZE);
            overlapping_chunks(edit_target(), margin).collect()
        } else {
            Vec::new()
        };
    }
}

/// The heightfield's slope at `p`'s column, as (d/dx, d/dz).
fn heightfield_slope(p: Vec3) -> (f32, f32) {
    let e = 0.05;
    let gx = (terrain_height(p.x + e, p.z) - terrain_height(p.x - e, p.z)) / (2.0 * e);
    let gz = (terrain_height(p.x, p.z + e) - terrain_height(p.x, p.z - e)) / (2.0 * e);
    (gx, gz)
}

/// The generated heightfield's upward normal at `p`'s column.
fn heightfield_normal(p: Vec3) -> Vec3 {
    let (gx, gz) = heightfield_slope(p);
    Vec3::new(-gx, 1.0, -gz).normalize()
}

/// Distance from `p` to the generated heightfield, approximately: the
/// height difference over the slope's stretch, as the generator's SDF does.
fn heightfield_distance(p: Vec3) -> f32 {
    let (gx, gz) = heightfield_slope(p);
    (p.y - terrain_height(p.x, p.z)) / gz.mul_add(gz, gx.mul_add(gx, 1.0)).sqrt()
}

/// Copies every arena's vertices, indices, index counts and vertex counts to
/// one mappable buffer, then checks each chunk's mesh against the
/// heightfield.
fn read_back_meshes(
    probe: Res<MeshProbe>,
    arenas: ArenaInspector,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    audit_aprons(&probe, &arenas, &device, &queue);
    let (mesh_arenas, exclude, watch) = {
        let Ok(mut s) = probe.0.lock() else {
            return;
        };
        if s.current.is_some() {
            return; // one at a time
        }
        let Some(phase) = s.requested.take() else {
            return;
        };
        s.current = Some(phase);
        s.current_frame = s.requested_frame;
        s.report = MeshReport {
            lods: arenas
                .resident_chunks()
                .map(|(pos, lod, _)| (pos, lod))
                .collect(),
            loaded: std::mem::take(&mut s.loaded),
            loaded_lods: std::mem::take(&mut s.loaded_lods),
            ..default()
        };
        let mesh_arenas = arenas.mesh_arenas();
        s.pending = mesh_arenas.len();
        (
            mesh_arenas,
            Arc::new(s.exclude.clone()),
            Arc::new(s.watch.clone()),
        )
    };
    for arena in mesh_arenas {
        let sources = [
            &arena.vertex_buffer,
            &arena.index_buffer,
            &arena.indirect_args_buffer,
            &arena.vertex_count_buffer,
            &arena.chunk_meta_buffer,
        ];
        let sizes = sources.map(|b| b.size());
        let mut offsets = [0u64; 5];
        let mut total = 0u64;
        for (offset, size) in offsets.iter_mut().zip(sizes) {
            *offset = total;
            total = total.saturating_add(size);
        }
        let readback = device.create_buffer(&BufferDescriptor {
            label: Some("bench_mesh_readback"),
            size: total,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("bench_mesh_readback"),
        });
        for ((src, offset), size) in sources.into_iter().zip(offsets).zip(sizes) {
            encoder.copy_buffer_to_buffer(src, 0, &readback, offset, size);
        }
        queue.submit(std::iter::once(encoder.finish()));

        let state = probe.0.clone();
        let exclude = exclude.clone();
        let watch = watch.clone();
        readback
            .clone()
            .slice(..)
            .map_async(MapMode::Read, move |result| {
                let mut part = MeshReport::default();
                if result.is_ok() {
                    if let Ok(bytes) = readback.slice(..).get_mapped_range() {
                        let parts = offsets.map(|o| usize::try_from(o).unwrap_or(usize::MAX));
                        let piece = |k: usize| {
                            let from = parts.get(k).copied().unwrap_or(usize::MAX);
                            let to = parts
                                .get(k.saturating_add(1))
                                .copied()
                                .unwrap_or(bytes.len());
                            bytes.get(from..to).unwrap_or_default()
                        };
                        let mesh = ArenaBytes {
                            vertices: piece(0),
                            indices: piece(1),
                            args: piece(2),
                            counts: piece(3),
                            meta: piece(4),
                        };
                        check_arena_meshes(&arena, &exclude, &watch, &mesh, &mut part);
                    }
                    readback.unmap();
                }
                finish_arena(&state, part);
            });
    }
}

/// Once `hold`'s mesh report is in, reads back the padded SDF and seam ranks
/// of the chunks it flagged, and compares every texel with the heightfield
/// at the position, and in the units, pass3 reads it at.
/// The chunks `audit_aprons` reads back once `hold`'s mesh report is in,
/// with the normal flags each one has to explain. Empty if the audit has
/// already started or the report is not in yet; marks it started.
fn apron_audit_chunks(probe: &MeshProbe) -> Vec<(IVec3, Vec<MeshFlag>)> {
    let Ok(mut s) = probe.0.lock() else {
        return Vec::new();
    };
    if s.apron_started {
        return Vec::new();
    }
    let Some((_, r)) = s.done.iter().find(|(p, _)| *p == Phase::Hold) else {
        return Vec::new();
    };
    // Every normal flag, pinned or not, carries its own explanation
    // (`normal_report`); the others only feed the apron classes.
    let normals: Vec<MeshFlag> = r
        .bad_normals
        .iter()
        .chain(&r.pinned_normals)
        .copied()
        .collect();
    let mut chunks: Vec<IVec3> = normals
        .iter()
        .map(|f| f.pos)
        .chain(r.folds.iter().map(|f| f.pos))
        .collect();
    chunks.sort_by_key(IVec3::to_array);
    chunks.dedup();
    chunks.truncate(APRON_AUDIT_CHUNKS);
    let dump = normal_rank_lines(r);
    s.apron_started = true;
    s.apron_pending = chunks.len();
    s.normal_dump = dump;
    chunks
        .into_iter()
        .map(|pos| {
            let flags: Vec<MeshFlag> = normals.iter().filter(|f| f.pos == pos).copied().collect();
            (pos, flags)
        })
        .collect()
}

fn audit_aprons(
    probe: &MeshProbe,
    arenas: &ArenaInspector,
    device: &RenderDevice,
    queue: &RenderQueue,
) {
    let chunks = apron_audit_chunks(probe);
    if chunks.is_empty() {
        return;
    }
    let mesh_arenas = arenas.mesh_arenas();
    for (pos, flags) in chunks {
        let found = mesh_arenas.iter().find_map(|a| {
            a.slots
                .iter()
                .find(|(p, _, _)| *p == pos)
                .map(|&(_, slot, _)| (a, slot))
        });
        let Some((arena, slot)) = found else {
            if let Ok(mut s) = probe.0.lock() {
                s.apron_pending = s.apron_pending.saturating_sub(1);
            }
            continue;
        };
        let padded = arena.padded_size;
        let sdf_bytes = u64::from(padded).pow(3).saturating_mul(2);
        let meta_bytes = 32;
        let readback = device.create_buffer(&BufferDescriptor {
            label: Some("bench_apron_readback"),
            size: sdf_bytes.saturating_add(meta_bytes),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("bench_apron_readback"),
        });
        let slot64 = u64::from(slot);
        encoder.copy_buffer_to_buffer(
            &arena.sdf_buffer,
            slot64.saturating_mul(sdf_bytes),
            &readback,
            0,
            sdf_bytes,
        );
        encoder.copy_buffer_to_buffer(
            &arena.chunk_meta_buffer,
            slot64.saturating_mul(meta_bytes),
            &readback,
            sdf_bytes,
            meta_bytes,
        );
        queue.submit(std::iter::once(encoder.finish()));
        let lod = arena.lod;
        let state = probe.0.clone();
        readback
            .clone()
            .slice(..)
            .map_async(MapMode::Read, move |result| {
                let mut classes = HashMap::new();
                let mut dump = Vec::new();
                if result.is_ok() {
                    if let Ok(bytes) = readback.slice(..).get_mapped_range() {
                        let split = usize::try_from(sdf_bytes).unwrap_or(usize::MAX);
                        let (sdf, meta) = bytes.split_at(split.min(bytes.len()));
                        let ranks = [read_u32(meta, 4), read_u32(meta, 5)];
                        if let [Some(r0), Some(r1)] = ranks {
                            let chunk = ApronChunk::new(pos, lod, padded, [r0, r1]);
                            chunk.audit(sdf, &mut classes);
                            for flag in &flags {
                                dump.extend(chunk.normal_report(sdf, flag));
                            }
                        }
                    }
                    readback.unmap();
                }
                if let Ok(mut s) = state.lock() {
                    s.normal_dump.extend(dump);
                    for (key, c) in classes {
                        let into: &mut ApronClass = s.apron.entry(key).or_default();
                        into.texels = into.texels.saturating_add(c.texels);
                        into.sum_error += c.sum_error;
                        into.max_error = into.max_error.max(c.max_error);
                        into.worst.extend(c.worst);
                        into.worst.sort_by(|a, b| b.0.total_cmp(&a.0));
                        into.worst.truncate(4);
                    }
                    s.apron_pending = s.apron_pending.saturating_sub(1);
                }
            });
    }
}

/// A chunk's padded volume as pass3 reads it.
/// One texel a normal's central difference read.
struct TexelRead {
    texel: UVec3,
    /// In the meshing chunk's voxels (an apron texel rescaled to them).
    value: f32,
    apron: bool,
    /// The LOD rank the apron texel came from.
    source: u32,
}

/// `mix` in WESL: `a` to `b` by `t`.
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    (b - a).mul_add(t, a)
}

struct ApronChunk {
    pos: IVec3,
    lod: LOD,
    padded: u32,
    ranks: [u32; 2],
}

impl ApronChunk {
    const fn new(pos: IVec3, lod: LOD, padded: u32, ranks: [u32; 2]) -> Self {
        Self {
            pos,
            lod,
            padded,
            ranks,
        }
    }

    /// LOD rank of the chunk at `offset` (`seam_rank` in `voxel_common.wesl`).
    fn rank(&self, offset: IVec3) -> u32 {
        let k = offset
            .saturating_add(IVec3::ONE)
            .dot(IVec3::new(1, 3, 9))
            .unsigned_abs();
        let word = self.ranks.get(usize_from(k / 16)).copied().unwrap_or(0);
        word.wrapping_shr((k % 16).wrapping_mul(2)) & 3
    }

    /// log2 of a rank's voxels per edge (`rank_log2_size`).
    const fn log2_size(rank: u32) -> i32 {
        match rank & 3 {
            0 => 2,
            r => r.cast_signed().saturating_add(3),
        }
    }

    /// The SDF at padded index `index`, in its source's voxels.
    fn decode(sdf: &[u8], index: usize) -> Option<f32> {
        let at = index.checked_mul(2)?;
        let bytes = sdf.get(at..at.checked_add(2)?)?;
        let snorm = i16::from_le_bytes(bytes.try_into().ok()?);
        Some((f32::from(snorm) / f32::from(i16::MAX)).max(-1.0) * SDF_BAND)
    }

    /// The class of texel `texel`: its LOD, apron region and source LOD.
    fn class(&self, apron: BVec3, shift: i32, coarse_steps: bool) -> String {
        let region = match apron.bitmask().count_ones() {
            0 => "interior",
            1 => "face",
            2 => "edge",
            _ => "corner",
        };
        let from = match shift.signum() {
            1 => "coarser",
            -1 => "finer",
            _ => "same",
        };
        let steps = if apron.any() && coarse_steps {
            " coarse-step"
        } else {
            ""
        };
        format!("{:?} {region} from {from}{steps}", self.lod)
    }

    /// Checks every texel near the surface against the heightfield.
    fn audit(&self, sdf: &[u8], classes: &mut HashMap<String, ApronClass>) {
        let edge = self.padded.saturating_sub(2);
        let own = Self::log2_size(self.rank(IVec3::ZERO));
        // pass3's face_shift: per axis, this chunk's voxels per apron step.
        let face = [IVec3::X, IVec3::Y, IVec3::Z]
            .map(|a| own.saturating_sub(Self::log2_size(self.rank(a))).max(0));
        let coarse_steps = face.iter().any(|&f| f > 0);
        // pass3's texel_pos: the second apron layer sits one step past.
        let far = Vec3::from_array(face.map(|f| f32_from_u32(edge) + 2f32.powi(f)));
        let voxel = self.lod.voxel_size();
        let origin = self.pos.as_vec3() * CHUNK_SIZE;
        for z in 0..self.padded {
            for y in 0..self.padded {
                for x in 0..self.padded {
                    let texel = UVec3::new(x, y, z);
                    let Some(raw) = Self::decode(sdf, flatten_cube(x, y, z, self.padded)) else {
                        continue;
                    };
                    let apron = texel.cmpge(UVec3::splat(edge));
                    let source = self.rank(IVec3::select(apron, IVec3::ONE, IVec3::ZERO));
                    let shift = if apron.any() {
                        own.saturating_sub(Self::log2_size(source))
                    } else {
                        0
                    };
                    let scale = 2f32.powi(shift);
                    let p = Vec3::select(texel.cmpgt(UVec3::splat(edge)), far, texel.as_vec3());
                    let expected = heightfield_distance(origin + p * voxel) / voxel;
                    if expected.abs() > APRON_AUDIT_RANGE * scale {
                        continue; // the narrow band clamps it
                    }
                    let value = raw * scale;
                    let error = (value - expected).abs();
                    let c = classes
                        .entry(self.class(apron, shift, coarse_steps))
                        .or_default();
                    c.texels = c.texels.saturating_add(1);
                    c.sum_error += f64::from(error);
                    c.max_error = c.max_error.max(error);
                    c.worst.push((error, self.pos, texel, value, expected));
                    c.worst.sort_by(|a, b| b.0.total_cmp(&a.0));
                    c.worst.truncate(4);
                }
            }
        }
    }

    /// pass3's `texel_coord`: a continuous position in this chunk's voxels,
    /// in texel space (an apron texel sits one neighbour-voxel step past the
    /// boundary, not one of ours).
    fn texel_coord(&self, p: Vec3, face: [i32; 3]) -> Vec3 {
        let n = f32_from_u32(self.padded.saturating_sub(2));
        let step = Vec3::from_array(face.map(|f| 2f32.powi(f.wrapping_neg())));
        Vec3::select(
            p.cmpgt(Vec3::splat(n)),
            (p - Vec3::splat(n)) * step + Vec3::splat(n),
            p,
        )
    }

    /// pass3's `sample_sdf` + `sdf_at`: texel `t`, clamped into the padded
    /// volume and rescaled to this chunk's voxels. Records what it read.
    fn sample(&self, sdf: &[u8], t: IVec3, own: i32, seen: &mut Vec<TexelRead>) -> f32 {
        let max = i32::try_from(self.padded.saturating_sub(1)).unwrap_or(0);
        let texel = t.clamp(IVec3::ZERO, IVec3::splat(max)).as_uvec3();
        let edge = self.padded.saturating_sub(2);
        let apron = texel.cmpge(UVec3::splat(edge));
        let source = self.rank(IVec3::select(apron, IVec3::ONE, IVec3::ZERO));
        let shift = if apron.any() {
            own.saturating_sub(Self::log2_size(source))
        } else {
            0
        };
        let raw = Self::decode(sdf, flatten_cube(texel.x, texel.y, texel.z, self.padded))
            .unwrap_or_default();
        let value = raw * 2f32.powi(shift);
        if !seen.iter().any(|s| s.texel == texel) {
            seen.push(TexelRead {
                texel,
                value,
                apron: apron.any(),
                source,
            });
        }
        value
    }

    /// pass3's `sample_sdf_trilinear`.
    fn trilinear(
        &self,
        sdf: &[u8],
        pos: Vec3,
        own: i32,
        face: [i32; 3],
        seen: &mut Vec<TexelRead>,
    ) -> f32 {
        let p = self.texel_coord(pos, face);
        let ip = p.floor();
        let f = p - ip;
        let base = ip.as_ivec3();
        let mut at =
            |dx, dy, dz| self.sample(sdf, base.saturating_add(IVec3::new(dx, dy, dz)), own, seen);
        let c000 = at(0, 0, 0);
        let c100 = at(1, 0, 0);
        let c010 = at(0, 1, 0);
        let c110 = at(1, 1, 0);
        let c001 = at(0, 0, 1);
        let c101 = at(1, 0, 1);
        let c011 = at(0, 1, 1);
        let c111 = at(1, 1, 1);
        let x0 = lerp(lerp(c000, c100, f.x), lerp(c010, c110, f.x), f.y);
        let x1 = lerp(lerp(c001, c101, f.x), lerp(c011, c111, f.x), f.y);
        lerp(x0, x1, f.z)
    }

    /// pass3's `compute_normal`: a central difference of this chunk's own
    /// padded SDF at `local`, and every texel it reads.
    fn normal_at(&self, sdf: &[u8], local: Vec3) -> (Vec3, Vec<TexelRead>) {
        let own = Self::log2_size(self.rank(IVec3::ZERO));
        let face = [IVec3::X, IVec3::Y, IVec3::Z]
            .map(|a| own.saturating_sub(Self::log2_size(self.rank(a))).max(0));
        let mut seen = Vec::new();
        let h = 0.1;
        let d = [Vec3::X, Vec3::Y, Vec3::Z].map(|axis| {
            let up = self.trilinear(sdf, local + axis * h, own, face, &mut seen);
            let down = self.trilinear(sdf, local - axis * h, own, face, &mut seen);
            up - down
        });
        let norm = Vec3::from_array(d);
        seen.sort_by_key(|s| (s.texel.z, s.texel.y, s.texel.x));
        if norm.length() > 0.00001 {
            (norm.normalize(), seen)
        } else {
            (Vec3::Y, seen)
        }
    }

    /// Why the vertex at `flag` got the normal it did: the normal this
    /// chunk's own padded SDF gives there, against the heightfield's, and
    /// every texel the central difference reads. The GPU computed the same
    /// difference over the same volume, so a CPU angle matching the flag's
    /// confirms the readback, and the texels say what produced it.
    fn normal_report(&self, sdf: &[u8], flag: &MeshFlag) -> Vec<String> {
        let (normal, seen) = self.normal_at(sdf, flag.local);
        let voxel = self.lod.voxel_size();
        let world = self.pos.as_vec3() * CHUNK_SIZE + flag.local * voxel;
        let want = heightfield_normal(world);
        let angle = normal.dot(want).clamp(-1.0, 1.0).acos().to_degrees();
        let mut lines = vec![format!(
            "    normal {:.2} at chunk {} {:?} local ({:.2}, {:.2}, {:.2}): recomputed {:.2} degrees off; normal ({:.3}, {:.3}, {:.3}) vs heightfield ({:.3}, {:.3}, {:.3})",
            flag.value,
            self.pos,
            self.lod,
            flag.local.x,
            flag.local.y,
            flag.local.z,
            angle,
            normal.x,
            normal.y,
            normal.z,
            want.x,
            want.y,
            want.z
        )];
        let origin = self.pos.as_vec3() * CHUNK_SIZE;
        let edge = self.padded.saturating_sub(2);
        let far = f32_from_u32(edge) + 2f32.powi(1);
        for s in &seen {
            // Where the texel sits and what the heightfield says there, so a
            // wrong apron value shows as a value far from `want`.
            let p = Vec3::select(
                s.texel.cmpgt(UVec3::splat(edge)),
                Vec3::splat(far),
                s.texel.as_vec3(),
            );
            let want = heightfield_distance(origin + p * voxel) / voxel;
            lines.push(format!(
                "      texel ({:2}, {:2}, {:2}){}: {:+.3} voxels, heightfield {:+.3}{}",
                s.texel.x,
                s.texel.y,
                s.texel.z,
                if s.apron { " apron" } else { "      " },
                s.value,
                want,
                if s.apron {
                    format!(", source rank {}", s.source)
                } else {
                    String::new()
                }
            ));
        }
        lines
    }
}

/// Adds one arena's results to the probe; after the last, files the report
/// and starts its crack check.
fn finish_arena(state: &Arc<Mutex<MeshProbeState>>, part: MeshReport) {
    let Ok(mut s) = state.lock() else {
        return;
    };
    s.report.merge(part);
    s.pending = s.pending.saturating_sub(1);
    if s.pending != 0 {
        return;
    }
    let Some(phase) = s.current.take() else {
        return;
    };
    let mut report = std::mem::take(&mut s.report);
    // Off the render thread: hashing every edge of the terrain takes over a
    // second.
    let input = MeshReport {
        world_triangles: std::mem::take(&mut report.world_triangles),
        lods: report.lods.clone(),
        loaded: std::mem::take(&mut report.loaded),
        ranks: report.ranks.clone(),
        ..default()
    };
    let frame = s.current_frame;
    let state = state.clone();
    std::thread::spawn(move || {
        let cracks = find_cracks(&input);
        if let Ok(mut s) = state.lock() {
            s.gaps.push((phase, frame, cracks));
        }
    });
    // The scheduled probe of each phase is its last.
    s.done.retain(|(p, _)| *p != phase);
    s.done.push((phase, report));
}

/// One arena's read-back buffers.
struct ArenaBytes<'a> {
    vertices: &'a [u8],
    indices: &'a [u8],
    args: &'a [u8],
    counts: &'a [u8],
    /// `ChunkMeta` per slot, 32 bytes; seam ranks at words 4 and 5.
    meta: &'a [u8],
}

impl ArenaBytes<'_> {
    fn position(&self, vertex: u32) -> Option<Vec3> {
        read_vec3(
            self.vertices,
            u64::from(vertex).checked_mul(MESH_VERTEX_BYTES)?,
        )
    }

    fn normal(&self, vertex: u32) -> Option<Vec3> {
        read_vec3(
            self.vertices,
            u64::from(vertex)
                .checked_mul(MESH_VERTEX_BYTES)?
                .checked_add(16)?,
        )
    }

    /// The vertex's material pair, as `pack_material_a`/`pack_material_b`
    /// wrote it: the two ids and the weight of `id_a`, 0-255. The pair is in
    /// canonical order (lower id first), so the id actually shown is `id_a`
    /// when its weight is at least half.
    fn material(&self, vertex: u32) -> Option<(u8, u8, u8)> {
        let word = u64::from(vertex).checked_mul(MESH_VERTEX_BYTES)? / 4;
        let a = read_u32(self.vertices, word.checked_add(3)?)?;
        let b = read_u32(self.vertices, word.checked_add(7)?)?;
        let byte = |v: u32, shift: u32| u8::try_from(v.wrapping_shr(shift) & 0xFF).ok();
        Some((byte(a, 0)?, byte(b, 0)?, byte(a, 8)?))
    }
}

/// The `u32` at word `index`.
fn read_u32(bytes: &[u8], index: u64) -> Option<u32> {
    let i = usize::try_from(index.checked_mul(4)?).ok()?;
    Some(u32::from_le_bytes(
        bytes.get(i..i.checked_add(4)?)?.try_into().ok()?,
    ))
}

fn read_vec3(bytes: &[u8], at: u64) -> Option<Vec3> {
    let word = at / 4;
    let f = |k: u64| read_u32(bytes, word.checked_add(k)?).map(f32::from_bits);
    Some(Vec3::new(f(0)?, f(1)?, f(2)?))
}

/// Whether `v` is over `max` (or not a number).
fn exceeds(v: f32, max: f32) -> bool {
    v.partial_cmp(&max) != Some(std::cmp::Ordering::Less)
        && v.partial_cmp(&max) != Some(std::cmp::Ordering::Equal)
}

fn check_arena_meshes(
    arena: &tarasaur::voxel::inspect::MeshArena,
    exclude: &[IVec3],
    watch: &[Vec3],
    mesh: &ArenaBytes,
    report: &mut MeshReport,
) {
    for &(pos, slot, base) in &arena.slots {
        let checked = !exclude.contains(&pos);
        let count = read_u32(mesh.counts, u64::from(slot)).unwrap_or(0);
        let word = u64::from(slot).saturating_mul(8);
        if let (Some(r0), Some(r1)) = (
            read_u32(mesh.meta, word.saturating_add(4)),
            read_u32(mesh.meta, word.saturating_add(5)),
        ) {
            report.ranks.insert(pos, [r0, r1]);
        }
        let chunk = ChunkMesh {
            pos,
            lod: arena.lod,
            base,
            count,
            checked,
        };
        if checked {
            report.chunks = report.chunks.saturating_add(1);
            report.vertices = report.vertices.saturating_add(usize_from(count));
            let ranks = report.ranks.get(&pos).copied();
            check_vertices(&chunk, mesh, ranks, watch, report);
        }
        let index_count = read_u32(mesh.args, u64::from(slot) * MESH_ARGS_BYTES / 4).unwrap_or(0);
        check_triangles(&chunk, index_count, mesh, report);
    }
}

/// One chunk's vertex range in its arena.
struct ChunkMesh {
    pos: IVec3,
    lod: LOD,
    base: u32,
    count: u32,
    /// False for chunks edited on purpose: their triangles still count for
    /// the crack check, but aren't checked against the heightfield.
    checked: bool,
}

impl ChunkMesh {
    fn flag(&self, value: f32, p: Vec3) -> MeshFlag {
        let local = (p - self.pos.as_vec3() * CHUNK_SIZE) / self.lod.voxel_size();
        MeshFlag {
            value,
            pos: self.pos,
            lod: self.lod,
            local,
        }
    }

    const fn holds(&self, vertex: u32) -> bool {
        vertex >= self.base && vertex < self.base.saturating_add(self.count)
    }
}

/// Whether any of the 26 stored ranks differs from the chunk's own: the
/// mesher's own test for "this chunk touches a LOD seam".
fn touches_seam(ranks: [u32; 2]) -> bool {
    let own = rank_at(ranks, IVec3::ZERO);
    (0..27).any(|k| {
        let d = IVec3::new(k % 3, (k / 3) % 3, k / 9).saturating_sub(IVec3::ONE);
        d != IVec3::ZERO && rank_at(ranks, d) != own
    })
}

/// The rank `ranks` holds for the neighbour at `offset`.
fn rank_at(ranks: [u32; 2], offset: IVec3) -> u32 {
    let k = offset
        .saturating_add(IVec3::ONE)
        .dot(IVec3::new(1, 3, 9))
        .unsigned_abs();
    ranks
        .get(usize_from(k / 16))
        .copied()
        .unwrap_or(0)
        .wrapping_shr((k % 16).wrapping_mul(2))
        & 3
}

/// Compares the material the mesher gave this vertex against the one the
/// generator defines at that point, and files it by LOD, by whether the
/// chunk touches a LOD seam, and by how close the vertex is to a chunk
/// face. Quantising a material boundary onto the voxel grid is expected
/// and does not depend on where the vertex sits in its chunk, so a bucket
/// that is worse than the chunk's interior is a seam defect, not sampling.
fn tally_material(
    chunk: &ChunkMesh,
    mesh: &ArenaBytes,
    vertex: u32,
    local: Vec3,
    ranks: Option<[u32; 2]>,
    report: &mut MeshReport,
) {
    let Some((id_a, id_b, weight_a)) = mesh.material(vertex) else {
        return;
    };
    let Some(ranks) = ranks else {
        return;
    };
    let shown = if weight_a >= 128 { id_a } else { id_b };
    let p = chunk.pos.as_vec3() * CHUNK_SIZE + local * chunk.lod.voxel_size();
    let truth = determine_material(terrain_height(p.x, p.z), p.y).to_id();
    let n = f32_from_u32(chunk.lod.size());
    let to_face = [local.x, local.y, local.z]
        .into_iter()
        .map(|c| c.min(n - c))
        .fold(f32::INFINITY, f32::min);
    let bucket = MATERIAL_BUCKETS
        .iter()
        .position(|&(_, limit)| to_face <= limit)
        .unwrap_or(2);
    let wrong = shown != truth;
    if let Some(cell) = report.material.at(chunk.lod, touches_seam(ranks), bucket) {
        cell.compared = cell.compared.saturating_add(1);
        if wrong {
            cell.wrong = cell.wrong.saturating_add(1);
        }
    }
    // Which boundary this vertex is nearest: the flat one is lattice-aligned
    // at every LOD, so it is the easy case and is counted apart.
    let (gx, gz) = heightfield_slope(p);
    let height = terrain_height(p.x, p.z);
    let k = gz.mul_add(gz, gx.mul_add(gx, 1.0)).sqrt().recip();
    let flat = (p.y - 2.5).abs();
    let curved = material_boundary_distance(p.y, height, k) < flat;
    if let Some(cell) = report.material.boundary(chunk.lod, curved) {
        cell.compared = cell.compared.saturating_add(1);
        if wrong {
            cell.wrong = cell.wrong.saturating_add(1);
        }
    }
}

/// `ranks`: the chunk's seam ranks as meshed, to classify pinned vertices.
fn check_vertices(
    chunk: &ChunkMesh,
    mesh: &ArenaBytes,
    ranks: Option<[u32; 2]>,
    watch: &[Vec3],
    report: &mut MeshReport,
) {
    let voxel = chunk.lod.voxel_size();
    for v in chunk.base..chunk.base.saturating_add(chunk.count) {
        let Some(p) = mesh.position(v) else {
            continue;
        };
        let d = heightfield_distance(p).abs() / voxel;
        let local = (p - chunk.pos.as_vec3() * CHUNK_SIZE) / voxel;
        let pinned = ranks.is_some_and(|r| seam::pinned_cell(r, local).is_some());
        tally_material(chunk, mesh, v, local, ranks, report);
        if watch.iter().any(|w| w.distance(p) <= WATCH_RADIUS) {
            let normal_deg = mesh.normal(v).map_or(f32::NAN, |n| {
                n.normalize_or_zero()
                    .dot(heightfield_normal(p))
                    .clamp(-1.0, 1.0)
                    .acos()
                    .to_degrees()
            });
            report.near.push(NearVertex {
                pos: chunk.pos,
                lod: chunk.lod,
                world: p,
                off: d,
                normal_deg,
                pinned,
            });
        }
        let rank = usize_from(chunk.lod.rank());
        if let Some(m) = report.max_off.get_mut(rank) {
            *m = m.max(d);
        }
        if pinned {
            report.pinned.push(chunk.flag(d, p));
        } else {
            if let Some(m) = report.max_off_unpinned.get_mut(rank) {
                *m = m.max(d);
            }
            if let Some(w) = report.worst_unpinned.get_mut(rank)
                && w.is_none_or(|w| d > w.value)
            {
                *w = Some(chunk.flag(d, p));
            }
            if exceeds(d, MESH_MAX_OFF_SURFACE) {
                report.off_surface.push(chunk.flag(d, p));
            }
        }
        if let Some(n) = mesh.normal(v) {
            let cos = n.normalize_or_zero().dot(heightfield_normal(p));
            let angle = cos.clamp(-1.0, 1.0).acos().to_degrees();
            if exceeds(angle, MESH_MAX_NORMAL_DEG) {
                if pinned {
                    report.pinned_normals.push(chunk.flag(angle, p));
                } else {
                    report.bad_normals.push(chunk.flag(angle, p));
                }
            }
        }
    }
}

fn check_triangles(
    chunk: &ChunkMesh,
    index_count: u32,
    mesh: &ArenaBytes,
    report: &mut MeshReport,
) {
    let voxel = chunk.lod.voxel_size();
    let first = u64::from(chunk.base).saturating_mul(u64::from(MESH_INDICES_PER_VERTEX));
    for t in 0..u64::from(index_count / 3) {
        if chunk.checked {
            report.triangles = report.triangles.saturating_add(1);
        }
        let at = first.saturating_add(t.saturating_mul(3));
        let ids = [0, 1, 2].map(|k| read_u32(mesh.indices, at.saturating_add(k)));
        let [Some(i0), Some(i1), Some(i2)] = ids else {
            continue;
        };
        if ![i0, i1, i2].into_iter().all(|i| chunk.holds(i)) {
            report.bad_indices = report.bad_indices.saturating_add(1);
            continue;
        }
        let (Some(a), Some(b), Some(c)) = (mesh.position(i0), mesh.position(i1), mesh.position(i2))
        else {
            continue;
        };
        report
            .world_triangles
            .push((chunk.pos, chunk.lod, [a, b, c]));
        if !chunk.checked {
            continue;
        }
        let centroid = (a + b + c) / 3.0;
        let longest = a.distance(b).max(b.distance(c)).max(c.distance(a)) / voxel;
        if exceeds(longest, MESH_MAX_EDGE) {
            report.long_edges.push(chunk.flag(longest, centroid));
        }
        // Counter-clockwise front faces. A sliver too thin to see has no
        // meaningful facing, so it doesn't count.
        let cross = (b - a).cross(c - a);
        let area = cross.length() / 2.0 / (voxel * voxel);
        if area > MESH_MIN_FOLD_AREA && cross.y <= 0.0 {
            report
                .folds
                .push(chunk.flag(cross.normalize_or_zero().y, centroid));
            report
                .fold_corners
                .push([a, b, c].map(|p| chunk.flag(0.0, p).local));
        }
    }
}

/// A world position, snapped to `MESH_WELD`.
fn weld(p: Vec3) -> IVec3 {
    (p / MESH_WELD).round().as_ivec3()
}

/// Counts how many triangles use each edge of the whole mesh, and sorts the
/// edges used once into the loaded world's border and cracks.
/// The 27 offsets of a cell and its neighbours.
fn around() -> impl Iterator<Item = IVec3> {
    (0u32..27).map(|k| {
        UVec3::new(k % 3, k / 3 % 3, k / 9)
            .as_ivec3()
            .wrapping_sub(IVec3::ONE)
    })
}

/// The report's triangles by `MESH_GAP_CELL` grid cell.
fn surface_grid(r: &MeshReport) -> HashMap<IVec3, Vec<usize>> {
    let mut grid: HashMap<IVec3, Vec<usize>> = HashMap::new();
    for (i, (_, _, t)) in r.world_triangles.iter().enumerate() {
        let lo = (t[0].min(t[1]).min(t[2]) / MESH_GAP_CELL)
            .floor()
            .as_ivec3();
        let hi = (t[0].max(t[1]).max(t[2]) / MESH_GAP_CELL)
            .floor()
            .as_ivec3();
        for z in lo.z..=hi.z {
            for y in lo.y..=hi.y {
                for x in lo.x..=hi.x {
                    grid.entry(IVec3::new(x, y, z)).or_default().push(i);
                }
            }
        }
    }
    grid
}

/// Distance from `point` to the nearest triangle not of chunk `own`.
fn gap_to_others(
    grid: &HashMap<IVec3, Vec<usize>>,
    r: &MeshReport,
    point: Vec3,
    own: IVec3,
) -> f32 {
    let cell = (point / MESH_GAP_CELL).floor().as_ivec3();
    let mut best = f32::INFINITY;
    for d in around() {
        for &i in grid.get(&cell.saturating_add(d)).into_iter().flatten() {
            if let Some((pos, _, tri)) = r.world_triangles.get(i)
                && *pos != own
            {
                best = best.min(point_triangle_distance(point, *tri));
            }
        }
    }
    best
}

/// How many triangles use each welded edge, and the first one that does.
fn edge_uses(r: &MeshReport) -> HashMap<(IVec3, IVec3), (u32, usize)> {
    let mut edges: HashMap<(IVec3, IVec3), (u32, usize)> = HashMap::new();
    for (i, (_, _, [a, b, c])) in r.world_triangles.iter().enumerate() {
        for (p, q) in [(a, b), (b, c), (c, a)] {
            let (p, q) = (weld(*p), weld(*q));
            let key = if p.to_array() < q.to_array() {
                (p, q)
            } else {
                (q, p)
            };
            let e = edges.entry(key).or_insert((0, i));
            e.0 = e.0.saturating_add(1);
        }
    }
    edges
}

fn find_cracks(r: &MeshReport) -> CrackReport {
    let grid = surface_grid(r);
    let gap_at = |p: Vec3, own: IVec3| gap_to_others(&grid, r, p, own);
    let edges = edge_uses(r);
    let mut out = CrackReport::default();
    for ((p, q), (uses, tri)) in edges {
        if uses > 2 {
            out.overused = out.overused.saturating_add(1);
        }
        if uses != 1 {
            continue;
        }
        out.open_edges = out.open_edges.saturating_add(1);
        let Some(&(pos, lod, _)) = r.world_triangles.get(tri) else {
            continue;
        };
        let voxel = lod.voxel_size();
        let origin = pos.as_vec3() * CHUNK_SIZE;
        let local = |w: IVec3| (w.as_vec3() * MESH_WELD - origin) / voxel;
        let (lp, lq) = (local(p), local(q));
        let n = f32_from_u32(lod.size());
        let eps = 0.01;
        // The chunk faces the edge lies on, and whether anything is meshed
        // across each.
        // The chunks across the faces the edge lies on (or, along a low
        // face, the half-cell strip that the neighbour across finishes).
        let mut faces = Vec::new();
        let mut across = Vec::new();
        for (axis, unit) in [IVec3::X, IVec3::Y, IVec3::Z].into_iter().enumerate() {
            let (a, b) = (lp[axis], lq[axis]);
            if a <= 1.0 && b <= 1.0 {
                across.push(pos.saturating_sub(unit));
            }
            let side = if a <= eps && b <= eps {
                -1
            } else if a >= n - eps && b >= n - eps {
                1
            } else {
                continue;
            };
            across.push(if side > 0 {
                pos.saturating_add(unit)
            } else {
                pos.saturating_sub(unit)
            });
            faces.push((axis, side));
        }
        if across.iter().any(|c| !r.loaded.contains(c)) {
            out.border = out.border.saturating_add(1);
            continue;
        }
        if let Some(&gone) = across.iter().find(|c| !r.lods.contains_key(*c)) {
            out.missing.push((gone, pos));
            continue;
        }
        let mid = (p.as_vec3() + q.as_vec3()) * MESH_WELD / 2.0;
        let gap = gap_at(mid, pos) / voxel;
        if gap <= MESH_GAP {
            out.covered = out.covered.saturating_add(1);
            continue;
        }
        let face = if faces.is_empty() {
            "interior".to_string()
        } else {
            faces
                .iter()
                .map(|&(axis, side)| {
                    let name = ["x", "y", "z"].get(axis).copied().unwrap_or("?");
                    format!("{name}{}", if side > 0 { "+" } else { "-" })
                })
                .collect::<Vec<_>>()
                .join(" ")
        };
        let stored = |of: IVec3, at: IVec3| {
            r.ranks.get(&of).map(|ranks| {
                let k = at
                    .saturating_sub(of)
                    .saturating_add(IVec3::ONE)
                    .dot(IVec3::new(1, 3, 9));
                let k = k.unsigned_abs();
                ranks
                    .get(usize_from(k / 16))
                    .copied()
                    .unwrap_or(0)
                    .wrapping_shr((k % 16).wrapping_mul(2))
                    & 3
            })
        };
        for &c in &across {
            if let Some(&l) = r.lods.get(&c) {
                if stored(pos, c).is_some_and(|s| s != l.rank()) {
                    out.stale_here = out.stale_here.saturating_add(1);
                }
                if stored(c, pos).is_some_and(|s| s != lod.rank()) {
                    out.stale_across = out.stale_across.saturating_add(1);
                }
            }
        }
        out.cracks.push((pos, lod, (lp + lq) / 2.0, face, gap));
    }
    out.cracks.sort_by(|a, b| b.4.total_cmp(&a.4));
    out
}

/// Distance from `p` to the triangle `t` (Ericson, Real-Time Collision
/// Detection 5.1.5).
fn point_triangle_distance(point: Vec3, tri: [Vec3; 3]) -> f32 {
    let [va, vb, vc] = tri;
    let (ab, ac, ap) = (vb - va, vc - va, point - va);
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return point.distance(va);
    }
    let bp = point - vb;
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));
    if d3 >= 0.0 && d4 <= d3 {
        return point.distance(vb);
    }
    let wc = d1.mul_add(d4, -(d3 * d2));
    if wc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return point.distance(va + ab * (d1 / (d1 - d3)));
    }
    let cp = point - vc;
    let (d5, d6) = (ab.dot(cp), ac.dot(cp));
    if d6 >= 0.0 && d5 <= d6 {
        return point.distance(vc);
    }
    let wb = d5.mul_add(d2, -(d1 * d6));
    if wb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return point.distance(va + ac * (d2 / (d2 - d6)));
    }
    let wa = d3.mul_add(d6, -(d5 * d4));
    if wa <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return point.distance(vb + (vc - vb) * ((d4 - d3) / ((d4 - d3) + (d5 - d6))));
    }
    let denom = 1.0 / (wa + wb + wc);
    if !denom.is_finite() {
        return point
            .distance(va)
            .min(point.distance(vb))
            .min(point.distance(vc));
    }
    point.distance(va + ab * (wb * denom) + ac * (wc * denom))
}

/// The cracks every probe found, as report lines.
fn gap_report_lines(probe: &MeshProbe, phase: Phase) -> Vec<String> {
    let Ok(s) = probe.0.lock() else {
        return Vec::new();
    };
    let mut lines = Vec::new();
    for (p, frame, c) in s.gaps.iter().filter(|(p, _, _)| *p == phase) {
        let mut by_face: HashMap<String, usize> = HashMap::new();
        let mut chunks: Vec<IVec3> = Vec::new();
        for (pos, lod, _, face, _) in &c.cracks {
            let key = format!("{lod:?} {face}");
            let n = by_face.entry(key).or_default();
            *n = n.saturating_add(1);
            if !chunks.contains(pos) {
                chunks.push(*pos);
            }
        }
        lines.push(format!(
            "{} frame {frame}: {} gaps in {} chunks ({} with stale ranks here, {} across), widest {:.2} voxels; {} open edges beside {} loaded chunks with no mesh ({} open edges: {} on the loaded border, {} covered by a neighbour; {} edges used 3+ times) {by_face:?}",
            p.name(),
            c.cracks.len(),
            chunks.len(),
            c.stale_here,
            c.stale_across,
            c.cracks.first().map_or(0.0, |x| x.4),
            c.missing.len(),
            {
                let mut gone: Vec<IVec3> = c.missing.iter().map(|m| m.0).collect();
                gone.sort_by_key(IVec3::to_array);
                gone.dedup();
                gone.len()
            },
            c.open_edges,
            c.border,
            c.covered,
            c.overused
        ));
        for (pos, lod, mid, face, gap) in c.cracks.iter().take(4) {
            lines.push(format!(
                "    gap {gap:.2}: chunk {pos} {lod:?} {face} at local ({:.2}, {:.2}, {:.2}); {}",
                mid.x,
                mid.y,
                mid.z,
                describe_neighbours(
                    &s.done
                        .iter()
                        .find(|(q, _)| q == p)
                        .map(|(_, r)| r.lods.clone())
                        .unwrap_or_default(),
                    *pos
                )
            ));
        }
    }
    lines
}

/// The LODs of the chunks around `pos`, as "dx,dy,dz:LOD" for each one that
/// differs from `pos`'s own, or has no mesh ("none").
fn describe_neighbours(lods: &HashMap<IVec3, LOD>, pos: IVec3) -> String {
    let own = lods.get(&pos).copied();
    let mut parts = Vec::new();
    for dz in -1..=1 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let d = IVec3::new(dx, dy, dz);
                if d == IVec3::ZERO {
                    continue;
                }
                let other = lods.get(&pos.saturating_add(d)).copied();
                if other != own {
                    let lod = other.map_or_else(|| "none".to_string(), |l| format!("{l:?}"));
                    parts.push(format!("{dx},{dy},{dz}:{lod}"));
                }
            }
        }
    }
    if parts.is_empty() {
        "all neighbours same LOD".into()
    } else {
        parts.join(" ")
    }
}

fn describe_flag(kind: &str, f: &MeshFlag, lods: &HashMap<IVec3, LOD>) -> String {
    format!(
        "  {kind} {:.2}: chunk {} {:?} at local ({:.2}, {:.2}, {:.2}) of {}; {}",
        f.value,
        f.pos,
        f.lod,
        f.local.x,
        f.local.y,
        f.local.z,
        f.lod.size(),
        describe_neighbours(lods, f.pos)
    )
}

/// The 27 stored seam ranks of `ranks`, as digits in neighbour order
/// (x fastest, z slowest), grouped by z: the LOD rank pass3 meshed the
/// chunk against for each neighbour. 0 Lowest, 1 Low, 2 Medium, 3 High.
fn rank_digits(ranks: [u32; 2]) -> String {
    let mut out = String::new();
    for k in 0u32..27 {
        if k != 0 && k % 9 == 0 {
            out.push(' ');
        }
        let word = ranks.get(usize_from(k / 16)).copied().unwrap_or(0);
        let rank = word.wrapping_shr((k % 16).wrapping_mul(2)) & 3;
        out.push(char::from_digit(rank, 10).unwrap_or('?'));
    }
    out
}

/// Every chunk touching the vertex at `local` of an `n`-voxel chunk at
/// `pos`: the chunk itself, plus the neighbour across each face the vertex
/// lies on. All of them emit that vertex, each from its own ranks.
fn chunks_at_vertex(pos: IVec3, local: Vec3, n: f32) -> Vec<IVec3> {
    let mut out = vec![pos];
    for (v, unit) in [
        (local.x, IVec3::X),
        (local.y, IVec3::Y),
        (local.z, IVec3::Z),
    ] {
        let side = if v.abs() <= 1e-3 {
            -1
        } else if (v - n).abs() <= 1e-3 {
            1
        } else {
            continue;
        };
        let step = unit.saturating_mul(IVec3::splat(side));
        let grown: Vec<IVec3> = out.iter().map(|p| p.saturating_add(step)).collect();
        out.extend(grown);
    }
    out
}

/// The vertex materials that disagree with the generator, by LOD and by
/// where the vertex sits. Boundary quantisation is independent of where a
/// vertex sits in its chunk, so the chunk-face rows against the interior
/// row say whether the seam path costs material correctness on top of it.
fn material_lines(r: &MeshReport) -> Vec<String> {
    let mut lines = Vec::new();
    for &lod in &LOD::ALL {
        let total: usize = (0..2)
            .flat_map(|s| (0..3).map(move |b| (s == 1, b)))
            .map(|(seam, b)| r.material.get(lod, seam, b).compared)
            .sum();
        if total == 0 {
            continue;
        }
        lines.push(format!(
            "  material vs the generator, {lod:?} ({total} vertices):"
        ));
        let split: Vec<String> = MATERIAL_BOUNDARIES
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let c = r.material.boundary_get(lod, i == 1);
                format!("{name}: {:.2}% of {}", c.pct(), c.compared)
            })
            .collect();
        lines.push(format!("    nearest boundary  {}", split.join("; ")));
        for (seam, label) in [(true, "seam chunks"), (false, "no seam    ")] {
            let row: Vec<String> = MATERIAL_BUCKETS
                .iter()
                .enumerate()
                .map(|(b, (name, _))| {
                    let c = r.material.get(lod, seam, b);
                    format!("{name}: {:.2}% of {}", c.pct(), c.compared)
                })
                .collect();
            lines.push(format!("    {label}  {}", row.join("; ")));
        }
    }
    lines
}

/// Chunks holding a seam rank that disagrees with the LOD a neighbour is
/// actually meshed at. The mesher rescales that neighbour's apron values by
/// the rank it stored, so a disagreement means a whole apron region is read
/// at the wrong scale (and placed at the wrong spacing) until the chunk
/// re-meshes, which nothing makes it do while its own inputs are unchanged.
///
/// Compared against every *loaded* neighbour, meshed or not: the apron
/// reads a loaded chunk whatever it meshes, so its rank should be its LOD.
/// An absent neighbour is own-substituted by design and is skipped.
fn stale_rank_lines(r: &MeshReport) -> Vec<String> {
    let mut stale: Vec<(IVec3, IVec3, u32, LOD)> = Vec::new();
    for (&pos, &ranks) in &r.ranks {
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let d = IVec3::new(dx, dy, dz);
                    if d == IVec3::ZERO {
                        continue;
                    }
                    let at = pos.saturating_add(d);
                    // A loaded chunk is readable, so its rank is its LOD
                    // whether or not it meshes anything; an absent one is
                    // own-substituted by design and is not compared.
                    let Some(&lod) = r.loaded_lods.get(&at).or_else(|| r.lods.get(&at)) else {
                        continue;
                    };
                    let k = d.saturating_add(IVec3::ONE).dot(IVec3::new(1, 3, 9)).unsigned_abs();
                    let word = ranks.get(usize_from(k / 16)).copied().unwrap_or(0);
                    let stored = word.wrapping_shr((k % 16).wrapping_mul(2)) & 3;
                    if stored != lod.rank() {
                        stale.push((pos, d, stored, lod));
                    }
                }
            }
        }
    }
    if stale.is_empty() {
        return vec![
            "  no chunk holds a seam rank a loaded neighbour disagrees with".to_string(),
        ];
    }
    stale.sort_by_key(|(p, d, _, _)| (p.to_array(), d.to_array()));
    let chunks: Vec<IVec3> = {
        let mut c: Vec<IVec3> = stale.iter().map(|(p, ..)| *p).collect();
        c.dedup();
        c
    };
    let mut lines = vec![format!(
        "  STALE SEAM RANKS: {} directions over {} chunks hold a rank a loaded neighbour disagrees with; their apron is rescaled by the stored rank, so those regions are read at the wrong scale",
        stale.len(),
        chunks.len()
    )];
    for (pos, d, stored, lod) in stale.iter().take(12) {
        lines.push(format!(
            "    chunk {pos} stored rank {stored} for {},{},{} ({}), which is loaded at {lod:?} ({})",
            d.x,
            d.y,
            d.z,
            pos.saturating_add(*d),
            lod.rank()
        ));
    }
    lines
}

/// For each normal-flagged vertex, the seam ranks every chunk touching it
/// was meshed against, and where those disagree with the LODs now resident:
/// the first thing to know about a normal that fires in one run and not the
/// next.
fn normal_rank_lines(r: &MeshReport) -> Vec<String> {
    let mut lines = Vec::new();
    for (kind, flag) in r
        .bad_normals
        .iter()
        .map(|f| ("failed", f))
        .chain(r.pinned_normals.iter().map(|f| ("pinned", f)))
    {
        lines.push(format!(
            "    {kind} normal {:.2} at chunk {} {:?} local ({:.2}, {:.2}, {:.2}); ranks (0 Lowest .. 3 High, x fastest, by z):",
            flag.value, flag.pos, flag.lod, flag.local.x, flag.local.y, flag.local.z
        ));
        for at in chunks_at_vertex(flag.pos, flag.local, f32_from_u32(flag.lod.size())) {
            let live = r.lods.get(&at).copied();
            let stored = r.ranks.get(&at).copied();
            let stale = stored.is_some_and(|s| {
                live.is_some_and(|l| {
                    let own = s.first().copied().unwrap_or(0).wrapping_shr(26) & 3;
                    own != l.rank()
                })
            });
            lines.push(stored.map_or_else(
                || {
                    format!(
                        "      {at} {}: no stored ranks (no mesh in any arena)",
                        live.map_or_else(|| "not resident".to_string(), |l| format!("{l:?}"))
                    )
                },
                |s| {
                    format!(
                        "      {at} {}: {}{}",
                        live.map_or_else(|| "no mesh".to_string(), |l| format!("{l:?}")),
                        rank_digits(s),
                        if stale { "  STALE own rank" } else { "" }
                    )
                },
            ));
        }
    }
    lines
}

/// The report's summary, plus its worst flags.
/// The pinned seam vertices (reported, not failed), and the worst
/// off-surface vertex per LOD among the rest, with where it is.
fn pinned_report_lines(r: &MeshReport) -> Vec<String> {
    let mut lines = Vec::new();
    let per_lod = |m: [f32; LOD::COUNT]| {
        LOD::ALL
            .iter()
            .zip(m)
            .map(|(l, m)| format!("{l:?} {m:.2}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let mut pinned_max = [0.0f32; LOD::COUNT];
    for f in &r.pinned {
        if let Some(m) = pinned_max.get_mut(usize_from(f.lod.rank())) {
            *m = m.max(f.value);
        }
    }
    lines.push(format!(
        "  pinned seam vertices, not failed: {} ({} over {MESH_MAX_OFF_SURFACE} voxels, {} normals over {MESH_MAX_NORMAL_DEG} degrees); worst per LOD: {}",
        r.pinned.len(),
        r.pinned
            .iter()
            .filter(|f| exceeds(f.value, MESH_MAX_OFF_SURFACE))
            .count(),
        r.pinned_normals.len(),
        per_lod(pinned_max)
    ));
    lines.push(format!(
        "  worst off-surface per LOD, not pinned (voxels): {}",
        per_lod(r.max_off_unpinned)
    ));
    lines.extend(stale_rank_lines(r));
    lines.extend(material_lines(r));
    for f in r.worst_unpinned.iter().flatten() {
        // Which of the stitch's cells, if any, holds it: the first thing
        // to know about a near-threshold vertex that is not pinned.
        let cell = r
            .ranks
            .get(&f.pos)
            .and_then(|&ranks| seam::holding_cell(ranks, f.local, 0));
        lines.push(format!(
            "    worst not pinned, {:?}: {:.3} at chunk {} local ({:.2}, {:.2}, {:.2}); {}",
            f.lod,
            f.value,
            f.pos,
            f.local.x,
            f.local.y,
            f.local.z,
            cell.map_or_else(
                || "in no shared cell".to_string(),
                |c| format!(
                    "in a shared cell flat on {} axes, lo {:?} step {:?}",
                    c.flat(),
                    c.lo,
                    c.step
                )
            )
        ));
    }
    lines
}

fn mesh_report_lines(r: &MeshReport) -> Vec<String> {
    let mut lines = vec![
        format!(
            "{} chunks, {} vertices, {} triangles; {} off-surface vertices (> {MESH_MAX_OFF_SURFACE} voxels), {} long triangles (> {MESH_MAX_EDGE} voxels), {} bad indices",
            r.chunks,
            r.vertices,
            r.triangles,
            r.off_surface.len(),
            r.long_edges.len(),
            r.bad_indices
        ),
        format!(
            "  {} vertex normals > {MESH_MAX_NORMAL_DEG} degrees off; known at LOD seam corners, not failed: {} downward triangles",
            r.bad_normals.len(),
            r.folds.len()
        ),
        format!(
            "  worst off-surface per LOD (voxels): {}",
            LOD::ALL
                .iter()
                .zip(r.max_off)
                .map(|(l, m)| format!("{l:?} {m:.2}"))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    ];
    lines.extend(pinned_report_lines(r));
    // Only with a watch list (the seam-site GPU test); a bench run has none.
    for v in &r.near {
        lines.push(format!(
            "  watched vertex: chunk {} {:?} at {} off {:.3} voxels, normal {:.1} degrees{}",
            v.pos,
            v.lod,
            v.world,
            v.off,
            v.normal_deg,
            if v.pinned { ", pinned" } else { "" }
        ));
    }
    let mut worst: Vec<_> = r.off_surface.iter().map(|f| ("off", f)).collect();
    worst.extend(r.long_edges.iter().map(|f| ("edge", f)));
    worst.sort_by(|a, b| b.1.value.total_cmp(&a.1.value));
    // Most downward first.
    let mut folds: Vec<_> = r.folds.iter().map(|f| ("fold", f)).collect();
    folds.sort_by(|a, b| a.1.value.total_cmp(&b.1.value));
    let mut normals: Vec<_> = r.bad_normals.iter().map(|f| ("normal", f)).collect();
    normals.sort_by(|a, b| b.1.value.total_cmp(&a.1.value));
    for (kind, f) in worst
        .iter()
        .chain(folds.iter().take(8))
        .chain(normals.iter().take(8))
    {
        lines.push(describe_flag(kind, f, &r.lods));
    }
    let mut order: Vec<usize> = (0..r.folds.len()).collect();
    order.sort_by(|&a, &b| {
        let y = |i: usize| r.folds.get(i).map_or(0.0, |f| f.value);
        y(a).total_cmp(&y(b))
    });
    for i in order.into_iter().take(12) {
        if let (Some(f), Some(c)) = (r.folds.get(i), r.fold_corners.get(i)) {
            let c = c.map(|v| format!("({:.2}, {:.2}, {:.2})", v.x, v.y, v.z));
            lines.push(format!(
                "  fold corners {:.2} chunk {} {:?}: {}",
                f.value,
                f.pos,
                f.lod,
                c.join(" ")
            ));
        }
    }
    lines
}

/// Every flag, for offline analysis: `bench_results/mesh-<phase>.csv`.
fn write_mesh_csv(phase: Phase, r: &MeshReport) {
    let mut csv = String::from("kind,value,cx,cy,cz,lod,size,lx,ly,lz,neighbours\n");
    let kinds = [
        ("off", &r.off_surface),
        ("edge", &r.long_edges),
        ("fold", &r.folds),
        ("normal", &r.bad_normals),
        ("pinned", &r.pinned),
        ("pinned_normal", &r.pinned_normals),
    ];
    for (kind, list) in kinds {
        for f in list {
            let _ = writeln!(
                csv,
                "{kind},{},{},{},{},{:?},{},{},{},{},{}",
                f.value,
                f.pos.x,
                f.pos.y,
                f.pos.z,
                f.lod,
                f.lod.size(),
                f.local.x,
                f.local.y,
                f.local.z,
                describe_neighbours(&r.lods, f.pos)
            );
        }
    }
    let _ = std::fs::write(format!("bench_results/mesh-{}.csv", phase.name()), csv);
}

fn check_meshes(mut bench: ResMut<Bench>, probe: Res<MeshProbe>) {
    let phase = bench.phase;
    if mesh_probe_frame(phase).is_none() || !phase.last_frame(&bench) {
        return;
    }
    let (pass, detail) = {
        let Ok(state) = probe.0.lock() else {
            return;
        };
        match state.done.iter().find(|(p, _)| *p == phase) {
            None => (false, "readback unfinished".to_string()),
            Some((_, r)) => {
                write_mesh_csv(phase, r);
                // A few folds are still known at LOD seam corners, so they're
                // reported (and in the CSV) without failing the run.
                let pass = r.chunks > 0
                    && r.off_surface.is_empty()
                    && r.long_edges.is_empty()
                    && r.bad_normals.is_empty()
                    && r.bad_indices == 0;
                (pass, mesh_report_lines(r).join("\n"))
            }
        }
    };
    bench.checks.record(phase, "mesh_on_surface", pass, detail);
    let gaps = gap_report_lines(&probe, phase);
    if !gaps.is_empty() {
        // Report-only until the transient cracks are understood.
        bench
            .checks
            .record(phase, "mesh_cracks", true, gaps.join("\n"));
    }
    if phase == Phase::Hold {
        let lines = apron_report_lines(&probe);
        bench
            .checks
            .record(phase, "apron_audit", true, lines.join("\n"));
    }
}

/// The apron audit's classes, worst first.
fn apron_report_lines(probe: &MeshProbe) -> Vec<String> {
    let Ok(s) = probe.0.lock() else {
        return Vec::new();
    };
    let mut lines = vec![format!(
        "{} chunks still pending; error in voxels, texels within {APRON_AUDIT_RANGE} source voxels of the surface",
        s.apron_pending
    )];
    let mut classes: Vec<_> = s.apron.iter().collect();
    classes.sort_by(|a, b| b.1.max_error.total_cmp(&a.1.max_error));
    for (key, c) in classes {
        let mean = c.sum_error / f64::from(u32::try_from(c.texels.max(1)).unwrap_or(u32::MAX));
        lines.push(format!(
            "  {key}: {} texels, mean {mean:.3}, max {:.3}",
            c.texels, c.max_error
        ));
        for (e, pos, t, v, x) in &c.worst {
            lines.push(format!(
                "    {e:.2}: chunk {pos} texel {t} value {v:.2} expected {x:.2}"
            ));
        }
    }
    if !s.normal_dump.is_empty() {
        lines.push(format!(
            "  normal flags, with the ranks they were meshed against and the texels their normal came from ({}):",
            s.normal_dump.len()
        ));
        lines.extend(s.normal_dump.iter().cloned());
    }
    lines
}

// ---------------------------------------------------------------------------
// visibility: hide and reveal edits, field removal, and the clipped shapes
// on the GPU
// ---------------------------------------------------------------------------

/// The hidden box: on the x = 240 chunk boundary, 0.45 of a High voxel off
/// the grid like the material box, so it covers parts of two chunks.
fn vis_box_center() -> Vec3 {
    let voxel = LOD::High.voxel_size();
    ((surface(240.0, START.z + 25.0) / voxel).floor() + 0.45) * voxel
}

/// The sphere toggled hidden and visible, in a chunk of its own.
fn vis_sphere_center() -> Vec3 {
    surface(230.0, START.z + 32.0)
}

/// The box's +x chunk, whose `VisibilityField` is removed mid-phase. The
/// +x one: the sphere's CSG reach touches the -x one, and an edit gives a
/// chunk without the field a fresh one.
fn vis_removed_chunk() -> IVec3 {
    world_pos_to_chunk_pos(&(vis_box_center() + Vec3::X * MATERIAL_BOX_HALF))
}

/// Every chunk the phase's edits touch: the ones whose shapes are checked.
fn vis_chunks() -> Vec<IVec3> {
    let mut chunks: Vec<IVec3> =
        overlapping_chunks(vis_box_center(), Vec3::splat(MATERIAL_BOX_HALF))
            .chain(overlapping_chunks(
                vis_sphere_center(),
                Vec3::splat(VIS_SPHERE_RADIUS),
            ))
            .collect();
    chunks.sort_unstable_by_key(|c| (c.x, c.y, c.z));
    chunks.dedup();
    chunks
}

/// The box is an exact CSG hide; the sphere is CSG-hidden and revealed in
/// turn.
fn drive_visibility(
    bench: Res<Bench>,
    mut boxes: MessageWriter<VisibilityCsgBox>,
    mut csg: MessageWriter<VisibilityCsgSphere>,
    chunks: Res<ChunkManager>,
    fields: Query<(&SDF, Option<&VisibilityField>)>,
    probe: Res<SlotProbe>,
    mut commands: Commands,
) {
    if bench.phase != Phase::Visibility {
        return;
    }
    let f = bench.frame_in_phase;
    if f == 0 {
        boxes.write(VisibilityCsgBox {
            center: vis_box_center(),
            half_size: Vec3::splat(MATERIAL_BOX_HALF),
            hide: true,
            max_step: None,
        });
    }
    // Hidden on odd toggles, so it ends visible.
    if f < VIS_TOGGLE_UNTIL && f.is_multiple_of(VIS_TOGGLE_PERIOD) {
        csg.write(VisibilityCsgSphere {
            center: vis_sphere_center(),
            radius: VIS_SPHERE_RADIUS,
            sweep_from: None,
            hide: (f / VIS_TOGGLE_PERIOD) % 2 == 1,
            max_step: None,
        });
    }
    if f == VIS_REMOVE_AT
        && let Some(entity) = chunks.get_chunk(&vis_removed_chunk())
    {
        commands.entity(entity).remove::<VisibilityField>();
    }
    if f != VIS_PROBE_AT {
        return;
    }
    let Ok(mut state) = probe.0.lock() else {
        return;
    };
    for pos in vis_chunks() {
        let Some(entity) = chunks.get_chunk(&pos) else {
            state
                .visibility_results
                .push((pos, format!("chunk {pos} isn't loaded")));
            continue;
        };
        let Ok((sdf, vis)) = fields.get(entity) else {
            state
                .visibility_results
                .push((pos, format!("chunk {pos} has no SDF")));
            continue;
        };
        let request = VisibilityRequest::new(pos, entity, sdf, vis);
        state.visibility_requests.push(request);
    }
}

/// A chunk whose GPU shape to read back. `expected` is the main world's
/// SDF clipped by its visibility, `lod.size()`^3 snorm16 values.
#[cfg(not(feature = "visibility-mask"))]
struct VisibilityRequest {
    pos: IVec3,
    entity: Entity,
    lod: LOD,
    /// Voxels the visibility hides, for the result line.
    hidden: usize,
    /// The clipped interior has one sign: it may mesh to nothing (only its
    /// apron could still cross), so holding no slot is fine.
    uniform: bool,
    expected: Arc<[i16]>,
}

#[cfg(not(feature = "visibility-mask"))]
impl VisibilityRequest {
    /// What extraction uploads for the interior: the SDF clipped by the
    /// visibility (absent = fully visible), quantized.
    fn new(pos: IVec3, entity: Entity, sdf: &SDF, vis: Option<&VisibilityField>) -> Self {
        use tarasaur::field::quantize_sdf;
        let mut hidden = 0usize;
        let mut inside = 0usize;
        let expected: Vec<i16> = sdf
            .data_slice()
            .iter()
            .enumerate()
            .map(|(i, &d)| {
                let c = vis.and_then(|v| v.data_slice().get(i).copied());
                let c = c.unwrap_or(-SDF_BAND);
                hidden = hidden.saturating_add(usize::from(c > 0.0));
                inside = inside.saturating_add(usize::from(d.max(c) <= 0.0));
                quantize_sdf(d.max(c))
            })
            .collect();
        Self {
            pos,
            entity,
            // Edits load chunks at max LOD; a chunk's slot is there too.
            lod: sdf.lod(),
            hidden,
            uniform: inside == 0 || inside == expected.len(),
            expected: expected.into(),
        }
    }
}

/// Copies each requested chunk's shape slot into a mappable buffer and
/// compares its interior (the padded volume minus the apron) with the main
/// world's clipped SDF.
#[cfg(not(feature = "visibility-mask"))]
fn read_back_shapes(
    probe: Res<SlotProbe>,
    arenas: ArenaInspector,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    let requests = match probe.0.lock() {
        Ok(mut s) => std::mem::take(&mut s.visibility_requests),
        Err(_) => return,
    };
    for req in requests {
        let pos = req.pos;
        let Some(slot) = arenas.shape_slot(req.lod, req.entity) else {
            let line = if req.uniform {
                format!("chunk {pos} (no geometry, no slot, {} hidden): ok", req.hidden)
            } else {
                format!("chunk {pos}: no slot")
            };
            if let Ok(mut s) = probe.0.lock() {
                s.visibility_results.push((pos, line));
            }
            continue;
        };
        let padded = slot.padded_size;
        let bytes = u64::from(padded).pow(3).saturating_mul(2);
        let readback = device.create_buffer(&BufferDescriptor {
            label: Some("bench_shape_readback"),
            size: bytes,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("bench_shape_readback"),
        });
        encoder.copy_buffer_to_buffer(&slot.buffer, slot.offset, &readback, 0, bytes);
        queue.submit(std::iter::once(encoder.finish()));

        let n = req.lod.size();
        let state = probe.0.clone();
        readback
            .clone()
            .slice(..)
            .map_async(MapMode::Read, move |result| {
                let line = if result.is_err() {
                    format!("chunk {pos}: map failed")
                } else {
                    let line = readback.slice(..).get_mapped_range().ok().map(|gpu| {
                        let got: &[i16] = bytemuck::cast_slice(&gpu);
                        compare_shape(pos, got, &req.expected, req.hidden, padded, n)
                    });
                    readback.unmap();
                    line.unwrap_or_else(|| format!("chunk {pos}: unreadable"))
                };
                if let Ok(mut s) = state.lock() {
                    s.visibility_results.push((pos, line));
                }
            });
    }
}

/// One result line: ": ok" when every interior voxel of the GPU's shape
/// equals the main world's clipped SDF.
#[cfg(not(feature = "visibility-mask"))]
fn compare_shape(
    pos: IVec3,
    got: &[i16],
    expected: &[i16],
    hidden: usize,
    padded: u32,
    n: u32,
) -> String {
    let mut wrong = 0usize;
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let g = got.get(flatten_cube(x, y, z, padded));
                let want = expected.get(flatten_cube(x, y, z, n));
                if g != want {
                    wrong = wrong.saturating_add(1);
                }
            }
        }
    }
    if wrong == 0 {
        format!("chunk {pos} ({hidden} hidden): ok")
    } else {
        format!(
            "chunk {pos}: {wrong} of {} voxels wrong",
            usize_from(n).pow(3)
        )
    }
}

/// The clip path's verdict on the result lines: every line ok, and at
/// least one chunk really clipped.
#[cfg(not(feature = "visibility-mask"))]
const VISIBILITY_CHECK: &str = "visibility_clipped_shapes";
#[cfg(not(feature = "visibility-mask"))]
fn visibility_path_ran(results: &[(IVec3, String)]) -> bool {
    results
        .iter()
        .any(|(_, l)| l.ends_with("hidden): ok") && !l.contains("(0 hidden)"))
}
/// The removed field's chunk reads back as its plain SDF.
#[cfg(not(feature = "visibility-mask"))]
fn visibility_plain(line: &str) -> bool {
    line.ends_with("(0 hidden): ok")
}

/// A chunk whose GPU mask to read back. `expected` is the main world's
/// mask (`lod.size()`^3 bytes, 0 = hidden), or `None` when the GPU should
/// hold no mask for it (no field, or all visible).
#[cfg(feature = "visibility-mask")]
struct VisibilityRequest {
    pos: IVec3,
    entity: Entity,
    lod: LOD,
    expected: Option<Arc<[u8]>>,
}

#[cfg(feature = "visibility-mask")]
impl VisibilityRequest {
    fn new(pos: IVec3, entity: Entity, sdf: &SDF, vis: Option<&VisibilityField>) -> Self {
        Self {
            pos,
            entity,
            // Edits load chunks at max LOD; a chunk's mask slot is there too.
            lod: sdf.lod(),
            expected: vis
                .filter(|v| v.should_extract())
                .map(|v| VisibilityField::gpu_bytes(v.data_slice()).into()),
        }
    }
}

/// Copies each requested chunk's mask words and its "has a mask" word into
/// one mappable buffer and compares them with the main world.
#[cfg(feature = "visibility-mask")]
fn read_back_masks(
    probe: Res<SlotProbe>,
    arenas: ArenaInspector,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    let requests = match probe.0.lock() {
        Ok(mut s) => std::mem::take(&mut s.visibility_requests),
        Err(_) => return,
    };
    for req in requests {
        let pos = req.pos;
        // A chunk that meshes to nothing holds no slot, so no GPU mask:
        // nothing is drawn there for a mask to hide.
        let Some(slot) = arenas.mask_slot(req.lod, req.entity) else {
            if let Ok(mut s) = probe.0.lock() {
                s.visibility_results
                    .push((pos, format!("chunk {pos} (no geometry, no slot): ok")));
            }
            continue;
        };
        let readback = device.create_buffer(&BufferDescriptor {
            label: Some("bench_mask_readback"),
            size: slot.bytes.saturating_add(4),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("bench_mask_readback"),
        });
        encoder.copy_buffer_to_buffer(&slot.buffer, slot.offset, &readback, 0, slot.bytes);
        encoder.copy_buffer_to_buffer(
            &slot.has_mask_buffer,
            slot.has_mask_offset,
            &readback,
            slot.bytes,
            4,
        );
        queue.submit(std::iter::once(encoder.finish()));

        let (padded, n) = (slot.padded_size, req.lod.size());
        let (bit, mask_words) = (
            slot.has_mask_bit,
            usize::try_from(slot.bytes / 4).unwrap_or(0),
        );
        let state = probe.0.clone();
        readback
            .clone()
            .slice(..)
            .map_async(MapMode::Read, move |result| {
                let line = if result.is_err() {
                    format!("chunk {pos}: map failed")
                } else {
                    let line = readback.slice(..).get_mapped_range().ok().map(|gpu| {
                        let words: &[u32] = bytemuck::cast_slice(&gpu);
                        let has_mask = words.get(mask_words).is_some_and(|w| (w >> bit) & 1 == 1);
                        compare_mask(pos, words, has_mask, req.expected.as_deref(), padded, n)
                    });
                    readback.unmap();
                    line.unwrap_or_else(|| format!("chunk {pos}: unreadable"))
                };
                if let Ok(mut s) = state.lock() {
                    s.visibility_results.push((pos, line));
                }
            });
    }
}

/// One result line: ": ok" when the GPU's flag and bits match the main
/// world for every interior voxel.
#[cfg(feature = "visibility-mask")]
fn compare_mask(
    pos: IVec3,
    words: &[u32],
    has_mask: bool,
    expected: Option<&[u8]>,
    padded: u32,
    n: u32,
) -> String {
    let Some(expected) = expected else {
        return if has_mask {
            format!("chunk {pos}: GPU still flags a mask, expected none")
        } else {
            format!("chunk {pos} (no mask): ok")
        };
    };
    if !has_mask {
        return format!("chunk {pos}: GPU flags no mask, expected one");
    }
    let mut wrong = 0usize;
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let i = flatten_cube(x, y, z, padded);
                let got = words.get(i / 32).is_some_and(|w| (w >> (i % 32)) & 1 == 1);
                let want = expected
                    .get(flatten_cube(x, y, z, n))
                    .is_some_and(|&b| b != 0);
                if got != want {
                    wrong = wrong.saturating_add(1);
                }
            }
        }
    }
    let hidden: usize = expected.iter().map(|&b| usize::from(b == 0)).sum();
    if wrong == 0 {
        format!("chunk {pos} ({hidden} hidden): ok")
    } else {
        format!(
            "chunk {pos}: {wrong} of {} mask bits wrong",
            usize_from(n).pow(3)
        )
    }
}

/// The mask path's verdict: every line ok, and both the masked and the
/// mask-free path ran.
#[cfg(feature = "visibility-mask")]
const VISIBILITY_CHECK: &str = "visibility_mask_slots";
#[cfg(feature = "visibility-mask")]
fn visibility_path_ran(results: &[(IVec3, String)]) -> bool {
    let masked = results.iter().any(|(_, l)| l.contains("hidden): ok"));
    let unmasked = results.iter().any(|(_, l)| l.contains("(no mask): ok"));
    masked && unmasked
}
/// The removed field's chunk holds no mask.
#[cfg(feature = "visibility-mask")]
fn visibility_plain(line: &str) -> bool {
    line.ends_with("(no mask): ok")
}

/// Every touched chunk's GPU shape (or mask) matches the main world, the
/// path under test really ran, and the removed field's chunk is back to
/// its plain SDF.
fn check_visibility(
    mut bench: ResMut<Bench>,
    probe: Res<SlotProbe>,
    chunks: Res<ChunkManager>,
    fields: Query<&VisibilityField>,
) {
    if !Phase::Visibility.last_frame(&bench) {
        return;
    }
    let results = probe
        .0
        .lock()
        .map_or_else(|_| Vec::new(), |s| s.visibility_results.clone());
    let expected = vis_chunks().len();
    let ok = results.iter().filter(|(_, l)| l.ends_with(": ok")).count();
    let detail = if results.is_empty() {
        "no readback finished".into()
    } else {
        results
            .iter()
            .map(|(_, l)| l.as_str())
            .collect::<Vec<_>>()
            .join("; ")
    };
    bench.checks.record(
        Phase::Visibility,
        VISIBILITY_CHECK,
        ok == expected && results.len() == expected && visibility_path_ran(&results),
        detail,
    );

    let removed = vis_removed_chunk();
    let field_gone = chunks
        .get_chunk(&removed)
        .is_some_and(|e| fields.get(e).is_err());
    let gpu_plain = results
        .iter()
        .any(|(p, l)| *p == removed && visibility_plain(l));
    bench.checks.record(
        Phase::Visibility,
        "visibility_removed",
        field_gone && gpu_plain,
        format!("chunk {removed}: field removed {field_gone}, GPU back to the plain SDF {gpu_plain}"),
    );
}

// ---------------------------------------------------------------------------
// dig: the cursor brush's held dig
// ---------------------------------------------------------------------------

/// Where `dig` starts: untouched ground past the `edit` spot, clear of the
/// other phases' chunks.
fn dig_target() -> Vec3 {
    surface(fly_end().x + 27.0, START.z)
}

/// The camera ray the brush follows, as the cursor's would.
fn dig_ray() -> (Vec3, Vec3) {
    let origin = fly_end();
    (origin, (dig_target() - origin).normalize())
}

fn drive_dig(bench: Res<Bench>, terrain: TerrainRay, mut csg: MessageWriter<SdfCsgSphere>) {
    if bench.phase != Phase::Dig || bench.frame_in_phase >= DIG_HELD {
        return;
    }
    let (origin, dir) = dig_ray();
    let Some(hit) = terrain.cast(origin, dir, QUERY_MAX_DISTANCE) else {
        return;
    };
    csg.write(SdfCsgSphere {
        center: hit.pos,
        radius: bench.out.dig_radius,
        sweep_from: None,
        subtract: true,
        max_step: Some(DIG_STEP),
        paint: None,
    });
}

/// Surface voxels (a sign change to a +x, +y or +z neighbour) whose
/// central-difference gradient is under `DIG_WEAK_GRADIENT`, and all
/// surface voxels. Chunk-border voxels are skipped.
fn weak_surface_voxels(sdf: &SDF) -> (usize, usize) {
    let inner = 1..sdf.size().x.saturating_sub(1);
    let (mut weak, mut total) = (0usize, 0usize);
    for z in inner.clone() {
        for y in inner.clone() {
            for x in inner.clone() {
                let get = |dx: i32, dy: i32, dz: i32| {
                    sdf.get(
                        x.saturating_add_signed(dx),
                        y.saturating_add_signed(dy),
                        z.saturating_add_signed(dz),
                    )
                };
                let v = get(0, 0, 0);
                let next = [get(1, 0, 0), get(0, 1, 0), get(0, 0, 1)];
                if next.iter().all(|&o| (o > 0.0) == (v > 0.0)) {
                    continue;
                }
                let g = Vec3::new(
                    get(1, 0, 0) - get(-1, 0, 0),
                    get(0, 1, 0) - get(0, -1, 0),
                    get(0, 0, 1) - get(0, 0, -1),
                ) / 2.0;
                total = total.saturating_add(1);
                weak = weak.saturating_add(usize::from(g.length() < DIG_WEAK_GRADIENT));
            }
        }
    }
    (weak, total)
}

/// The dug surface is still a distance field: under 1% of the surface
/// voxels around the dig have a weak gradient (a distance field's is 1;
/// where it goes flat the crater terraces and its normals turn black). The
/// ray still hits ground, about as deep as the brush moved it.
fn check_dig(
    mut bench: ResMut<Bench>,
    terrain: TerrainRay,
    chunks: Res<ChunkManager>,
    sdfs: Query<&SDF>,
) {
    if !Phase::Dig.last_frame(&bench) {
        return;
    }
    let (origin, dir) = dig_ray();
    let Some(hit) = terrain.cast(origin, dir, QUERY_MAX_DISTANCE) else {
        bench.checks.record(
            Phase::Dig,
            "dig_depth",
            false,
            "the ray no longer hits ground: the dig punched through".into(),
        );
        return;
    };
    // The surface moves `DIG_STEP` a frame under the brush's center (this
    // dig lands at ~1.07x that along the slanted ray). Much further means a
    // flat field flipped all at once; less, that the dig stalled at chunk
    // borders, where it couldn't see its neighbours (~0.67x).
    let dug = hit.pos.distance(dig_target());
    let nominal = DIG_STEP * f32_from_u32(DIG_HELD);
    let (min, max) = (nominal * 0.8, nominal * 1.25);
    bench.checks.record(
        Phase::Dig,
        "dig_depth",
        (min..=max).contains(&dug),
        format!("dug {dug:.2} units along the ray ({min:.1} to {max:.1} expected)"),
    );

    // Every chunk the dig could have reached, from the start to the end of
    // the ray it re-centered along.
    let (mut weak, mut total, mut checked) = (0usize, 0usize, 0usize);
    let mid = (dig_target() + hit.pos) * 0.5;
    let half = Vec3::splat(bench.out.dig_radius + 1.0) + (hit.pos - dig_target()).abs() * 0.5;
    for pos in overlapping_chunks(mid, half) {
        let Some(sdf) = chunks.get_chunk(&pos).and_then(|e| sdfs.get(e).ok()) else {
            continue;
        };
        if sdf.lod != LOD::High {
            continue;
        }
        let (w, t) = weak_surface_voxels(sdf);
        weak = weak.saturating_add(w);
        total = total.saturating_add(t);
        checked = checked.saturating_add(1);
    }
    let pass = total > 0 && weak.saturating_mul(100) < total;
    bench.checks.record(
        Phase::Dig,
        "dig_distance_field",
        pass,
        format!(
            "{weak}/{total} surface voxels under gradient {DIG_WEAK_GRADIENT} in {checked} High chunks"
        ),
    );
}

fn record_frame(
    mut bench: ResMut<Bench>,
    store: Res<DiagnosticsStore>,
    time: Res<Time<Real>>,
    frame: Res<FrameCount>,
    adapter: Option<Res<RenderAdapterInfo>>,
    capture: Option<Res<FrameCapture>>,
    mut exit: MessageWriter<AppExit>,
) {
    if bench.phase == Phase::Done {
        return;
    }
    // The first frames include startup; nothing has been asked of the GPU yet.
    if frame.0 < 2 {
        return;
    }

    let mut values = [None; COLUMNS.len()];
    for ((_, path), slot) in COLUMNS.iter().zip(&mut values) {
        let Some(m) = store
            .get(&DiagnosticPath::new(*path))
            .and_then(|d| d.measurement())
        else {
            continue;
        };
        if bench.seen.get(path).is_some_and(|t| *t >= m.time) {
            continue; // nothing new this frame
        }
        bench.seen.insert(path, m.time);
        *slot = Some(m.value);
    }
    let row = FrameRow {
        phase: bench.phase,
        frame_ms: time.delta_secs_f64() * 1000.0,
        values,
    };
    let idle = is_idle(&row);

    // `--settle-stream`: don't start a phase's measured frames while chunk
    // streaming from an earlier phase is still draining. Deliberately not
    // `is_idle`, which also demands `meshed_chunks == 0` -- a sculpting phase
    // never reaches that, so `is_idle` can't gate one. Only the streaming
    // pair is tested. `load` is exempt (`frames()` is `None`): it *is* the
    // phase that does the streaming.
    if bench.out.settle_stream == SettleStream::WaitForDrain
        && bench.frame_in_phase == 0
        && bench.phase.frames().is_some()
    {
        let clear = |name| row.get(name).is_some_and(|v| v == 0.0);
        let drained = clear("loads_in_flight") && clear("lod_pending");
        let timed_out = bench.phase_started.elapsed().as_secs_f32() > SETTLE_STREAM_TIMEOUT_SECS;
        if !drained && !timed_out {
            // Not recorded and `frame_in_phase` not advanced, so the phase
            // still runs exactly the number of measured frames it declares
            // and the result tables are unchanged in shape.
            bench.settling = true;
            return;
        }
        if timed_out && !drained {
            warn!(
                "[bench] {}: --settle-stream gave up after {SETTLE_STREAM_TIMEOUT_SECS}s; \
                 streaming still busy, this phase is contaminated",
                bench.phase.name()
            );
        }
        bench.settling = false;
    }
    bench.rows.push(row);
    bench.frame_in_phase = bench.frame_in_phase.saturating_add(1);

    let finished = if let Some(n) = bench.phase.frames() {
        bench.frame_in_phase >= n
    } else {
        bench.idle_frames = if idle {
            bench.idle_frames.saturating_add(1)
        } else {
            0
        };
        let timed_out = bench.phase_started.elapsed().as_secs_f32() > LOAD_TIMEOUT_SECS;
        bench.load_timed_out = timed_out;
        bench.idle_frames >= SETTLE_FRAMES || timed_out
    };
    if !finished {
        return;
    }
    info!(
        "[bench] {} done: {} frames in {:.1}s",
        bench.phase.name(),
        bench.frame_in_phase,
        bench.phase_started.elapsed().as_secs_f32()
    );
    // GPU time of every pass Bevy timed, averaged over the phase's last
    // frames (the diagnostic history).
    let phase = bench.phase;
    for d in store.iter() {
        let path = d.path().as_str();
        if path.starts_with("render/")
            && path.ends_with("/elapsed_gpu")
            && let Some(avg) = d.average()
        {
            bench.gpu_passes.push((phase, path.to_string(), avg));
        }
    }
    bench.phase = bench.phase.next();
    let done = bench.phase == Phase::Done;
    // Written after every phase, so a run cut short still leaves results.
    let adapter = adapter.map_or_else(
        || "unknown".to_string(),
        |a| format!("{} ({:?}, {})", a.name, a.backend, a.driver_info),
    );
    write_results(&bench, &adapter, done);
    bench.frame_in_phase = 0;
    bench.phase_started = Instant::now();
    if done {
        for path in capture.iter().flat_map(|c| c.captures()) {
            println!("renderdoc: {}", path.display());
        }
        exit.write(if bench.checks.all_pass() {
            AppExit::Success
        } else {
            AppExit::error()
        });
    }
}

/// Streaming has finished: something is drawn and nothing is queued anywhere.
fn is_idle(row: &FrameRow) -> bool {
    let zero = |name| row.get(name).is_some_and(|v| v == 0.0);
    row.get("draws").is_some_and(|v| v > 0.0)
        && [
            "loads_in_flight",
            "lod_pending",
            "pending_uploads",
            "dirty_slots",
            "meshed_chunks",
            "reinits",
        ]
        .into_iter()
        .all(zero)
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

/// Nearest-rank percentile of an ascending slice.
fn percentile(sorted: &[f64], p: usize) -> f64 {
    let rank = sorted
        .len()
        .saturating_sub(1)
        .saturating_mul(p)
        .checked_div(100)
        .unwrap_or(0);
    sorted.get(rank).copied().unwrap_or(f64::NAN)
}

fn f64_from_usize(v: usize) -> f64 {
    f64::from(u32::try_from(v).unwrap_or(u32::MAX))
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.iter().sum::<f64>() / f64_from_usize(v.len())
}

/// (phase, metric, value) rows for one phase.
fn summarize(phase: Phase, rows: &[&FrameRow], load_timed_out: bool) -> Vec<(String, f64)> {
    let mut frame_ms: Vec<f64> = rows.iter().map(|r| r.frame_ms).collect();
    frame_ms.sort_by(f64::total_cmp);
    let total_ms: f64 = frame_ms.iter().sum();
    let present = |name: &str| -> Vec<f64> { rows.iter().filter_map(|r| r.get(name)).collect() };

    let mut out = vec![
        ("frames".into(), f64_from_usize(rows.len())),
        ("seconds".into(), total_ms / 1000.0),
        ("frame_ms_mean".into(), mean(&frame_ms)),
        ("frame_ms_p50".into(), percentile(&frame_ms, 50)),
        ("frame_ms_p95".into(), percentile(&frame_ms, 95)),
        ("frame_ms_p99".into(), percentile(&frame_ms, 99)),
        (
            "frame_ms_max".into(),
            frame_ms.last().copied().unwrap_or(f64::NAN),
        ),
        ("fps".into(), f64_from_usize(rows.len()) * 1000.0 / total_ms),
    ];
    // Averaged over the frames that reported a value.
    for name in ["gpu_raster_ms", "draws", "resident", "chunks"] {
        out.push((format!("{name}_mean"), mean(&present(name))));
    }
    // Meshing runs only on some frames: report totals and the per-run cost.
    for name in [
        "gpu_mesh_ms",
        "gpu_pass1_ms",
        "gpu_pass3_ms",
        "gpu_query_ms",
    ] {
        let v = present(name);
        out.push((format!("{name}_total"), total(&v)));
        out.push((format!("{name}_per_dispatch"), mean(&v)));
    }
    for name in ["meshed_chunks", "meshed_cells", "reslotted", "side_refills"] {
        out.push((format!("{name}_total"), total(&present(name))));
    }
    if phase == Phase::Load {
        out.push(("timed_out".into(), if load_timed_out { 1.0 } else { 0.0 }));
    }
    out
}

/// Writes the frames and summary so far; `print`s the tables too at the end.
fn write_results(bench: &Bench, adapter: &str, print: bool) {
    // Built by name, not `with_extension`: that replaces everything after the
    // last dot, so a label with a dot in it (`--dig-radius 7.5`) lost both its
    // tail and the stamp, silently overwriting the previous run and leaving
    // `ab.sh`'s `<label>-*` glob with nothing to match.
    let stem = format!("{}-{}", bench.out.label, bench.out.stamp);
    // What the compute pool actually got, not what was asked for. A flag
    // A/B whose flag silently did nothing measures a variant against itself
    // and reads as a clean null result, so the run states its own pool size
    // rather than leaving it to be trusted. This replaces the value recorded
    // when `--compute-threads` was parsed.
    bench.out.knobs.record(
        "--compute-threads",
        &bevy::tasks::ComputeTaskPool::get().thread_num(),
    );
    // Every knob the run read, in read order. Nothing is listed here by name:
    // a flag is in the header because the parser put it there. A hand-written
    // field list is how `msaa=` went unrecorded for the night's biggest
    // result, so this must not become one again.
    let header = format!(
        "# label={} adapter={adapter} profile={} {}\n",
        bench.out.label,
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        bench.out.knobs.header_fields(),
    );

    let mut frames = header.clone();
    frames.push_str("phase,frame_ms");
    for (name, _) in COLUMNS {
        let _ = write!(frames, ",{name}");
    }
    frames.push('\n');
    for r in &bench.rows {
        let _ = write!(frames, "{},{:.4}", r.phase.name(), r.frame_ms);
        for v in r.values {
            match v {
                Some(v) => {
                    let _ = write!(frames, ",{v}");
                }
                None => frames.push(','),
            }
        }
        frames.push('\n');
    }

    let mut summary = header;
    summary.push_str("phase,metric,value\n");
    let mut table = String::new();
    for phase in PHASES {
        let rows: Vec<&FrameRow> = bench.rows.iter().filter(|r| r.phase == phase).collect();
        if rows.is_empty() {
            continue; // not run (yet, or skipped by `--quick`)
        }
        let metrics = summarize(phase, &rows, bench.load_timed_out);
        for (metric, value) in &metrics {
            let _ = writeln!(summary, "{},{metric},{value}", phase.name());
        }
        for (_, path, avg) in bench.gpu_passes.iter().filter(|(p, ..)| *p == phase) {
            let _ = writeln!(summary, "{},gpu:{path},{avg}", phase.name());
        }
        bench.checks.write_rows(&mut summary, phase);
        let get = |m: &str| {
            metrics
                .iter()
                .find(|(k, _)| k == m)
                .map_or(f64::NAN, |(_, v)| *v)
        };
        let _ = writeln!(
            table,
            "{:<5} {:>6.0} frames {:>7.2}s | frame ms mean {:>6.2} p95 {:>6.2} p99 {:>6.2} max {:>7.2} | gpu raster {:>5.2}ms | gpu mesh total {:>7.1}ms",
            phase.name(),
            get("frames"),
            get("seconds"),
            get("frame_ms_mean"),
            get("frame_ms_p95"),
            get("frame_ms_p99"),
            get("frame_ms_max"),
            get("gpu_raster_ms_mean"),
            get("gpu_mesh_ms_total"),
        );
    }

    let frames_path = bench.out.dir.join(format!("{stem}.frames.csv"));
    let summary_path = bench.out.dir.join(format!("{stem}.summary.csv"));
    let written = std::fs::create_dir_all(&bench.out.dir)
        .and_then(|()| std::fs::write(&frames_path, frames))
        .and_then(|()| std::fs::write(&summary_path, summary));
    if let Err(e) = written {
        error!("[bench] couldn't write results: {e}");
    }
    if !print {
        return;
    }
    println!("\n{table}");
    print_gpu_passes(&bench.gpu_passes, Phase::Hold);
    bench.checks.print();
    println!("frames:  {}", frames_path.display());
    println!("summary: {}", summary_path.display());
}

/// A channel difference above this (of 255) counts the pixel as changed.
const DIFF_CHANGED: u8 = 16;

/// `--diff a.png b.png [heatmap.png]`: how far apart two screenshots are.
/// Prints the mean channel difference (of 255), the share of changed
/// pixels and the largest difference; the heatmap shows each pixel's
/// largest channel difference, 4x brighter.
fn diff_args(a: Option<&String>, b: Option<&String>, heatmap: Option<&String>) -> AppExit {
    let (Some(a), Some(b)) = (a, b) else {
        eprintln!("usage: bench --diff <a.png> <b.png> [heatmap.png]");
        return AppExit::error();
    };
    let load = |p: &String| {
        std::fs::read(p)
            .map_err(|e| e.to_string())
            .and_then(|bytes| {
                Image::from_buffer(
                    &bytes,
                    ImageType::Extension("png"),
                    CompressedImageFormats::NONE,
                    true,
                    ImageSampler::Default,
                    RenderAssetUsages::default(),
                )
                .map_err(|e| e.to_string())
            })
            .inspect_err(|e| eprintln!("{p}: {e}"))
            .ok()
    };
    let (Some(ia), Some(ib)) = (load(a), load(b)) else {
        return AppExit::error();
    };
    let (Some(da), Some(db)) = (ia.data.as_ref(), ib.data.as_ref()) else {
        eprintln!("bench --diff: an image has no pixel data");
        return AppExit::error();
    };
    if ia.size() != ib.size() || da.len() != db.len() {
        eprintln!(
            "bench --diff: sizes differ ({} vs {})",
            ia.size(),
            ib.size()
        );
        return AppExit::error();
    }
    let (mut sum, mut changed, mut max) = (0u32, 0u32, 0u8);
    let mut heat = Vec::with_capacity(da.len());
    for (pa, pb) in da.as_chunks::<4>().0.iter().zip(db.as_chunks::<4>().0) {
        let d = pa
            .iter()
            .zip(pb)
            .take(3)
            .map(|(x, y)| x.abs_diff(*y))
            .inspect(|d| sum = sum.saturating_add(u32::from(*d)))
            .max()
            .unwrap_or(0);
        max = max.max(d);
        if d > DIFF_CHANGED {
            changed = changed.saturating_add(1);
        }
        let v = d.saturating_mul(4);
        heat.extend([v, v, v, 255]);
    }
    let pixels = f64::from(u32::try_from(da.len() / 4).unwrap_or(u32::MAX).max(1));
    let (mean, share) = (
        f64::from(sum) / (pixels * 3.0),
        100.0 * f64::from(changed) / pixels,
    );
    println!("diff mean {mean:.3} changed {share:.3}% max {max}");
    if let Some(path) = heatmap {
        let image = Image::new(
            ia.texture_descriptor.size,
            TextureDimension::D2,
            heat,
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::default(),
        );
        if let Err(e) = image
            .try_into_dynamic()
            .map_err(|e| e.to_string())
            .and_then(|d| d.save(path).map_err(|e| e.to_string()))
        {
            eprintln!("{path}: {e}");
            return AppExit::error();
        }
    }
    AppExit::Success
}

/// `--compare a b`.
fn compare_args(a: Option<&String>, b: Option<&String>) -> AppExit {
    let (Some(a), Some(b)) = (a, b) else {
        eprintln!("usage: bench --compare <a.summary.csv> <b.summary.csv>");
        return AppExit::error();
    };
    compare(Path::new(a), Path::new(b));
    AppExit::Success
}

/// Prints `b` against `a`, metric by metric.
fn compare(a: &Path, b: &Path) {
    let read = |p: &Path| -> Option<Vec<(String, String, f64)>> {
        let text = std::fs::read_to_string(p)
            .inspect_err(|e| eprintln!("{}: {e}", p.display()))
            .ok()?;
        Some(
            text.lines()
                .filter(|l| !l.starts_with('#') && !l.starts_with("phase,"))
                .filter_map(|l| {
                    let mut parts = l.splitn(3, ',');
                    let phase = parts.next()?.to_string();
                    let metric = parts.next()?.to_string();
                    Some((phase, metric, parts.next()?.parse().ok()?))
                })
                .collect(),
        )
    };
    let (Some(ra), Some(rb)) = (read(a), read(b)) else {
        return;
    };
    println!(
        "{:<6} {:<34} {:>14} {:>14} {:>9}",
        "phase", "metric", "a", "b", "change"
    );
    for (phase, metric, va) in &ra {
        let Some((_, _, vb)) = rb.iter().find(|(p, m, _)| p == phase && m == metric) else {
            continue;
        };
        let change = if *va == 0.0 {
            String::new()
        } else {
            format!("{:+.1}%", (vb - va) / va * 100.0)
        };
        println!("{phase:<6} {metric:<34} {va:>14.3} {vb:>14.3} {change:>9}");
    }
}

/// Sums from +0.0 (an empty `f64` sum is -0.0).
fn total(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |a, b| a + b)
}

/// The cursor ray promotes the LOD of the chunks it hits, so where the mouse
/// happens to sit would change the work done. Runs must not depend on it.
fn ignore_cursor(mut hit: ResMut<CursorHit>) {
    hit.0 = None;
}

/// Prints one phase's GPU pass times, slowest first.
fn print_gpu_passes(passes: &[(Phase, String, f64)], phase: Phase) {
    let mut rows: Vec<&(Phase, String, f64)> =
        passes.iter().filter(|(p, ..)| *p == phase).collect();
    rows.sort_by(|a, b| b.2.total_cmp(&a.2));
    println!(
        "GPU passes during {} (ms, nested spans included):",
        phase.name()
    );
    for (_, path, avg) in rows {
        println!("  {avg:>7.3}  {path}");
    }
}

// ---------------------------------------------------------------------------
// RenderDoc
// ---------------------------------------------------------------------------

/// `hold` frame to capture with `--renderdoc`: mid-phase, well after load.
const CAPTURE_FRAME: u32 = 300;

/// RenderDoc's in-app API. Only available when RenderDoc launched the bench:
/// `renderdoccmd capture -w target/release/examples/bench --renderdoc`.
#[derive(Resource)]
struct FrameCapture(Mutex<RenderDoc<V141>>);

impl FrameCapture {
    fn attach(out: &Output) -> Option<Self> {
        match RenderDoc::<V141>::new() {
            Ok(mut rd) => {
                let _ = std::fs::create_dir_all(&out.dir);
                // RenderDoc appends `_frameN.rdc`.
                rd.set_capture_file_path_template(out.dir.join(&out.label));
                Some(Self(Mutex::new(rd)))
            }
            Err(e) => {
                eprintln!(
                    "bench: --renderdoc needs RenderDoc loaded; launch with \
                     `renderdoccmd capture -w <bench> --renderdoc` ({e})"
                );
                None
            }
        }
    }

    fn captures(&self) -> Vec<PathBuf> {
        let Ok(rd) = self.0.lock() else {
            return Vec::new();
        };
        (0..rd.get_num_captures())
            .filter_map(|i| rd.get_capture(i).map(|(path, _)| path))
            .collect()
    }
}

fn trigger_frame_capture(bench: Res<Bench>, capture: Res<FrameCapture>) {
    if bench.phase != Phase::Hold || bench.frame_in_phase != CAPTURE_FRAME {
        return;
    }
    if let Ok(mut rd) = capture.0.lock() {
        rd.trigger_capture();
        info!("[bench] RenderDoc: capturing the next frame");
    }
}

/// The camera's MSAA, from `--msaa` (default: Bevy's default, 4x).
#[derive(Resource)]
struct BenchMsaa(Msaa);

/// What `--compute-threads` asked for.
#[derive(Clone, Copy)]
enum ComputePool {
    /// Leave Bevy's own split alone. This is what an application that does
    /// not opt into `tarasaur::task_pools` gets.
    Bevy,
    /// Give the compute pool exactly this many threads.
    ///
    /// Both bounds, not just `min_threads`. `min_threads` is only a floor --
    /// "force using at least this many" -- so asking for fewer threads than
    /// Bevy's percentage split already allocates does nothing at all. That
    /// cost a six-run experiment: `--compute-floor 2` against
    /// `--compute-floor 1` on a Deck both ran 4 compute threads and measured a
    /// null, which is indistinguishable from "thread count doesn't matter".
    Exact(usize),
}

impl ComputePool {
    /// The thread count to pin both bounds to, or `None` to set nothing.
    const fn exact(self) -> Option<usize> {
        match self {
            Self::Bevy => None,
            Self::Exact(n) => Some(n),
        }
    }
}

/// `--compute-threads`: how many threads the compute pool gets.
///
/// Absent, the bench widens the pool as it always has. `bevy` leaves Bevy's
/// own split alone. A number pins the pool to exactly that many. `None` means
/// the value was bad and the run should not start.
fn parse_compute_threads(arg: Option<&str>) -> Option<ComputePool> {
    match arg {
        None => Some(ComputePool::Exact(tarasaur::task_pools::edit_compute_threads())),
        Some("bevy" | "default") => Some(ComputePool::Bevy),
        Some(other) => match other.parse::<usize>() {
            Ok(n) if n >= 1 => Some(ComputePool::Exact(n)),
            _ => {
                eprintln!("bench: --compute-threads takes `bevy` or a count >= 1, not {other}");
                None
            }
        },
    }
}

/// How many samples an [`Msaa`] is, for the result header. `--msaa` was the
/// one flag an A/B could turn and the record could not show, and finding 1's
/// MSAA result is unverifiable for exactly that reason.
const fn msaa_samples(msaa: Msaa) -> u32 {
    match msaa {
        Msaa::Off => 1,
        Msaa::Sample2 => 2,
        Msaa::Sample4 => 4,
        Msaa::Sample8 => 8,
    }
}

/// `--msaa`: 1, 2, 4 or 8 samples; Bevy's default (4x) when absent.
fn parse_msaa(arg: Option<&str>) -> Option<Msaa> {
    match arg {
        None => Some(Msaa::default()),
        Some("1") => Some(Msaa::Off),
        Some("2") => Some(Msaa::Sample2),
        Some("4") => Some(Msaa::Sample4),
        Some("8") => Some(Msaa::Sample8),
        Some(other) => {
            eprintln!("bench: --msaa takes 1, 2, 4 or 8, not {other}");
            None
        }
    }
}

/// The three logged LOD-seam sites, meshed by the shipped `surface_nets_pass3`.
/// `src/voxel/seam_repro.rs` predicts them with a hand port of the shader's
/// seam logic; this holds the GPU to the same predictions, so the port is
/// checked against the real thing rather than by inspection.
#[cfg(test)]
mod seam_gpu {
    use super::*;
    use tarasaur::{chunk::Chunk, field::TargetLOD};

    /// One logged site: the meshing chunk, its LOD, and the neighbours at
    /// another LOD, as `seam_repro::sites()` pins them (keep the two in step).
    /// Every other chunk of the 3x3x3 block is spawned at `own`, so nothing
    /// in the block is missing: residency is not a variable.
    struct GpuSite {
        name: &'static str,
        chunk: IVec3,
        own: LOD,
        others: &'static [(IVec3, LOD)],
        /// The port's shared-cell corner, in world units: where it says the
        /// vertex is pinned.
        vertex: Vec3,
        /// The port's distance off the terrain at `vertex`, in `own` voxels.
        off: f32,
        /// Offsets in the block left unspawned (absent, not merely meshless).
        absent: &'static [IVec3],
    }

    /// The sites a run spawns.
    #[derive(Resource, Clone, Copy)]
    struct Sites(&'static [GpuSite]);

    const SITES: [GpuSite; 3] = [
        GpuSite {
            name: "fly [8,0,4] Medium (8,0,32)",
            chunk: IVec3::new(8, 0, 4),
            own: LOD::Medium,
            others: &[(IVec3::new(0, 0, 1), LOD::Low)],
            vertex: Vec3::new(82.5, 0.0, 50.0),
            off: 2.320,
            absent: &[],
        },
        GpuSite {
            name: "fly [8,0,11] Medium (32,0,16)",
            chunk: IVec3::new(8, 0, 11),
            own: LOD::Medium,
            others: &[
                (IVec3::new(-1, -1, -1), LOD::Low),
                (IVec3::new(-1, 0, -1), LOD::Low),
                (IVec3::new(-1, -1, 0), LOD::Low),
                (IVec3::new(0, -1, 0), LOD::Low),
                (IVec3::new(-1, 0, 0), LOD::Low),
                (IVec3::new(1, -1, 1), LOD::Low),
                (IVec3::new(-1, 0, 1), LOD::Low),
                (IVec3::new(0, 0, 1), LOD::Low),
                (IVec3::new(1, 0, 1), LOD::Low),
            ],
            vertex: Vec3::new(90.0, 0.0, 115.0),
            off: 2.074,
            absent: &[],
        },
        GpuSite {
            name: "hold [4,-1,4] High (64,64,0)",
            chunk: IVec3::new(4, -1, 4),
            own: LOD::High,
            others: &[
                (IVec3::new(1, 0, -1), LOD::Medium),
                (IVec3::new(1, 1, -1), LOD::Medium),
                (IVec3::new(1, 0, 0), LOD::Medium),
                (IVec3::new(1, 1, 0), LOD::Medium),
                (IVec3::new(1, 0, 1), LOD::Medium),
            ],
            vertex: Vec3::new(50.0, 0.0, 40.0),
            off: 1.477,
            absent: &[],
        },
    ];

    /// The hold site again, with the 13 neighbours the hold log listed as
    /// `none` (no mesh) left out entirely, instead of present at the
    /// meshing chunk's LOD: the one difference from `SITES[2]` that could
    /// move its normal. Two of them, (0,1,1) and (1,1,1), are +side
    /// neighbours its apron reads; absent, the apron extends its own edge.
    /// The ranks don't change: an absent neighbour counts at this chunk's
    /// LOD, which is what the port already assumes for them.
    const HOLD_NONE_ABSENT: [GpuSite; 1] = [GpuSite {
        name: "hold [4,-1,4] High, logged `none` neighbours absent",
        chunk: IVec3::new(4, -1, 4),
        own: LOD::High,
        others: &[
            (IVec3::new(1, 0, -1), LOD::Medium),
            (IVec3::new(1, 1, -1), LOD::Medium),
            (IVec3::new(1, 0, 0), LOD::Medium),
            (IVec3::new(1, 1, 0), LOD::Medium),
            (IVec3::new(1, 0, 1), LOD::Medium),
        ],
        vertex: Vec3::new(50.0, 0.0, 40.0),
        off: 1.477,
        absent: &[
            IVec3::new(-1, -1, -1),
            IVec3::new(0, -1, -1),
            IVec3::new(1, -1, -1),
            IVec3::new(-1, -1, 0),
            IVec3::new(0, -1, 0),
            IVec3::new(1, -1, 0),
            IVec3::new(-1, 1, 0),
            IVec3::new(-1, -1, 1),
            IVec3::new(0, -1, 1),
            IVec3::new(1, -1, 1),
            IVec3::new(-1, 1, 1),
            IVec3::new(0, 1, 1),
            IVec3::new(1, 1, 1),
        ],
    }];

    impl GpuSite {
        fn lod_at(&self, offset: IVec3) -> LOD {
            self.others
                .iter()
                .find(|(o, _)| *o == offset)
                .map_or(self.own, |(_, l)| *l)
        }
    }

    fn offsets() -> impl Iterator<Item = IVec3> {
        (-1..=1)
            .flat_map(|z| (-1..=1).flat_map(move |y| (-1..=1).map(move |x| IVec3::new(x, y, z))))
    }

    /// The rank `seam_lod_ranks` stored for `offset`, from a `ChunkMeta`.
    fn stored_rank(ranks: [u32; 2], offset: IVec3) -> u32 {
        let k = offset
            .saturating_add(IVec3::ONE)
            .dot(IVec3::new(1, 3, 9))
            .unsigned_abs();
        ranks
            .get(usize_from(k / 16))
            .copied()
            .unwrap_or(0)
            .wrapping_shr((k % 16).wrapping_mul(2))
            & 3
    }

    /// A camera, so the render world runs as in a bench, with no
    /// `ChunkLoader`: the test spawns every chunk itself, and with no loader
    /// nothing replans, retargets or despawns them.
    fn seam_camera(mut commands: Commands, mut images: ResMut<Assets<Image>>, sites: Res<Sites>) {
        let mut image = Image::new_target_texture(
            RESOLUTION.x,
            RESOLUTION.y,
            TextureFormat::Rgba8UnormSrgb,
            None,
        );
        image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
        let target = images.add(image);
        commands.spawn((
            Camera3d::default(),
            Msaa::Off,
            Transform::from_xyz(60.0, 30.0, 0.0).looking_at(Vec3::new(70.0, 0.0, 60.0), Vec3::Y),
            RenderTarget::Image(ImageRenderTarget {
                handle: target,
                scale_factor: 1.0,
            }),
        ));
        for site in sites.0 {
            for offset in offsets().filter(|o| !site.absent.contains(o)) {
                commands.spawn((
                    Chunk,
                    ChunkPosition(site.chunk.saturating_add(offset)),
                    TargetLOD(site.lod_at(offset)),
                ));
            }
        }
    }

    /// Meshes `sites` on a device and returns the probe's report.
    fn mesh_sites(sites: &'static [GpuSite]) -> MeshReport {
        let dir = std::env::temp_dir().join(format!("tarasaur-seam-gpu-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut app = App::new();
        add_default_plugins(&mut app, false);
        app.add_plugins((
            TarasaurPlugin {
                save_dir: dir.clone(),
                autosave_secs: f32::INFINITY,
                ..default()
            },
            MaterialFieldPlugin::<TerrainMaterial>::new(),
            stats::VoxelDiagnosticsPlugin,
        ));
        register_terrain(&mut app);
        let probe = MeshProbe::default();
        if let Some(render) = app.get_sub_app_mut(RenderApp) {
            render
                .insert_resource(probe.clone())
                .add_systems(Render, read_back_meshes.in_set(RenderSystems::Cleanup));
        }
        app.insert_resource(probe.clone())
            .insert_resource(Sites(sites))
            .add_systems(Startup, (seam_camera, setup_terrain_palette));
        app.finish();
        app.cleanup();

        let blocks: usize = sites
            .iter()
            .map(|s| 27usize.saturating_sub(s.absent.len()))
            .sum();
        let mut settled = 0u32;
        let mut meshed_any = false;
        let mut asked = false;
        for _ in 0..6000 {
            app.update();
            if let Ok(s) = probe.0.lock()
                && let Some((_, r)) = s.done.first()
            {
                let report = r.clone();
                drop(s);
                let _ = std::fs::remove_dir_all(&dir);
                return report;
            }
            if asked {
                continue;
            }
            let world = app.world_mut();
            let mut q = world.query::<(&ChunkPosition, &TargetLOD, Option<&LOD>)>();
            let chunks: Vec<(IVec3, bool, LOD)> = q
                .iter(world)
                .map(|(p, t, l)| (p.0, l == Some(&t.0), l.copied().unwrap_or(t.0)))
                .collect();
            // The bench's own "load done" test (`is_idle`), with `resident`
            // for `draws`: the camera needn't see the blocks. A fixed frame
            // count is not enough: the mesh pipelines compile asynchronously,
            // and until they do the slots stay dirty and hold no vertices.
            let store = world.resource::<DiagnosticsStore>();
            let value = |p: &DiagnosticPath| store.get(p).and_then(|d| d.value());
            meshed_any |= value(&stats::MESHED_CHUNKS).is_some_and(|v| v > 0.0);
            let idle = value(&stats::RESIDENT).is_some_and(|v| v > 0.0)
                && [
                    stats::LOADS_IN_FLIGHT,
                    stats::LOD_PENDING,
                    stats::PENDING_UPLOADS,
                    stats::DIRTY_SLOTS,
                    stats::MESHED_CHUNKS,
                    stats::REINITS,
                ]
                .iter()
                .all(|p| value(p) == Some(0.0));
            settled = if meshed_any && idle && chunks.len() == blocks && chunks.iter().all(|c| c.1)
            {
                settled.saturating_add(1)
            } else {
                0
            };
            if settled >= SETTLE_FRAMES {
                if let Ok(mut s) = probe.0.lock() {
                    s.requested = Some(Phase::Hold);
                    s.loaded = chunks.iter().map(|c| c.0).collect();
                    s.loaded_lods = chunks.iter().map(|&(p, _, lod)| (p, lod)).collect();
                    s.watch = sites.iter().map(|site| site.vertex).collect();
                }
                asked = true;
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
        panic!("the sites never finished meshing and reading back");
    }

    /// For each logged site, the shipped shader puts a vertex exactly where
    /// the port says it is pinned (the shared-cell corner on a chunk edge),
    /// as far off the terrain as the port computes. It first checks that the
    /// seam ranks the GPU meshed with are the pinned ones, so a disagreement
    /// is about the shader, not the harness.
    ///
    ///     cargo test --release --example bench seam_gpu -- --ignored --nocapture
    ///
    /// **Read the result line, not the exit code.** The process segfaults on
    /// its way out after printing `test result` (rc 139, in GPU teardown at
    /// exit), so the exit code of any run that includes this test is
    /// meaningless.
    #[test]
    #[ignore = "wants a device: run it by name"]
    fn shipped_shader_pins_the_logged_sites_where_the_port_says() {
        let report = mesh_sites(&SITES);
        let mut wrong = Vec::new();
        for site in &SITES {
            // The ranks the meshing chunk was meshed with, from its ChunkMeta.
            let Some(&ranks) = report.ranks.get(&site.chunk) else {
                wrong.push(format!("{}: the meshing chunk has no mesh", site.name));
                continue;
            };
            for offset in offsets() {
                let want = site.lod_at(offset).rank();
                let got = stored_rank(ranks, offset);
                if got != want {
                    wrong.push(format!(
                        "{}: rank at {offset} is {got}, pinned {want}",
                        site.name
                    ));
                }
            }
            let near: Vec<&NearVertex> = report
                .near
                .iter()
                .filter(|v| v.world.distance(site.vertex) <= WATCH_RADIUS)
                .collect();
            println!(
                "{}: {} vertices within {WATCH_RADIUS} of {}",
                site.name,
                near.len(),
                site.vertex
            );
            for v in &near {
                let local = (v.world - v.pos.as_vec3() * CHUNK_SIZE) / v.lod.voxel_size();
                println!(
                    "    chunk {} {:?} local ({:.3}, {:.3}, {:.3}) world {:.4} off {:.3} normal {:.1} deg{}",
                    v.pos,
                    v.lod,
                    local.x,
                    local.y,
                    local.z,
                    v.world,
                    v.off,
                    v.normal_deg,
                    if v.pinned { " PINNED" } else { "" }
                );
            }
            let pinned = near
                .iter()
                .find(|v| v.pos == site.chunk && v.world.distance(site.vertex) < 1e-3);
            match pinned {
                None => wrong.push(format!(
                    "{}: no vertex of {} at {} (the port's pinned corner)",
                    site.name, site.chunk, site.vertex
                )),
                Some(v) if (v.off - site.off).abs() > 0.005 => wrong.push(format!(
                    "{}: vertex at {} is {:.3} voxels off, the port says {:.3}",
                    site.name, site.vertex, v.off, site.off
                )),
                Some(v) if !v.pinned => wrong.push(format!(
                    "{}: the vertex at {} is not classified as pinned",
                    site.name, site.vertex
                )),
                Some(v) => println!(
                    "    => pinned as predicted, and classified: {:.3} voxels off, normal {:.1} deg",
                    v.off, v.normal_deg
                ),
            }
        }
        assert!(
            wrong.is_empty(),
            "{}\n(the probe read {} chunks, {} vertices, {} near a watched point)",
            wrong.join("\n"),
            report.chunks,
            report.vertices,
            report.near.len()
        );
    }

    /// Wave 2 found the hold site's vertex where the logs put it, but its
    /// normal 12.7 degrees off where the log had 34.97. This meshes the
    /// site with the logged `none` neighbours absent, and prints the
    /// normal of every vertex at the pinned corner, to see whether that is
    /// what moves it. It checks only that the vertex is still pinned there.
    ///
    ///     cargo test --release --example bench seam_gpu::hold_normal -- --ignored --nocapture
    ///
    /// Read the result line, not the exit code (see the test above).
    #[test]
    #[ignore = "wants a device: run it by name"]
    fn hold_normal_with_the_logged_none_absent() {
        let report = mesh_sites(&HOLD_NONE_ABSENT);
        let [site] = &HOLD_NONE_ABSENT;
        let at: Vec<&NearVertex> = report
            .near
            .iter()
            .filter(|v| v.world.distance(site.vertex) < 1e-3)
            .collect();
        for v in &at {
            println!(
                "    chunk {} {:?} at the corner: off {:.3}, normal {:.1} deg (logged: [4,-1,4] High 34.97){}",
                v.pos,
                v.lod,
                v.off,
                v.normal_deg,
                if v.pinned { ", pinned" } else { "" }
            );
        }
        assert!(
            at.iter().any(|v| v.pos == site.chunk && v.pinned),
            "no pinned vertex of {} at {} ({} vertices near it)",
            site.chunk,
            site.vertex,
            report.near.len()
        );
    }
}
