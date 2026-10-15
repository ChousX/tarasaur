//! Repeatable end-to-end benchmark.
//!
//! Runs a fixed scenario on the sandbox terrain and writes per-frame and
//! per-phase results:
//!
//! ```text
//! cargo run --release --example bench -- --label baseline [--msaa 1|2|4|8] [--max-loads N] [--depth-prepass] [--small-arenas]
//! cargo run --release --example bench -- --compare bench_results/A.summary.csv bench_results/B.summary.csv
//! ```
//!
//! Phases, in order:
//! - `load`:  stream the world in from a still camera until nothing is pending.
//! - `hold`:  the same view, nothing changing: steady-state render cost.
//! - `spin`:  turn in place: view-dependent cost (culling).
//! - `fly`:   fly a fixed distance per frame: streaming cost while moving.
//! - `edit`:  sculpt one spot every frame: edit -> re-mesh cost.
//! - `box`:   box brushes: an SDF box sculpted every frame, plus one material
//!   box off the voxel grid. Checks that only voxels inside it changed.
//! - `query`: GPU ray queries every frame. Checks each result batch holds
//!   exactly the queries sent, and that the hits match the CPU ray. With
//!   `--small-arenas`, also on chunks stored past a LOD's first arena.
//! - `reslot`: chunks filled solid (no mesh, slot released) and dug out again
//!   (a new slot), every 60 frames. Checks their material slots are
//!   refilled, by reading them back from the GPU.
//! - `visibility`: a hidden box across a chunk boundary, a sphere toggled
//!   hidden and visible, and one chunk's mask removed. Checks every touched
//!   chunk's mask on the GPU against the main world.
//! - `dig`: the cursor brush's gradual dig, held at a slant and re-centered
//!   on the surface every frame, like a player holding the mouse. Checks
//!   the surface keeps a real distance field (no terraced, black crater)
//!   and that the dig doesn't punch through.
//!
//! A failed check is printed, written to the summary as `check:<name>` = 0,
//! and makes the bench exit with an error.
//!
//! `--small-arenas` shrinks the GPU arenas so High terrain spans several,
//! which the `query` phase then checks. More arenas cost frame time (~7-11%
//! in hold/spin/fly), so compare only runs made with the same setting.
//!
//! Camera motion is per frame, not per second, so every run does the same
//! work whatever its frame rate. Vsync is off. The world is generated into a
//! scratch directory, so your saves are never read or written.
//!
//! GPU times need timestamp-query support (Vulkan/DX12). They arrive a few
//! frames late, so a value is attributed to the frame in which it arrived.

#[allow(dead_code)] // shared with the sandbox, which uses the fly cam
mod common;

use std::{
    collections::HashMap,
    fmt::Write as _,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use bevy::{
    diagnostic::{DiagnosticPath, DiagnosticsStore, FrameCount},
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        diagnostic::RenderDiagnosticsPlugin,
        render_resource::{BufferDescriptor, BufferUsages, CommandEncoderDescriptor, MapMode},
        renderer::{RenderAdapterInfo, RenderDevice, RenderQueue},
    },
    window::{PresentMode, WindowResolution},
};
use common::{
    TerrainMaterial, register_terrain, render_plugin, setup_terrain_palette, terrain_height,
    terrain_materials,
};
use renderdoc::{RenderDoc, V141};
use tarasaur::{
    MaterialFieldPlugin, TarasaurPlugin,
    chunk::{CHUNK_SIZE, ChunkLoader, ChunkManager, overlapping_chunks, world_pos_to_chunk_pos},
    field::{
        Field, LOD, MaterialField, SDF, VisibilityField, VoxelField, VoxelMaterial,
        editor::{SdfCsgSphere, WorldEditor},
    },
    grid::flatten_cube,
    num::{f32_from_u32, usize_from},
    raycast::{
        cursor::{CursorBrush, CursorHit, resolve_cursor_hit, update_cursor_focus},
        gpu::{HitResult, PendingVoxelQueries, RayQuery, VoxelQueryResults},
        terrain_ray::TerrainRay,
    },
    voxel::{ArenaConfig, inspect::ArenaInspector, stats},
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
const EDIT_RADIUS: f32 = 3.0;
const EDIT_DELTA: f32 = 0.15; // SDF change per frame; flips dig/build every 30 frames
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
/// When the masks are read back (well after the last edit has landed).
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
    Hold,
    Spin,
    Fly,
    Edit,
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
            Self::Hold => "hold",
            Self::Spin => "spin",
            Self::Fly => "fly",
            Self::Edit => "edit",
            Self::BoxEdit => "box",
            Self::Query => "query",
            Self::Reslot => "reslot",
            Self::Visibility => "visibility",
            Self::Dig => "dig",
            Self::Done => "done",
        }
    }

    const fn next(self) -> Self {
        match self {
            Self::Load => Self::Hold,
            Self::Hold => Self::Spin,
            Self::Spin => Self::Fly,
            Self::Fly => Self::Edit,
            Self::Edit => Self::BoxEdit,
            Self::BoxEdit => Self::Query,
            Self::Query => Self::Reslot,
            Self::Reslot => Self::Visibility,
            Self::Visibility => Self::Dig,
            Self::Dig | Self::Done => Self::Done,
        }
    }

    /// Fixed length, or `None` for `load` (runs until settled).
    const fn frames(self) -> Option<u32> {
        match self {
            Self::Load => None,
            Self::Hold => Some(HOLD_FRAMES),
            Self::Spin => Some(SPIN_FRAMES),
            Self::Fly => Some(FLY_FRAMES),
            Self::Edit => Some(EDIT_FRAMES),
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

const PHASES: [Phase; 10] = [
    Phase::Load,
    Phase::Hold,
    Phase::Spin,
    Phase::Fly,
    Phase::Edit,
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
    out: Output,
}

struct Output {
    dir: PathBuf,
    label: String,
    /// `--depth-prepass`: draw voxel depth before shading.
    depth_prepass: bool,
    /// `--small-arenas`: see `BENCH_ARENAS`.
    small_arenas: bool,
}

#[derive(Component)]
struct BenchCamera;

/// Exits with an error if setup fails or any check fails.
fn main() -> AppExit {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Some(i) = args.iter().position(|a| a == "--compare") {
        return compare_args(args.get(i.saturating_add(1)), args.get(i.saturating_add(2)));
    }
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i.saturating_add(1)))
            .cloned()
    };
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let out = Output {
        dir: flag("--out").map_or_else(|| manifest.join("bench_results"), PathBuf::from),
        label: flag("--label").unwrap_or_else(|| "run".into()),
        depth_prepass: args.iter().any(|a| a == "--depth-prepass"),
        small_arenas: args.iter().any(|a| a == "--small-arenas"),
    };
    let out = Output {
        dir: std::path::absolute(&out.dir).unwrap_or(out.dir),
        ..out
    };

    // An empty scratch save directory, so every run generates the same
    // terrain from scratch and your saves are never read or written.
    let world_dir = manifest.join("target").join("bench-world");
    let _ = std::fs::remove_dir_all(&world_dir);

    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(render_plugin()).set(WindowPlugin {
        primary_window: Some(Window {
            title: "tarasaur bench".into(),
            // Steam Deck's native resolution.
            resolution: WindowResolution::new(1280, 800),
            present_mode: PresentMode::AutoNoVsync,
            resizable: false,
            ..default()
        }),
        ..default()
    }))
    .add_plugins((
        TarasaurPlugin {
            save_dir: world_dir,
            // Disk writes would only add noise; nothing here needs to persist.
            autosave_secs: f32::INFINITY,
            depth_prepass: out.depth_prepass,
            max_loads_in_flight: flag("--max-loads")
                .and_then(|n| n.parse().ok())
                .unwrap_or_else(|| TarasaurPlugin::default().max_loads_in_flight),
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
    let Some(msaa) = parse_msaa(flag("--msaa").as_deref()) else {
        return AppExit::error();
    };
    app.insert_resource(BenchMsaa(msaa));
    if args.iter().any(|a| a == "--renderdoc") {
        let Some(capture) = FrameCapture::attach(&out) else {
            return AppExit::error();
        };
        app.insert_resource(capture)
            .add_systems(Update, trigger_frame_capture);
    }

    add_checked_phases(&mut app, out.small_arenas);
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
            out,
            gpu_passes: Vec::new(),
            checks: Checks::default(),
        })
        .add_systems(Startup, (spawn_camera, setup_terrain_palette))
        .add_systems(
            Update,
            (
                drive_camera,
                drive_edits,
                ignore_cursor
                    .after(resolve_cursor_hit)
                    .before(update_cursor_focus),
            ),
        )
        .add_systems(Last, record_frame)
        .run()
}

/// Whether the run uses `BENCH_ARENAS` (`--small-arenas`).
#[derive(Resource, Clone, Copy)]
struct SmallArenas(bool);

/// The `box`, `query` and `reslot` phases: their drivers, checks, and the
/// render-world readbacks.
fn add_checked_phases(app: &mut App, small_arenas: bool) {
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
                    read_back_masks,
                    read_back_meshes,
                    report_arenas,
                )
                    .in_set(RenderSystems::Cleanup),
            );
    }
    app.insert_resource(probe)
        .insert_resource(mesh_probe)
        .insert_resource(SmallArenas(small_arenas))
        .init_resource::<QueryWatch>()
        .add_systems(
            Update,
            (
                drive_box_edits,
                drive_queries,
                drive_reslots,
                drive_visibility,
                drive_dig,
                drive_mesh_probe,
                watch_queries,
                check_material_box,
                check_queries,
                check_reslots,
                check_visibility,
                check_dig,
                check_meshes,
            ),
        );
}

fn spawn_camera(mut commands: Commands, msaa: Res<BenchMsaa>) {
    commands.spawn((
        BenchCamera,
        Camera3d::default(),
        msaa.0,
        Transform::from_translation(START).looking_at(LOOK_AT, Vec3::Y),
        // Same bands as the sandbox, so this measures what it runs.
        ChunkLoader {
            lod_bands: vec![(LOD::High, 4), (LOD::Medium, 3), (LOD::Low, 8)],
            hysteresis: 1,
        },
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 10000.0,
            ..default()
        },
        Transform::from_xyz(10.0, 40.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

fn fly_end() -> Vec3 {
    START + Vec3::X * FLY_STEP * f32_from_u32(FLY_FRAMES)
}

/// Positions the camera from the phase and frame alone.
fn drive_camera(bench: Res<Bench>, mut cams: Query<&mut Transform, With<BenchCamera>>) {
    let f = f32_from_u32(bench.frame_in_phase);
    for mut t in &mut cams {
        *t = match bench.phase {
            Phase::Load | Phase::Hold => {
                Transform::from_translation(START).looking_at(LOOK_AT, Vec3::Y)
            }
            Phase::Spin => {
                let base = Transform::from_translation(START).looking_at(LOOK_AT, Vec3::Y);
                base.with_rotation(Quat::from_rotation_y(f.to_radians()) * base.rotation)
            }
            Phase::Fly => {
                let pos = START + Vec3::X * FLY_STEP * f;
                Transform::from_translation(pos)
                    .looking_at(pos + Vec3::new(1.0, -0.4, 0.0), Vec3::Y)
            }
            Phase::Edit
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
                subtract: false,
                max_step: None,
                paint: None,
            });
        } else if at == RESLOT_DIG_AT {
            csg.write(SdfCsgSphere {
                center,
                radius: RESLOT_DIG_RADIUS,
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
    /// The main world's field, `lod.size()`^3 ids.
    expected: Arc<[u8]>,
}

#[derive(Default)]
struct ProbeState {
    requests: Vec<SlotRequest>,
    /// One line per request, ending in ": ok" when the slot matched.
    results: Vec<String>,
    /// Visibility masks to read back, and one result line per request.
    mask_requests: Vec<MaskRequest>,
    mask_results: Vec<(IVec3, String)>,
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
        let bytes = u64::from(padded).pow(3);
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
                                    let got = gpu.get(flatten_cube(x, y, z, padded));
                                    let want = req.expected.get(flatten_cube(x, y, z, n));
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
/// readback takes a few frames, so each is checked on its phase's last.
const MESH_PROBES: [(Phase, u32); 3] = [
    (Phase::Hold, 10),
    (Phase::Fly, FLY_FRAMES - 30),
    (Phase::Edit, EDIT_FRAMES - 30),
];
/// A vertex further than this from the heightfield, in its chunk's voxels,
/// is off the surface. Surface nets keeps a vertex inside its cell, and a
/// seam cell spans at most a few of this chunk's voxels.
const MESH_MAX_OFF_SURFACE: f32 = 2.0;
/// A triangle edge longer than this, in its chunk's voxels, is a sliver.
const MESH_MAX_EDGE: f32 = 6.0;
/// A vertex normal further than this from the heightfield's, in degrees.
const MESH_MAX_NORMAL_DEG: f32 = 30.0;
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
    report: MeshReport,
    done: Vec<(Phase, MeshReport)>,
}

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
    /// Vertex normals more than `MESH_MAX_NORMAL_DEG` degrees off the
    /// heightfield's.
    bad_normals: Vec<MeshFlag>,
    /// Largest off-surface distance per LOD rank, in that LOD's voxels.
    max_off: [f32; LOD::COUNT],
    /// Every resident chunk's LOD, to describe the neighbours of a flag.
    lods: HashMap<IVec3, LOD>,
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
        self.bad_normals.extend(part.bad_normals);
        for (m, p) in self.max_off.iter_mut().zip(part.max_off) {
            *m = m.max(p);
        }
    }
}

/// Shared by both worlds: the main world asks, the render world answers.
#[derive(Resource, Clone, Default)]
struct MeshProbe(Arc<Mutex<MeshProbeState>>);

fn drive_mesh_probe(bench: Res<Bench>, probe: Res<MeshProbe>) {
    if !MESH_PROBES.contains(&(bench.phase, bench.frame_in_phase)) {
        return;
    }
    if let Ok(mut s) = probe.0.lock() {
        s.requested = Some(bench.phase);
        s.exclude = if bench.phase == Phase::Edit {
            let margin = Vec3::splat(EDIT_RADIUS + CHUNK_SIZE);
            overlapping_chunks(edit_target() - margin, edit_target() + margin).collect()
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
    let (mesh_arenas, exclude) = {
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
        s.report = MeshReport {
            lods: arenas
                .resident_chunks()
                .map(|(pos, lod, _)| (pos, lod))
                .collect(),
            ..default()
        };
        let mesh_arenas = arenas.mesh_arenas();
        s.pending = mesh_arenas.len();
        (mesh_arenas, Arc::new(s.exclude.clone()))
    };
    for arena in mesh_arenas {
        let sources = [
            &arena.vertex_buffer,
            &arena.index_buffer,
            &arena.indirect_args_buffer,
            &arena.vertex_count_buffer,
        ];
        let sizes = sources.map(|b| b.size());
        let mut offsets = [0u64; 4];
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
                        };
                        check_arena_meshes(&arena, &exclude, &mesh, &mut part);
                    }
                    readback.unmap();
                }
                if let Ok(mut s) = state.lock() {
                    s.report.merge(part);
                    s.pending = s.pending.saturating_sub(1);
                    if s.pending == 0
                        && let Some(phase) = s.current.take()
                    {
                        let report = std::mem::take(&mut s.report);
                        s.done.push((phase, report));
                    }
                }
            });
    }
}

/// One arena's read-back buffers.
struct ArenaBytes<'a> {
    vertices: &'a [u8],
    indices: &'a [u8],
    args: &'a [u8],
    counts: &'a [u8],
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
    mesh: &ArenaBytes,
    report: &mut MeshReport,
) {
    for &(pos, slot, base) in &arena.slots {
        if exclude.contains(&pos) {
            continue;
        }
        let count = read_u32(mesh.counts, u64::from(slot)).unwrap_or(0);
        report.chunks = report.chunks.saturating_add(1);
        report.vertices = report.vertices.saturating_add(usize_from(count));
        let chunk = ChunkMesh {
            pos,
            lod: arena.lod,
            base,
            count,
        };
        check_vertices(&chunk, mesh, report);
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

fn check_vertices(chunk: &ChunkMesh, mesh: &ArenaBytes, report: &mut MeshReport) {
    let voxel = chunk.lod.voxel_size();
    for v in chunk.base..chunk.base.saturating_add(chunk.count) {
        let Some(p) = mesh.position(v) else {
            continue;
        };
        let d = heightfield_distance(p).abs() / voxel;
        if let Some(m) = report.max_off.get_mut(usize_from(chunk.lod.rank())) {
            *m = m.max(d);
        }
        if exceeds(d, MESH_MAX_OFF_SURFACE) {
            report.off_surface.push(chunk.flag(d, p));
        }
        if let Some(n) = mesh.normal(v) {
            let cos = n.normalize_or_zero().dot(heightfield_normal(p));
            let angle = cos.clamp(-1.0, 1.0).acos().to_degrees();
            if exceeds(angle, MESH_MAX_NORMAL_DEG) {
                report.bad_normals.push(chunk.flag(angle, p));
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
        report.triangles = report.triangles.saturating_add(1);
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
        let centroid = (a + b + c) / 3.0;
        let longest = a.distance(b).max(b.distance(c)).max(c.distance(a)) / voxel;
        if exceeds(longest, MESH_MAX_EDGE) {
            report.long_edges.push(chunk.flag(longest, centroid));
        }
        // Counter-clockwise front faces.
        let face = (b - a).cross(c - a).normalize_or_zero();
        if face != Vec3::ZERO && face.y <= 0.0 {
            report.folds.push(chunk.flag(face.y, centroid));
        }
    }
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

/// The report's summary, plus its worst flags.
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
            "  known at LOD seams, not failed: {} downward triangles, {} vertex normals > {MESH_MAX_NORMAL_DEG} degrees off ({} > 90)",
            r.folds.len(),
            r.bad_normals.len(),
            r.bad_normals.iter().filter(|f| f.value > 90.0).count()
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
    if !MESH_PROBES.iter().any(|&(p, _)| p == phase) || !phase.last_frame(&bench) {
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
                // Folds and bent normals are still known at LOD seams, so
                // they're reported (and in the CSV) without failing the run.
                let pass = r.chunks > 0
                    && r.off_surface.is_empty()
                    && r.long_edges.is_empty()
                    && r.bad_indices == 0;
                (pass, mesh_report_lines(r).join("\n"))
            }
        }
    };
    bench.checks.record(phase, "mesh_on_surface", pass, detail);
}

// ---------------------------------------------------------------------------
// visibility: mask edits, mask removal, and the masks on the GPU
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

/// The box's -x chunk, whose `VisibilityField` is removed mid-phase.
fn vis_removed_chunk() -> IVec3 {
    world_pos_to_chunk_pos(&(vis_box_center() - Vec3::X * MATERIAL_BOX_HALF))
}

/// Every chunk the phase's edits touch: the ones whose masks are checked.
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

fn drive_visibility(
    bench: Res<Bench>,
    mut visibility: WorldEditor<VisibilityField, bool>,
    chunks: Res<ChunkManager>,
    fields: Query<&VisibilityField>,
    probe: Res<SlotProbe>,
    mut commands: Commands,
) {
    if bench.phase != Phase::Visibility {
        return;
    }
    let f = bench.frame_in_phase;
    if f == 0 {
        visibility.fill_box(vis_box_center(), Vec3::splat(MATERIAL_BOX_HALF), false);
    }
    // Hidden on odd toggles, so it ends visible: an all-visible chunk sends
    // no mask at all.
    if f < VIS_TOGGLE_UNTIL && f.is_multiple_of(VIS_TOGGLE_PERIOD) {
        let hide = (f / VIS_TOGGLE_PERIOD) % 2 == 1;
        visibility.fill_sphere(vis_sphere_center(), VIS_SPHERE_RADIUS, !hide);
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
                .mask_results
                .push((pos, format!("chunk {pos} isn't loaded")));
            continue;
        };
        let field = fields.get(entity).ok();
        state.mask_requests.push(MaskRequest {
            pos,
            entity,
            // Edits load chunks at max LOD; a chunk's mask slot is there too.
            lod: field.map_or(LOD::High, VoxelField::lod),
            expected: field
                .filter(|f| f.should_extract())
                .map(|f| f.data_slice().into()),
        });
    }
}

/// A chunk whose GPU mask to read back. `expected` is the main world's
/// mask (`lod.size()`^3 bytes, 0 = hidden), or `None` when the GPU should
/// hold no mask for it (no field, or all visible).
struct MaskRequest {
    pos: IVec3,
    entity: Entity,
    lod: LOD,
    expected: Option<Arc<[u8]>>,
}

/// Copies each requested chunk's mask words and its "has a mask" word into
/// one mappable buffer and compares them with the main world.
fn read_back_masks(
    probe: Res<SlotProbe>,
    arenas: ArenaInspector,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    let requests = match probe.0.lock() {
        Ok(mut s) => std::mem::take(&mut s.mask_requests),
        Err(_) => return,
    };
    for req in requests {
        let pos = req.pos;
        // A chunk that meshes to nothing holds no slot, so no GPU mask:
        // nothing is drawn there for a mask to hide.
        let Some(slot) = arenas.mask_slot(req.lod, req.entity) else {
            if let Ok(mut s) = probe.0.lock() {
                s.mask_results
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
                    s.mask_results.push((pos, line));
                }
            });
    }
}

/// One result line: ": ok" when the GPU's flag and bits match the main
/// world for every interior voxel.
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

/// Every touched chunk's GPU mask matches the main world, both the masked
/// and the mask-free path ran, and the removed field's mask is gone.
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
        .map_or_else(|_| Vec::new(), |s| s.mask_results.clone());
    let expected = vis_chunks().len();
    let ok = results.iter().filter(|(_, l)| l.ends_with(": ok")).count();
    let masked = results
        .iter()
        .filter(|(_, l)| l.contains("hidden): ok"))
        .count();
    let unmasked = results
        .iter()
        .filter(|(_, l)| l.contains("(no mask): ok"))
        .count();
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
        "visibility_mask_slots",
        ok == expected && results.len() == expected && masked > 0 && unmasked > 0,
        detail,
    );

    let removed = vis_removed_chunk();
    let field_gone = chunks
        .get_chunk(&removed)
        .is_some_and(|e| fields.get(e).is_err());
    let gpu_cleared = results
        .iter()
        .any(|(p, l)| *p == removed && l.ends_with("(no mask): ok"));
    bench.checks.record(
        Phase::Visibility,
        "visibility_removed",
        field_gone && gpu_cleared,
        format!("chunk {removed}: field removed {field_gone}, GPU mask cleared {gpu_cleared}"),
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
        radius: DIG_RADIUS,
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
    let half = Vec3::splat(DIG_RADIUS + 1.0) + (hit.pos - dig_target()).abs() * 0.5;
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
    bench.frame_in_phase = 0;
    bench.phase_started = Instant::now();
    if bench.phase == Phase::Done {
        let adapter = adapter.map_or_else(
            || "unknown".to_string(),
            |a| format!("{} ({:?}, {})", a.name, a.backend, a.driver_info),
        );
        write_results(&bench, &adapter);
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

fn write_results(bench: &Bench, adapter: &str) {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let base = bench.out.dir.join(format!("{}-{stamp}", bench.out.label));
    let commit = std::process::Command::new("git")
        .args([
            "-C",
            env!("CARGO_MANIFEST_DIR"),
            "rev-parse",
            "--short",
            "HEAD",
        ])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map_or_else(|| "unknown".into(), |s| s.trim().to_string());
    let header = format!(
        "# label={} commit={commit} adapter={adapter} profile={} depth_prepass={} small_arenas={}\n",
        bench.out.label,
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        bench.out.depth_prepass,
        bench.out.small_arenas,
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

    let frames_path = base.with_extension("frames.csv");
    let summary_path = base.with_extension("summary.csv");
    let written = std::fs::create_dir_all(&bench.out.dir)
        .and_then(|()| std::fs::write(&frames_path, frames))
        .and_then(|()| std::fs::write(&summary_path, summary));
    if let Err(e) = written {
        error!("[bench] couldn't write results: {e}");
    }
    println!("\n{table}");
    print_gpu_passes(&bench.gpu_passes, Phase::Hold);
    bench.checks.print();
    println!("frames:  {}", frames_path.display());
    println!("summary: {}", summary_path.display());
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
