//! Repeatable end-to-end benchmark.
//!
//! Runs a fixed scenario on the sandbox terrain and writes per-frame and
//! per-phase results:
//!
//! ```text
//! cargo run --release --example bench -- --label baseline [--msaa 1|2|4|8] [--max-loads N] [--depth-prepass]
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
//!   exactly the queries sent, and that the hits match the CPU ray, also on
//!   chunks stored past a LOD's first arena (the bench uses small arenas so
//!   High terrain spans several).
//! - `reslot`: chunks filled solid (no mesh, slot released) and dug out again
//!   (a new slot), every 60 frames. Checks their material slots are
//!   refilled, by reading them back from the GPU.
//!
//! A failed check is printed, written to the summary as `check:<name>` = 0,
//! and makes the bench exit with an error.
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
        sync_world::MainEntity,
    },
    window::{PresentMode, WindowResolution},
};
use common::{
    TerrainMaterial, register_terrain, render_plugin, setup_terrain_palette, terrain_height,
    terrain_materials,
};
use renderdoc::{RenderDoc, V141};
use tarasaur::{
    FieldLOD, LOD, MaterialField, SDF, TarasaurPlugin, VoxelDataSlice, VoxelMaterial,
    chunk::{
        CHUNK_SIZE, ChunkLoader, ChunkManager, overlapping_chunks, update_cursor_focus,
        world_pos_to_chunk_pos,
    },
    editor::{SdfCsgSphere, WorldEditor},
    field::{Field, MaterialFieldPlugin, flatten_cube},
    loading::MaxInFlightLoads,
    num::{f32_from_u32, usize_from},
    persistence::AutosaveIntervalSecs,
    voxel::{
        arena::{ArenaConfig, VoxelChunkArenaSet},
        cursor::{CursorBrush, CursorHit, resolve_cursor_hit},
        pipeline::VoxelDepthPrepass,
        query::{PendingVoxelQueries, VoxelQueryResults},
        stats,
        terrain_ray::TerrainRay,
        types::{HitResult, RayQuery},
    },
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
/// Arenas small enough that High terrain spans several, with enough of them
/// that no LOD has less room than with the defaults.
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
            Self::Reslot | Self::Done => Self::Done,
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

const PHASES: [Phase; 8] = [
    Phase::Load,
    Phase::Hold,
    Phase::Spin,
    Phase::Fly,
    Phase::Edit,
    Phase::BoxEdit,
    Phase::Query,
    Phase::Reslot,
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
    };
    let out = Output {
        dir: std::path::absolute(&out.dir).unwrap_or(out.dir),
        ..out
    };

    // Saves resolve against the working directory: point it at an empty
    // scratch world so every run generates the same terrain from scratch.
    let world_dir = manifest.join("target").join("bench-world");
    let _ = std::fs::remove_dir_all(&world_dir);
    if let Err(e) =
        std::fs::create_dir_all(&world_dir).and_then(|()| std::env::set_current_dir(&world_dir))
    {
        eprintln!(
            "bench: can't use scratch world {}: {e}",
            world_dir.display()
        );
        return AppExit::error();
    }

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
        TarasaurPlugin,
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
    app.insert_resource(VoxelDepthPrepass(out.depth_prepass));
    if let Some(n) = flag("--max-loads").and_then(|n| n.parse().ok()) {
        app.insert_resource(MaxInFlightLoads(n));
    }
    if args.iter().any(|a| a == "--renderdoc") {
        let Some(capture) = FrameCapture::attach(&out) else {
            return AppExit::error();
        };
        app.insert_resource(capture)
            .add_systems(Update, trigger_frame_capture);
    }

    add_checked_phases(&mut app);
    app.insert_resource(CursorBrush { enabled: false })
        // Disk writes would only add noise; nothing here needs to persist.
        .insert_resource(AutosaveIntervalSecs(f32::INFINITY))
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

/// The `box`, `query` and `reslot` phases: their drivers, checks, and the
/// render-world slot readback.
fn add_checked_phases(app: &mut App) {
    let probe = SlotProbe::default();
    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app
            .world_mut()
            .resource_mut::<VoxelChunkArenaSet>()
            .config = BENCH_ARENAS;
        render_app.insert_resource(probe.clone()).add_systems(
            Render,
            (read_back_slots, report_arenas).in_set(RenderSystems::Cleanup),
        );
    }
    app.insert_resource(probe)
        .init_resource::<QueryWatch>()
        .add_systems(
            Update,
            (
                drive_box_edits,
                drive_queries,
                drive_reslots,
                watch_queries,
                check_material_box,
                check_queries,
                check_reslots,
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
            Phase::Edit | Phase::BoxEdit | Phase::Query | Phase::Reslot | Phase::Done => {
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
) {
    if bench.phase != Phase::Query {
        return;
    }
    if bench.frame_in_phase == ARENA_REPORT_AT
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
    // per arena finds those.
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
            });
        } else if at == RESLOT_DIG_AT {
            csg.write(SdfCsgSphere {
                center,
                radius: RESLOT_DIG_RADIUS,
                subtract: true,
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
    /// Set by the main world to ask for an `ArenaReport`.
    arena_request: bool,
    arena_report: Option<ArenaReport>,
}

/// Which arena holds each resident chunk.
struct ArenaReport {
    /// Arenas per LOD rank.
    counts: [usize; LOD::COUNT],
    /// Chunk position -> (LOD rank, arena index).
    chunk_arena: HashMap<IVec3, (usize, usize)>,
}

/// Answers an arena-report request with the arena of every resident chunk.
fn report_arenas(probe: Res<SlotProbe>, arena_set: Res<VoxelChunkArenaSet>) {
    let Ok(mut state) = probe.0.lock() else {
        return;
    };
    if !std::mem::take(&mut state.arena_request) {
        return;
    }
    let mut chunk_arena = HashMap::new();
    for (rank, arenas) in arena_set.arenas.iter().enumerate() {
        for (arena_idx, arena) in arenas.iter().enumerate() {
            for &pos in arena.slot_of_chunk_pos.keys() {
                chunk_arena.insert(pos, (rank, arena_idx));
            }
        }
    }
    state.arena_report = Some(ArenaReport {
        counts: arena_set.arenas.each_ref().map(Vec::len),
        chunk_arena,
    });
}

/// Shared by both worlds: the main world asks, the render world answers.
#[derive(Resource, Clone, Default)]
struct SlotProbe(Arc<Mutex<ProbeState>>);

/// Copies each requested chunk's material slot to a mappable buffer and
/// compares its interior (the padded volume minus the apron) with the field.
fn read_back_slots(
    probe: Res<SlotProbe>,
    arena_set: Res<VoxelChunkArenaSet>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    let requests = match probe.0.lock() {
        Ok(mut s) => std::mem::take(&mut s.requests),
        Err(_) => return,
    };
    for req in requests {
        let pos = req.pos;
        let slot = arena_set.find_slot(req.lod, MainEntity::from(req.entity));
        let arena = slot.and_then(|s| {
            arena_set
                .arenas
                .get(VoxelChunkArenaSet::lod_index(req.lod))
                .and_then(|a| a.get(s.arena_idx))
                .map(|a| (s.slot, a))
        });
        let Some((slot, arena)) = arena else {
            if let Ok(mut s) = probe.0.lock() {
                s.results.push(format!("chunk {pos}: no slot"));
            }
            continue;
        };
        let bytes = u64::from(arena.sdf_elems_per_chunk);
        let readback = device.create_buffer(&BufferDescriptor {
            label: Some("bench_material_readback"),
            size: bytes,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("bench_material_readback"),
        });
        encoder.copy_buffer_to_buffer(
            &arena.material_buffer,
            u64::from(slot).saturating_mul(bytes),
            &readback,
            0,
            bytes,
        );
        queue.submit(std::iter::once(encoder.finish()));

        let padded = arena.texture_size;
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
        "# label={} commit={commit} adapter={adapter} profile={} depth_prepass={}\n",
        bench.out.label,
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        bench.out.depth_prepass,
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
