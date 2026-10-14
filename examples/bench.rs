//! Repeatable end-to-end benchmark.
//!
//! Runs a fixed scenario on the sandbox terrain and writes per-frame and
//! per-phase results:
//!
//! ```text
//! cargo run --release --example bench -- --label baseline [--msaa 1|2|4|8]
//! cargo run --release --example bench -- --compare bench_results/A.summary.csv bench_results/B.summary.csv
//! ```
//!
//! Phases, in order:
//! - `load`:  stream the world in from a still camera until nothing is pending.
//! - `hold`:  the same view, nothing changing: steady-state render cost.
//! - `spin`:  turn in place: view-dependent cost (culling).
//! - `fly`:   fly a fixed distance per frame: streaming cost while moving.
//! - `edit`:  sculpt one spot every frame: edit -> re-mesh cost.
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
    sync::Mutex,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use bevy::{
    diagnostic::{DiagnosticPath, DiagnosticsStore, FrameCount},
    prelude::*,
    render::{diagnostic::RenderDiagnosticsPlugin, renderer::RenderAdapterInfo},
    window::{PresentMode, WindowResolution},
};
use common::{
    TerrainMaterial, register_terrain, render_plugin, setup_terrain_palette, terrain_height,
};
use renderdoc::{RenderDoc, V141};
use tarasaur::{
    LOD, SDF, TarasaurPlugin,
    chunk::{ChunkLoader, update_cursor_focus},
    editor::WorldEditor,
    field::MaterialFieldPlugin,
    num::f32_from_u32,
    persistence::AutosaveIntervalSecs,
    voxel::{
        cursor::{CursorBrush, CursorHit, resolve_cursor_hit},
        stats,
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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Phase {
    Load,
    Hold,
    Spin,
    Fly,
    Edit,
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
            Self::Done => "done",
        }
    }

    const fn next(self) -> Self {
        match self {
            Self::Load => Self::Hold,
            Self::Hold => Self::Spin,
            Self::Spin => Self::Fly,
            Self::Fly => Self::Edit,
            Self::Edit | Self::Done => Self::Done,
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
            Self::Done => Some(0),
        }
    }
}

// ---------------------------------------------------------------------------
// Recorded columns
// ---------------------------------------------------------------------------

/// Diagnostics recorded every frame. Render spans only produce a value on
/// frames they ran in, so a column is empty when nothing new arrived.
const COLUMNS: [(&str, &str); 15] = [
    ("gpu_mesh_ms", "render/voxel_mesh/elapsed_gpu"),
    ("gpu_pass1_ms", "render/voxel_mesh/pass1/elapsed_gpu"),
    ("gpu_scan_ms", "render/voxel_mesh/scan/elapsed_gpu"),
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
    out: Output,
}

struct Output {
    dir: PathBuf,
    label: String,
}

#[derive(Component)]
struct BenchCamera;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Some(i) = args.iter().position(|a| a == "--compare") {
        match (args.get(i.saturating_add(1)), args.get(i.saturating_add(2))) {
            (Some(a), Some(b)) => compare(Path::new(a), Path::new(b)),
            _ => eprintln!("usage: bench --compare <a.summary.csv> <b.summary.csv>"),
        }
        return;
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
        return;
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
    let msaa = match flag("--msaa").as_deref() {
        None => Msaa::default(),
        Some("1") => Msaa::Off,
        Some("2") => Msaa::Sample2,
        Some("4") => Msaa::Sample4,
        Some("8") => Msaa::Sample8,
        Some(other) => {
            eprintln!("bench: --msaa takes 1, 2, 4 or 8, not {other}");
            return;
        }
    };
    app.insert_resource(BenchMsaa(msaa));
    if args.iter().any(|a| a == "--renderdoc") {
        let Some(capture) = FrameCapture::attach(&out) else {
            return;
        };
        app.insert_resource(capture)
            .add_systems(Update, trigger_frame_capture);
    }

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
        .run();
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
            Phase::Edit | Phase::Done => {
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
        exit.write(AppExit::Success);
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
    for name in ["gpu_mesh_ms", "gpu_pass1_ms", "gpu_scan_ms", "gpu_pass3_ms"] {
        let v = present(name);
        out.push((format!("{name}_total"), total(&v)));
        out.push((format!("{name}_per_dispatch"), mean(&v)));
    }
    for name in ["meshed_chunks", "meshed_cells"] {
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
        "# label={} commit={commit} adapter={adapter} profile={}\n",
        bench.out.label,
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
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
    for phase in [
        Phase::Load,
        Phase::Hold,
        Phase::Spin,
        Phase::Fly,
        Phase::Edit,
    ] {
        let rows: Vec<&FrameRow> = bench.rows.iter().filter(|r| r.phase == phase).collect();
        let metrics = summarize(phase, &rows, bench.load_timed_out);
        for (metric, value) in &metrics {
            let _ = writeln!(summary, "{},{metric},{value}", phase.name());
        }
        for (_, path, avg) in bench.gpu_passes.iter().filter(|(p, ..)| *p == phase) {
            let _ = writeln!(summary, "{},gpu:{path},{avg}", phase.name());
        }
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
    println!("frames:  {}", frames_path.display());
    println!("summary: {}", summary_path.display());
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
