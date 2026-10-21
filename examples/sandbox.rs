//! Main playground: streamed, textured heightfield terrain around a fly
//! camera, edited with the library's `CursorBrush`. Hold right mouse to
//! look, left mouse to apply the brush. 1-9 pick the tool (dig, build,
//! smooth, flatten, paint, fill, scrape, hide, reveal), Q/E the material,
//! scroll the radius and Ctrl + scroll the strength; Ctrl + left mouse
//! swaps dig and build (fill and scrape, hide and reveal), R toggles the
//! flatten, fill and scrape plane between level and along the slope, and B
//! toggles hide and reveal between a sphere and a cube.
//!
//! Capture: F12 saves a screenshot, F10 starts and stops recording frames
//! (turn them into a GIF with `tools/make-gif.sh`), H hides the HUD. Both
//! write under `captures/`.

mod common;

use std::time::{SystemTime, UNIX_EPOCH};

use bevy::{
    input::mouse::MouseWheel,
    prelude::*,
    render::view::screenshot::{Screenshot, save_to_disk},
};
use common::{
    FlyCam, TerrainMaterial, fly_cam_system, register_terrain, render_plugin, setup_terrain_palette,
};
use tarasaur::{
    MaterialFieldPlugin, TarasaurPlugin,
    chunk::ChunkLoader,
    field::{FieldSet, LOD, VoxelMaterial},
    raycast::cursor::{BrushTool, CursorBrush},
};

fn main() {
    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(render_plugin()))
        // The default save directory (./saves/chunks): the sandbox is the
        // main playground, so its world is the one kept there. Other
        // examples save under saves/<example>.
        // TARASAUR_TELEMETRY=1 turns on the GPU guard readbacks in release
        // builds too (they're on in debug builds).
        .add_plugins(TarasaurPlugin {
            guard_telemetry: std::env::var_os("TARASAUR_TELEMETRY").is_some()
                || cfg!(debug_assertions),
            ..default()
        })
        .add_plugins(MaterialFieldPlugin::<TerrainMaterial>::new());
    register_terrain(&mut app);

    app.insert_resource(CursorBrush {
        radius: 4.0,
        ..default()
    })
    .init_resource::<Recording>()
    .add_systems(Startup, (spawn_player, spawn_hud, setup_terrain_palette))
    .add_systems(
        Update,
        (
            fly_cam_system,
            brush_controls.before(FieldSet::Brush),
            update_hud.after(brush_controls),
            capture_controls,
        ),
    )
    .run();
}

// ---------------------------------------------------------------------------
// Player
// ---------------------------------------------------------------------------

fn spawn_player(mut commands: Commands) {
    let t = Transform::from_xyz(0.0, 30.0, 40.0).looking_at(Vec3::new(0.0, 8.0, 0.0), Vec3::Y);
    let (yaw, pitch, _) = t.rotation.to_euler(EulerRot::YXZ);

    commands.spawn((
        Camera3d::default(),
        t,
        FlyCam {
            move_speed: 20.0,
            sensitivity: 0.002,
            pitch,
            yaw,
        },
        // Band widths are additive, finest first (Chebyshev distance in chunks):
        //   d <= 4 High, d <= 7 Medium, d <= 15 Low.
        // `hysteresis` delays LOD downgrades and despawn by that many chunks.
        ChunkLoader {
            lod_bands: vec![
                (LOD::High, 4),
                (LOD::Medium, 3),
                (LOD::Low, 8),
                //(LOD::Lowest, 2),
            ],
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

// ---------------------------------------------------------------------------
// Terrain editing
// ---------------------------------------------------------------------------

const TOOL_KEYS: [KeyCode; 9] = [
    KeyCode::Digit1,
    KeyCode::Digit2,
    KeyCode::Digit3,
    KeyCode::Digit4,
    KeyCode::Digit5,
    KeyCode::Digit6,
    KeyCode::Digit7,
    KeyCode::Digit8,
    KeyCode::Digit9,
];

/// Number keys pick the tool, Q/E cycle the material, R toggles level
/// flattening, B the hide/reveal shape, scroll resizes the brush and Ctrl +
/// scroll changes its strength.
fn brush_controls(
    keys: Res<ButtonInput<KeyCode>>,
    mut wheel: MessageReader<MouseWheel>,
    mut brush: ResMut<CursorBrush>,
) {
    for (key, tool) in TOOL_KEYS.into_iter().zip(BrushTool::ALL) {
        if keys.just_pressed(key) {
            brush.tool = tool;
        }
    }

    let count = TerrainMaterial::COUNT;
    if keys.just_pressed(KeyCode::KeyE) {
        brush.material = brush.material.saturating_add(1) % count;
    }
    if keys.just_pressed(KeyCode::KeyQ) {
        brush.material = brush
            .material
            .checked_sub(1)
            .unwrap_or_else(|| count.saturating_sub(1));
    }

    if keys.just_pressed(KeyCode::KeyR) {
        brush.flatten = brush.flatten.toggled();
    }
    if keys.just_pressed(KeyCode::KeyB) {
        brush.shape = brush.shape.toggled();
    }

    let ctrl = keys.any_pressed([KeyCode::ControlLeft, KeyCode::ControlRight]);
    for ev in wheel.read() {
        if ctrl {
            brush.strength = f32::mul_add(ev.y, 0.5, brush.strength).clamp(0.5, 20.0);
        } else {
            brush.radius = f32::mul_add(ev.y, 0.5, brush.radius).clamp(1.0, 20.0);
        }
    }
}

#[derive(Component)]
struct Hud;

fn spawn_hud(mut commands: Commands) {
    commands.spawn((
        Hud,
        Text::default(),
        TextFont {
            font_size: FontSize::Px(16.0),
            ..default()
        },
        Node {
            position_type: PositionType::Absolute,
            top: px(8),
            left: px(8),
            ..default()
        },
    ));
}

fn update_hud(brush: Res<CursorBrush>, mut hud: Query<&mut Text, With<Hud>>) {
    if !brush.is_changed() {
        return;
    }
    let tools: String = BrushTool::ALL
        .iter()
        .zip(1..)
        .map(|(t, n)| {
            if *t == brush.tool {
                format!("[{n} {}] ", t.name())
            } else {
                format!(" {n} {}  ", t.name())
            }
        })
        .collect();
    let material = TerrainMaterial::from_id(brush.material);
    for mut text in &mut hud {
        text.0 = format!(
            "{tools}\nmaterial {material:?} (Q/E)   radius {:.1} (scroll)   strength {:.1} (ctrl+scroll)   \
             plane {:?} (R)   hide/reveal shape {:?} (B)\n\
             LMB apply, ctrl+LMB swaps dig/build, fill/scrape and hide/reveal, RMB look",
            brush.radius, brush.strength, brush.flatten, brush.shape
        );
    }
}

// ---------------------------------------------------------------------------
// Capture
// ---------------------------------------------------------------------------

/// Frames per second while recording; `tools/make-gif.sh` plays them back at
/// the same rate.
const RECORD_FPS: f32 = 15.0;

/// An active recording: the folder its frames go to and the next frame number.
#[derive(Resource)]
struct Recording {
    dir: Option<String>,
    frame: u32,
    timer: Timer,
}

impl Default for Recording {
    fn default() -> Self {
        Self {
            dir: None,
            frame: 0,
            timer: Timer::from_seconds(RECORD_FPS.recip(), TimerMode::Repeating),
        }
    }
}

fn unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// F12 saves a screenshot, F10 toggles recording, H toggles the HUD.
fn capture_controls(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut rec: ResMut<Recording>,
    mut hud: Query<&mut Visibility, With<Hud>>,
) {
    if keys.just_pressed(KeyCode::KeyH) {
        for mut vis in &mut hud {
            vis.toggle_visible_hidden();
        }
    }

    if keys.just_pressed(KeyCode::F12) {
        if let Err(e) = std::fs::create_dir_all("captures") {
            warn!("Can't create captures/: {e}");
        }
        let path = format!("captures/shot-{}.png", unix_secs());
        info!("Screenshot: {path}");
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(path));
    }

    if keys.just_pressed(KeyCode::F10) {
        if let Some(dir) = rec.dir.take() {
            info!(
                "Recorded {} frames to {dir}; make a GIF with: tools/make-gif.sh {dir}",
                rec.frame
            );
        } else {
            let dir = format!("captures/rec-{}", unix_secs());
            match std::fs::create_dir_all(&dir) {
                Ok(()) => {
                    info!("Recording to {dir} (F10 to stop)");
                    rec.dir = Some(dir);
                    rec.frame = 0;
                    rec.timer.reset();
                }
                Err(e) => warn!("Can't create {dir}: {e}"),
            }
        }
    }

    let Some(dir) = rec.dir.clone() else {
        return;
    };
    rec.timer.tick(time.delta());
    // One frame per tick, even after a hitch, so playback stays at the
    // recorded speed except for the hitch itself.
    if rec.timer.just_finished() {
        let path = format!("{dir}/frame-{:05}.png", rec.frame);
        rec.frame = rec.frame.saturating_add(1);
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(path));
    }
}
