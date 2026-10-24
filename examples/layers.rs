//! A second surface layer over the terrain: a flat water table, its own
//! shape field meshed by the same GPU surface nets and drawn with its own
//! shader (`common/flat_layer.wesl`). Hold right mouse to look, WASD and
//! Space/Shift to fly.
//!
//! The water layer is what `bench --layer` adds, so this is also where to
//! look at what that measures. `--shot <file.png>` runs headless (as
//! ornithochory's grove: no window, the camera renders into a fixed-size
//! image), saves a screenshot after the world has streamed in
//! (`--shot-frame <n>`, default 400) and exits.

mod common;

use std::time::Duration;

use bevy::{
    app::ScheduleRunnerPlugin,
    camera::{ImageRenderTarget, RenderTarget},
    diagnostic::FrameCount,
    prelude::*,
    render::{
        render_resource::{TextureFormat, TextureUsages},
        view::screenshot::{Screenshot, save_to_disk},
    },
    window::ExitCondition,
    winit::WinitPlugin,
};
use common::{
    FlyCam, TerrainMaterial, fly_cam_system, register_terrain, render_plugin,
    setup_terrain_palette, water::add_water_layer,
};
use tarasaur::{MaterialFieldPlugin, TarasaurPlugin, chunk::ChunkLoader};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let shot = args
        .iter()
        .position(|a| a == "--shot")
        .and_then(|i| args.get(i.saturating_add(1)));
    let mut app = App::new();
    if shot.is_some() {
        app.add_plugins((
            DefaultPlugins
                .set(render_plugin())
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                .disable::<WinitPlugin>(),
            // With a window plugin, DefaultPlugins leaves the runner to winit.
            // 60 Hz, as a vsynced window, so frames count time.
            ScheduleRunnerPlugin::run_loop(Duration::from_secs_f64(1.0 / 60.0)),
        ));
    } else {
        app.add_plugins(DefaultPlugins.set(render_plugin()));
    }
    app.add_plugins(TarasaurPlugin {
            // Its own saves, so other examples' chunks never load here.
            save_dir: "saves/layers".into(),
            ..default()
        })
        .add_plugins(MaterialFieldPlugin::<TerrainMaterial>::new());
    register_terrain(&mut app);
    add_water_layer(&mut app);

    app.add_systems(Startup, (spawn_player, setup_terrain_palette))
        .add_systems(Update, fly_cam_system);
    if let Some(path) = shot {
        let frame = args
            .iter()
            .position(|a| a == "--shot-frame")
            .and_then(|i| args.get(i.saturating_add(1)))
            .and_then(|n| n.parse().ok())
            .unwrap_or(SHOT_FRAME);
        app.insert_resource(Shot {
            path: path.clone(),
            frame,
            target: Handle::default(),
        })
        .add_systems(PreStartup, shot_target)
        .add_observer(retarget_camera)
        .add_systems(Update, take_shot);
    }
    app.run();
}

/// `--shot`: where to save the screenshot, at which frame, and the image the
/// camera renders into (a fixed size whatever the window would have been).
#[derive(Resource)]
struct Shot {
    path: String,
    frame: u32,
    target: Handle<Image>,
}

const SHOT_SIZE: (u32, u32) = (1280, 720);

fn shot_target(mut shot: ResMut<Shot>, mut images: ResMut<Assets<Image>>) {
    let mut image = Image::new_target_texture(
        SHOT_SIZE.0,
        SHOT_SIZE.1,
        TextureFormat::Rgba8UnormSrgb,
        None,
    );
    image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    shot.target = images.add(image);
}

/// Points a window camera at the shot target, and the UI with it.
fn retarget_camera(
    add: On<Add<Camera>>,
    mut commands: Commands,
    shot: Res<Shot>,
    cameras: Query<&RenderTarget>,
) {
    if let Ok(RenderTarget::Window(_)) = cameras.get(add.entity) {
        commands.entity(add.entity).insert((
            RenderTarget::Image(ImageRenderTarget {
                handle: shot.target.clone(),
                scale_factor: 1.0,
            }),
            IsDefaultUiCamera,
        ));
    }
}

/// Frames to let the world stream in before the shot.
const SHOT_FRAME: u32 = 400;

fn take_shot(
    frame: Res<FrameCount>,
    shot: Res<Shot>,
    mut commands: Commands,
    mut exit: MessageWriter<AppExit>,
) {
    if frame.0 == shot.frame {
        commands
            .spawn(Screenshot::image(shot.target.clone()))
            .observe(save_to_disk(shot.path.clone()));
    }
    if frame.0 == shot.frame.saturating_add(60) {
        exit.write(AppExit::Success);
    }
}

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
        ChunkLoader::default(),
    ));
}
