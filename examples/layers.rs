//! A second surface layer over the terrain: a flat water table, its own
//! shape field meshed by the same GPU surface nets and drawn with its own
//! shader (`common/flat_layer.wesl`). Hold right mouse to look, WASD and
//! Space/Shift to fly.
//!
//! The water layer is what `bench --layer` adds, so this is also where to
//! look at what that measures. `--shot <file.png>` saves a screenshot after
//! the world has streamed in and exits.

mod common;

use bevy::{
    diagnostic::FrameCount,
    prelude::*,
    render::view::screenshot::{Screenshot, save_to_disk},
};
use common::{
    FlyCam, TerrainMaterial, fly_cam_system, register_terrain, render_plugin,
    setup_terrain_palette, water::add_water_layer,
};
use tarasaur::{MaterialFieldPlugin, TarasaurPlugin, chunk::ChunkLoader};

fn main() {
    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(render_plugin()))
        .add_plugins(TarasaurPlugin {
            // Its own saves, so other examples' chunks never load here.
            save_dir: "saves/layers".into(),
            ..default()
        })
        .add_plugins(MaterialFieldPlugin::<TerrainMaterial>::new());
    register_terrain(&mut app);
    add_water_layer(&mut app);

    app.add_systems(Startup, (spawn_player, setup_terrain_palette))
        .add_systems(Update, fly_cam_system);
    let args: Vec<String> = std::env::args().collect();
    if let Some(path) = args
        .iter()
        .position(|a| a == "--shot")
        .and_then(|i| args.get(i.saturating_add(1)))
    {
        app.insert_resource(Shot(path.clone()))
            .add_systems(Update, take_shot);
    }
    app.run();
}

/// `--shot`: where to save the screenshot.
#[derive(Resource)]
struct Shot(String);

/// Frames to let the world stream in before the shot.
const SHOT_FRAME: u32 = 400;

fn take_shot(
    frame: Res<FrameCount>,
    shot: Res<Shot>,
    mut commands: Commands,
    mut exit: MessageWriter<AppExit>,
) {
    if frame.0 == SHOT_FRAME {
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(shot.0.clone()));
    }
    if frame.0 == SHOT_FRAME.saturating_add(60) {
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
