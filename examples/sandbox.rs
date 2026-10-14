mod common;

use bevy::{input::mouse::MouseWheel, prelude::*};
use common::{
    FlyCam, TerrainMaterial, fly_cam_system, register_terrain, render_plugin, setup_terrain_palette,
};
use tarasaur::{
    LOD, SDF, TarasaurPlugin,
    chunk::ChunkLoader,
    editor::WorldEditor,
    field::MaterialFieldPlugin,
    voxel::cursor::{CursorBrush, CursorHit, resolve_cursor_hit},
};

fn main() {
    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(render_plugin()))
        // TarasaurPlugin now includes PalettePlugin.
        .add_plugins(TarasaurPlugin)
        .add_plugins(MaterialFieldPlugin::<TerrainMaterial>::new());
    register_terrain(&mut app);

    app.init_resource::<Brush>()
        .insert_resource(CursorBrush { enabled: false })
        .add_systems(Startup, (spawn_player, setup_terrain_palette))
        .add_systems(
            Update,
            (fly_cam_system, terrain_edit.after(resolve_cursor_hit)),
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
        //   d <= 1 High, d <= 3 Medium, d <= 5 Low, d <= 8 Lowest.
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

#[derive(Resource)]
struct Brush {
    radius: f32,
    /// SDF change per second while a button is held.
    strength: f32,
}

impl Default for Brush {
    fn default() -> Self {
        Self {
            radius: 4.0,
            strength: 12.0,
        }
    }
}

/// Brush on top of the library's resolved `CursorHit` (backend picked by
/// `CursorQueryMode`, F8 cycles). The built-in CSG brush is disabled in
/// `main` so a click only edits once.
fn terrain_edit(
    hit: Res<CursorHit>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut wheel: MessageReader<MouseWheel>,
    time: Res<Time>,
    mut brush: ResMut<Brush>,
    mut sdf: WorldEditor<SDF, f32>,
    mut gizmos: Gizmos,
) {
    for ev in wheel.read() {
        brush.radius = f32::mul_add(ev.y, 0.5, brush.radius).clamp(1.0, 20.0);
    }

    let Some(hit) = hit.0.map(|h| h.pos) else {
        return;
    };

    gizmos.sphere(
        Isometry3d::from_translation(hit),
        brush.radius,
        Color::WHITE,
    );

    let amount = brush.strength * time.delta_secs();
    if buttons.pressed(MouseButton::Left) {
        sdf.accumulate_sphere(hit, brush.radius, amount); // dig (raise SDF)
    }
    if buttons.pressed(MouseButton::Right) {
        sdf.accumulate_sphere(hit, brush.radius, -amount); // build (lower SDF)
    }
}
