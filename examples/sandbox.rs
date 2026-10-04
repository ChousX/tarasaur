mod common;

use std::any::type_name;

use bevy::{
    input::mouse::MouseWheel,
    prelude::*,
    render::{
        RenderPlugin,
        settings::{RenderCreation, WgpuFeatures, WgpuLimits, WgpuSettings},
    },
    window::{CursorGrabMode, CursorOptions, PrimaryWindow},
};
use common::{
    FlyCam, TerrainMaterial, determine_material, fly_cam_system, setup_terrain_palette,
    terrain_height,
};
use tarasaur::{
    LOD, MaterialField, SDF, TarasaurPlugin, VoxelMaterial,
    chunk::{CHUNK_SIZE, ChunkLoader},
    editor::WorldEditor,
    field::{MaterialFieldPlugin, generator::ChunkGeneratorRegistry},
};

fn main() {
    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins.set(RenderPlugin {
            render_creation: RenderCreation::Automatic(Box::new(WgpuSettings {
                features: WgpuFeatures::MULTI_DRAW_INDIRECT_COUNT,
                limits: WgpuLimits {
                    max_buffer_size: 1024 * 1024 * 1024,
                    max_storage_buffer_binding_size: 512 * 1024 * 1024,
                    ..default()
                }
                .into(),
                ..default()
            })),
            ..default()
        }),
    )
    // TarasaurPlugin now includes PalettePlugin.
    .add_plugins(TarasaurPlugin)
    .add_plugins(MaterialFieldPlugin::<TerrainMaterial>::new());

    {
        let mut g = app.world_mut().resource_mut::<ChunkGeneratorRegistry>();
        g.register_sdf_distance(terrain_sdf);
        // The key must equal the type_name the persistence registry uses for
        // this field (register_field::<F> uses std::any::type_name::<F>()).
        g.register::<u8>(
            type_name::<MaterialField<TerrainMaterial>>(),
            terrain_materials,
        );
        // No VisibilityField generator: a chunk without one is fully visible.
    }

    app.init_resource::<Brush>()
        .add_systems(Startup, (spawn_player, setup_terrain_palette))
        .add_systems(Update, (fly_cam_system, terrain_edit))
        .run();
}

// ---------------------------------------------------------------------------
// Generators (run off-thread, pure functions of chunk_pos + lod)
// ---------------------------------------------------------------------------

/// The heightfield only depends on (x, z), so evaluate it once per column
/// instead of once per voxel. Both the SDF and material generators share it.
/// Voxel `i` samples `origin + i * voxel`, so the same chunk generated at any
/// LOD describes the same surface.
struct Columns {
    size: u32,
    voxel: f32,
    origin: Vec3,
    height: Vec<f32>,
    /// 1/sqrt(1+|∇h|²): rescales (y - h) into an approximate true distance,
    /// so brush edits that min/max against the SDF blend sensibly on slopes.
    inv_norm: Vec<f32>,
}

impl Columns {
    fn new(chunk_pos: IVec3, lod: LOD) -> Self {
        let size = lod.size();
        let voxel = CHUNK_SIZE / size as f32;
        let origin = chunk_pos.as_vec3() * CHUNK_SIZE;
        let n = size as usize;
        let (mut height, mut inv_norm) = (Vec::with_capacity(n * n), Vec::with_capacity(n * n));
        let e = 0.5;
        for z in 0..n {
            for x in 0..n {
                let (wx, wz) = (origin.x + x as f32 * voxel, origin.z + z as f32 * voxel);
                let gx = (terrain_height(wx + e, wz) - terrain_height(wx - e, wz)) / (2.0 * e);
                let gz = (terrain_height(wx, wz + e) - terrain_height(wx, wz - e)) / (2.0 * e);
                height.push(terrain_height(wx, wz));
                inv_norm.push(1.0 / (1.0 + gx * gx + gz * gz).sqrt());
            }
        }
        Self {
            size,
            voxel,
            origin,
            height,
            inv_norm,
        }
    }

    /// x-fastest, then y, then z ordering (matches the lod_streaming example
    /// and what the fields expect).
    fn fill<T>(&self, f: impl Fn(f32, f32, f32) -> T) -> Vec<T> {
        let n = self.size as usize;
        let mut out = Vec::with_capacity(n * n * n);
        for z in 0..n {
            for y in 0..n {
                let wy = self.origin.y + y as f32 * self.voxel;
                for x in 0..n {
                    let c = z * n + x;
                    out.push(f(wy, self.height[c], self.inv_norm[c]));
                }
            }
        }
        out
    }
}

/// Sign volume; `register_sdf` turns it into real distances with a jump flood.
fn terrain_sdf(pos: IVec3, lod: LOD) -> Vec<f32> {
    Columns::new(pos, lod).fill(|y, h, k| (y - h) * k)
}

fn terrain_materials(pos: IVec3, lod: LOD) -> Vec<u8> {
    Columns::new(pos, lod).fill(|y, h, _| determine_material(h, y).to_id())
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
                (LOD::High, 2),
                (LOD::Medium, 3),
                (LOD::Low, 4),
                (LOD::Lowest, 5),
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

/// Marches the ray until it dips below the heightfield, then bisects to
/// refine the hit point.
fn pick_terrain(ray: Ray3d, max_dist: f32) -> Option<Vec3> {
    const STEP: f32 = 0.25;
    let at = |t: f32| ray.origin + *ray.direction * t;
    let below = |p: Vec3| p.y < terrain_height(p.x, p.z);

    if below(ray.origin) {
        return None;
    }
    let mut t = 0.0;
    while t < max_dist {
        let next = t + STEP;
        if below(at(next)) {
            let (mut lo, mut hi) = (t, next);
            for _ in 0..8 {
                let mid = 0.5 * (lo + hi);
                if below(at(mid)) { hi = mid } else { lo = mid }
            }
            return Some(at(hi));
        }
        t = next;
    }
    None
}

fn terrain_edit(
    windows: Query<(&Window, &CursorOptions), With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform), With<FlyCam>>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut wheel: MessageReader<MouseWheel>,
    time: Res<Time>,
    mut brush: ResMut<Brush>,
    mut sdf: WorldEditor<SDF, f32>,
    mut gizmos: Gizmos,
) {
    for ev in wheel.read() {
        brush.radius = (brush.radius + ev.y * 0.5).clamp(1.0, 20.0);
    }

    let (Ok((window, cursor)), Ok((camera, cam_tf))) = (windows.single(), cameras.single()) else {
        return;
    };

    // With a grabbed/locked cursor (mouse-look), aim from the screen centre.
    let screen_pos = if cursor.grab_mode == CursorGrabMode::None {
        window.cursor_position().unwrap_or(window.size() / 2.0)
    } else {
        window.size() / 2.0
    };

    let Ok(ray) = camera.viewport_to_world(cam_tf, screen_pos) else {
        return;
    };
    let Some(hit) = pick_terrain(ray, 200.0) else {
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
