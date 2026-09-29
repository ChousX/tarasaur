mod common;

use std::any::type_name;

use bevy::{
    prelude::*,
    render::{
        RenderPlugin,
        settings::{RenderCreation, WgpuLimits, WgpuSettings},
    },
};
use common::{
    FlyCam, TerrainMaterial, determine_material, fly_cam_system, setup_terrain_palette,
    terrain_height,
};
use tarasaur::{
    LOD, MaterialField, TarasaurPlugin, VisibilityField, VoxelMaterial,
    chunk::{CHUNK_SIZE, ChunkLoader},
    field::{MaterialFieldPlugin, generator::ChunkGeneratorRegistry},
    texture_palette::plugin::PalettePlugin,
};

fn main() {
    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins.set(RenderPlugin {
            render_creation: RenderCreation::Automatic(Box::new(WgpuSettings {
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
    .add_plugins(TarasaurPlugin)
    .add_plugins(PalettePlugin::new())
    .add_plugins(MaterialFieldPlugin::<TerrainMaterial>::new());

    {
        let mut g = app.world_mut().resource_mut::<ChunkGeneratorRegistry>();
        g.register_sdf(terrain_sdf);
        // ASSUMPTION: material fields register under their type_name like
        // VisibilityField does. Need generator.rs to confirm.
        g.register::<u8>(
            type_name::<MaterialField<TerrainMaterial>>(),
            terrain_materials,
        );
        g.register::<u8>(type_name::<VisibilityField>(), all_visible);
    }

    app.add_systems(Startup, (spawn_player, setup_terrain_palette))
        .add_systems(Update, fly_cam_system)
        .run();
}

// ---------------------------------------------------------------------------
// Generators (run off-thread, pure functions of chunk_pos + lod)
// ---------------------------------------------------------------------------

/// The heightfield only depends on (x, z), so evaluate it once per column
/// instead of once per voxel. Both the SDF and material generators share it.
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

    /// Same x-fastest, then y, then z ordering as the lod_streaming example.
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

fn terrain_sdf(pos: IVec3, lod: LOD) -> Vec<f32> {
    Columns::new(pos, lod).fill(|y, h, k| (y - h) * k)
}

fn terrain_materials(pos: IVec3, lod: LOD) -> Vec<u8> {
    Columns::new(pos, lod).fill(|y, h, _| determine_material(h, y).to_id())
}

fn all_visible(_pos: IVec3, lod: LOD) -> Vec<u8> {
    vec![1u8; lod.volume()]
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
        ChunkLoader {
            high_distance: 1,
            medium_distance: 3,
            low_distance: 5,
            lowest_distance: 8,
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
