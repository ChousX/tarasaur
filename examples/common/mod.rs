//! Shared by the examples and the bench: render settings, fly camera,
//! heightfield terrain generators and its material palette.

use std::any::type_name;

use bevy::{
    asset::RenderAssetUsages,
    input::mouse::MouseMotion,
    prelude::*,
    render::{
        RenderPlugin,
        render_resource::{Extent3d, TextureDimension, TextureFormat},
        settings::{RenderCreation, WgpuFeatures, WgpuSettings},
    },
};
use tarasaur::{
    LOD, MaterialField, VoxelMaterial,
    chunk::CHUNK_SIZE,
    field::generator::ChunkGeneratorRegistry,
    num::{f32_from_u32, i16_from_f32, u8_from_f32, usize_from},
    ops::{AccumulateExt, BlendExt},
    texture_palette::{PaletteBuilder, PaletteMaterial, TexturePalette, plugin::ActivePalette},
};

/// Renderer settings shared by the examples (and the benchmark, so it
/// measures what the sandbox runs).
pub fn render_plugin() -> RenderPlugin {
    RenderPlugin {
        render_creation: RenderCreation::Automatic(Box::new(WgpuSettings {
            features: WgpuFeatures::MULTI_DRAW_INDIRECT_COUNT,
            ..default()
        })),
        ..default()
    }
}

/// Registers the heightfield terrain's SDF and material generators.
pub fn register_terrain(app: &mut App) {
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

#[derive(Component)]
pub struct FlyCam {
    pub move_speed: f32,
    pub sensitivity: f32,
    pub pitch: f32,
    pub yaw: f32,
}

pub fn fly_cam_system(
    time: Res<Time>,
    mouse_button: Res<ButtonInput<MouseButton>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut mouse_motion: MessageReader<MouseMotion>,
    mut query: Query<(&mut Transform, &mut FlyCam)>,
) {
    let dt = time.delta_secs();
    for (mut transform, mut cam) in query.iter_mut() {
        if mouse_button.pressed(MouseButton::Right) {
            for ev in mouse_motion.read() {
                cam.yaw = f32::mul_add(ev.delta.x, -cam.sensitivity, cam.yaw);
                cam.pitch = f32::mul_add(ev.delta.y, -cam.sensitivity, cam.pitch);
                cam.pitch = cam.pitch.clamp(-1.54, 1.54);
            }
            transform.rotation = Quat::from_euler(EulerRot::YXZ, cam.yaw, cam.pitch, 0.0);
        }
        let mut velocity = Vec3::ZERO;
        let forward = transform.forward();
        let right = transform.right();
        if keyboard.pressed(KeyCode::KeyW) {
            velocity += *forward;
        }
        if keyboard.pressed(KeyCode::KeyS) {
            velocity -= *forward;
        }
        if keyboard.pressed(KeyCode::KeyA) {
            velocity -= *right;
        }
        if keyboard.pressed(KeyCode::KeyD) {
            velocity += *right;
        }
        if keyboard.pressed(KeyCode::Space) {
            velocity += Vec3::Y;
        }
        if keyboard.pressed(KeyCode::ShiftLeft) {
            velocity -= Vec3::Y;
        }
        if velocity != Vec3::ZERO {
            transform.translation += velocity.normalize() * cam.move_speed * dt;
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Default, Debug)]
#[repr(u8)]
pub enum TerrainMaterial {
    #[default]
    Grass = 0,
    Stone = 1,
    Dirt = 2,
    Sand = 3,
}

impl VoxelMaterial for TerrainMaterial {
    const COUNT: u8 = 4;

    fn to_id(self) -> u8 {
        match self {
            Self::Grass => 0,
            Self::Stone => 1,
            Self::Dirt => 2,
            Self::Sand => 3,
        }
    }

    /// Ids past `COUNT` (never written by `to_id`) read as the default.
    fn from_id(id: u8) -> Self {
        match id {
            1 => Self::Stone,
            2 => Self::Dirt,
            3 => Self::Sand,
            _ => Self::Grass,
        }
    }
}

impl AccumulateExt for TerrainMaterial {
    fn accumulate(self, delta: Self) -> Self {
        delta
    }
}

impl BlendExt for TerrainMaterial {
    fn blend_towards(self, target: Self, factor: f32) -> Self {
        if factor >= 0.5 { target } else { self }
    }
}

pub fn determine_material(height: f32, world_y: f32) -> TerrainMaterial {
    if world_y < 2.5 {
        TerrainMaterial::Sand
    } else if height > 13.5 {
        TerrainMaterial::Stone
    } else if world_y < height - 0.8 {
        TerrainMaterial::Dirt
    } else {
        TerrainMaterial::Grass
    }
}

pub fn setup_terrain_palette(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut palettes: ResMut<Assets<TexturePalette>>,
) {
    let size: u32 = 64;

    // Per layer: base color, and its ARM roughness (0-255).
    let colors: [([u8; 3], u8); 4] = [
        ([34, 139, 34], 215),   // Forest Green
        ([128, 128, 128], 140), // Stone Gray, the smoothest
        ([112, 66, 20], 225),   // Dirt Brown
        ([225, 198, 153], 200), // Sand Beige
    ];
    let layers = u32::try_from(colors.len()).unwrap_or(u32::MAX);

    // RGBA8, layer-major then row-major, as `Image::new` expects.
    let bytes = usize_from(size)
        .pow(2)
        .saturating_mul(4)
        .saturating_mul(colors.len());
    let mut albedo_data = Vec::with_capacity(bytes);
    let mut arm_data = Vec::with_capacity(bytes);

    for (seed, (base_col, roughness)) in (1u32..).zip(colors) {
        for y in 0..size {
            for x in 0..size {
                let noise = hash2(x.cast_signed(), y.cast_signed(), seed);
                let var = i16_from_f32(noise.mul_add(30.0, -15.0));
                let tint = |c: u8| {
                    let v = i16::from(c).saturating_add(var).clamp(0, 255);
                    u8::try_from(v).unwrap_or(u8::MAX)
                };
                albedo_data.extend([tint(base_col[0]), tint(base_col[1]), tint(base_col[2]), 255]);

                // Darker texels read as crevices: occluded and rougher.
                let ao = u8_from_f32(noise.mul_add(90.0, 165.0));
                let rough = roughness.saturating_add(u8_from_f32((1.0 - noise) * 30.0));
                arm_data.extend([ao, rough, 0, 255]);
            }
        }
    }

    let extent = Extent3d {
        width: size,
        height: size,
        depth_or_array_layers: layers,
    };

    let albedo_image = Image::new(
        extent,
        TextureDimension::D2,
        albedo_data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );

    let arm_image = Image::new(
        extent,
        TextureDimension::D2,
        arm_data,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::default(),
    );

    let albedo_handle = images.add(albedo_image);
    let arm_handle = images.add(arm_image);

    let palette = PaletteBuilder::new(albedo_handle)
        .with_arm(arm_handle)
        .add_material(
            PaletteMaterial::new("grass")
                .with_texture_scale(1.0)
                .with_blend_sharpness(4.0),
        )
        .add_material(
            PaletteMaterial::new("stone")
                .with_texture_scale(0.5)
                .with_blend_sharpness(8.0),
        )
        .add_material(
            PaletteMaterial::new("dirt")
                .with_texture_scale(1.0)
                .with_blend_sharpness(4.0),
        )
        .add_material(
            PaletteMaterial::new("sand")
                .with_texture_scale(1.2)
                .with_blend_sharpness(3.0),
        )
        .build();

    let palette_handle = palettes.add(palette);
    commands.insert_resource(ActivePalette(palette_handle));
}

pub fn terrain_height(x: f32, z: f32) -> f32 {
    let mut h = 0.0;
    let mut amp = 6.0;
    let mut freq = 0.03;
    for _ in 0..4 {
        h = ((x * freq).sin() * (z * freq * 1.3).cos()).mul_add(amp, h);
        amp *= 0.5;
        freq *= 2.1;
    }
    h
}

pub fn hash2(x: i32, z: i32, seed: u32) -> f32 {
    let mut h = x.cast_unsigned().wrapping_mul(374_761_393)
        ^ z.cast_unsigned().wrapping_mul(668_265_263)
        ^ seed.wrapping_mul(2_246_822_519);
    h ^= h >> 15;
    h = h.wrapping_mul(2_246_822_519);
    h ^= h >> 13;
    h = h.wrapping_mul(3_266_489_917);
    h ^= h >> 16;
    f32_from_u32(h) / f32_from_u32(u32::MAX)
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
        let voxel = CHUNK_SIZE / f32_from_u32(size);
        let origin = chunk_pos.as_vec3() * CHUNK_SIZE;
        let columns = usize_from(size).pow(2);
        let (mut height, mut inv_norm) = (Vec::with_capacity(columns), Vec::with_capacity(columns));
        let e = 0.5;
        for z in 0..size {
            for x in 0..size {
                let wx = f32_from_u32(x).mul_add(voxel, origin.x);
                let wz = f32_from_u32(z).mul_add(voxel, origin.z);
                let gx = (terrain_height(wx + e, wz) - terrain_height(wx - e, wz)) / (2.0 * e);
                let gz = (terrain_height(wx, wz + e) - terrain_height(wx, wz - e)) / (2.0 * e);
                height.push(terrain_height(wx, wz));
                inv_norm.push(1.0 / gz.mul_add(gz, gx.mul_add(gx, 1.0)).sqrt());
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

    /// x-fastest, then y, then z ordering (matches the `lod_streaming` example
    /// and what the fields expect).
    fn fill<T>(&self, f: impl Fn(f32, f32, f32) -> T) -> Vec<T> {
        let n = usize_from(self.size);
        let mut out = Vec::with_capacity(n.pow(3));
        // Columns are stored z-major: one row of x per z.
        let rows = self
            .height
            .chunks_exact(n)
            .zip(self.inv_norm.chunks_exact(n));
        for (heights, norms) in rows {
            for y in 0..self.size {
                let wy = f32_from_u32(y).mul_add(self.voxel, self.origin.y);
                for (&h, &k) in heights.iter().zip(norms) {
                    out.push(f(wy, h, k));
                }
            }
        }
        out
    }
}

/// Sign volume; `register_sdf` turns it into real distances with a jump flood.
pub fn terrain_sdf(pos: IVec3, lod: LOD) -> Vec<f32> {
    Columns::new(pos, lod).fill(|y, h, k| (y - h) * k)
}

pub fn terrain_materials(pos: IVec3, lod: LOD) -> Vec<u8> {
    Columns::new(pos, lod).fill(|y, h, _| determine_material(h, y).to_id())
}
