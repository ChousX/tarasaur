use bevy::{
    asset::RenderAssetUsages,
    input::mouse::MouseMotion,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};
use tarasaur::{
    VoxelMaterial,
    ops::{AccumulateExt, BlendExt},
    texture_palette::{PaletteBuilder, PaletteMaterial, TexturePalette, plugin::ActivePalette},
};

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
                cam.yaw -= ev.delta.x * cam.sensitivity;
                cam.pitch -= ev.delta.y * cam.sensitivity;
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
        self as u8
    }

    fn from_id(id: u8) -> Self {
        match id {
            0 => Self::Grass,
            1 => Self::Stone,
            2 => Self::Dirt,
            3 => Self::Sand,
            _ => unreachable!(
                "id {id} out of range for TerrainMaterial::COUNT = {}",
                Self::COUNT
            ),
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
    let size = 64;
    let layers = 4;
    let bytes_per_pixel = 4;
    let layer_size = (size * size * bytes_per_pixel) as usize;

    let colors: [[u8; 3]; 4] = [
        [34, 139, 34],   // Forest Green
        [128, 128, 128], // Stone Gray
        [112, 66, 20],   // Dirt Brown
        [225, 198, 153], // Sand Beige
    ];

    let mut albedo_data = vec![0u8; layer_size * layers as usize];
    let mut normal_data = vec![0u8; layer_size * layers as usize];

    for layer in 0..layers as usize {
        let layer_offset = layer * layer_size;
        let base_col = colors[layer];

        for y in 0..size {
            for x in 0..size {
                let pixel_offset = layer_offset + ((y * size + x) * bytes_per_pixel) as usize;

                let noise = hash2(x as i32, y as i32, (layer + 1) as u32);
                let var = (noise * 30.0 - 15.0) as i16;

                albedo_data[pixel_offset] = (base_col[0] as i16 + var).clamp(0, 255) as u8;
                albedo_data[pixel_offset + 1] = (base_col[1] as i16 + var).clamp(0, 255) as u8;
                albedo_data[pixel_offset + 2] = (base_col[2] as i16 + var).clamp(0, 255) as u8;
                albedo_data[pixel_offset + 3] = 255;

                normal_data[pixel_offset] = (128.0 + (noise - 0.5) * 40.0) as u8;
                normal_data[pixel_offset + 1] = (128.0 + (noise - 0.5) * 40.0) as u8;
                normal_data[pixel_offset + 2] = 255;
                normal_data[pixel_offset + 3] = 255;
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

    let normal_image = Image::new(
        extent,
        TextureDimension::D2,
        normal_data,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::default(),
    );

    let albedo_handle = images.add(albedo_image);
    let normal_handle = images.add(normal_image);

    let palette = PaletteBuilder::new()
        .with_albedo(albedo_handle)
        .with_normal(normal_handle)
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
        h += (x * freq).sin() * (z * freq * 1.3).cos() * amp;
        amp *= 0.5;
        freq *= 2.1;
    }
    h
}

pub fn hash2(x: i32, z: i32, seed: u32) -> f32 {
    let mut h = (x as u32).wrapping_mul(374761393)
        ^ (z as u32).wrapping_mul(668265263)
        ^ seed.wrapping_mul(2246822519);
    h ^= h >> 15;
    h = h.wrapping_mul(2246822519);
    h ^= h >> 13;
    h = h.wrapping_mul(3266489917);
    h ^= h >> 16;
    (h as f32) / (u32::MAX as f32)
}
