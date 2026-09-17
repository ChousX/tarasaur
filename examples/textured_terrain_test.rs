use bevy::{
    asset::RenderAssetUsages,
    input::mouse::MouseMotion,
    prelude::*,
    render::{
        RenderPlugin,
        render_resource::{Extent3d, TextureDimension, TextureFormat},
        settings::{RenderCreation, WgpuLimits, WgpuSettings},
    },
};
use tarasaur::{
    Field, LOD, MaterialField, SDFField, TarasaurPlugin, VoxelDataSlice, VoxelMaterial,
    chunk::{CHUNK_SIZE, Chunk, ChunkPosition},
    field::plugin::MaterialFieldPlugin,
    ops::{AccumulateExt, BlendExt},
    texture_palette::{
        asset::TexturePalette,
        builder::PaletteBuilder,
        plugin::{ActivePalette, PalettePlugin},
        properties::PaletteMaterial,
    },
};

#[derive(Copy, Clone, PartialEq, Eq, Default, Debug)]
#[repr(u8)]
enum TerrainMaterial {
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

fn determine_material(height: f32, world_y: f32) -> TerrainMaterial {
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

const GRID_RADIUS: i32 = 5;

const LOD_BANDS: &[(i32, LOD)] = &[
    (1, LOD::High),
    (3, LOD::Medium),
    (4, LOD::Low),
    (i32::MAX, LOD::Lowest),
];

fn lod_for_distance(chebyshev_dist: i32) -> LOD {
    for &(radius, lod) in LOD_BANDS {
        if chebyshev_dist <= radius {
            return lod;
        }
    }
    unreachable!("LOD_BANDS must end with an i32::MAX catch-all");
}

fn chunk_count() -> i32 {
    (2 * GRID_RADIUS + 1).pow(2) * 2
}

#[derive(Resource, Default)]
struct TerrainReady(bool);

#[derive(Component)]
struct FlyCam {
    pub move_speed: f32,
    pub sensitivity: f32,
    pub pitch: f32,
    pub yaw: f32,
}

fn main() {
    App::new()
        .add_plugins(
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
        .add_plugins(MaterialFieldPlugin::<TerrainMaterial>::new())
        .init_resource::<TerrainReady>()
        .add_systems(
            Startup,
            (
                spawn_camera_and_light,
                spawn_terrain_chunks,
                setup_terrain_palette,
            ),
        )
        .add_systems(
            Update,
            (
                log_material_distribution,
                fly_cam_system,
                generate_terrain,
                log_lod_distribution,
            )
                .chain(),
        )
        .run();
}

fn log_material_distribution(
    mut frame: Local<u32>,
    mut logged: Local<bool>,
    ready: Res<TerrainReady>,
    query: Query<&MaterialField<TerrainMaterial>>,
) {
    if *logged || !ready.0 {
        return;
    }
    *frame += 1;
    if *frame < 5 {
        return;
    }
    *logged = true;

    let mut counts = [0u64; TerrainMaterial::COUNT as usize];
    for field in query.iter() {
        for &id in field.data_slice() {
            counts[id as usize] += 1;
        }
    }

    info!(
        "[lod_terrain_test] material voxel counts: grass={} stone={} dirt={} sand={}",
        counts[0], counts[1], counts[2], counts[3]
    );
}

fn setup_terrain_palette(
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

fn spawn_camera_and_light(mut commands: Commands) {
    let initial_transform =
        Transform::from_xyz(0.0, 35.0, 30.0).looking_at(Vec3::new(0.0, 8.0, 0.0), Vec3::Y);
    let (yaw, pitch, _) = initial_transform.rotation.to_euler(EulerRot::YXZ);

    commands.spawn((
        Camera3d::default(),
        initial_transform,
        FlyCam {
            move_speed: 20.0,
            sensitivity: 0.002,
            pitch,
            yaw,
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

fn fly_cam_system(
    time: Res<Time>,
    mouse_button: Res<ButtonInput<MouseButton>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut mouse_motion: MessageReader<MouseMotion>,
    mut query: Query<(&mut Transform, &mut FlyCam)>,
) {
    let delta_time = time.delta_secs();

    for (mut transform, mut fly_cam) in query.iter_mut() {
        if mouse_button.pressed(MouseButton::Right) {
            for ev in mouse_motion.read() {
                fly_cam.yaw -= ev.delta.x * fly_cam.sensitivity;
                fly_cam.pitch -= ev.delta.y * fly_cam.sensitivity;
                fly_cam.pitch = fly_cam.pitch.clamp(-1.54, 1.54);
            }
            transform.rotation = Quat::from_euler(EulerRot::YXZ, fly_cam.yaw, fly_cam.pitch, 0.0);
        } else {
            mouse_motion.clear();
        }

        let mut velocity = Vec3::ZERO;
        let forward = *transform.forward();
        let right = *transform.right();

        if keyboard.pressed(KeyCode::KeyW) {
            velocity += forward;
        }
        if keyboard.pressed(KeyCode::KeyS) {
            velocity -= forward;
        }
        if keyboard.pressed(KeyCode::KeyD) {
            velocity += right;
        }
        if keyboard.pressed(KeyCode::KeyA) {
            velocity -= right;
        }
        if keyboard.pressed(KeyCode::Space) {
            velocity += Vec3::Y;
        }
        if keyboard.pressed(KeyCode::ShiftLeft) {
            velocity -= Vec3::Y;
        }

        if velocity != Vec3::ZERO {
            let speed_multiplier = if keyboard.pressed(KeyCode::ControlLeft) {
                2.5
            } else {
                1.0
            };
            transform.translation +=
                velocity.normalize() * fly_cam.move_speed * speed_multiplier * delta_time;
        }
    }
}

fn spawn_terrain_chunks(mut commands: Commands) {
    let mut counts: std::collections::HashMap<LOD, u32> = std::collections::HashMap::new();

    for x in -GRID_RADIUS..=GRID_RADIUS {
        for z in -GRID_RADIUS..=GRID_RADIUS {
            for y in 0..=1 {
                let dist = x.abs().max(z.abs());
                let lod = lod_for_distance(dist);
                *counts.entry(lod).or_insert(0) += 1;
                commands.spawn((Chunk, ChunkPosition(IVec3::new(x, y, z)), lod));
            }
        }
    }

    info!(
        "[lod_terrain_test] spawned {} chunks across bands {:?} -> counts: {:?}",
        chunk_count(),
        LOD_BANDS,
        counts
    );
}

fn hash2(x: i32, z: i32, seed: u32) -> f32 {
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

#[inline]
fn smoothstep(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

fn value_noise(x: f32, z: f32, seed: u32) -> f32 {
    let x0 = x.floor() as i32;
    let z0 = z.floor() as i32;
    let tx = smoothstep(x - x0 as f32);
    let tz = smoothstep(z - z0 as f32);

    let v00 = hash2(x0, z0, seed);
    let v10 = hash2(x0 + 1, z0, seed);
    let v01 = hash2(x0, z0 + 1, seed);
    let v11 = hash2(x0 + 1, z0 + 1, seed);

    let a = v00 + (v10 - v00) * tx;
    let b = v01 + (v11 - v01) * tx;
    a + (b - a) * tz
}

fn fbm(x: f32, z: f32, octaves: u32, seed: u32) -> f32 {
    let mut amplitude = 0.5;
    let mut frequency = 1.0;
    let mut sum = 0.0;
    let mut max = 0.0;
    for i in 0..octaves {
        sum += value_noise(x * frequency, z * frequency, seed.wrapping_add(i)) * amplitude;
        max += amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    sum / max
}

fn terrain_height(world_x: f32, world_z: f32) -> f32 {
    const BASE_HEIGHT: f32 = 10.0;
    const AMPLITUDE: f32 = 7.5;
    const NOISE_SCALE: f32 = 0.03;

    let base = fbm(world_x * NOISE_SCALE, world_z * NOISE_SCALE, 5, 1337);
    let ridge = (0.5 - (base - 0.5).abs()) * 2.0;
    let mountain_mask = fbm(world_x * 0.01, world_z * 0.01, 3, 42);

    let combined_noise = base + (ridge * mountain_mask * 0.6);

    BASE_HEIGHT + (combined_noise - 0.5) * 2.0 * AMPLITUDE
}

fn generate_terrain(
    mut generated: Local<bool>,
    mut ready: ResMut<TerrainReady>,
    mut query: Query<(
        &ChunkPosition,
        &mut SDFField,
        &mut MaterialField<TerrainMaterial>,
    )>,
) {
    if *generated || query.iter().count() as i32 != chunk_count() {
        return;
    }
    *generated = true;

    for (ChunkPosition(chunk_pos), mut sdf, mut mat_field) in query.iter_mut() {
        let size = sdf.size();
        let voxel_size = CHUNK_SIZE / size.x as f32;
        let chunk_origin = chunk_pos.as_vec3() * CHUNK_SIZE;

        for z in 0..size.z {
            for y in 0..size.y {
                for x in 0..size.x {
                    let world = chunk_origin + Vec3::new(x as f32, y as f32, z as f32) * voxel_size;
                    let height = terrain_height(world.x, world.z);
                    let value = if world.y < height { -1.0 } else { 1.0 };

                    let mat = determine_material(height, world.y);

                    sdf.set(x, y, z, value);
                    mat_field.set(x, y, z, mat);
                }
            }
        }

        sdf.reinit();
    }

    ready.0 = true;
    info!(
        "[lod_terrain_test] terrain generated across all {} chunks",
        chunk_count()
    );
}

fn log_lod_distribution(
    mut frame: Local<u32>,
    mut logged: Local<bool>,
    ready: Res<TerrainReady>,
    query: Query<(&ChunkPosition, &LOD, &SDFField)>,
) {
    if *logged || !ready.0 {
        return;
    }
    *frame += 1;
    if *frame < 5 {
        return;
    }
    *logged = true;

    let mut per_lod: std::collections::HashMap<LOD, (u32, f32, f32)> =
        std::collections::HashMap::new();

    for (_pos, lod, sdf) in query.iter() {
        let data = sdf.data_slice();
        let (mut min, mut max) = (f32::MAX, f32::MIN);
        for &v in data {
            min = min.min(v);
            max = max.max(v);
        }
        let entry = per_lod.entry(*lod).or_insert((0, f32::MAX, f32::MIN));
        entry.0 += 1;
        entry.1 = entry.1.min(min);
        entry.2 = entry.2.max(max);
    }

    for (lod, (count, min, max)) in per_lod.iter() {
        info!(
            "[lod_terrain_test] LOD {:?}: {} chunks, sdf range=[{:.3}, {:.3}]",
            lod, count, min, max
        );
    }
}
