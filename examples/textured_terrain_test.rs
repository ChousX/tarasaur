//! A fixed grid of chunks with generated, palette-textured terrain at
//! mixed LODs; logs material and LOD distribution.

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
use tarasaur::editor::WorldEditor;
use tarasaur::num::{f32_from_i32, f32_from_u32, i16_from_f32, i32_from_f32, usize_from};
use tarasaur::{
    Field, LOD, MaterialField, SDF, TarasaurPlugin, VisibilityField, VoxelDataSlice, VoxelMaterial,
    chunk::{CHUNK_SIZE, Chunk, ChunkPosition},
    field::MaterialFieldPlugin,
    ops::{AccumulateExt, BlendExt},
    texture_palette::{
        asset::TexturePalette, builder::PaletteBuilder, plugin::ActivePalette,
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
    // LOD_BANDS ends with an i32::MAX catch-all, so the fallback never applies.
    LOD_BANDS
        .iter()
        .find(|&&(radius, _)| chebyshev_dist <= radius)
        .map_or(LOD::Lowest, |&(_, lod)| lod)
}

/// A (2r+1)^2 grid, two chunks tall.
const CHUNK_COUNT: i32 = (2 * GRID_RADIUS + 1).pow(2) * 2;

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
        .add_plugins(DefaultPlugins.set(RenderPlugin {
            render_creation: RenderCreation::Automatic(Box::new(WgpuSettings {
                limits: WgpuLimits {
                    max_buffer_size: 1024 * 1024 * 1024,
                    max_storage_buffer_binding_size: 512 * 1024 * 1024,
                    ..default()
                },
                ..default()
            })),
            ..default()
        }))
        .add_plugins(TarasaurPlugin)
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
    *frame = frame.saturating_add(1);
    if *frame < 5 {
        return;
    }
    *logged = true;

    let mut counts = [0u64; 4]; // one per TerrainMaterial variant
    for field in query.iter() {
        for &id in field.data_slice() {
            if let Some(c) = counts.get_mut(usize::from(id)) {
                *c = c.saturating_add(1);
            }
        }
    }

    let [grass, stone, dirt, sand] = counts;
    info!(
        "[lod_terrain_test] material voxel counts: grass={grass} stone={stone} dirt={dirt} sand={sand}"
    );
}

fn setup_terrain_palette(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut palettes: ResMut<Assets<TexturePalette>>,
) {
    let size: u32 = 64;

    let colors: [[u8; 3]; 4] = [
        [34, 139, 34],   // Forest Green
        [128, 128, 128], // Stone Gray
        [112, 66, 20],   // Dirt Brown
        [225, 198, 153], // Sand Beige
    ];
    let layers = u32::try_from(colors.len()).unwrap_or(u32::MAX);

    // RGBA8, layer-major then row-major, as `Image::new` expects.
    let bytes = usize_from(size)
        .pow(2)
        .saturating_mul(4)
        .saturating_mul(colors.len());
    let mut albedo_data = Vec::with_capacity(bytes);

    for (seed, base_col) in (1u32..).zip(colors) {
        for y in 0..size {
            for x in 0..size {
                let noise = hash2(x.cast_signed(), y.cast_signed(), seed);
                let var = i16_from_f32(noise.mul_add(30.0, -15.0));
                let tint = |c: u8| {
                    let v = i16::from(c).saturating_add(var).clamp(0, 255);
                    u8::try_from(v).unwrap_or(u8::MAX)
                };
                albedo_data.extend([tint(base_col[0]), tint(base_col[1]), tint(base_col[2]), 255]);
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

    let albedo_handle = images.add(albedo_image);

    let palette = PaletteBuilder::new(albedo_handle)
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
                fly_cam.yaw = f32::mul_add(ev.delta.x, -fly_cam.sensitivity, fly_cam.yaw);
                fly_cam.pitch = f32::mul_add(ev.delta.y, -fly_cam.sensitivity, fly_cam.pitch);
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
                let count = counts.entry(lod).or_insert(0);
                *count = count.saturating_add(1);
                commands.spawn((Chunk, ChunkPosition(IVec3::new(x, y, z)), lod));
            }
        }
    }

    info!(
        "[lod_terrain_test] spawned {} chunks across bands {:?} -> counts: {:?}",
        CHUNK_COUNT, LOD_BANDS, counts
    );
}

fn hash2(x: i32, z: i32, seed: u32) -> f32 {
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

#[inline]
fn smoothstep(t: f32) -> f32 {
    t * t * 2.0f32.mul_add(-t, 3.0)
}

fn value_noise(x: f32, z: f32, seed: u32) -> f32 {
    let x0 = i32_from_f32(x.floor());
    let z0 = i32_from_f32(z.floor());
    let tx = smoothstep(x - f32_from_i32(x0));
    let tz = smoothstep(z - f32_from_i32(z0));
    let (x1, z1) = (x0.wrapping_add(1), z0.wrapping_add(1));

    let v00 = hash2(x0, z0, seed);
    let v10 = hash2(x1, z0, seed);
    let v01 = hash2(x0, z1, seed);
    let v11 = hash2(x1, z1, seed);

    let a = (v10 - v00).mul_add(tx, v00);
    let b = (v11 - v01).mul_add(tx, v01);
    (b - a).mul_add(tz, a)
}

fn fbm(x: f32, z: f32, octaves: u32, seed: u32) -> f32 {
    let mut amplitude = 0.5;
    let mut frequency = 1.0;
    let mut sum = 0.0;
    let mut max = 0.0;
    for i in 0..octaves {
        sum =
            value_noise(x * frequency, z * frequency, seed.wrapping_add(i)).mul_add(amplitude, sum);
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

    let combined_noise = (ridge * mountain_mask).mul_add(0.6, base);

    ((combined_noise - 0.5) * 2.0).mul_add(AMPLITUDE, BASE_HEIGHT)
}

fn generate_terrain(
    mut generated: Local<bool>,
    mut ready: ResMut<TerrainReady>,
    mut query: Query<(
        &ChunkPosition,
        &mut SDF,
        &mut MaterialField<TerrainMaterial>,
    )>,
    mut visibility_editor: WorldEditor<VisibilityField, bool>,
) {
    if *generated || i32::try_from(query.iter().count()).ok() != Some(CHUNK_COUNT) {
        return;
    }
    *generated = true;

    for (ChunkPosition(chunk_pos), mut sdf, mut mat_field) in query.iter_mut() {
        let size = sdf.size();
        let voxel_size = CHUNK_SIZE / f32_from_u32(size.x);
        let chunk_origin = chunk_pos.as_vec3() * CHUNK_SIZE;

        for z in 0..size.z {
            for y in 0..size.y {
                for x in 0..size.x {
                    let world = chunk_origin + UVec3::new(x, y, z).as_vec3() * voxel_size;
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

    // Both edits use fill_box — same message type (EditFieldMessage
    // VisibilityField, Cuboid, bool>), same processing system
    // (process_box_edits), which drains its MessageReader in write order.
    // That guarantees "mark everything visible" applies before "hide this
    // region" within this one frame — no race, no delay needed. Mixing
    // box + sphere would put the two edits in different systems with no
    // defined relative order.
    let radius = f32_from_i32(GRID_RADIUS);
    let world_min = Vec3::new(-radius * CHUNK_SIZE, 0.0, -radius * CHUNK_SIZE);
    let world_max = Vec3::new(
        (radius + 1.0) * CHUNK_SIZE,
        2.0 * CHUNK_SIZE,
        (radius + 1.0) * CHUNK_SIZE,
    );
    let world_center = (world_min + world_max) * 0.5;
    //let world_half_extent = (world_max - world_min) * 0.5;
    //visibility_editor.fill_box(world_center, world_half_extent, true);
    visibility_editor.fill_sphere(world_center, 10.0, true);
    ready.0 = true;
    info!(
        "[lod_terrain_test] terrain generated across all {} chunks",
        CHUNK_COUNT
    );
}

fn log_lod_distribution(
    mut frame: Local<u32>,
    mut logged: Local<bool>,
    ready: Res<TerrainReady>,
    query: Query<(&ChunkPosition, &LOD, &SDF)>,
) {
    if *logged || !ready.0 {
        return;
    }
    *frame = frame.saturating_add(1);
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
        entry.0 = entry.0.saturating_add(1);
        entry.1 = entry.1.min(min);
        entry.2 = entry.2.max(max);
    }

    for (lod, (count, min, max)) in &per_lod {
        info!(
            "[lod_terrain_test] LOD {:?}: {} chunks, sdf range=[{:.3}, {:.3}]",
            lod, count, min, max
        );
    }
}
