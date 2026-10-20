//! A second surface layer for the `layers` example and `bench --layer`: a
//! flat water table. Its own shape field (signed distances like the
//! `SDF`'s), generated per chunk, meshed by the GPU surface nets and drawn
//! with one flat color (`flat_layer.wesl`).
#![allow(dead_code)]

use std::any::type_name;

use bevy::{
    asset::{load_internal_asset, uuid_handle},
    prelude::*,
    render::{
        render_resource::{
            BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, BindingType, BufferBindingType,
            BufferInitDescriptor, BufferUsages, ShaderStages,
        },
        renderer::RenderDevice,
    },
};
use tarasaur::{
    AppSurfaceLayerExt, LayerRaster, SurfaceLayer,
    chunk::CHUNK_SIZE,
    field::{Field, GpuChannel, LOD, SDF_BAND, VoxelField, quantize_sdf},
    grid::{flatten_cube, sample_neighbor},
    num::{f32_from_i32, f32_from_u32},
    storage::generator::ChunkGeneratorRegistry,
    voxel::{LayerMaterial, NoChannel},
};

/// Height of the water table, world units.
pub const WATER_LEVEL: f32 = 3.0;

/// The water's color, linear RGBA.
const WATER_COLOR: [f32; 4] = [0.15, 0.45, 0.8, 1.0];

pub const FLAT_LAYER_SHADER: Handle<Shader> = uuid_handle!("7a8b9c0d-1e2f-4a5b-8c6d-7e8f9a0b1c2d");

/// Signed distance to the water table in voxels of the chunk's LOD, stored
/// like the `SDF` (and saved and uploaded the same way).
#[derive(Component, Clone)]
pub struct WaterField {
    lod: LOD,
    data: Box<[f32]>,
    version: u64,
}

impl Field<f32> for WaterField {
    fn size(&self) -> UVec3 {
        UVec3::splat(self.lod.size())
    }

    fn get(&self, x: u32, y: u32, z: u32) -> f32 {
        self.data
            .get(flatten_cube(x, y, z, self.lod.size()))
            .copied()
            .unwrap_or(SDF_BAND)
    }

    fn set(&mut self, x: u32, y: u32, z: u32, value: f32) {
        let i = flatten_cube(x, y, z, self.lod.size());
        let value = value.clamp(-SDF_BAND, SDF_BAND);
        if let Some(slot) = self.data.get_mut(i)
            && slot.to_bits() != value.to_bits()
        {
            *slot = value;
            self.version = self.version.wrapping_add(1);
        }
    }
}

impl VoxelField for WaterField {
    type Value = f32;
    type Elem = f32;
    const GPU: GpuChannel = GpuChannel::Shape;

    fn new_for_lod(lod: LOD) -> Self {
        Self::from_raw(lod, vec![SDF_BAND; lod.volume()])
    }

    fn from_raw(lod: LOD, data: Vec<f32>) -> Self {
        Self {
            lod,
            data: data.into_boxed_slice(),
            version: 0,
        }
    }

    fn lod(&self) -> LOD {
        self.lod
    }

    fn data_slice(&self) -> &[f32] {
        &self.data
    }

    fn version(&self) -> u64 {
        self.version
    }

    fn gpu_bytes(vol: &[f32]) -> Vec<u8> {
        let packed: Vec<i16> = vol.iter().map(|&v| quantize_sdf(v)).collect();
        bytemuck::cast_slice(&packed).to_vec()
    }

    fn rescale(v: f32, ratio: f32) -> f32 {
        v / ratio
    }

    fn sample_apron(data: &[f32], size: u32, x: f32, y: f32, z: f32) -> f32 {
        sample_neighbor(data, size, x, y, z)
    }

    fn extend_edge(edge: f32, inward: [f32; 3], past: UVec3) -> f32 {
        inward
            .iter()
            .zip(past.to_array())
            .fold(edge, |v, (&i, k)| (edge - i).mul_add(f32_from_u32(k), v))
    }
}

/// The water table's distances for one chunk: a horizontal plane.
fn water_sdf(pos: IVec3, lod: LOD) -> Vec<f32> {
    let voxel = lod.voxel_size();
    let origin_y = f32_from_i32(pos.y) * CHUNK_SIZE;
    let n = lod.size();
    let mut out = Vec::with_capacity(lod.volume());
    for _z in 0..n {
        for y in 0..n {
            let wy = f32_from_u32(y).mul_add(voxel, origin_y);
            let d = ((wy - WATER_LEVEL) / voxel).clamp(-SDF_BAND, SDF_BAND);
            out.extend(std::iter::repeat_n(d, n as usize));
        }
    }
    out
}

/// One flat color, lit by a fixed light; opaque, like the terrain.
pub struct FlatRaster;

impl LayerRaster for FlatRaster {
    fn shader() -> Handle<Shader> {
        FLAT_LAYER_SHADER
    }

    fn material_layout_entries() -> Vec<BindGroupLayoutEntry> {
        vec![BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }]
    }

    fn material(
        world: &mut World,
        layout: &BindGroupLayout,
        current: Option<&LayerMaterial>,
    ) -> Option<LayerMaterial> {
        if current.is_some() {
            return None; // the color never changes
        }
        let device = world.resource::<RenderDevice>();
        let color = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("flat_layer_color"),
            contents: bytemuck::bytes_of(&WATER_COLOR),
            usage: BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(
            Some("flat_layer_material"),
            layout,
            &[BindGroupEntry {
                binding: 0,
                resource: color.as_entire_binding(),
            }],
        );
        Some(LayerMaterial::plain(bind_group))
    }
}

pub struct WaterLayer;

impl SurfaceLayer for WaterLayer {
    type Shape = WaterField;
    type Material = NoChannel;
    type Mask = NoChannel;
    type Raster = FlatRaster;
}

/// Adds the water table: its shader, its generator and the layer.
pub fn add_water_layer(app: &mut App) {
    load_internal_asset!(app, FLAT_LAYER_SHADER, "flat_layer.wesl", Shader::from_wesl);
    app.world_mut()
        .resource_mut::<ChunkGeneratorRegistry>()
        .register::<f32>(type_name::<WaterField>(), water_sdf);
    app.add_surface_layer::<WaterLayer>();
}
