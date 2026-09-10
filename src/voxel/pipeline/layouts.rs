use std::num::NonZeroU64;

use bevy::render::render_resource::*;

/// A read-write or read-only storage buffer binding, bound to `ShaderStages`.
pub fn storage_buffer_entry(
    binding: u32,
    visibility: ShaderStages,
    read_only: bool,
) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// A uniform buffer binding. Pass `min_binding_size` when the shader
/// requires a guaranteed minimum size (e.g. a fixed-size struct binding
/// shared across draw calls); otherwise pass `None`.
pub fn uniform_buffer_entry(
    binding: u32,
    visibility: ShaderStages,
    min_binding_size: Option<NonZeroU64>,
) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size,
        },
        count: None,
    }
}

/// A 3D storage texture binding (used for the SDF volume across all passes).
pub fn storage_texture_3d_entry(
    binding: u32,
    visibility: ShaderStages,
    format: TextureFormat,
    access: StorageTextureAccess,
) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility,
        ty: BindingType::StorageTexture {
            access,
            format,
            view_dimension: TextureViewDimension::D3,
        },
        count: None,
    }
}
/// Binding layout shared by all three stream-compaction shader entry points
/// (`scan_workgroup`, `scan_block_sums`, `resolve_block_offsets`).
pub fn compaction_entries() -> Vec<BindGroupLayoutEntry> {
    vec![
        uniform_buffer_entry(0, ShaderStages::COMPUTE, None),
        storage_buffer_entry(1, ShaderStages::COMPUTE, false),
        storage_buffer_entry(2, ShaderStages::COMPUTE, false),
        storage_buffer_entry(3, ShaderStages::COMPUTE, false),
    ]
}

/// Binding layout for `surface_nets_pass1`.
pub fn pass1_surface_entries() -> Vec<BindGroupLayoutEntry> {
    vec![
        storage_texture_3d_entry(
            0,
            ShaderStages::COMPUTE,
            TextureFormat::R32Float,
            StorageTextureAccess::ReadOnly,
        ),
        storage_buffer_entry(1, ShaderStages::COMPUTE, false),
        storage_buffer_entry(2, ShaderStages::COMPUTE, false),
        storage_buffer_entry(3, ShaderStages::COMPUTE, false),
        storage_buffer_entry(4, ShaderStages::COMPUTE | ShaderStages::VERTEX, false),
        storage_buffer_entry(5, ShaderStages::COMPUTE, false),
        uniform_buffer_entry(6, ShaderStages::COMPUTE, None),
    ]
}

/// Binding layout for `surface_nets_pass3`.
pub fn pass3_surface_entries() -> Vec<BindGroupLayoutEntry> {
    vec![
        storage_texture_3d_entry(
            0,
            ShaderStages::COMPUTE,
            TextureFormat::R32Float,
            StorageTextureAccess::ReadOnly,
        ),
        storage_buffer_entry(1, ShaderStages::COMPUTE, true),
        storage_buffer_entry(2, ShaderStages::COMPUTE, true),
        storage_buffer_entry(3, ShaderStages::COMPUTE, false),
        storage_buffer_entry(4, ShaderStages::COMPUTE, false),
        storage_buffer_entry(5, ShaderStages::COMPUTE | ShaderStages::VERTEX, false),
        uniform_buffer_entry(6, ShaderStages::COMPUTE, NonZeroU64::new(32)),
    ]
}
