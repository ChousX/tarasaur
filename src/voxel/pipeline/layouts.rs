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

/// Binding layout shared by all four stream-compaction shader entry points
/// (`scan_workgroup`, `scan_block_sums`, `resolve_block_offsets`,
/// `write_chunk_active_count`). Binding 4 (chunk_active_counts) is only
/// written by `write_chunk_active_count`, but the layout must be shared
/// across all entry points that use this bind group, so it's declared
/// read_write here even though most entry points leave it untouched.
pub fn compaction_entries() -> Vec<BindGroupLayoutEntry> {
    vec![
        uniform_buffer_entry(0, ShaderStages::COMPUTE, None),
        storage_buffer_entry(1, ShaderStages::COMPUTE, false),
        storage_buffer_entry(2, ShaderStages::COMPUTE, false),
        storage_buffer_entry(3, ShaderStages::COMPUTE, false),
        storage_buffer_entry(4, ShaderStages::COMPUTE, false), // chunk_active_counts
    ]
}

/// Binding layout for `surface_nets_pass1`.
pub fn pass1_surface_entries() -> Vec<BindGroupLayoutEntry> {
    vec![
        storage_buffer_entry(0, ShaderStages::COMPUTE, true), // sdf_buffer: now array<f32>, not a texture
        storage_buffer_entry(1, ShaderStages::COMPUTE, false),
        storage_buffer_entry(2, ShaderStages::COMPUTE, false),
        storage_buffer_entry(3, ShaderStages::COMPUTE, false),
        storage_buffer_entry(4, ShaderStages::COMPUTE | ShaderStages::VERTEX, false),
        storage_buffer_entry(5, ShaderStages::COMPUTE, false),
        uniform_buffer_entry(6, ShaderStages::COMPUTE, NonZeroU64::new(16)),
        storage_buffer_entry(7, ShaderStages::COMPUTE, true), // chunk_meta
        storage_buffer_entry(8, ShaderStages::COMPUTE, true), // active_slot_map: dispatch-order pos -> real arena slot
    ]
}

/// Binding layout for `surface_nets_pass3`.
pub fn pass3_surface_entries() -> Vec<BindGroupLayoutEntry> {
    vec![
        storage_buffer_entry(0, ShaderStages::COMPUTE, true), // sdf_buffer: now array<f32>, not a texture
        storage_buffer_entry(1, ShaderStages::COMPUTE, true),
        storage_buffer_entry(2, ShaderStages::COMPUTE, true),
        storage_buffer_entry(3, ShaderStages::COMPUTE, false),
        storage_buffer_entry(4, ShaderStages::COMPUTE, false),
        storage_buffer_entry(5, ShaderStages::COMPUTE | ShaderStages::VERTEX, false),
        uniform_buffer_entry(6, ShaderStages::COMPUTE, NonZeroU64::new(16)),
        storage_buffer_entry(7, ShaderStages::COMPUTE, true), // chunk_meta
        storage_buffer_entry(8, ShaderStages::COMPUTE, true), // chunk_vertex_base
        storage_buffer_entry(9, ShaderStages::COMPUTE, true), // chunk_index_base
        storage_buffer_entry(10, ShaderStages::COMPUTE, true), // active_slot_map: dispatch-order pos -> real arena slot
        // material_buffer: packed as array<u32>, 4 material-id bytes per
        // word — WGSL storage buffers have no array<u8>. Only pass3 reads
        // it (material tallying happens per-vertex, not per-cell), so no
        // change needed to pass1_surface_entries() or compaction_entries().
        storage_buffer_entry(11, ShaderStages::COMPUTE, true),
    ]
}

/// Binding layout for `compute_chunk_bases` (single-workgroup, serial prefix
/// sum over active chunks). Binding 6 (overflow_flag) uses an atomic store
/// in the shader, so it must be storage read_write even though the buffer
/// is logically a single u32.
pub fn chunk_bases_entries() -> Vec<BindGroupLayoutEntry> {
    vec![
        uniform_buffer_entry(0, ShaderStages::COMPUTE, NonZeroU64::new(16)),
        storage_buffer_entry(1, ShaderStages::COMPUTE, true), // chunk_active_counts
        storage_buffer_entry(2, ShaderStages::COMPUTE, true), // active_slot_map
        storage_buffer_entry(3, ShaderStages::COMPUTE, false), // chunk_vertex_base
        storage_buffer_entry(4, ShaderStages::COMPUTE, false), // chunk_index_base
        storage_buffer_entry(5, ShaderStages::COMPUTE, false), // indirect_args
        storage_buffer_entry(6, ShaderStages::COMPUTE, false), // overflow_flag
    ]
}
