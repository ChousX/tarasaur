use std::num::NonZeroU64;

use bevy::render::render_resource::*;

use crate::LOD;

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
        storage_buffer_entry(7, ShaderStages::COMPUTE, true),
        storage_buffer_entry(8, ShaderStages::COMPUTE, true),
        storage_buffer_entry(9, ShaderStages::COMPUTE, true),
        storage_buffer_entry(10, ShaderStages::COMPUTE, true),
    ]
}

/// Binding layout for `surface_nets_pass3`. Back to 12 entries — visibility
/// is no longer sampled here. Pass3 only needs to know *which slot* each
/// vertex belongs to (packed into the vertex itself, see pack_material_b in
/// the shader); the actual visibility test now happens per-fragment in
/// voxel_raster.wgsl against raster_chunk_visibility_entries() below.
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

/// Binding layout for group 2 of the raster pipeline — per-fragment
/// visibility sampling. Bound once per arena (all buffers here are
/// per-arena, not per-material), alongside group 0 (view) and group 1
/// (material). All FRAGMENT-only: the vertex stage doesn't need any of
/// this, it only carries `real_slot` through as a packed flat attribute.
///
/// Binding 3 reuses the same BatchUniforms layout as pass3's own uniform
/// (cell_count, texture_size, wg_per_chunk_z, voxel_size) — only
/// texture_size/voxel_size are read here, but no reason to duplicate the
/// buffer when pass3's `batch_uniform_buffer_pass3` already holds the
/// right values for this arena and never changes after arena creation.
pub fn raster_chunk_visibility_entries() -> Vec<BindGroupLayoutEntry> {
    vec![
        storage_buffer_entry(0, ShaderStages::FRAGMENT, true), // chunk_meta
        storage_buffer_entry(1, ShaderStages::FRAGMENT, true), // visibility_mask_buffer
        storage_buffer_entry(2, ShaderStages::FRAGMENT, true), // chunk_has_mask_buffer
        uniform_buffer_entry(3, ShaderStages::FRAGMENT, NonZeroU64::new(16)), // BatchUniforms
    ]
}

/// Binding layout for the cursor/raycast query pass (`cursor_query.wgsl`).
/// Binding 1 (hits) is read_write since the shader writes results in
/// place. Everything else is read-only from the shader's perspective.
pub fn query_entries() -> Vec<BindGroupLayoutEntry> {
    let mut entries = vec![
        storage_buffer_entry(0, ShaderStages::COMPUTE, true), // queries
        storage_buffer_entry(1, ShaderStages::COMPUTE, false), // hits
        uniform_buffer_entry(2, ShaderStages::COMPUTE, NonZeroU64::new(80)), // QueryUniforms, grew
        storage_buffer_entry(3, ShaderStages::COMPUTE, true), // merged chunk_lookup
    ];
    for lod_rank in 0..LOD::COUNT as u32 {
        let base = 4 + lod_rank * 2;
        entries.push(storage_buffer_entry(base, ShaderStages::COMPUTE, true)); // sdf_buffer[lod]
        entries.push(storage_buffer_entry(base + 1, ShaderStages::COMPUTE, true)); // chunk_meta[lod]
    }
    entries
}
