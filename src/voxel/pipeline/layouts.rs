//! Bind group layout entries; binding numbers must match the shaders.

use std::num::NonZeroU64;

use bevy::render::render_resource::{
    BindGroupLayoutEntry, BindingType, BufferBindingType, ShaderStages,
};

use crate::LOD;

/// A read-write or read-only storage buffer binding, bound to `ShaderStages`.
pub const fn storage_buffer_entry(
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
pub const fn uniform_buffer_entry(
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

/// Binding layout for `surface_nets_pass1`.
pub fn pass1_surface_entries() -> Vec<BindGroupLayoutEntry> {
    vec![
        storage_buffer_entry(0, ShaderStages::COMPUTE, true), // sdf_buffer: snorm16, 2 voxels per u32
        storage_buffer_entry(1, ShaderStages::COMPUTE, false), // active_cells: [vertex] -> cell
        storage_buffer_entry(2, ShaderStages::COMPUTE, false), // cell_vertex: [cell] -> vertex
        storage_buffer_entry(3, ShaderStages::COMPUTE, false), // chunk_counts: per-chunk atomics
        storage_buffer_entry(4, ShaderStages::COMPUTE | ShaderStages::VERTEX, false),
        storage_buffer_entry(5, ShaderStages::COMPUTE, false),
        uniform_buffer_entry(6, ShaderStages::COMPUTE, NonZeroU64::new(16)),
        storage_buffer_entry(7, ShaderStages::COMPUTE, true),
        storage_buffer_entry(8, ShaderStages::COMPUTE, true),
        storage_buffer_entry(9, ShaderStages::COMPUTE, true),
        storage_buffer_entry(10, ShaderStages::COMPUTE, true),
    ]
}

/// Binding layout for `surface_nets_pass3`. Visibility isn't sampled here:
/// pass3 only packs each vertex's slot into the vertex (`pack_material_b`
/// in the shader), and the visibility test happens per fragment in
/// `voxel_raster.wesl` against `raster_chunk_visibility_entries()` below.
pub fn pass3_surface_entries() -> Vec<BindGroupLayoutEntry> {
    vec![
        storage_buffer_entry(0, ShaderStages::COMPUTE, true), // sdf_buffer: snorm16, 2 voxels per u32
        storage_buffer_entry(1, ShaderStages::COMPUTE, true),
        storage_buffer_entry(2, ShaderStages::COMPUTE, true),
        storage_buffer_entry(3, ShaderStages::COMPUTE, false),
        storage_buffer_entry(4, ShaderStages::COMPUTE, false),
        storage_buffer_entry(5, ShaderStages::COMPUTE | ShaderStages::VERTEX, false),
        uniform_buffer_entry(6, ShaderStages::COMPUTE, NonZeroU64::new(16)),
        storage_buffer_entry(7, ShaderStages::COMPUTE, true), // chunk_meta
        storage_buffer_entry(8, ShaderStages::COMPUTE, true), // slot_vertex_base
        storage_buffer_entry(9, ShaderStages::COMPUTE, true), // slot_vertex_count
        storage_buffer_entry(10, ShaderStages::COMPUTE, true), // active_slot_map: dispatch-order pos -> real arena slot
        // material_buffer: packed as array<u32>, 4 material-id bytes per
        // word — WGSL storage buffers have no array<u8>. Pass1 doesn't
        // read it; the raster pass does (binding 4 of its group 3).
        storage_buffer_entry(11, ShaderStages::COMPUTE, true),
        storage_buffer_entry(12, ShaderStages::COMPUTE, false), // guard_debug: range-guard telemetry
        storage_buffer_entry(13, ShaderStages::COMPUTE, true),  // chunk_counts from pass1
    ]
}

/// Binding layout for group 3 of the raster pipeline — per-fragment
/// visibility and material sampling. Bound once per arena (all buffers here
/// are per-arena, not per-palette), alongside groups 0 and 1 (Bevy's mesh
/// view) and group 2 (the palette). All FRAGMENT-only: the vertex stage doesn't need any of
/// this, it only carries `real_slot` through as a packed flat attribute.
///
/// Binding 3 reuses the same `BatchUniforms` layout as pass3's own uniform
/// (`cell_count`, `texture_size`, `wg_per_chunk_z`, `voxel_size`) — only
/// `texture_size/voxel_size` are read here, but no reason to duplicate the
/// buffer when pass3's `batch_uniform_buffer_pass3` already holds the
/// right values for this arena and never changes after arena creation.
pub fn raster_chunk_visibility_entries() -> Vec<BindGroupLayoutEntry> {
    vec![
        storage_buffer_entry(0, ShaderStages::FRAGMENT, true), // chunk_meta
        storage_buffer_entry(1, ShaderStages::FRAGMENT, true), // visibility_mask_buffer
        storage_buffer_entry(2, ShaderStages::FRAGMENT, true), // chunk_has_mask_buffer
        uniform_buffer_entry(3, ShaderStages::FRAGMENT, NonZeroU64::new(16)), // BatchUniforms
        storage_buffer_entry(4, ShaderStages::FRAGMENT, true), // material_buffer (packed u8 ids in u32 words)
    ]
}

/// Binding layout for the cursor/raycast query pass (`cursor_query.wgsl`).
/// Binding 1 (hits) is `read_write` since the shader writes results in
/// place. Everything else is read-only from the shader's perspective.
pub fn query_entries() -> Vec<BindGroupLayoutEntry> {
    let mut entries = vec![
        storage_buffer_entry(0, ShaderStages::COMPUTE, true), // queries
        storage_buffer_entry(1, ShaderStages::COMPUTE, false), // hits
        uniform_buffer_entry(2, ShaderStages::COMPUTE, NonZeroU64::new(80)), // QueryUniforms
        storage_buffer_entry(3, ShaderStages::COMPUTE, true), // merged chunk_lookup
    ];
    // Bindings 4, 5 for LOD rank 0; 6, 7 for rank 1; ...
    for base in (4u32..).step_by(2).take(LOD::COUNT) {
        entries.push(storage_buffer_entry(base, ShaderStages::COMPUTE, true)); // sdf_buffer[lod]
        entries.push(storage_buffer_entry(
            base.wrapping_add(1),
            ShaderStages::COMPUTE,
            true,
        )); // chunk_meta[lod]
    }
    // This pass's arena index.
    entries.push(uniform_buffer_entry(
        12,
        ShaderStages::COMPUTE,
        NonZeroU64::new(16),
    ));
    entries
}
