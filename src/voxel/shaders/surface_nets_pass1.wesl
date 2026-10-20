// Finds the active cells (some but not all corners inside) and gives each a
// vertex slot, replacing a prefix scan: each workgroup counts its active
// cells in shared memory and claims a contiguous range of the chunk's slots
// with one atomic. Vertex order within a chunk is therefore arbitrary; pass3
// only needs the cell <-> vertex mapping, which this writes both ways.

import tarasaur::voxel::shaders::voxel_common::{BatchUniforms, decode_sdf, flatten_sdf_idx};

@group(0) @binding(0) var<storage, read> sdf_buffer: array<u32>;
// Per-dispatch scratch, indexed by the chunk's position in the dispatch:
@group(0) @binding(1) var<storage, read_write> active_cells: array<u32>; // [vertex] -> cell
@group(0) @binding(2) var<storage, read_write> cell_vertex: array<u32>;  // [cell] -> vertex (active cells only)
@group(0) @binding(3) var<storage, read_write> chunk_counts: array<atomic<u32>>; // [chunk] -> active cells; zeroed before each dispatch

struct IndirectDrawArgs {
    index_count: atomic<u32>,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}
@group(0) @binding(5) var<storage, read_write> indirect_args: array<IndirectDrawArgs>;

@group(0) @binding(6) var<uniform> uniforms: BatchUniforms;

@group(0) @binding(8) var<storage, read> active_slot_map: array<u32>;

fn flatten_cell_idx(coord: vec3<u32>, cell_count: u32) -> u32 {
    return coord.z * cell_count * cell_count + coord.y * cell_count + coord.x;
}

fn sdf_offset_for(slot: u32) -> u32 {
    return slot * uniforms.texture_size * uniforms.texture_size * uniforms.texture_size;
}

fn cell_offset_for(chunk_idx: u32) -> u32 {
    return chunk_idx * uniforms.cell_count * uniforms.cell_count * uniforms.cell_count;
}

var<workgroup> wg_active: atomic<u32>;
var<workgroup> wg_base: u32;

@compute @workgroup_size(4, 4, 4)
fn cs_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    let chunk_idx = wg_id.z / uniforms.wg_per_chunk_z;
    let local_wg_z = wg_id.z % uniforms.wg_per_chunk_z;
    let cell_coord = vec3<u32>(global_id.x, global_id.y, local_wg_z * 4u + local_id.z);
    let real_slot = active_slot_map[chunk_idx];
    let cell_count = uniforms.cell_count;

    // Workgroup memory isn't zero-initialized (see the pipeline descriptor).
    if (local_index == 0u) {
        atomicStore(&wg_active, 0u);
    }
    if (all(cell_coord == vec3<u32>(0u))) {
        atomicStore(&indirect_args[real_slot].index_count, 0u); // indirect_args is per slot
    }

    // No early return: the barriers below need every invocation.
    var is_active = false;
    if (all(cell_coord < vec3<u32>(cell_count))) {
        let sdf_base = sdf_offset_for(real_slot);
        let offsets = array<vec3<u32>, 8>(
            vec3<u32>(0u, 0u, 0u), vec3<u32>(1u, 0u, 0u),
            vec3<u32>(0u, 1u, 0u), vec3<u32>(1u, 1u, 0u),
            vec3<u32>(0u, 0u, 1u), vec3<u32>(1u, 0u, 1u),
            vec3<u32>(0u, 1u, 1u), vec3<u32>(1u, 1u, 1u)
        );
        var inside_count = 0u;
        for (var i = 0u; i < 8u; i = i + 1u) {
            let idx = sdf_base + flatten_sdf_idx(cell_coord + offsets[i], uniforms.texture_size);
            if (decode_sdf(sdf_buffer[idx >> 1u], idx) <= 0.0) {
                inside_count = inside_count + 1u;
            }
        }
        // Depends ONLY on the sign crossing (the shape arrives already
        // clipped by its layer's clip field), so it can never desync from
        // pass3.
        is_active = inside_count > 0u && inside_count < 8u;
    }

    workgroupBarrier();
    var slot_in_group = 0u;
    if (is_active) {
        slot_in_group = atomicAdd(&wg_active, 1u);
    }
    workgroupBarrier();
    if (local_index == 0u) {
        let n = atomicLoad(&wg_active);
        wg_base = select(0u, atomicAdd(&chunk_counts[chunk_idx], n), n > 0u);
    }
    workgroupBarrier();

    if (is_active) {
        let cell_base = cell_offset_for(chunk_idx);
        let cell = flatten_cell_idx(cell_coord, cell_count);
        let vertex = wg_base + slot_in_group;
        cell_vertex[cell_base + cell] = vertex;
        active_cells[cell_base + vertex] = cell;
    }
}
