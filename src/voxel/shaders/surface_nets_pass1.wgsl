@group(0) @binding(0) var<storage, read> sdf_buffer: array<f32>;
@group(0) @binding(1) var<storage, read_write> flags_buffer: array<u32>;
struct IndirectDrawArgs {
    index_count: atomic<u32>,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}
@group(0) @binding(5) var<storage, read_write> indirect_args: array<IndirectDrawArgs>;

struct BatchUniforms {
    cell_count: u32,
    texture_size: u32,
    wg_per_chunk_z: u32,
    _pad0: u32,
};
@group(0) @binding(6) var<uniform> uniforms: BatchUniforms;

struct ChunkMeta {
    chunk_world_origin: vec3<f32>,
    voxel_size: f32,
    sdf_offset: u32,
    cell_offset: u32,
    active_list_pos: u32,
    _pad: u32,
}
@group(0) @binding(7) var<storage, read> chunk_meta: array<ChunkMeta>;
@group(0) @binding(8) var<storage, read> active_slot_map: array<u32>;

fn flatten_cell_idx(coord: vec3<u32>, cell_count: u32) -> u32 {
    return coord.z * cell_count * cell_count + coord.y * cell_count + coord.x;
}

fn flatten_sdf_idx(coord: vec3<u32>, texture_size: u32) -> u32 {
    return coord.z * texture_size * texture_size + coord.y * texture_size + coord.x;
}

@compute @workgroup_size(4, 4, 4)
fn cs_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let chunk_idx = wg_id.z / uniforms.wg_per_chunk_z;
    let local_wg_z = wg_id.z % uniforms.wg_per_chunk_z;
    let cell_coord = vec3<u32>(global_id.x, global_id.y, local_wg_z * 4u + local_id.z);

    let real_slot = active_slot_map[chunk_idx];
    let cmeta = chunk_meta[real_slot];
    let cell_count = uniforms.cell_count;

    if (all(cell_coord == vec3<u32>(0u))) {
        atomicStore(&indirect_args[chunk_idx].index_count, 0u);
    }

    if (cell_coord.x >= cell_count || cell_coord.y >= cell_count || cell_coord.z >= cell_count) {
        return;
    }

    let offsets = array<vec3<u32>, 8>(
        vec3<u32>(0u, 0u, 0u), vec3<u32>(1u, 0u, 0u),
        vec3<u32>(0u, 1u, 0u), vec3<u32>(1u, 1u, 0u),
        vec3<u32>(0u, 0u, 1u), vec3<u32>(1u, 0u, 1u),
        vec3<u32>(0u, 1u, 1u), vec3<u32>(1u, 1u, 1u)
    );

    var inside_count = 0u;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let pos = cell_coord + offsets[i];
        let val = sdf_buffer[cmeta.sdf_offset + flatten_sdf_idx(pos, uniforms.texture_size)];
        if (val <= 0.0) {
            inside_count = inside_count + 1u;
        }
    }

    let flat_idx = cmeta.cell_offset + flatten_cell_idx(cell_coord, cell_count);
    flags_buffer[flat_idx] = select(0u, 1u, inside_count > 0u && inside_count < 8u);
}
