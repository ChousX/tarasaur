@group(0) @binding(0) var sdf_texture: texture_storage_3d<r32float, read>;
@group(0) @binding(1) var<storage, read_write> flags_buffer: array<u32>;
struct Pass1Uniforms {
    cell_count: u32,
    texture_size: u32,
    _pad0: u32,
    _pad1: u32,
};
@group(0) @binding(6) var<uniform> uniforms: Pass1Uniforms;

fn flatten_cell_idx(coord: vec3<u32>, cell_count: u32) -> u32 {
    return coord.z * cell_count * cell_count + coord.y * cell_count + coord.x;
}

@compute @workgroup_size(4, 4, 4)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_count = uniforms.cell_count;

    if (global_id.x >= cell_count || global_id.y >= cell_count || global_id.z >= cell_count) {
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
        let pos = global_id + offsets[i];
        let val = textureLoad(sdf_texture, vec3<i32>(pos)).x;
        if (val <= 0.0) {
            inside_count = inside_count + 1u;
        }
    }

    let flat_idx = flatten_cell_idx(global_id, cell_count);
    flags_buffer[flat_idx] = select(0u, 1u, inside_count > 0u && inside_count < 8u);
}
