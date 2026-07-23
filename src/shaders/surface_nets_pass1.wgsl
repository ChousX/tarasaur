@group(0) @binding(0) var sdf_texture: texture_storage_3d<r32float, read>;

struct FlagsBuffer {
    data: array<u32>,
}

@group(0) @binding(1) var<storage, read_write> flags_buffer: FlagsBuffer;

fn flatten_cell_idx(coord: vec3<u32>, size: u32) -> u32 {
    return coord.z * size * size + coord.y * size + coord.x;
}

@compute @workgroup_size(4, 4, 4)
fn cs_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let size = num_workgroups.x * 4u; // Grid extent (e.g. 32)

    // Clamp execution inside grid bounds (Surface Nets inspects global_id + 1)
    if (global_id.x >= size - 1u || global_id.y >= size - 1u || global_id.z >= size - 1u) {
        let flat_idx = flatten_cell_idx(global_id, size);
        flags_buffer.data[flat_idx] = 0u;
        return;
    }

    // 8 Corners of the local dual cell
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
        if (val < 0.0) {
            inside_count = inside_count + 1u;
        }
    }

    let flat_idx = flatten_cell_idx(global_id, size);

    // Flag cell as active (1) if it straddles the surface boundary
    if (inside_count > 0u && inside_count < 8u) {
        flags_buffer.data[flat_idx] = 1u;
    } else {
        flags_buffer.data[flat_idx] = 0u;
    }
}
