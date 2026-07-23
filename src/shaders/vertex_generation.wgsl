struct CompactionUniforms {
    chunk_size: u32,
    total_cells: u32,
    iso_level: f32,
    _pad0: u32,
};

struct Vertex {
    position: vec4<f32>,
    normal: vec4<f32>,
};

struct IndirectArgs {
    index_count: atomic<u32>,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
};

@group(0) @binding(0) var sdf_volume: texture_storage_3d<r32float, read>;
@group(0) @binding(1) var<storage, read_write> flags: array<u32>;
@group(0) @binding(2) var<storage, read_write> compacted_offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> scattered_vertices: array<Vertex>;
@group(0) @binding(4) var<storage, read_write> final_vertices: array<Vertex>;
@group(0) @binding(5) var<storage, read_write> indirect_args: IndirectArgs;

@group(1) @binding(0) var<uniform> config: CompactionUniforms;

// Converts a flat 1D buffer index back into local 3D chunk voxel coordinates
fn index_to_coord(index: u32, size: u32) -> vec3<u32> {
    let z = index / (size * size);
    let rem = index % (size * size);
    let y = rem / size;
    let x = rem % size;
    return vec3<u32>(x, y, z);
}

@compute @workgroup_size(64, 1, 1)
fn generate_vertices(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_index = global_id.x;
    if (cell_index >= config.total_cells) {
        return;
    }

    // Identify if this cell has a surface crossing verified by Pass 1 & 2
    let is_active = flags[cell_index];
    if (is_active == 0u) {
        return;
    }

    let coord = index_to_coord(cell_index, config.chunk_size);
    
    // Bounds checking boundary limits for cell parsing
    if (coord.x >= config.chunk_size - 1u || 
        coord.y >= config.chunk_size - 1u || 
        coord.z >= config.chunk_size - 1u) {
        return;
    }

    // Read values at the 8 cell corners
    let s000 = textureLoad(sdf_volume, vec3<i32>(coord) + vec3<i32>(0, 0, 0)).x;
    let s100 = textureLoad(sdf_volume, vec3<i32>(coord) + vec3<i32>(1, 0, 0)).x;
    let s010 = textureLoad(sdf_volume, vec3<i32>(coord) + vec3<i32>(0, 1, 0)).x;
    let s110 = textureLoad(sdf_volume, vec3<i32>(coord) + vec3<i32>(1, 1, 0)).x;
    let s001 = textureLoad(sdf_volume, vec3<i32>(coord) + vec3<i32>(0, 0, 1)).x;
    let s101 = textureLoad(sdf_volume, vec3<i32>(coord) + vec3<i32>(1, 0, 1)).x;
    let s011 = textureLoad(sdf_volume, vec3<i32>(coord) + vec3<i32>(0, 1, 1)).x;
    let s111 = textureLoad(sdf_volume, vec3<i32>(coord) + vec3<i32>(1, 1, 1)).x;

    // Linear Interpolation calculation for accurate surface tracking (Surface Nets)
    var surface_pos = vec3<f32>(0.0);
    var intersections = 0.0;

    // Evaluate the X edge crossings
    if ((s000 < config.iso_level) != (s100 < config.iso_level)) {
        let t = (config.iso_level - s000) / (s100 - s000);
        surface_pos += vec3<f32>(coord) + vec3<f32>(t, 0.0, 0.0);
        intersections += 1.0;
    }
    // Evaluate the Y edge crossings
    if ((s000 < config.iso_level) != (s010 < config.iso_level)) {
        let t = (config.iso_level - s000) / (s010 - s000);
        surface_pos += vec3<f32>(coord) + vec3<f32>(0.0, t, 0.0);
        intersections += 1.0;
    }
    // Evaluate the Z edge crossings
    if ((s000 < config.iso_level) != (s001 < config.iso_level)) {
        let t = (config.iso_level - s000) / (s001 - s000);
        surface_pos += vec3<f32>(coord) + vec3<f32>(0.0, 0.0, t);
        intersections += 1.0;
    }

    if (intersections > 0.0) {
        surface_pos = surface_pos / intersections;
    } else {
        surface_pos = vec3<f32>(coord) + vec3<f32>(0.5);
    }

    // Compute central differences gradient mapping for the vertex normal field
    let eps = 1u;
    let nx = textureLoad(sdf_volume, vec3<i32>(coord) + vec3<i32>(1, 0, 0)).x - textureLoad(sdf_volume, vec3<i32>(coord) - vec3<i32>(1, 0, 0)).x;
    let ny = textureLoad(sdf_volume, vec3<i32>(coord) + vec3<i32>(0, 1, 0)).x - textureLoad(sdf_volume, vec3<i32>(coord) - vec3<i32>(0, 1, 0)).x;
    let nz = textureLoad(sdf_volume, vec3<i32>(coord) + vec3<i32>(0, 0, 1)).x - textureLoad(sdf_volume, vec3<i32>(coord) - vec3<i32>(0, 0, 1)).x;
    let normal = normalize(vec4<f32>(nx, ny, nz, 0.0));

    var generated_vertex: Vertex;
    generated_vertex.position = vec4<f32>(surface_pos, 1.0);
    generated_vertex.normal = normal;

    // Cache to uncompacted array for structural topology reference 
    scattered_vertices[cell_index] = generated_vertex;

    // STREAM COMPACTION: Fetch your exclusive packed array location calculated by Pass 2
    let target_packed_index = compacted_offsets[cell_index];

    // Push the vertex down into the zero-bubble tightly packed draw block
    final_vertices[target_packed_index] = generated_vertex;
}
