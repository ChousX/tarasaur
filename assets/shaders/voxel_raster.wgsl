struct ChunkUniforms {
    chunk_size: u32,
    total_cells: u32,
    iso_level: f32,
    _pad0: u32,
};

@group(0) @binding(0) var sdf_volume: texture_storage_3d<r32float, read>;

struct VertexInput {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    // Pass spatial configurations downstream to avoid any structural bottlenecks
    out.world_position = input.position.xyz;
    out.world_normal = normalize(input.normal.xyz);
    
    // Clip space conversions (Assuming identity layout projections for base milestone isolation testing)
    out.clip_position = vec4<f32>(input.position.xyz * 0.1 - 1.0, 1.0);
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Basic verification visualization: output surface normals directly to screen
    let clean_normal = normalize(in.world_normal) * 0.5 + 0.5;
    return vec4<f32>(clean_normal, 1.0);
}
