// --- Bind Group 0: Standard Bevy View Uniform Layout Mapping ---
struct ViewUniform {
    view_proj: mat4x4<f32>,
    inverse_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    inverse_view: mat4x4<f32>,
    projection: mat4x4<f32>,
    inverse_projection: mat4x4<f32>,
    world_position: vec3<f32>,
    // Explicit padding to respect WebGPU 16-byte uniform structures alignment rules
    _padding: f32, 
};

@group(0) @binding(0) var<uniform> view: ViewUniform;

// --- Bind Group 1: Triplanar Material Assets ---
@group(1) @binding(0) var texture_array: texture_2d<f32>;
@group(1) @binding(1) var texture_sampler: sampler;

// Fixed: Swapped to vec4<f32> to match Float32x4 layout specifications 
// and 32-byte host array strides exactly.
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
    
    // Extract xyz components from packed buffer array layout
    out.world_position = input.position.xyz;
    out.world_normal = input.normal.xyz;
    
    // Transform position into projection clip space
    out.clip_position = view.view_proj * vec4<f32>(input.position.xyz, 1.0);
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let N = normalize(in.world_normal);
    
    // Calculate Triplanar Blending Weights from absolute normal vector
    var weights = abs(N);
    
    // Apply sharpening contrast factor to eliminate muddy blending zones
    weights = max(weights - 0.2, vec3<f32>(0.0)); 
    
    // Secure safe normalization boundary check to eliminate any potential divide-by-zero anomalies
    let total_weight = weights.x + weights.y + weights.z;
    if (total_weight > 0.0) {
        weights = weights / total_weight;
    } else {
        // Fallback to equal blending if weights collapse completely to zero
        weights = vec3<f32>(0.3333);
    }

    // Project coordinates on the three major cartesian planes
    let tex_scale = 0.1; 
    let coord_x = in.world_position.yz * tex_scale;
    let coord_y = in.world_position.xz * tex_scale;
    let coord_z = in.world_position.xy * tex_scale;

    // Sample texture across all directional channels
    let sample_x = textureSample(texture_array, texture_sampler, coord_x);
    let sample_y = textureSample(texture_array, texture_sampler, coord_y);
    let sample_z = textureSample(texture_array, texture_sampler, coord_z);

    // Accumulate weighted triplanar color profile
    let blended_color = (sample_x * weights.x) + (sample_y * weights.y) + (sample_z * weights.z);

    // Calculate a directional diffuse + ambient lighting contribution
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.2));
    let diffuse = max(dot(N, light_dir), 0.0);
    let ambient = 0.2;
    let lighting = diffuse + ambient;
    
    return vec4<f32>(blended_color.rgb * lighting, 1.0);
}
