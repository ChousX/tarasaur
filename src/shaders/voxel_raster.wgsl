// Import Bevy's built-in view uniform layout to ensure absolute memory layout alignment
#import bevy_render::view::View

@group(0) @binding(0) var<uniform> view: View;

// --- Bind Group 1: Triplanar Material Assets ---
@group(1) @binding(0) var texture_array: texture_2d<f32>;
@group(1) @binding(1) var texture_sampler: sampler;

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
    
    out.world_position = input.position.xyz;
    out.world_normal = input.normal.xyz;
    
    // Transformed using Bevy's official view transform
    out.clip_position = view.clip_from_world * vec4<f32>(input.position.xyz, 1.0);
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let N = normalize(in.world_normal);
    
    var weights = abs(N);
    weights = max(weights - 0.2, vec3<f32>(0.0)); 
    
    let total_weight = weights.x + weights.y + weights.z;
    if (total_weight > 0.0) {
        weights = weights / total_weight;
    } else {
        weights = vec3<f32>(0.3333);
    }

    let tex_scale = 0.1; 
    let coord_x = in.world_position.yz * tex_scale;
    let coord_y = in.world_position.xz * tex_scale;
    let coord_z = in.world_position.xy * tex_scale;

    let sample_x = textureSample(texture_array, texture_sampler, coord_x);
    let sample_y = textureSample(texture_array, texture_sampler, coord_y);
    let sample_z = textureSample(texture_array, texture_sampler, coord_z);

    let blended_color = (sample_x * weights.x) + (sample_y * weights.y) + (sample_z * weights.z);

    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.2));
    let diffuse = max(dot(N, light_dir), 0.0);
    let ambient = 0.2;
    let lighting = diffuse + ambient;
    
    return vec4<f32>(blended_color.rgb * lighting, 1.0);
}
