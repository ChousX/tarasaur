struct ViewUniform {
    view_proj: mat4x4<f32>,
    world_position: vec3<f32>,
};
@group(0) @binding(0) var<uniform> view: ViewUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.world_position = input.position;
    out.world_normal = input.normal;
    out.clip_position = view.view_proj * vec4<f32>(input.position, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Basic visualization normal shading (Milestone 6 will implement full triplanar mapping)
    let N = normalize(in.world_normal);
    let lighting = max(dot(N, normalize(vec3<f32>(0.5, 1.0, 0.2))), 0.2);
    return vec4<f32>(vec3<f32>(lighting), 1.0);
}
