// A second layer's raster: one flat color, lit by a fixed light, drawn
// over the terrain. The vertex stage comes from the voxel_raster_vertex
// library; group 2 is this layer's color.

import tarasaur::voxel::shaders::voxel_raster_vertex::{VertexInput, VertexOutput, voxel_vertex};

struct FlatColor {
    color: vec4<f32>,
}

@group(2) @binding(0) var<uniform> flat: FlatColor;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    return voxel_vertex(input);
}

const LIGHT_DIR: vec3<f32> = vec3<f32>(0.4, 0.8, 0.45);

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let lit = 0.35 + 0.65 * max(dot(n, normalize(LIGHT_DIR)), 0.0);
    return vec4<f32>(flat.color.rgb * lit, flat.color.a);
}

// The depth prepass: depth only, no color.
@fragment
fn fs_prepass(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(0.0);
}
