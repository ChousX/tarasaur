// A second layer's raster: one flat, translucent color per sub-surface,
// lit by a fixed light. The vertex stage and the per-slot sub-surface
// lookup come from the voxel_raster_vertex library; group 2 is this
// layer's colors.

import tarasaur::voxel::shaders::voxel_raster_vertex::{
    VertexInput, VertexOutput, voxel_vertex, sub_surface,
};

struct FlatColors {
    color: array<vec4<f32>, 2>, // premultiplied; indexed by sub-surface
}

@group(2) @binding(0) var<uniform> flat: FlatColors;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    return voxel_vertex(input);
}

const LIGHT_DIR: vec3<f32> = vec3<f32>(0.4, 0.8, 0.45);

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let lit = 0.35 + 0.65 * abs(dot(n, normalize(LIGHT_DIR)));
    let c = flat.color[min(sub_surface(in.real_slot), 1u)];
    return vec4<f32>(c.rgb * lit, c.a);
}

// The depth prepass: depth only, no color.
@fragment
fn fs_prepass(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(0.0);
}
