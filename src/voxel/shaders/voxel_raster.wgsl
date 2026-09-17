// Import Bevy's built-in view uniform layout to ensure absolute memory layout alignment
#import bevy_render::view::View

@group(0) @binding(0) var<uniform> view: View;

@group(1) @binding(0) var texture_array: texture_2d_array<f32>;
@group(1) @binding(1) var texture_sampler: sampler;

struct VertexInput {
    @location(0) position: vec4<f32>, // .xyz = world position, .w = bitcast-packed material_id_a (low byte) + blend weight (next byte)
    @location(1) normal: vec4<f32>,   // .xyz = normal, .w = bitcast-packed material_id_b (low byte)
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    // Raw bitcast bit patterns must never hit the rasterizer's perspective
    // interpolator as floats — garbage results. Unpacked to real u32s here
    // in vs_main and marked flat so every fragment in a triangle sees the
    // same (correct) provoking-vertex id, no interpolation across ids.
    @location(2) @interpolate(flat) material_id_a: u32,
    @location(3) @interpolate(flat) material_id_b: u32,
    // Unlike the ids, the weight is a genuine scalar meant to blend
    // continuously across the triangle — smooth (default) interpolation.
    @location(4) blend_weight: f32,
};

struct MaterialProperties {
    texture_scale: f32,
    blend_sharpness: f32,
    roughness_override: f32,
    metallic_override: f32,
}

// Bound and populated correctly; fs_main doesn't read from it yet
// (tex_scale below is still hardcoded) — wiring per-material scale/
// sharpness into the triplanar sampling is a separate fs_main change.
@group(1) @binding(2) var<storage, read> material_properties: array<MaterialProperties>;

// Standard power-based triplanar sharpening: higher blend_sharpness
// concentrates weight onto the axis most aligned with N, producing a
// crisper seam between projection planes. Replaces the old fixed
// subtract-and-renormalize approach, which had no per-material hook.
fn triplanar_weights(N: vec3<f32>, sharpness: f32) -> vec3<f32> {
    var w = pow(abs(N), vec3<f32>(max(sharpness, 0.0001)));
    let total = w.x + w.y + w.z;
    if (total > 0.0) {
        w = w / total;
    } else {
        w = vec3<f32>(0.3333);
    }
    return w;
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    out.world_position = input.position.xyz;
    out.world_normal = input.normal.xyz;

    let packed_a = bitcast<u32>(input.position.w);
    let packed_b = bitcast<u32>(input.normal.w);

    out.material_id_a = packed_a & 0xFFu;
    let weight_u8 = (packed_a >> 8u) & 0xFFu;
    out.blend_weight = f32(weight_u8) / 255.0; // 0.0 = fully id_b, 1.0 = fully id_a

    out.material_id_b = packed_b & 0xFFu;

    // Transformed using Bevy's official view transform
    out.clip_position = view.clip_from_world * vec4<f32>(input.position.xyz, 1.0);

    return out;
}


@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let N = normalize(in.world_normal);

    let mat_a = material_properties[in.material_id_a];
    let mat_b = material_properties[in.material_id_b];

    // texture_scale is "world units per repeat" (see PaletteMaterial's doc
    // comment) — dividing world position by it, not multiplying, is what
    // makes larger values stretch the texture rather than shrink it.
    let inv_scale_a = 1.0 / max(mat_a.texture_scale, 0.0001);
    let inv_scale_b = 1.0 / max(mat_b.texture_scale, 0.0001);

    let weights_a = triplanar_weights(N, mat_a.blend_sharpness);
    let layer_a = i32(in.material_id_a);
    let a_x = textureSample(texture_array, texture_sampler, in.world_position.yz * inv_scale_a, layer_a);
    let a_y = textureSample(texture_array, texture_sampler, in.world_position.xz * inv_scale_a, layer_a);
    let a_z = textureSample(texture_array, texture_sampler, in.world_position.xy * inv_scale_a, layer_a);
    let color_a = (a_x * weights_a.x) + (a_y * weights_a.y) + (a_z * weights_a.z);

    let weights_b = triplanar_weights(N, mat_b.blend_sharpness);
    let layer_b = i32(in.material_id_b);
    let b_x = textureSample(texture_array, texture_sampler, in.world_position.yz * inv_scale_b, layer_b);
    let b_y = textureSample(texture_array, texture_sampler, in.world_position.xz * inv_scale_b, layer_b);
    let b_z = textureSample(texture_array, texture_sampler, in.world_position.xy * inv_scale_b, layer_b);
    let color_b = (b_x * weights_b.x) + (b_y * weights_b.y) + (b_z * weights_b.z);

    // blend_weight is id_a's share — mix(x, y, t) = x*(1-t) + y*t.
    let blended_color = mix(color_b, color_a, in.blend_weight);

    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.2));
    let diffuse = max(dot(N, light_dir), 0.0);
    let ambient = 0.2;
    let lighting = diffuse + ambient;

    return vec4<f32>(blended_color.rgb * lighting, 1.0);
}
