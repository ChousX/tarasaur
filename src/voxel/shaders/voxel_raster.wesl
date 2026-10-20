// The palette raster: triplanar palette texturing (albedo, plus ARM under
// VOXEL_ARM and normal maps under VOXEL_NORMAL) with material blending,
// per-pixel discard against the chunk's visibility mask, and Bevy's PBR
// lighting. Chunks without a mask are drawn without VOXEL_MASK, so without
// the mask test. The vertex stage and the group 3 lookups come from the
// voxel_raster_vertex library; group 2 is the palette.

import tarasaur::voxel::shaders::voxel_raster_vertex::{
    VertexInput, VertexOutput, voxel_vertex, FragBlend, fragment_material_blend,
    fragment_is_visible,
};
import bevy_pbr::render::{
    mesh_view_bindings::view,
    mesh_types::MESH_FLAGS_SHADOW_RECEIVER_BIT,
    pbr_types::{pbr_input_new, STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BIT},
    pbr_functions::{apply_pbr_lighting, calculate_view, main_pass_post_lighting_processing},
};

@group(2) @binding(0) var texture_array: texture_2d_array<f32>;
@group(2) @binding(1) var texture_sampler: sampler;
@if(VOXEL_ARM) @group(2) @binding(3) var arm_array: texture_2d_array<f32>;
@if(VOXEL_NORMAL) @group(2) @binding(4) var normal_array: texture_2d_array<f32>;

struct MaterialProperties {
    texture_scale: f32,
    blend_sharpness: f32,
    roughness_override: f32,
    metallic_override: f32,
    normal_fade_start: f32,
    normal_fade_end: f32,
    macro_variation: f32,
    _pad: f32,
}

@group(2) @binding(2) var<storage, read> material_properties: array<MaterialProperties>;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    return voxel_vertex(input);
}

// Width of the transition band, measured in |normal| units. Smaller = harder,
// narrower seams. 0.15 is a moderate band; try 0.05-0.25.
const TRIPLANAR_BLEND_WIDTH: f32 = 0.15;

fn triplanar_weights(N: vec3<f32>, sharpness: f32) -> vec3<f32> {
    let a = abs(N);
    let m = max(a.x, max(a.y, a.z));

    // Per-material blend_sharpness now also narrows the band (1.0 = default width).
    let width = max(TRIPLANAR_BLEND_WIDTH / max(sharpness, 1.0), 0.02);

    // The dominant axis gets 1.0. Other axes get 0 until they come within
    // `width` of the dominant one, then ramp up linearly.
    var w = clamp((a - vec3<f32>(m - width)) / width, vec3<f32>(0.0), vec3<f32>(1.0));

    // Optional: smooth the ramp so the seam has no visible kink.
    w = w * w * (3.0 - 2.0 * w);

    let total = w.x + w.y + w.z; // always >= 1 because the dominant axis is 1
    return w / total;
}

// Weights below this are dropped (and the rest renormalized).
// triplanar_weights already returns exact 0 for non-dominant axes outside the
// blend band, so most pixels end up with a single projection.
const MIN_PROJ_WEIGHT: f32 = 0.01;

// Without an ARM texture: unoccluded, fully rough, not metallic.
const DEFAULT_ARM: vec3<f32> = vec3<f32>(1.0, 1.0, 0.0);

// One material's surface at this pixel. arm = (AO, perceptual roughness,
// metallic), with the material's overrides already applied. normal is a
// world-space shading normal, unnormalized (only filled under VOXEL_NORMAL).
// From sample_projection, normal is instead the tangent-space sample.
struct Surface {
    albedo: vec4<f32>,
    arm: vec3<f32>,
    normal: vec3<f32>,
}

// One projection's sample: albedo, plus ARM under VOXEL_ARM and, if
// `normal`, the tangent-space normal under VOXEL_NORMAL. Those fetches reuse
// the albedo's coordinates and gradients.
fn sample_projection(
    layer: i32,
    uv: vec2<f32>,
    ddx_uv: vec2<f32>,
    ddy_uv: vec2<f32>,
    normal: bool,
) -> Surface {
    var s = Surface(
        textureSampleGrad(texture_array, texture_sampler, uv, layer, ddx_uv, ddy_uv),
        DEFAULT_ARM,
        vec3<f32>(0.0, 0.0, 1.0),
    );
    @if(VOXEL_ARM)
    s.arm = textureSampleGrad(arm_array, texture_sampler, uv, layer, ddx_uv, ddy_uv).rgb;
    @if(VOXEL_NORMAL)
    s.normal = sample_normal(layer, uv, ddx_uv, ddy_uv, normal);
    return s;
}

// The tangent-space normal, or flat without fetching when `fetch` is false.
// Explicit gradients make the fetch safe in non-uniform control flow.
@if(VOXEL_NORMAL)
fn sample_normal(
    layer: i32,
    uv: vec2<f32>,
    ddx_uv: vec2<f32>,
    ddy_uv: vec2<f32>,
    fetch: bool,
) -> vec3<f32> {
    if (!fetch) {
        return vec3<f32>(0.0, 0.0, 1.0);
    }
    return decode_normal(textureSampleGrad(normal_array, texture_sampler, uv, layer, ddx_uv, ddy_uv).rg);
}

// Tangent-space normal from its X and Y; Z is rebuilt, so two-channel
// formats (BC5, RG8) work. Maps are OpenGL convention like Bevy's
// StandardMaterial (green = up the image), and +V runs down the image, so
// green is flipped.
fn decode_normal(rg: vec2<f32>) -> vec3<f32> {
    let xy = (rg * 2.0 - 1.0) * vec2<f32>(1.0, -1.0);
    return vec3<f32>(xy, sqrt(max(1.0 - dot(xy, xy), 0.0)));
}

// Whiteout blend of a projection's tangent-space normal `t` onto the mesh
// normal, where the projection's U and V run along world axes with mesh
// normal components `n_uv`, and `n_w` is the component along its own axis.
// Returns (U, V, W) world components; callers swizzle them back to xyz.
// UVs aren't mirrored on back faces, so the tangent frame needs no sign flips.
fn whiteout(t: vec3<f32>, n_uv: vec2<f32>, n_w: f32) -> vec3<f32> {
    return vec3<f32>(t.xy + n_uv, abs(t.z) * n_w);
}

fn sample_material(
    id: u32,
    p: vec3<f32>,
    ddx_p: vec3<f32>,
    ddy_p: vec3<f32>,
    N: vec3<f32>,
    view_dist: f32,
    macro_n: f32,
) -> Surface {
    let mat = material_properties[id];
    // 1 near the camera, 0 past normal_fade_end, where the normal map isn't
    // sampled at all. smoothstep by hand: WGSL's is undefined when
    // start == end, which here should just cut off at that distance.
    let fade_t = saturate(
        (view_dist - mat.normal_fade_start) / max(mat.normal_fade_end - mat.normal_fade_start, 1e-4),
    );
    let normal_fade = 1.0 - fade_t * fade_t * (3.0 - 2.0 * fade_t);
    let normal = normal_fade > 0.0;
    let layer = i32(id);
    let inv_scale = 1.0 / max(mat.texture_scale, 0.0001);

    var w = triplanar_weights(N, mat.blend_sharpness);
    w = select(vec3<f32>(0.0), w, w > vec3<f32>(MIN_PROJ_WEIGHT));
    w = w / (w.x + w.y + w.z); // dominant axis is always kept, so the sum is > 0

    let q = p * inv_scale;
    let dx = ddx_p * inv_scale;
    let dy = ddy_p * inv_scale;
    var s = Surface(vec4<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
    if (w.x > 0.0) {
        let x = sample_projection(layer, q.yz, dx.yz, dy.yz, normal);
        s.albedo += w.x * x.albedo;
        s.arm += w.x * x.arm;
        @if(VOXEL_NORMAL)
        s.normal += w.x * whiteout(x.normal, N.yz, N.x).zxy;
    }
    if (w.y > 0.0) {
        let y = sample_projection(layer, q.xz, dx.xz, dy.xz, normal);
        s.albedo += w.y * y.albedo;
        s.arm += w.y * y.arm;
        @if(VOXEL_NORMAL)
        s.normal += w.y * whiteout(y.normal, N.xz, N.y).xzy;
    }
    if (w.z > 0.0) {
        let z = sample_projection(layer, q.xy, dx.xy, dy.xy, normal);
        s.albedo += w.z * z.albedo;
        s.arm += w.z * z.arm;
        @if(VOXEL_NORMAL)
        s.normal += w.z * whiteout(z.normal, N.xy, N.z);
    }
    @if(VOXEL_NORMAL)
    s.normal = mix(N, s.normal, normal_fade);

    s.albedo = vec4<f32>(s.albedo.rgb * (1.0 + mat.macro_variation * macro_n), s.albedo.a);

    // A negative override means "use the texture".
    s.arm.y = select(s.arm.y, mat.roughness_override, mat.roughness_override >= 0.0);
    s.arm.z = select(s.arm.z, mat.metallic_override, mat.metallic_override >= 0.0);
    return s;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Derivatives must be taken in uniform control flow, so do it before the
    // discard and before any branching on per-pixel data.
    let ddx_p = dpdx(in.world_position);
    let ddy_p = dpdy(in.world_position);

    // Only chunks with a visibility mask are drawn with VOXEL_MASK: a
    // shader that can discard loses early depth testing on tiling GPUs.
    @if(VOXEL_MASK)
    if (!fragment_is_visible(in.real_slot, in.world_position)) {
        discard;
    }

    let N = normalize(in.world_normal);
    let view_dist = distance(in.world_position, view.world_position);
    let macro_n = in.macro_n;
    // The same result fragment_material_blend gives when all 8 corners agree,
    // without its 8 lookups.
    var blend: FragBlend;
    if (in.uniform_mat.x > 0.9999) {
        let id = u32(round(in.uniform_mat.y / in.uniform_mat.x));
        blend = FragBlend(id, id, 1.0);
    } else {
        blend = fragment_material_blend(in.real_slot, in.world_position);
    }

    // Decide which materials are actually needed.
    let t = smoothstep(0.35, 0.65, blend.w);
    let same = blend.id_a == blend.id_b;
    let need_a = same || t > 0.001;
    let need_b = !same && t < 0.999;

    var surface: Surface;
    if (need_a && need_b) {
        let a = sample_material(blend.id_a, in.world_position, ddx_p, ddy_p, N, view_dist, macro_n);
        let b = sample_material(blend.id_b, in.world_position, ddx_p, ddy_p, N, view_dist, macro_n);
        surface = Surface(
            mix(b.albedo, a.albedo, t),
            mix(b.arm, a.arm, t),
            mix(b.normal, a.normal, t),
        );
    } else if (need_a) {
        surface = sample_material(blend.id_a, in.world_position, ddx_p, ddy_p, N, view_dist, macro_n);
    } else {
        surface = sample_material(blend.id_b, in.world_position, ddx_p, ddy_p, N, view_dist, macro_n);
    }

    var pbr = pbr_input_new();
    pbr.material.base_color = vec4<f32>(surface.albedo.rgb, 1.0);
    pbr.material.perceptual_roughness = surface.arm.y;
    pbr.material.metallic = surface.arm.z;
    pbr.material.flags |= STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BIT;
    pbr.diffuse_occlusion = vec3<f32>(surface.arm.x);
    pbr.frag_coord = in.clip_position;
    pbr.world_position = vec4<f32>(in.world_position, 1.0);
    // world_normal stays the mesh normal (Bevy uses it for shadow biasing);
    // N is the shading normal.
    pbr.world_normal = N;
    pbr.N = N;
    @if(VOXEL_NORMAL)
    pbr.N = normalize(surface.normal);
    pbr.is_orthographic = view.clip_from_view[3].w == 1.0;
    pbr.V = calculate_view(pbr.world_position, pbr.is_orthographic);
    pbr.flags = MESH_FLAGS_SHADOW_RECEIVER_BIT;

    let lit = apply_pbr_lighting(pbr);
    return main_pass_post_lighting_processing(pbr, lit);
}

// Depth prepass: depth only (no color writes), but hidden voxels must not
// write depth either, or they'd occlude what's behind them.
@fragment
fn fs_prepass(in: VertexOutput) -> @location(0) vec4<f32> {
    @if(VOXEL_MASK)
    if (!fragment_is_visible(in.real_slot, in.world_position)) {
        discard;
    }
    return vec4<f32>(0.0);
}
