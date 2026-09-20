// Import Bevy's built-in view uniform layout to ensure absolute memory layout alignment
#import bevy_render::view::View

@group(0) @binding(0) var<uniform> view: View;

@group(1) @binding(0) var texture_array: texture_2d_array<f32>;
@group(1) @binding(1) var texture_sampler: sampler;

struct MaterialProperties {
    texture_scale: f32,
    blend_sharpness: f32,
    roughness_override: f32,
    metallic_override: f32,
}

@group(1) @binding(2) var<storage, read> material_properties: array<MaterialProperties>;

// --- Group 2: per-arena visibility data, fragment-stage only. ---
struct ChunkMeta {
    chunk_world_origin: vec3<f32>,
    active_list_pos: u32,
}

struct VisBatchUniforms {
    cell_count: u32,
    texture_size: u32,
    wg_per_chunk_z: u32, // unused here — reusing pass3's uniform buffer as-is
    voxel_size: f32,
}

@group(2) @binding(0) var<storage, read> chunk_meta: array<ChunkMeta>;
@group(2) @binding(1) var<storage, read> visibility_mask_buffer: array<u32>;
@group(2) @binding(2) var<storage, read> chunk_has_mask_buffer: array<u32>;
@group(2) @binding(3) var<uniform> vis_uniforms: VisBatchUniforms;

struct VertexInput {
    @location(0) position: vec4<f32>, // .xyz = world position, .w = bitcast-packed material_id_a (low byte) + blend weight (next byte)
    @location(1) normal: vec4<f32>,   // .xyz = normal, .w = bitcast-packed material_id_b (low byte) + real_slot (next 15 bits)
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) @interpolate(flat) material_id_a: u32,
    @location(3) @interpolate(flat) material_id_b: u32,
    @location(4) blend_weight: f32,
    // NEW — flat-interpolated (every vertex in a triangle is from the same
    // chunk, so this is constant across the triangle regardless of
    // interpolation mode; flat avoids any float-precision surprises).
    @location(5) @interpolate(flat) real_slot: u32,
};

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
    out.real_slot = (packed_b >> 8u) & 0x7FFFu; // see pack_material_b in pass3

    out.clip_position = view.clip_from_world * vec4<f32>(input.position.xyz, 1.0);

    return out;
}

// --- Visibility sampling, ported from pass1/pass3's identical logic ---
fn flatten_sdf_idx(coord: vec3<u32>, texture_size: u32) -> u32 {
    return coord.z * texture_size * texture_size + coord.y * texture_size + coord.x;
}

fn mask_words_per_chunk() -> u32 {
    let elems = vis_uniforms.texture_size * vis_uniforms.texture_size * vis_uniforms.texture_size;
    return (elems + 31u) / 32u;
}

fn chunk_has_mask(slot: u32) -> bool {
    let word = chunk_has_mask_buffer[slot / 32u];
    return ((word >> (slot % 32u)) & 1u) != 0u;
}

fn sample_visibility(slot: u32, coord: vec3<u32>) -> bool {
    let local_bit = flatten_sdf_idx(coord, vis_uniforms.texture_size);
    let word_idx = slot * mask_words_per_chunk() + local_bit / 32u;
    let word = visibility_mask_buffer[word_idx];
    return ((word >> (local_bit % 32u)) & 1u) != 0u;
}

// Nearest-voxel sample at the fragment's actual interpolated world
// position, converted back to this chunk's local texture-space coords.
// This is what makes the discard boundary follow the visibility grid at
// per-pixel resolution instead of being frozen per-triangle at vertex time.
fn fragment_is_visible(slot: u32, world_pos: vec3<f32>) -> bool {
    if (!chunk_has_mask(slot)) {
        return true; // uniformly-visible chunk — see pass1's identical comment
    }
    let origin = chunk_meta[slot].chunk_world_origin;
    let local = (world_pos - origin) / vis_uniforms.voxel_size;
    let max_coord = f32(vis_uniforms.texture_size) - 1.0;
    let rounded = clamp(round(local), vec3<f32>(0.0), vec3<f32>(max_coord));
    return sample_visibility(slot, vec3<u32>(rounded));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Per-pixel visibility test, using the smoothly-interpolated
    // world_position rather than a flat per-vertex flag — this is the
    // change from the previous per-vertex-flag approach: the discard
    // boundary now follows the visibility voxel grid at fragment
    // resolution instead of snapping to whole triangles.
    if (!fragment_is_visible(in.real_slot, in.world_position)) {
        discard;
    }

    let N = normalize(in.world_normal);

    let mat_a = material_properties[in.material_id_a];
    let mat_b = material_properties[in.material_id_b];

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

    let blended_color = mix(color_b, color_a, in.blend_weight);

    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.2));
    let diffuse = max(dot(N, light_dir), 0.0);
    let ambient = 0.2;
    let lighting = diffuse + ambient;

    return vec4<f32>(blended_color.rgb * lighting, 1.0);
}
