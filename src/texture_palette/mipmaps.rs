//! CPU mip chains for palette textures that come without one
//! (`TexturePalette::generate_mipmaps`).

use bevy::{
    prelude::*,
    render::render_resource::{TextureDimension, TextureFormat},
};

use crate::num::usize_from;

/// Whether `generate_mipmaps` can build `image`'s mip chain: an
/// uncompressed RGBA8 2D (array) texture with only its base level.
#[must_use]
pub fn can_generate_mipmaps(image: &Image) -> bool {
    let desc = &image.texture_descriptor;
    desc.mip_level_count == 1
        && desc.dimension == TextureDimension::D2
        && matches!(
            desc.format,
            TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb
        )
        && desc.size.width > 1
        && desc.size.width == desc.size.height
        && desc.size.width.is_power_of_two()
        && image.data.is_some()
}

/// Appends a full mip chain to `image` (see `can_generate_mipmaps`), each
/// level a 2x2 box filter of the one above. sRGB textures are averaged in
/// linear light. Returns whether it did.
pub fn generate_mipmaps(image: &mut Image) -> bool {
    if !can_generate_mipmaps(image) {
        return false;
    }
    let srgb = image.texture_descriptor.format == TextureFormat::Rgba8UnormSrgb;
    let size = image.texture_descriptor.size.width;
    let layers = image.texture_descriptor.size.depth_or_array_layers;
    let levels = size.ilog2().saturating_add(1);
    let Some(data) = image.data.as_ref() else {
        return false;
    };
    let base_len = usize_from(size).pow(2).saturating_mul(4);
    if data.len() != base_len.saturating_mul(usize_from(layers)) {
        return false;
    }

    // Layer-major, like Bevy's default `TextureDataOrder`: every level of
    // layer 0, then of layer 1, ...
    // A full chain adds under a third.
    let mut out = Vec::with_capacity(data.len().saturating_mul(2));
    for layer in data.chunks_exact(base_len) {
        out.extend_from_slice(layer);
        let mut prev = layer.to_vec();
        let mut prev_size = size;
        while prev_size > 1 {
            prev = downsample(&prev, prev_size, srgb);
            prev_size /= 2;
            out.extend_from_slice(&prev);
        }
    }
    image.texture_descriptor.mip_level_count = levels;
    image.data = Some(out);
    true
}

/// One 2x2 box-filtered level from a `size`-square RGBA8 level.
fn downsample(src: &[u8], size: u32, srgb: bool) -> Vec<u8> {
    let half = usize_from(size / 2);
    let row = usize_from(size).saturating_mul(4);
    let mut dst = Vec::with_capacity(half.pow(2).saturating_mul(4));
    for y in 0..half {
        for x in 0..half {
            let at = |dx: usize, dy: usize, c: usize| {
                let sx = x.saturating_mul(2).saturating_add(dx);
                let sy = y.saturating_mul(2).saturating_add(dy);
                let i = sy
                    .saturating_mul(row)
                    .saturating_add(sx.saturating_mul(4))
                    .saturating_add(c);
                src.get(i).copied().unwrap_or(0)
            };
            for c in 0..4 {
                let texels = [at(0, 0, c), at(1, 0, c), at(0, 1, c), at(1, 1, c)];
                dst.push(if srgb && c < 3 {
                    let sum: f32 = texels.iter().map(|&v| srgb_to_linear(v)).sum();
                    linear_to_srgb(sum / 4.0)
                } else {
                    let sum: u16 = texels.iter().map(|&v| u16::from(v)).sum();
                    u8::try_from(sum.saturating_add(2) / 4).unwrap_or(u8::MAX)
                });
            }
        }
    }
    dst
}

fn srgb_to_linear(v: u8) -> f32 {
    let c = f32::from(v) / 255.0;
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(l: f32) -> u8 {
    let c = if l <= 0.003_130_8 {
        l * 12.92
    } else {
        1.055f32.mul_add(l.powf(1.0 / 2.4), -0.055)
    };
    crate::num::u8_from_f32((c * 255.0).round())
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::{asset::RenderAssetUsages, render::render_resource::Extent3d};

    fn image(size: u32, layers: u32, format: TextureFormat, data: Vec<u8>) -> Image {
        Image::new(
            Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: layers,
            },
            TextureDimension::D2,
            data,
            format,
            RenderAssetUsages::default(),
        )
    }

    #[test]
    fn builds_a_layer_major_chain() {
        // Two 4x4 layers: layer 0 all 40, layer 1 all 200.
        let mut data = vec![40u8; 4 * 4 * 4];
        data.extend(vec![200u8; 4 * 4 * 4]);
        let mut img = image(4, 2, TextureFormat::Rgba8Unorm, data);
        assert!(generate_mipmaps(&mut img));
        assert_eq!(img.texture_descriptor.mip_level_count, 3);
        let data = img.data.unwrap();
        // Per layer: 16 + 4 + 1 texels.
        assert_eq!(data.len(), 2 * 21 * 4);
        assert!(data[..21 * 4].iter().all(|&v| v == 40));
        assert!(data[21 * 4..].iter().all(|&v| v == 200));
    }

    #[test]
    fn averages_texels() {
        // 2x2, one channel varying: 0, 100, 200, 100 -> 100.
        let mut data = Vec::new();
        for v in [0u8, 100, 200, 100] {
            data.extend([v, 0, 0, 255]);
        }
        let mut img = image(2, 1, TextureFormat::Rgba8Unorm, data);
        assert!(generate_mipmaps(&mut img));
        let data = img.data.unwrap();
        assert_eq!(&data[16..], &[100, 0, 0, 255]);
    }

    #[test]
    fn srgb_averages_in_linear_light() {
        // Black and white average to linear 0.5, which is sRGB ~188, not 128.
        let mut data = Vec::new();
        for v in [0u8, 255, 0, 255] {
            data.extend([v, v, v, 255]);
        }
        let mut img = image(2, 1, TextureFormat::Rgba8UnormSrgb, data);
        assert!(generate_mipmaps(&mut img));
        let data = img.data.unwrap();
        assert_eq!(&data[16..], &[188, 188, 188, 255]);
    }

    #[test]
    fn leaves_mipmapped_and_unsupported_images_alone() {
        let mut img = image(2, 1, TextureFormat::Rg8Unorm, vec![0; 2 * 2 * 2]);
        assert!(!generate_mipmaps(&mut img));
        assert_eq!(img.texture_descriptor.mip_level_count, 1);

        let mut img = image(2, 1, TextureFormat::Rgba8Unorm, vec![0; 2 * 2 * 4]);
        assert!(generate_mipmaps(&mut img));
        assert!(!generate_mipmaps(&mut img)); // already has its chain
    }
}
