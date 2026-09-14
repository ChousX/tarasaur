//! Palette validation utilities.

use bevy::image::Image;
use bevy::render::render_resource::{TextureDimension, TextureFormat};
use thiserror::Error;

use super::properties::MAX_MATERIALS;

/// Errors that can occur when validating a texture palette.
#[derive(Error, Debug, Clone)]
pub enum PaletteValidationError {
    #[error("Albedo texture not loaded")]
    AlbedoNotLoaded,

    #[error("Albedo texture must be a 2D array texture, got {found:?}")]
    AlbedoNotArray { found: TextureDimension },

    #[error("Albedo texture has invalid format: expected sRGB format, got {found:?}")]
    InvalidAlbedoFormat { found: TextureFormat },

    #[error("Normal texture has invalid format: expected linear format, got {found:?}")]
    InvalidNormalFormat { found: TextureFormat },

    #[error("ARM texture has invalid format: expected linear format, got {found:?}")]
    InvalidArmFormat { found: TextureFormat },

    #[error("Texture '{name}' is not a 2D array: dimension is {found:?}")]
    NotTextureArray {
        name: &'static str,
        found: TextureDimension,
    },

    #[error(
        "Texture layer count mismatch: albedo has {albedo} layers, {other_name} has {other} layers"
    )]
    LayerCountMismatch {
        albedo: u32,
        other_name: &'static str,
        other: u32,
    },

    #[error(
        "Texture size mismatch: albedo is {albedo_width}x{albedo_height}, {other_name} is {other_width}x{other_height}"
    )]
    SizeMismatch {
        albedo_width: u32,
        albedo_height: u32,
        other_name: &'static str,
        other_width: u32,
        other_height: u32,
    },

    #[error("Material count ({material_count}) exceeds texture layer count ({layer_count})")]
    TooManyMaterials {
        material_count: usize,
        layer_count: u32,
    },

    #[error("Material count ({count}) exceeds maximum ({max})")]
    ExceedsMaxMaterials { count: usize, max: usize },

    #[error("Texture is not square: {width}x{height}")]
    NotSquare { width: u32, height: u32 },

    #[error("Texture size {size} is not a power of two")]
    NotPowerOfTwo { size: u32 },
}

/// Check if a texture format is valid for sRGB albedo textures.
pub fn is_valid_srgb_format(format: TextureFormat) -> bool {
    matches!(
        format,
        // Uncompressed sRGB
        TextureFormat::Rgba8UnormSrgb
            | TextureFormat::Bgra8UnormSrgb
            // BC compressed sRGB (desktop)
            | TextureFormat::Bc1RgbaUnormSrgb
            | TextureFormat::Bc2RgbaUnormSrgb
            | TextureFormat::Bc3RgbaUnormSrgb
            | TextureFormat::Bc7RgbaUnormSrgb
            // ETC2 compressed sRGB (mobile/web)
            | TextureFormat::Etc2Rgb8UnormSrgb
            | TextureFormat::Etc2Rgb8A1UnormSrgb
            | TextureFormat::Etc2Rgba8UnormSrgb
    ) || matches!(format, TextureFormat::Astc { channel, .. } if channel == bevy::render::render_resource::AstcChannel::UnormSrgb)
}

/// Check if a texture format is valid for linear data textures (normal, ARM).
pub fn is_valid_linear_format(format: TextureFormat) -> bool {
    matches!(
        format,
        // Uncompressed linear
        TextureFormat::Rgba8Unorm
            | TextureFormat::Bgra8Unorm
            | TextureFormat::Rg8Unorm // 2-component normal maps
            | TextureFormat::Rg16Unorm
            // BC compressed linear (desktop)
            | TextureFormat::Bc1RgbaUnorm
            | TextureFormat::Bc3RgbaUnorm
            | TextureFormat::Bc4RUnorm
            | TextureFormat::Bc5RgUnorm // Normal maps
            | TextureFormat::Bc7RgbaUnorm
            // ETC2/EAC compressed linear (mobile/web)
            | TextureFormat::Etc2Rgb8Unorm
            | TextureFormat::Etc2Rgba8Unorm
            | TextureFormat::EacR11Unorm
            | TextureFormat::EacRg11Unorm // Normal maps
    ) || matches!(format, TextureFormat::Astc { channel, .. } if channel == bevy::render::render_resource::AstcChannel::Unorm)
}

/// Validate an albedo texture.
pub fn validate_albedo(image: &Image) -> Result<(), PaletteValidationError> {
    // Check dimension
    if image.texture_descriptor.dimension != TextureDimension::D2 {
        return Err(PaletteValidationError::AlbedoNotArray {
            found: image.texture_descriptor.dimension,
        });
    }

    // Check array layers
    let layers = image.texture_descriptor.size.depth_or_array_layers;
    if layers < 1 {
        return Err(PaletteValidationError::AlbedoNotArray {
            found: image.texture_descriptor.dimension,
        });
    }

    // Check format
    if !is_valid_srgb_format(image.texture_descriptor.format) {
        return Err(PaletteValidationError::InvalidAlbedoFormat {
            found: image.texture_descriptor.format,
        });
    }

    // Check square
    let width = image.texture_descriptor.size.width;
    let height = image.texture_descriptor.size.height;
    if width != height {
        return Err(PaletteValidationError::NotSquare { width, height });
    }

    // Check power of two
    if !width.is_power_of_two() {
        return Err(PaletteValidationError::NotPowerOfTwo { size: width });
    }

    Ok(())
}

/// Validate a linear texture (normal or ARM) against the albedo texture.
pub fn validate_linear_texture(
    image: &Image,
    albedo: &Image,
    name: &'static str,
) -> Result<(), PaletteValidationError> {
    // Check dimension
    if image.texture_descriptor.dimension != TextureDimension::D2 {
        return Err(PaletteValidationError::NotTextureArray {
            name,
            found: image.texture_descriptor.dimension,
        });
    }

    // Check format
    if !is_valid_linear_format(image.texture_descriptor.format) {
        if name == "normal" {
            return Err(PaletteValidationError::InvalidNormalFormat {
                found: image.texture_descriptor.format,
            });
        } else {
            return Err(PaletteValidationError::InvalidArmFormat {
                found: image.texture_descriptor.format,
            });
        }
    }

    // Check layer count matches albedo
    let albedo_layers = albedo.texture_descriptor.size.depth_or_array_layers;
    let layers = image.texture_descriptor.size.depth_or_array_layers;
    if layers != albedo_layers {
        return Err(PaletteValidationError::LayerCountMismatch {
            albedo: albedo_layers,
            other_name: name,
            other: layers,
        });
    }

    // Check size matches albedo
    let albedo_size = &albedo.texture_descriptor.size;
    let size = &image.texture_descriptor.size;
    if size.width != albedo_size.width || size.height != albedo_size.height {
        return Err(PaletteValidationError::SizeMismatch {
            albedo_width: albedo_size.width,
            albedo_height: albedo_size.height,
            other_name: name,
            other_width: size.width,
            other_height: size.height,
        });
    }

    Ok(())
}

/// Validate material count against texture layers and maximum.
pub fn validate_material_count(
    material_count: usize,
    layer_count: u32,
) -> Result<(), PaletteValidationError> {
    if material_count > MAX_MATERIALS {
        return Err(PaletteValidationError::ExceedsMaxMaterials {
            count: material_count,
            max: MAX_MATERIALS,
        });
    }

    if material_count > layer_count as usize {
        return Err(PaletteValidationError::TooManyMaterials {
            material_count,
            layer_count,
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srgb_formats() {
        assert!(is_valid_srgb_format(TextureFormat::Rgba8UnormSrgb));
        assert!(is_valid_srgb_format(TextureFormat::Bc7RgbaUnormSrgb));
        assert!(!is_valid_srgb_format(TextureFormat::Rgba8Unorm)); // Linear, not sRGB
    }

    #[test]
    fn test_linear_formats() {
        assert!(is_valid_linear_format(TextureFormat::Rgba8Unorm));
        assert!(is_valid_linear_format(TextureFormat::Bc5RgUnorm));
        assert!(!is_valid_linear_format(TextureFormat::Rgba8UnormSrgb)); // sRGB, not linear
    }
}
