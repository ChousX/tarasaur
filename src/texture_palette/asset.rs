//! Texture palette asset definition.

use bevy::asset::Assets;
use bevy::image::Image;
use bevy::prelude::*;

use super::properties::PaletteMaterial;
use super::validation::{self, PaletteValidationError};

/// A texture palette containing all materials for triplanar voxel rendering.
///
/// A palette consists of texture arrays (albedo, optionally normal and ARM)
/// where each layer corresponds to a material that can be applied to vertices.
///
/// # Texture Requirements
///
/// All textures must be:
/// - 2D array textures (KTX2 format recommended)
/// - Same layer count across all texture types
/// - Same resolution per layer
/// - Square and power-of-two dimensions
/// - Albedo: sRGB format (e.g., `Rgba8UnormSrgb`, `Bc7RgbaUnormSrgb`)
/// - Normal/ARM: Linear format (e.g., `Rgba8Unorm`, `Bc5RgUnorm`)
///
/// # Example
///
/// ```ignore
/// use bevy_painter::palette::{TexturePalette, PaletteMaterial};
///
/// let palette = TexturePalette {
///     albedo: asset_server.load("terrain/albedo.ktx2"),
///     normal: Some(asset_server.load("terrain/normal.ktx2")),
///     arm: Some(asset_server.load("terrain/arm.ktx2")),
///     materials: vec![
///         PaletteMaterial::new("grass").with_texture_scale(1.0),
///         PaletteMaterial::new("stone").with_texture_scale(0.5),
///         PaletteMaterial::new("dirt").with_texture_scale(1.0),
///     ],
///     generate_mipmaps: false, // KTX2 should have mipmaps baked in
/// };
/// ```
#[derive(Asset, TypePath, Clone, Debug)]
pub struct TexturePalette {
    /// Albedo (base color) texture array.
    ///
    /// **Required.** Must be sRGB format.
    pub albedo: Handle<Image>,

    /// Normal map texture array.
    ///
    /// Optional. Must be linear format.
    /// If provided, enables triplanar normal mapping.
    pub normal: Option<Handle<Image>>,

    /// ARM (Ambient Occlusion, Roughness, Metallic) texture array.
    ///
    /// Optional. Must be linear format.
    /// Channel layout: R = AO, G = Roughness, B = Metallic
    pub arm: Option<Handle<Image>>,

    /// Per-layer material properties.
    ///
    /// The length of this vector should match the layer count of the textures.
    /// Each material corresponds to one layer in the texture arrays.
    pub materials: Vec<PaletteMaterial>,

    /// Whether to generate mipmaps for textures that don't have them.
    ///
    /// When using pre-mipmapped KTX2 textures (recommended), set this to `false`.
    /// Default: `false`
    pub generate_mipmaps: bool,
}

impl Default for TexturePalette {
    fn default() -> Self {
        Self {
            albedo: Handle::default(),
            normal: None,
            arm: None,
            materials: Vec::new(),
            generate_mipmaps: false,
        }
    }
}

impl TexturePalette {
    /// Get the number of materials in this palette.
    pub fn material_count(&self) -> usize {
        self.materials.len()
    }

    /// Check if this palette has normal maps.
    pub fn has_normal_maps(&self) -> bool {
        self.normal.is_some()
    }

    /// Check if this palette has ARM textures.
    pub fn has_arm(&self) -> bool {
        self.arm.is_some()
    }

    /// Validate the palette against loaded image assets.
    ///
    /// This checks that:
    /// - All textures are loaded and have correct formats
    /// - All textures have matching layer counts and dimensions
    /// - Material count doesn't exceed layer count
    ///
    /// # Panics
    ///
    /// This method is intended to be called in a system where you can handle
    /// the error appropriately. For automatic validation with panics, see
    /// the palette validation system in the plugin.
    pub fn validate(&self, images: &Assets<Image>) -> Result<(), PaletteValidationError> {
        // Validate albedo (required)
        let albedo_image = images
            .get(&self.albedo)
            .ok_or(PaletteValidationError::AlbedoNotLoaded)?;

        validation::validate_albedo(albedo_image)?;

        let layer_count = albedo_image.texture_descriptor.size.depth_or_array_layers;

        // Validate normal (optional)
        if let Some(ref normal_handle) = self.normal {
            if let Some(normal_image) = images.get(normal_handle) {
                validation::validate_linear_texture(normal_image, albedo_image, "normal")?;
            }
            // If not loaded yet, skip validation (will be caught on next frame)
        }

        // Validate ARM (optional)
        if let Some(ref arm_handle) = self.arm {
            if let Some(arm_image) = images.get(arm_handle) {
                validation::validate_linear_texture(arm_image, albedo_image, "arm")?;
            }
        }

        // Validate material count
        validation::validate_material_count(self.materials.len(), layer_count)?;

        Ok(())
    }

    /// Get the layer count from the albedo texture.
    ///
    /// Returns `None` if the albedo texture isn't loaded.
    pub fn layer_count(&self, images: &Assets<Image>) -> Option<u32> {
        images
            .get(&self.albedo)
            .map(|img| img.texture_descriptor.size.depth_or_array_layers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_palette() {
        let palette = TexturePalette::default();
        assert_eq!(palette.material_count(), 0);
        assert!(!palette.has_normal_maps());
        assert!(!palette.has_arm());
    }
}
