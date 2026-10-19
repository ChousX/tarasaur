//! Texture palette asset definition.

use bevy::asset::Assets;
use bevy::image::Image;
use bevy::prelude::*;

use super::properties::PaletteMaterial;
use super::validation::{self, PaletteValidationError};

/// A texture palette containing all materials for triplanar voxel rendering.
///
/// A palette is an albedo texture array, plus optional ARM and normal map
/// arrays, where each layer corresponds to a material that can be applied to
/// vertices.
///
/// # Texture Requirements
///
/// The albedo texture must be:
/// - A 2D array texture (KTX2 format recommended)
/// - Square and power-of-two per layer
/// - sRGB format (e.g., `Rgba8UnormSrgb`, `Bc7RgbaUnormSrgb`)
///
/// The optional ARM and normal map textures must match the albedo's size and
/// layer count, in a linear format (e.g., `Rgba8Unorm`, `Bc7RgbaUnorm`, or
/// `Bc5RgUnorm` for normals).
///
/// # Example
///
/// ```ignore
/// use tarasaur::texture_palette::{TexturePalette, PaletteMaterial};
///
/// let palette = TexturePalette {
///     albedo: asset_server.load("terrain/albedo.ktx2"),
///     arm: Some(asset_server.load("terrain/arm.ktx2")),
///     normal: Some(asset_server.load("terrain/normal.ktx2")),
///     materials: vec![
///         PaletteMaterial::new("grass").with_texture_scale(1.0),
///         PaletteMaterial::new("stone").with_texture_scale(0.5),
///         PaletteMaterial::new("dirt").with_texture_scale(1.0),
///     ],
///     generate_mipmaps: false, // KTX2 should have mipmaps baked in
/// };
/// ```
#[derive(Asset, TypePath, Clone, Debug, Default)]
pub struct TexturePalette {
    /// Albedo (base color) texture array.
    ///
    /// **Required.** Must be sRGB format.
    pub albedo: Handle<Image>,

    /// ARM (Ambient Occlusion, Roughness, Metallic) texture array.
    ///
    /// Optional. Must be linear format, with the albedo's size and layer
    /// count. Channel layout: R = AO, G = perceptual roughness, B = metallic.
    /// Without it, every material is fully unoccluded, fully rough and
    /// non-metallic unless its overrides say otherwise.
    pub arm: Option<Handle<Image>>,

    /// Tangent-space normal map texture array.
    ///
    /// Optional. Must be linear format, with the albedo's size and layer
    /// count. OpenGL convention (green = +V): R and G hold the normal's X and
    /// Y, and Z is rebuilt in the shader, so a two-channel format such as
    /// `Bc5RgUnorm` works. Without it, terrain is lit by its mesh normals.
    pub normal: Option<Handle<Image>>,

    /// Per-layer material properties.
    ///
    /// The length of this vector should match the albedo layer count.
    /// Each material corresponds to one layer of the albedo array.
    pub materials: Vec<PaletteMaterial>,

    /// Whether to generate mipmaps for textures that don't have them.
    ///
    /// Built once on the CPU when each image has loaded, for uncompressed
    /// `Rgba8Unorm`/`Rgba8UnormSrgb` arrays with only a base level (see
    /// `mipmaps::generate_mipmaps`); other images are left as they are.
    /// Without mips, distant terrain shimmers. When using pre-mipmapped KTX2
    /// textures (recommended), set this to `false`.
    /// Default: `false`
    pub generate_mipmaps: bool,
}

impl TexturePalette {
    /// Get the number of materials in this palette.
    #[must_use]
    pub const fn material_count(&self) -> usize {
        self.materials.len()
    }

    /// Check if this palette has an ARM texture.
    #[must_use]
    pub const fn has_arm(&self) -> bool {
        self.arm.is_some()
    }

    /// Check if this palette has a normal map texture.
    #[must_use]
    pub const fn has_normal(&self) -> bool {
        self.normal.is_some()
    }

    /// Validate the palette against loaded image assets.
    ///
    /// This checks that:
    /// - The albedo texture is loaded and has a valid format
    /// - The ARM and normal map textures, once loaded, are linear and match
    ///   the albedo
    /// - Material count doesn't exceed layer count
    ///
    /// # Errors
    ///
    /// The first check that fails, as a [`PaletteValidationError`]. This
    /// method is intended to be called in a system where you can handle the
    /// error appropriately; the plugin's palette validation system does so.
    pub fn validate(&self, images: &Assets<Image>) -> Result<(), PaletteValidationError> {
        // Validate albedo (required)
        let albedo_image = images
            .get(&self.albedo)
            .ok_or(PaletteValidationError::AlbedoNotLoaded)?;

        validation::validate_albedo(albedo_image)?;

        let layer_count = albedo_image.texture_descriptor.size.depth_or_array_layers;

        // Validate ARM (optional). If not loaded yet, it's checked next time.
        if let Some(arm_image) = self.arm.as_ref().and_then(|h| images.get(h)) {
            validation::validate_arm(arm_image, albedo_image)?;
        }
        if let Some(normal_image) = self.normal.as_ref().and_then(|h| images.get(h)) {
            validation::validate_normal(normal_image, albedo_image)?;
        }

        // Validate material count
        validation::validate_material_count(self.materials.len(), layer_count)?;

        Ok(())
    }

    /// Get the layer count from the albedo texture.
    ///
    /// Returns `None` if the albedo texture isn't loaded.
    #[must_use]
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
        assert!(!palette.has_arm());
        assert!(!palette.has_normal());
    }
}
