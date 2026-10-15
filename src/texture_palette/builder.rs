//! Builder for constructing texture palettes.

use super::asset::TexturePalette;
use super::properties::PaletteMaterial;
use bevy::prelude::*;

/// Builder for creating [`TexturePalette`] instances.
///
/// # Example
///
/// ```ignore
/// use tarasaur::texture_palette::{PaletteBuilder, PaletteMaterial};
///
/// let palette = PaletteBuilder::new(asset_server.load("terrain/albedo.ktx2"))
///     .with_arm(asset_server.load("terrain/arm.ktx2"))
///     .with_normal(asset_server.load("terrain/normal.ktx2"))
///     .add_material(PaletteMaterial::new("grass").with_texture_scale(1.0))
///     .add_material(PaletteMaterial::new("stone").with_texture_scale(0.5))
///     .add_material(PaletteMaterial::new("dirt"))
///     .build();
/// ```
pub struct PaletteBuilder {
    albedo: Handle<Image>,
    arm: Option<Handle<Image>>,
    normal: Option<Handle<Image>>,
    materials: Vec<PaletteMaterial>,
    generate_mipmaps: bool,
}

impl PaletteBuilder {
    /// Create a palette builder around its albedo (base color) texture
    /// array, the one texture every palette needs.
    #[must_use]
    pub const fn new(albedo: Handle<Image>) -> Self {
        Self {
            albedo,
            arm: None,
            normal: None,
            materials: Vec::new(),
            generate_mipmaps: false,
        }
    }

    /// Set the ARM (AO/Roughness/Metallic) texture array.
    ///
    /// Optional. Channel layout: R = AO, G = perceptual roughness,
    /// B = metallic. Must match the albedo's size and layer count.
    #[must_use]
    pub fn with_arm(mut self, arm: Handle<Image>) -> Self {
        self.arm = Some(arm);
        self
    }

    /// Set the tangent-space normal map texture array.
    ///
    /// Optional. OpenGL convention (green = +V); R and G hold X and Y, and Z
    /// is rebuilt in the shader. Must match the albedo's size and layer count.
    #[must_use]
    pub fn with_normal(mut self, normal: Handle<Image>) -> Self {
        self.normal = Some(normal);
        self
    }

    /// Add a material to the palette.
    ///
    /// Materials are added in order, corresponding to texture array layers.
    #[must_use]
    pub fn add_material(mut self, material: PaletteMaterial) -> Self {
        self.materials.push(material);
        self
    }

    /// Add multiple materials at once.
    #[must_use]
    pub fn add_materials(mut self, materials: impl IntoIterator<Item = PaletteMaterial>) -> Self {
        self.materials.extend(materials);
        self
    }

    /// Add a material with just a name, using default properties.
    #[must_use]
    pub fn add_material_named(self, name: impl Into<String>) -> Self {
        self.add_material(PaletteMaterial::new(name))
    }

    /// Set whether to generate mipmaps for textures without them.
    ///
    /// Default: `false` (assumes pre-mipmapped KTX2 textures).
    #[must_use]
    pub const fn with_generate_mipmaps(mut self, generate: bool) -> Self {
        self.generate_mipmaps = generate;
        self
    }

    /// Build the texture palette.
    #[must_use]
    pub fn build(self) -> TexturePalette {
        TexturePalette {
            albedo: self.albedo,
            arm: self.arm,
            normal: self.normal,
            materials: self.materials,
            generate_mipmaps: self.generate_mipmaps,
        }
    }
}

/// Extension trait for quickly creating simple palettes.
pub trait QuickPalette {
    /// Create a palette with just an albedo texture and auto-generated material names.
    ///
    /// Material names will be "`material_0`", "`material_1`", etc.
    #[must_use]
    fn quick_palette(albedo: Handle<Image>, material_count: usize) -> TexturePalette {
        let materials = (0..material_count)
            .map(|i| PaletteMaterial::new(format!("material_{i}")))
            .collect();

        TexturePalette {
            albedo,
            arm: None,
            normal: None,
            materials,
            generate_mipmaps: false,
        }
    }
}

impl QuickPalette for TexturePalette {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_basic() {
        let palette = PaletteBuilder::new(Handle::default())
            .add_material_named("grass")
            .add_material_named("stone")
            .build();

        assert_eq!(palette.material_count(), 2);
        assert_eq!(palette.materials[0].name, "grass");
        assert_eq!(palette.materials[1].name, "stone");
    }

    #[test]
    fn test_builder_full() {
        let palette = PaletteBuilder::new(Handle::default())
            .with_arm(Handle::default())
            .with_normal(Handle::default())
            .add_material(PaletteMaterial::new("grass").with_texture_scale(2.0))
            .with_generate_mipmaps(true)
            .build();

        assert!(palette.has_arm());
        assert!(palette.has_normal());
        assert!(palette.generate_mipmaps);
        assert_eq!(palette.materials[0].texture_scale, 2.0);
    }

    #[test]
    fn test_quick_palette() {
        let palette = TexturePalette::quick_palette(Handle::default(), 3);

        assert_eq!(palette.material_count(), 3);
        assert_eq!(palette.materials[0].name, "material_0");
        assert_eq!(palette.materials[2].name, "material_2");
    }
}
