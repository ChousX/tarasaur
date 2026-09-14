use bevy::prelude::*;

#[derive(Asset, TypePath, Clone, Debug)]
pub struct TexturePalette {
    pub albedo: Handle<Image>,
    /// Optional 2D array texture for linear normal maps.
    pub normal: Option<Handle<Image>>,
    /// Optional 2D array texture for Occlusion (R), Roughness (G), and Metallic (B).
    pub arm: Option<Handle<Image>>,
    /// Per-layer properties (scale, sharpness, overrides) indexed by material ID.
    pub materials: Vec<PaletteMaterial>,
    /// Controls whether to generate mipmaps dynamically if not pre-baked.
    pub generate_mipmaps: bool,
}

/// Properties for a single material layer in the palette.
///
/// These properties control how the material is rendered, including
/// texture scaling and triplanar blend sharpness.
#[derive(Clone, Debug, Reflect)]
pub struct PaletteMaterial {
    /// Display name for debugging and tooling.
    pub name: String,

    /// Texture coordinate scale in world units per texture repeat.
    ///
    /// A value of 1.0 means the texture repeats every 1 world unit.
    /// Smaller values = more repetition, larger values = more stretched.
    ///
    /// Default: 1.0
    pub texture_scale: f32,

    /// Triplanar blend sharpness for this material.
    ///
    /// Higher values create sharper transitions between projection planes.
    /// Lower values create smoother but potentially blurrier transitions.
    ///
    /// Typical range: 1.0 - 16.0
    /// Default: 4.0
    pub blend_sharpness: f32,

    /// Optional roughness override.
    ///
    /// If `Some`, this value overrides the roughness from the ARM texture.
    /// If `None`, the ARM texture value is used.
    pub roughness_override: Option<f32>,

    /// Optional metallic override.
    ///
    /// If `Some`, this value overrides the metallic from the ARM texture.
    /// If `None`, the ARM texture value is used.
    pub metallic_override: Option<f32>,
}
