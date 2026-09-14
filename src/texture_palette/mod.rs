mod asset;
mod builder;
mod properties;
mod validation;

use bevy::prelude::*;

use properties::PaletteMaterial;

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
