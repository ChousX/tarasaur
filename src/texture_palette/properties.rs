//! Per-material properties within a palette.
use bevy::prelude::*;
use bevy::render::render_resource::ShaderType;
use bytemuck::{Pod, Zeroable};

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

impl Default for PaletteMaterial {
    fn default() -> Self {
        Self {
            name: String::new(),
            texture_scale: 1.0,
            blend_sharpness: 4.0,
            roughness_override: None,
            metallic_override: None,
        }
    }
}

impl PaletteMaterial {
    /// Create a new material with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    /// Set the texture scale.
    pub fn with_texture_scale(mut self, scale: f32) -> Self {
        self.texture_scale = scale;
        self
    }

    /// Set the blend sharpness.
    pub fn with_blend_sharpness(mut self, sharpness: f32) -> Self {
        self.blend_sharpness = sharpness;
        self
    }

    /// Set an optional roughness override.
    pub fn with_roughness(mut self, roughness: f32) -> Self {
        self.roughness_override = Some(roughness);
        self
    }

    /// Set an optional metallic override.
    pub fn with_metallic(mut self, metallic: f32) -> Self {
        self.metallic_override = Some(metallic);
        self
    }
}

/// GPU-side representation of material properties.
///
/// This is stored in a uniform buffer and indexed by material ID in the shader.
#[derive(Clone, Copy, Debug, Default, ShaderType, Pod, Zeroable)]
#[repr(C)]
pub struct MaterialPropertiesGpu {
    /// Texture scale (world units per repeat).
    pub texture_scale: f32,

    /// Triplanar blend sharpness.
    pub blend_sharpness: f32,

    /// Roughness override. Negative value means "use texture".
    pub roughness_override: f32,

    /// Metallic override. Negative value means "use texture".
    pub metallic_override: f32,
}

impl From<&PaletteMaterial> for MaterialPropertiesGpu {
    fn from(mat: &PaletteMaterial) -> Self {
        Self {
            texture_scale: mat.texture_scale,
            blend_sharpness: mat.blend_sharpness,
            roughness_override: mat.roughness_override.unwrap_or(-1.0),
            metallic_override: mat.metallic_override.unwrap_or(-1.0),
        }
    }
}

/// Maximum number of materials supported in a single palette.
///
/// This limit exists because we use a uniform buffer for material properties.
pub const MAX_MATERIALS: usize = u8::MAX as usize;

/// GPU-side array of all material properties.
///
/// Padded to MAX_MATERIALS for uniform buffer alignment.
#[derive(Clone, Debug, ShaderType)]
pub struct MaterialPropertiesArray {
    #[shader(size(runtime))]
    pub materials: Vec<MaterialPropertiesGpu>,
}

impl MaterialPropertiesArray {
    /// Create from a slice of palette materials.
    ///
    /// # Panics
    /// Panics if `materials.len() > MAX_MATERIALS`.
    pub fn from_materials(materials: &[PaletteMaterial]) -> Self {
        assert!(
            materials.len() <= MAX_MATERIALS,
            "Too many materials: {} > {}",
            materials.len(),
            MAX_MATERIALS
        );

        let gpu_materials: Vec<MaterialPropertiesGpu> =
            materials.iter().map(MaterialPropertiesGpu::from).collect();

        Self {
            materials: gpu_materials,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_palette_material_builder() {
        let mat = PaletteMaterial::new("grass")
            .with_texture_scale(2.0)
            .with_blend_sharpness(8.0)
            .with_roughness(0.9);

        assert_eq!(mat.name, "grass");
        assert_eq!(mat.texture_scale, 2.0);
        assert_eq!(mat.blend_sharpness, 8.0);
        assert_eq!(mat.roughness_override, Some(0.9));
        assert_eq!(mat.metallic_override, None);
    }

    #[test]
    fn test_gpu_conversion() {
        let mat = PaletteMaterial::new("stone").with_roughness(0.5);

        let gpu: MaterialPropertiesGpu = (&mat).into();

        assert_eq!(gpu.texture_scale, 1.0);
        assert_eq!(gpu.roughness_override, 0.5);
        assert!(gpu.metallic_override < 0.0);
    }

    #[test]
    fn test_material_properties_array() {
        let materials = vec![
            PaletteMaterial::new("grass"),
            PaletteMaterial::new("stone"),
            PaletteMaterial::new("dirt"),
        ];

        let array = MaterialPropertiesArray::from_materials(&materials);
        assert_eq!(array.materials.len(), 3);
    }

    #[test]
    #[should_panic(expected = "Too many materials")]
    fn test_too_many_materials() {
        let materials: Vec<PaletteMaterial> = (0..MAX_MATERIALS + 1)
            .map(|i| PaletteMaterial::new(format!("mat_{}", i)))
            .collect();

        MaterialPropertiesArray::from_materials(&materials);
    }
}
