pub mod asset;
pub mod builder;
pub mod plugin;
pub mod properties;
pub mod validation;

pub use asset::TexturePalette;
pub use builder::{PaletteBuilder, QuickPalette};
pub use properties::{
    MAX_MATERIALS, MaterialPropertiesArray, MaterialPropertiesGpu, PaletteMaterial,
};
pub use validation::PaletteValidationError;
