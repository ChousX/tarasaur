//! Registers `TexturePalette` asset support and extracts the app's active
//! palette (if any) into the render world each frame.

use bevy::asset::Assets;
use bevy::prelude::*;
use bevy::render::{Extract, ExtractSchedule, RenderApp};

use super::asset::TexturePalette;
use super::mipmaps::{can_generate_mipmaps, generate_mipmaps};
use super::properties::PaletteMaterial;

/// The end user's chosen palette.
///
/// Insert this yourself — via
/// `commands.insert_resource(ActivePalette(handle))` — whenever your
/// palette handle becomes available. Commonly a Startup system, since
/// palettes are usually built procedurally rather than known at
/// app-construction time. Not set by `PalettePlugin` itself; that plugin
/// runs before any Startup system has had a chance to create a handle, so
/// it has nothing to insert on your behalf.
#[derive(Resource, Clone)]
pub struct ActivePalette(pub Handle<TexturePalette>);

/// Registers `TexturePalette` as an asset type and wires up extraction.
/// Does not provide a palette itself — see `ActivePalette`.
#[derive(Default)]
pub struct PalettePlugin;

impl PalettePlugin {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Plugin for PalettePlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<TexturePalette>();
        app.add_systems(PostUpdate, generate_palette_mipmaps);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app.add_systems(ExtractSchedule, extract_palette);
    }
}

#[derive(Resource, Clone)]
pub struct ExtractedPalette {
    pub albedo: Handle<Image>,
    pub arm: Option<Handle<Image>>,
    pub normal: Option<Handle<Image>>,
    pub materials: Vec<PaletteMaterial>,
}

/// Builds the missing mip chains of every palette that asks for them
/// (`TexturePalette::generate_mipmaps`), once each image has loaded.
fn generate_palette_mipmaps(
    palettes: Res<Assets<TexturePalette>>,
    mut images: ResMut<Assets<Image>>,
) {
    for (_, palette) in palettes.iter().filter(|(_, p)| p.generate_mipmaps) {
        let handles = [
            Some(&palette.albedo),
            palette.arm.as_ref(),
            palette.normal.as_ref(),
        ];
        for handle in handles.into_iter().flatten() {
            // Look before `get_mut`, which would mark the image changed.
            if images.get(handle).is_some_and(can_generate_mipmaps)
                && let Some(mut image) = images.get_mut(handle)
            {
                generate_mipmaps(&mut image);
            }
        }
    }
}

fn extract_palette(
    mut commands: Commands,
    active: Extract<Option<Res<ActivePalette>>>,
    palettes: Extract<Res<Assets<TexturePalette>>>,
) {
    let Some(active) = active.as_ref() else {
        return; // No ActivePalette inserted yet — dummy material stays in use.
    };
    let Some(palette) = palettes.get(&active.0) else {
        return; // Handle set but asset not loaded/inserted yet.
    };
    commands.insert_resource(ExtractedPalette {
        albedo: palette.albedo.clone(),
        arm: palette.arm.clone(),
        normal: palette.normal.clone(),
        materials: palette.materials.clone(),
    });
}
