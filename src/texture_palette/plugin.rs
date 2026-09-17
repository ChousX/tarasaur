//! Registers TexturePalette asset support and extracts the app's active
//! palette (if any) into the render world each frame.

use bevy::asset::Assets;
use bevy::prelude::*;
use bevy::render::{Extract, ExtractSchedule, RenderApp};

use super::asset::TexturePalette;
use super::properties::PaletteMaterial;

/// The end user's chosen palette. Insert this yourself — via
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
    pub fn new() -> Self {
        Self
    }
}

impl Plugin for PalettePlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<TexturePalette>();

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app.add_systems(ExtractSchedule, extract_palette);
    }
}

#[derive(Resource, Clone)]
pub struct ExtractedPalette {
    pub albedo: Handle<Image>,
    pub materials: Vec<PaletteMaterial>,
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
        materials: palette.materials.clone(),
    });
}
