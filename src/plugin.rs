use bevy::prelude::*;

use crate::{chunk::ChunkPlugin, field::FieldsPlugin};

pub struct TarasaurPlugin;
impl Plugin for TarasaurPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((ChunkPlugin, FieldsPlugin));
    }
}
