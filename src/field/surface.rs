use std::collections::HashSet;

use bevy::prelude::*;

use crate::{
    SDF,
    chunk::{Chunk, ChunkManager, ChunkPosition},
    voxel::systems::NEIGHBORS_MASK,
};

/// Present on chunks whose padded SDF volume contains a sign change, i.e.
/// chunks that can produce geometry. Only these are extracted and meshed.
#[derive(Component, Default)]
pub struct HasSurface;

/// Conservative: uses each neighbour's whole-volume sign, not just the 2
/// apron layers. Missing neighbours are ignored, because the apron falls
/// back to this chunk's own edge data (same sign as its volume).
fn chunk_has_surface<'a>(own: &SDF, mut neighbours: impl Iterator<Item = &'a SDF>) -> bool {
    let Some(sign) = own.uniform_sign() else {
        return true;
    };
    neighbours.any(|n| n.uniform_sign() != Some(sign))
}

/// Chunk `p` depends on `p + off` for each `off` in NEIGHBORS_MASK, so a
/// change at `q` dirties `q` and `q - off`.
pub fn update_surface_flags(
    changed: Query<&ChunkPosition, (With<Chunk>, Changed<SDF>)>,
    sdfs: Query<(&SDF, Has<HasSurface>), With<Chunk>>,
    chunk_manager: Res<ChunkManager>,
    mut dirty: Local<HashSet<IVec3>>,
    mut commands: Commands,
) {
    dirty.clear();
    for pos in &changed {
        dirty.insert(pos.0);
        for off in NEIGHBORS_MASK {
            dirty.insert(pos.0 - off);
        }
    }

    for &pos in dirty.iter() {
        let Some(entity) = chunk_manager.get_chunk(&pos) else {
            continue;
        };
        let Ok((sdf, has)) = sdfs.get(entity) else {
            continue;
        };

        let neighbours = NEIGHBORS_MASK.iter().filter_map(|off| {
            let e = chunk_manager.get_chunk(&(pos + *off))?;
            sdfs.get(e).ok().map(|(s, _)| s)
        });
        let wants = chunk_has_surface(sdf, neighbours);

        match (wants, has) {
            (true, false) => {
                commands.entity(entity).try_insert(HasSurface);
            }
            (false, true) => {
                commands.entity(entity).remove::<HasSurface>();
            }
            _ => {}
        }
    }
}

pub struct SurfaceCullPlugin;
impl Plugin for SurfaceCullPlugin {
    fn build(&self, app: &mut App) {
        // PostUpdate: after every Update-time edit/install, and its commands
        // flush before the next ExtractSchedule.
        app.add_systems(PostUpdate, update_surface_flags);
    }
}
