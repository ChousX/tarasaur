//! Skips chunks that can't produce geometry: only chunks tagged
//! `HasSurface` (a sign change in their padded volume, or a visibility
//! boundary through their solid) are extracted.

use bevy::platform::collections::HashSet;

use bevy::prelude::*;

use crate::{
    chunk::{Chunk, ChunkManager, ChunkPosition},
    field::{SDF, VisibilityField},
    grid::NEIGHBORS_MASK,
};

/// Present on chunks whose padded volume can produce geometry: a sign
/// change in the `SDF`, or solid cut by the `VisibilityField`'s boundary
/// (the cap). Only these are extracted and meshed.
#[derive(Component, Default)]
pub struct HasSurface;

/// One chunk's signs: the `SDF`'s uniform sign (`None` = mixed) and the
/// visibility's (`None` = mixed; absent = all visible).
type Signs = (Option<bool>, Option<bool>);

fn signs(sdf: &SDF, visibility: Option<&VisibilityField>) -> Signs {
    // A mask (the `visibility-mask` feature) makes no cap: the SDF alone
    // decides.
    let visibility = visibility.filter(|_| !cfg!(feature = "visibility-mask"));
    (
        sdf.uniform_sign(),
        visibility.map_or(Some(true), VisibilityField::uniform),
    )
}

/// Conservative: uses each neighbour's whole-volume signs, not just the 2
/// apron layers. Missing neighbours are ignored, because the apron falls
/// back to this chunk's own edge data (same signs as its volume).
///
/// The visibility boundary only makes a surface through solid: in air,
/// `max(sdf, visibility)` stays outside either way.
fn chunk_has_surface(own: Signs, mut neighbours: impl Iterator<Item = Signs>) -> bool {
    let (sdf, vis) = own;
    let Some(inside) = sdf else {
        return true;
    };
    let mut any_solid = inside;
    let mut vis_mixed = vis.is_none();
    for (n_sdf, n_vis) in &mut neighbours {
        if n_sdf != Some(inside) {
            return true; // the SDF crosses zero between them
        }
        any_solid |= n_sdf == Some(true);
        vis_mixed |= n_vis.is_none() || n_vis != vis;
    }
    any_solid && vis_mixed
}

/// What can change a chunk's flag: its SDF or visibility, or a
/// neighbour's.
#[derive(bevy::ecs::system::SystemParam)]
pub struct SurfaceInputs<'w, 's> {
    sdf_changed: Query<'w, 's, &'static ChunkPosition, (With<Chunk>, Changed<SDF>)>,
    vis_changed: Query<'w, 's, &'static ChunkPosition, (With<Chunk>, Changed<VisibilityField>)>,
    vis_removed: RemovedComponents<'w, 's, VisibilityField>,
    positions: Query<'w, 's, &'static ChunkPosition>,
}

/// A chunk's fields and whether it is flagged.
type ChunkFields = (
    &'static SDF,
    Option<&'static VisibilityField>,
    Has<HasSurface>,
);

/// Chunk `p` depends on `p + off` for each `off` in `NEIGHBORS_MASK`, so a
/// change at `q` dirties `q` and `q - off`.
pub fn update_surface_flags(
    mut inputs: SurfaceInputs,
    chunks: Query<ChunkFields, With<Chunk>>,
    chunk_manager: Res<ChunkManager>,
    mut dirty: Local<HashSet<IVec3>>,
    mut commands: Commands,
) {
    dirty.clear();
    let removed: Vec<IVec3> = inputs
        .vis_removed
        .read()
        .filter_map(|e| inputs.positions.get(e).ok().map(|p| p.0))
        .collect();
    let changed = inputs
        .sdf_changed
        .iter()
        .chain(inputs.vis_changed.iter())
        .map(|p| p.0)
        .chain(removed);
    for pos in changed {
        dirty.insert(pos);
        for off in NEIGHBORS_MASK {
            dirty.insert(pos.wrapping_sub(off));
        }
    }

    for &pos in &dirty {
        let Some(entity) = chunk_manager.get_chunk(&pos) else {
            continue;
        };
        let Ok((sdf, vis, has)) = chunks.get(entity) else {
            continue;
        };

        let neighbours = NEIGHBORS_MASK.iter().filter_map(|off| {
            let e = chunk_manager.get_chunk(&pos.wrapping_add(*off))?;
            chunks.get(e).ok().map(|(s, v, _)| signs(s, v))
        });
        let wants = chunk_has_surface(signs(sdf, vis), neighbours);

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

#[cfg(test)]
mod tests {
    use super::chunk_has_surface;

    #[test]
    fn solid_needs_a_visibility_boundary_to_have_a_surface() {
        let solid = Some(true);
        let air = Some(false);
        let visible = Some(true);
        let hidden = Some(false);
        // Plain solid rock, fully visible: nothing to mesh.
        assert!(!chunk_has_surface((solid, visible), [(solid, visible)].into_iter()));
        // The SDF crosses zero: always.
        assert!(chunk_has_surface((None, visible), std::iter::empty()));
        assert!(chunk_has_surface((solid, visible), [(air, visible)].into_iter()));
        // Rock cut by the visibility boundary, in it or at a neighbour.
        assert!(chunk_has_surface((solid, None), std::iter::empty()));
        assert!(chunk_has_surface((solid, visible), [(solid, hidden)].into_iter()));
        assert!(chunk_has_surface((solid, hidden), [(solid, None)].into_iter()));
        // Air with a visibility boundary through it: still nothing.
        assert!(!chunk_has_surface((air, None), [(air, hidden)].into_iter()));
        // Hidden rock next to hidden rock: nothing.
        assert!(!chunk_has_surface((solid, hidden), [(solid, hidden)].into_iter()));
    }
}
