use super::{FieldLOD, FieldNew};
use crate::LOD;
use bevy::ecs::component::Mutable;
use bevy::prelude::*;

/// Whenever a chunk's LOD component changes — by any system, for any
/// reason — resets every out-of-sync field back to a correctly-sized,
/// empty instance at the new LOD. This is what makes LOD the single
/// source of truth: nothing else needs to know or care how a field's
/// internal size gets kept correct, it just always converges here.
///
/// Deliberately resets to empty rather than resampling old data —
/// real resize-and-resample (trilinear upsample / box-filter downsample)
/// is a real feature, not a bug fix, and belongs in its own pass. Chunks
/// with a registered generator (see ChunkGeneratorRegistry) get their
/// content back via the normal async regen path once resolve picks up
/// the freshly-inserted DirtyField-equivalent state; see note below.
pub fn sync_field_lod<F>(mut query: Query<(&LOD, &mut F), Changed<LOD>>)
where
    F: FieldLOD + FieldNew + Component<Mutability = Mutable>,
{
    for (lod, mut field) in query.iter_mut() {
        if field.lod() != *lod {
            *field = F::new_for_lod(*lod);
        }
    }
}
