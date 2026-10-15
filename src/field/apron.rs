//! Padded volumes: a chunk's voxels plus a `PADDING`-wide apron on its
//! +x/+y/+z sides, read from its neighbours, as the mesher and the GPU
//! queries consume them.

use bevy::prelude::*;

use crate::{
    chunk::{ChunkManager, ChunkPosition},
    field::{LOD, VoxelField},
    grid::{NEIGHBORS_MASK, PADDING, flatten_cube},
    num::f32_from_u32,
};

/// Fills the `+x/+y/+z` apron of a padded volume from the 7 neighbours in
/// `NEIGHBORS_MASK`, resampling across LODs.
///
/// A missing neighbour (or one mid-LOD-change, whose data doesn't match its
/// LOD) falls back to this chunk's own edge voxels. Every region writes only
/// apron voxels and reads only interior ones, so the regions are independent.
pub fn fill_apron<T>(
    vol: &mut [T::Elem],
    chunk_pos: IVec3,
    chunk_manager: &ChunkManager,
    query: &Query<(Entity, &ChunkPosition, &T, &LOD)>,
) where
    T: VoxelField,
{
    let Some((_, lod)) = field_at(chunk_pos, chunk_manager, query) else {
        return;
    };
    fill_apron_with::<T>(vol, lod.size(), |offset| {
        apron_source(chunk_pos.wrapping_add(offset), chunk_manager, query)
            .map(|(f, l)| (f.data_slice(), l.size()))
    });
}

fn field_at<'a, T: Component>(
    pos: IVec3,
    chunk_manager: &ChunkManager,
    query: &'a Query<(Entity, &ChunkPosition, &T, &LOD)>,
) -> Option<(&'a T, LOD)> {
    let (_, _, field, lod) = query.get(chunk_manager.get_chunk(&pos)?).ok()?;
    Some((field, *lod))
}

/// The neighbour at `pos` if the apron can read it: present, and its data
/// matching its LOD (not mid-LOD-change).
fn apron_source<'a, T: VoxelField>(
    pos: IVec3,
    chunk_manager: &ChunkManager,
    query: &'a Query<(Entity, &ChunkPosition, &T, &LOD)>,
) -> Option<(&'a T, LOD)> {
    field_at(pos, chunk_manager, query).filter(|(f, l)| f.data_slice().len() == l.volume())
}

/// The LODs pass3 needs to stitch seams, packed for `ChunkMeta::lod_ranks`:
///
/// - bits 0-1: this chunk's LOD rank;
/// - bits `2 + 2k`: the rank whose voxel units apron region `k`
///   (`NEIGHBORS_MASK` order) holds: the neighbour's, or this chunk's own
///   where the apron fell back to its edge;
/// - bits `16 + 2a`: the rank of the `-a` face neighbour if it is coarser
///   than this chunk, else this chunk's own.
///
/// Keep in sync with `fill_apron` (which neighbours it reads) and the
/// decoding in `surface_nets_pass3.wesl` and `cursor_query.wesl`.
#[must_use]
pub fn seam_lod_ranks<T: VoxelField>(
    chunk_pos: IVec3,
    lod: LOD,
    chunk_manager: &ChunkManager,
    query: &Query<(Entity, &ChunkPosition, &T, &LOD)>,
) -> u32 {
    let mut ranks = lod.rank();
    for (shift, offset) in (2u32..).step_by(2).zip(NEIGHBORS_MASK) {
        let source = apron_source(chunk_pos.wrapping_add(offset), chunk_manager, query)
            .map_or(lod, |(_, l)| l);
        ranks |= source.rank() << shift;
    }
    for (shift, axis) in (16u32..).step_by(2).zip([IVec3::X, IVec3::Y, IVec3::Z]) {
        let below = apron_source(chunk_pos.wrapping_sub(axis), chunk_manager, query)
            .map(|(_, l)| l)
            .filter(|l| l.size() < lod.size())
            .unwrap_or(lod);
        ranks |= below.rank() << shift;
    }
    ranks
}

/// [`fill_apron`] over plain data: `neighbor(offset)` gives that neighbour's
/// voxels and size, or `None` to fall back to this chunk's edge.
///
/// Same-LOD neighbours are copied as-is. Other LODs are resampled at this
/// chunk's spacing but keep the neighbour's voxel units, so SDF values never
/// clip at `SDF_BAND`; pass3 rescales them (see [`seam_lod_ranks`]).
pub fn fill_apron_with<'a, T>(
    vol: &mut [T::Elem],
    size: u32,
    neighbor: impl Fn(IVec3) -> Option<(&'a [T::Elem], u32)>,
) where
    T: VoxelField,
    T::Elem: 'a,
{
    let padded = size.wrapping_add(PADDING);
    let idx = |c: UVec3| flatten_cube(c.x, c.y, c.z, padded);
    let edge = UVec3::splat(size.saturating_sub(1));
    let size3 = UVec3::splat(size);

    for offset in NEIGHBORS_MASK {
        let dir = offset.as_uvec3();
        // Axes with dir = 1 cover the apron (n..n+PADDING), the rest 0..n.
        let origin = dir.wrapping_mul(size3);
        let end = size3.wrapping_add(dir.wrapping_mul(UVec3::splat(PADDING)));
        let source = neighbor(offset);
        for z in origin.z..end.z {
            for y in origin.y..end.y {
                for x in origin.x..end.x {
                    let cell = UVec3::new(x, y, z);
                    let local = cell.wrapping_sub(origin);
                    let value = match source {
                        // Same LOD: the sample lands exactly on a voxel, so
                        // read it. Equal to the interpolated path below (an
                        // exact lerp by 0), which costs 8 reads per voxel.
                        Some((data, n_size)) if n_size == size => T::rescale(
                            data.get(flatten_cube(local.x, local.y, local.z, n_size))
                                .copied()
                                .unwrap_or_default(),
                            1.0,
                        ),
                        // A coarser neighbour's 2 apron layers are its first
                        // 2 voxels, so a border cell spans one of its cells
                        // and pass3 can snap it to the coarse vertex.
                        Some((data, n_size)) => {
                            let ratio = f32_from_u32(n_size) / f32_from_u32(size);
                            let mut at = local.as_vec3() * ratio;
                            if n_size < size {
                                at = Vec3::select(dir.cmpeq(UVec3::ONE), local.as_vec3(), at);
                            }
                            T::sample_apron(data, n_size, at.x, at.y, at.z)
                        }
                        None => vol.get(idx(cell.min(edge))).copied().unwrap_or_default(),
                    };
                    if let Some(v) = vol.get_mut(idx(cell)) {
                        *v = value;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        field::SDF,
        field::VoxelField,
        grid::{NEIGHBORS_MASK, PADDING, flatten_cube},
        num::usize_from,
    };

    /// Each `NEIGHBORS_MASK` neighbour's voxels and size, if present.
    type Neighbors<E> = [Option<(Vec<E>, u32)>; 7];

    /// The original apron fill: always the interpolating `sample_apron`.
    fn reference_apron<T: VoxelField>(
        vol: &mut [T::Elem],
        size: u32,
        neighbors: &Neighbors<T::Elem>,
    ) {
        use crate::num::f32_from_u32;
        use bevy::math::{UVec3, Vec3};
        let padded = size.wrapping_add(PADDING);
        let idx = |c: UVec3| flatten_cube(c.x, c.y, c.z, padded);
        let edge = UVec3::splat(size.wrapping_sub(1));
        for (offset, n) in NEIGHBORS_MASK.iter().zip(neighbors) {
            let dir = offset.as_uvec3();
            let origin = dir.wrapping_mul(UVec3::splat(size));
            let end = UVec3::splat(size).wrapping_add(dir.wrapping_mul(UVec3::splat(PADDING)));
            for z in origin.z..end.z {
                for y in origin.y..end.y {
                    for x in origin.x..end.x {
                        let cell = UVec3::new(x, y, z);
                        vol[idx(cell)] = match n {
                            Some((data, n_size)) => {
                                let ratio = f32_from_u32(*n_size) / f32_from_u32(size);
                                let local = cell.wrapping_sub(origin).as_vec3();
                                // coarser: apron layers are its own voxels 0 and 1
                                let at = if *n_size < size {
                                    Vec3::select(dir.cmpeq(UVec3::ONE), local, local * ratio)
                                } else {
                                    local * ratio
                                };
                                T::sample_apron(data, *n_size, at.x, at.y, at.z)
                            }
                            None => vol[idx(cell.min(edge))],
                        };
                    }
                }
            }
        }
    }

    /// Deterministic noise in `0..bound` for test volumes.
    fn lcg(seed: &mut u64, bound: u32) -> u32 {
        *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        u32::try_from(*seed >> 33)
            .unwrap()
            .checked_rem(bound)
            .unwrap()
    }

    /// Fills a `size`^3 interior and the 7 neighbours, then checks the fast
    /// apron fill against the reference, compared as uploaded bytes.
    fn check_apron<T: VoxelField>(
        size: u32,
        neighbor_sizes: [Option<u32>; 7],
        value: impl Fn(&mut u64) -> T::Elem,
        bytes: impl Fn(&[T::Elem]) -> Vec<u8>,
    ) {
        let mut seed = u64::from(size).wrapping_mul(7919);
        let padded_size = size.wrapping_add(PADDING);
        let mut interior = vec![T::Elem::default(); usize_from(padded_size).pow(3)];
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    interior[flatten_cube(x, y, z, padded_size)] = value(&mut seed);
                }
            }
        }
        let neighbors: Neighbors<T::Elem> = neighbor_sizes.map(|s| {
            s.map(|n| {
                let data = (0..usize_from(n).pow(3)).map(|_| value(&mut seed));
                (data.collect(), n)
            })
        });

        let mut expected = interior.clone();
        reference_apron::<T>(&mut expected, size, &neighbors);
        let mut actual = interior;
        super::fill_apron_with::<T>(&mut actual, size, |offset| {
            let i = NEIGHBORS_MASK.iter().position(|o| *o == offset)?;
            neighbors[i].as_ref().map(|(d, n)| (d.as_slice(), *n))
        });
        assert_eq!(
            bytes(&actual),
            bytes(&expected),
            "size {size}, neighbours {neighbor_sizes:?}"
        );
    }

    #[test]
    fn fast_apron_matches_interpolated_apron() {
        use crate::{field::VisibilityField, num::f32_from_u32};
        // Equal LOD (fast path), coarser/finer (interpolated) and missing
        // neighbours, mixed.
        let layouts = [
            [Some(16); 7],
            [
                Some(16),
                Some(32),
                Some(4),
                None,
                Some(16),
                Some(64),
                Some(16),
            ],
            [None; 7],
        ];
        // Includes exact zeros of both signs: the fast path must not change
        // what the GPU gets for them.
        let sdf = |s: &mut u64| match lcg(s, 10) {
            0 => 0.0,
            1 => -0.0,
            _ => f32_from_u32(lcg(s, 1601)) / 100.0 - 8.0,
        };
        let id = |s: &mut u64| u8::try_from(lcg(s, 4)).unwrap();
        for layout in layouts {
            check_apron::<SDF>(16, layout, sdf, SDF::gpu_bytes);
            check_apron::<VisibilityField>(16, layout, id, <[u8]>::to_vec);
        }
    }
}
