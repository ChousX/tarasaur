//! Padded volumes: a chunk's voxels plus a `PADDING`-wide apron on its
//! +x/+y/+z sides, read from its neighbours, as the mesher and the GPU
//! queries consume them.

use bevy::prelude::*;

use crate::{
    chunk::{ChunkManager, ChunkPosition},
    field::{LOD, VoxelField},
    grid::{NEIGHBORS_MASK, PADDING, flatten_cube},
    num::{f32_from_u32, usize_from},
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

/// The LOD ranks pass3 needs to stitch seams, for `ChunkMeta::seam_ranks`:
/// this chunk's and each of its 26 neighbours', 2 bits each, 16 per word.
/// Neighbour `(dx, dy, dz)` is entry `(dx + 1) + 3 (dy + 1) + 9 (dz + 1)`,
/// so this chunk is entry 13.
///
/// A neighbour the apron can't read (missing, or mid-LOD-change) gets this
/// chunk's rank: the apron falls back to this chunk's own edge there, which
/// is what a same-LOD neighbour looks like. That also keeps a neighbour
/// loading in at this chunk's LOD from changing the ranks, and so from
/// re-meshing every chunk around it.
///
/// Keep in sync with `fill_apron` (which neighbours it reads) and the
/// decoding in `voxel_common.wesl`.
#[must_use]
pub fn seam_lod_ranks<T: VoxelField>(
    chunk_pos: IVec3,
    lod: LOD,
    chunk_manager: &ChunkManager,
    query: &Query<(Entity, &ChunkPosition, &T, &LOD)>,
) -> [u32; 2] {
    let mut ranks = [0u32; 2];
    for i in 0u32..27 {
        let offset = UVec3::new(i % 3, (i / 3) % 3, i / 9)
            .as_ivec3()
            .wrapping_sub(IVec3::ONE);
        // `apron_source` would read this chunk itself at offset 0.
        let rank = apron_source(chunk_pos.wrapping_add(offset), chunk_manager, query)
            .map_or(lod, |(_, l)| l)
            .rank();
        if let Some(word) = ranks.get_mut(usize_from(i / 16)) {
            *word |= rank << (i % 16).wrapping_mul(2);
        }
    }
    ranks
}

/// [`fill_apron`] over plain data: `neighbor(offset)` gives that neighbour's
/// voxels and size, or `None` to fall back to this chunk's edge.
///
/// Off the apron's axes, neighbours are resampled at this chunk's spacing.
/// Along an apron axis, the two layers are the face neighbour's boundary
/// and one step past it, where a step is this chunk's voxel or, if the face
/// neighbour across that axis is coarser, its voxel. So a coarser face
/// neighbour's layers are its first two voxels, and every region sharing
/// that axis (the edge and corner regions too) uses the same spacing:
/// pass3 then sees one grid per axis wherever its cells cross the boundary.
///
/// Values keep the neighbour's voxel units, so SDF values never clip at
/// `SDF_BAND`; pass3 rescales them (see [`seam_lod_ranks`]).
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
    // Per axis, this chunk's voxels per apron step (see above).
    let step = [IVec3::X, IVec3::Y, IVec3::Z].map(|axis| {
        neighbor(axis)
            .and_then(|(_, n_size)| size.checked_div(n_size))
            .map_or(1.0, |r| f32_from_u32(r.max(1)))
    });
    let step = Vec3::from_array(step);

    for offset in NEIGHBORS_MASK {
        let dir = offset.as_uvec3();
        let is_dir = dir.cmpeq(UVec3::ONE);
        // Axes with dir = 1 cover the apron (n..n+PADDING), the rest 0..n.
        let origin = dir.wrapping_mul(size3);
        let end = size3.wrapping_add(dir.wrapping_mul(UVec3::splat(PADDING)));
        let source = neighbor(offset);
        // Same LOD with unit steps: every sample lands exactly on a voxel.
        let direct = source.is_some_and(|(_, n_size)| n_size == size)
            && Vec3::select(is_dir, step, Vec3::ONE) == Vec3::ONE;
        let scale = Vec3::select(is_dir, step, Vec3::ONE);
        for z in origin.z..end.z {
            for y in origin.y..end.y {
                for x in origin.x..end.x {
                    let cell = UVec3::new(x, y, z);
                    let local = cell.wrapping_sub(origin);
                    let value = match source {
                        // Read it: equal to the interpolated path below (an
                        // exact lerp by 0), which costs 8 reads per voxel.
                        Some((data, n_size)) if direct => T::rescale(
                            data.get(flatten_cube(local.x, local.y, local.z, n_size))
                                .copied()
                                .unwrap_or_default(),
                            1.0,
                        ),
                        Some((data, n_size)) => {
                            let ratio = f32_from_u32(n_size) / f32_from_u32(size);
                            let at = local.as_vec3() * scale * ratio;
                            T::sample_apron(data, n_size, at.x, at.y, at.z)
                        }
                        None => extend_edge::<T>(vol, padded, cell, edge),
                    };
                    if let Some(v) = vol.get_mut(idx(cell)) {
                        *v = value;
                    }
                }
            }
        }
    }
}

/// The fallback for an apron voxel `cell` whose neighbour can't be read:
/// [`VoxelField::extend_edge`] from this chunk's nearest edge voxel. Reads
/// only interior voxels.
fn extend_edge<T: VoxelField>(vol: &[T::Elem], padded: u32, cell: UVec3, edge: UVec3) -> T::Elem {
    let at = |c: UVec3| {
        vol.get(flatten_cube(c.x, c.y, c.z, padded))
            .copied()
            .unwrap_or_default()
    };
    let base = cell.min(edge);
    let inward = [UVec3::X, UVec3::Y, UVec3::Z].map(|axis| at(base.saturating_sub(axis)));
    T::extend_edge(at(base), inward, cell.saturating_sub(edge))
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

    /// The apron fill done the slow way: always the interpolating
    /// `sample_apron`, with each apron axis stepping by the face neighbour's
    /// voxel where that is coarser.
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
        // NEIGHBORS_MASK starts with the +x, +y, +z faces.
        let step = |axis: usize| match &neighbors[axis] {
            Some((_, n_size)) if *n_size < size => size.checked_div(*n_size).map_or(1.0, f32_from_u32),
            _ => 1.0,
        };
        let step = Vec3::new(step(0), step(1), step(2));
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
                                let scale = Vec3::select(dir.cmpeq(UVec3::ONE), step, Vec3::ONE);
                                let at = local * scale * ratio;
                                T::sample_apron(data, *n_size, at.x, at.y, at.z)
                            }
                            None => super::extend_edge::<T>(vol, padded, cell, edge),
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

    /// A planar SDF carries on exactly into the apron of a missing
    /// neighbour, edges and corners included.
    #[test]
    fn missing_neighbour_extends_sdf() {
        use bevy::math::UVec3;
        let size = 8;
        let padded = size + PADDING;
        let plane = |x: u32, y: u32, z: u32| {
            UVec3::new(x, y, z)
                .as_vec3()
                .dot(bevy::math::Vec3::new(0.3, 0.8, -0.5))
                - 2.0
        };
        let mut vol = vec![0.0f32; usize_from(padded).pow(3)];
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    vol[flatten_cube(x, y, z, padded)] = plane(x, y, z);
                }
            }
        }
        super::fill_apron_with::<SDF>(&mut vol, size, |_| None);
        for z in 0..padded {
            for y in 0..padded {
                for x in 0..padded {
                    let got = vol[flatten_cube(x, y, z, padded)];
                    let want = plane(x, y, z);
                    assert!(
                        (got - want).abs() < 1e-4,
                        "({x}, {y}, {z}): {got} vs {want}"
                    );
                }
            }
        }
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
            // Same-LOD edge/corner regions beside coarser faces: their apron
            // axes step by the coarse voxel, so they can't take the fast path.
            [
                Some(4),
                Some(16),
                Some(8),
                Some(16),
                Some(16),
                Some(16),
                Some(16),
            ],
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
