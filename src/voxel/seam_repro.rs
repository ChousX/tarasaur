//! A deterministic reproducer for the LOD-seam collapse behind
//! `fly/check:mesh_on_surface` (2.32 voxels off) and `hold`'s bad normal.
//!
//! The defect is in `surface_nets_pass3.wesl`, which no test can run, but
//! the part that decides it is pure integer logic on the 27 seam ranks:
//! `seam_depth_shift`, `seam_shift`, `seam_collapse` and the head of
//! `seam_cell`. `super::seam` is a line-for-line port of those; this feeds
//! it the neighbour LODs the bench logged at each failing vertex, so
//! residency is pinned exactly instead of left to whichever frame the probe
//! catches. The bench's `seam_gpu` test holds the shipped shader to the
//! same answers.
//!
//! When two axes collapse, the shared cell is a segment on a chunk edge.
//! `relaxed_vertex` leaves step-0 axes where they are ("Flat (step 0) axes
//! stay put"), so the vertex lands on that edge however far the surface
//! is: that is the off-surface vertex and the bad normal. A port can
//! drift from the shader; keep it in step with pass3 by hand.

use bevy::math::Vec3;

use super::seam::{Seam, SharedCell as Shared, pinned_cell, seam_cell as port_cell};

type V3 = [i32; 3];
type U3 = [u32; 3];

const LOW: i32 = 4;
const MEDIUM: i32 = 5;
const HIGH: i32 = 6;

/// The shared cell `super::seam` (the port) builds for cell `id`.
fn seam_cell(seam: &Seam, id: U3, skip: impl Fn(V3) -> bool) -> Option<Shared> {
    port_cell(seam, id, &skip)
}

/// A logged failing vertex: the meshing chunk's neighbourhood and the cell.
struct Site {
    name: &'static str,
    /// The meshing chunk.
    chunk: V3,
    seam: Seam,
    id: U3,
    /// The shared cell pass3 builds today: its `lo` is the logged vertex.
    today: Shared,
    /// What the bench logged as `none` (no mesh in any arena) among the
    /// neighbours that matter here.
    none: &'static [V3],
    /// The bench's `off N` for this vertex, if it logged one.
    off: Option<f32>,
}

/// The three logged sites. Neighbour LODs are from the bench's
/// `describe_neighbours` lists; `none` (no mesh) is given the meshing
/// chunk's rank, as `seam_lod_ranks` would if the chunk were unreadable.
fn sites() -> [Site; 3] {
    [
        // pool-A2, fly: chunk [8,0,4] Medium, local (8, 0, 32) of 32, 2.32
        // voxels off. [8,0,5] is Low; [8,-1,4] logged `none`.
        Site {
            name: "fly [8,0,4] Medium (8,0,32)",
            chunk: [8, 0, 4],
            off: Some(2.32),
            seam: Seam::new(MEDIUM, &[([0, 0, 1], LOW)]),
            id: [8, 0, 32],
            today: Shared {
                lo: [8, 0, 32],
                step: [2, 0, 0],
            },
            none: &[[0, -1, 0]],
        },
        // Deck fly CSV: chunk [8,0,11] Medium, local (32, 0, 16) of 32,
        // 2.07 voxels off (pool-A2's second site). Every chunk on the edge
        // has a mesh: [9,0,11] and [9,-1,11] Medium, [8,-1,11] Low.
        Site {
            name: "fly [8,0,11] Medium (32,0,16)",
            chunk: [8, 0, 11],
            off: Some(2.07),
            seam: Seam::new(
                MEDIUM,
                &[
                    ([-1, -1, -1], LOW),
                    ([-1, 0, -1], LOW),
                    ([-1, -1, 0], LOW),
                    ([0, -1, 0], LOW),
                    ([-1, 0, 0], LOW),
                    ([1, -1, 1], LOW),
                    ([-1, 0, 1], LOW),
                    ([0, 0, 1], LOW),
                    ([1, 0, 1], LOW),
                ],
            ),
            id: [32, 0, 16],
            today: Shared {
                lo: [32, 0, 16],
                step: [0, 0, 2],
            },
            none: &[],
        },
        // shapebound B1, hold: chunk [4,-1,4] High, local (64, 64, 0) of 64,
        // normal 34.97 degrees off. All eight chunks at that corner have a
        // mesh; the twelve `none`s are elsewhere.
        Site {
            name: "hold [4,-1,4] High (64,64,0)",
            chunk: [4, -1, 4],
            off: None,
            seam: Seam::new(
                HIGH,
                &[
                    ([1, 0, -1], MEDIUM),
                    ([1, 1, -1], MEDIUM),
                    ([1, 0, 0], MEDIUM),
                    ([1, 1, 0], MEDIUM),
                    ([1, 0, 1], MEDIUM),
                ],
            ),
            id: [64, 64, 0],
            today: Shared {
                lo: [64, 64, 0],
                step: [1, 0, 0],
            },
            none: &[],
        },
    ]
}

/// Today's behaviour, pinned: each logged vertex comes from a shared cell
/// flat on two axes, i.e. a segment on a chunk edge, starting exactly at
/// the logged position. The free axis is clamped onto a face too (finding
/// 11, section 1.3), which is why all three coordinates are whole.
#[test]
fn logged_sites_collapse_onto_a_chunk_edge() {
    for site in sites() {
        let cell = seam_cell(&site.seam, site.id, |_| false);
        assert_eq!(cell.as_ref(), Some(&site.today), "{}", site.name);
        assert_eq!(site.today.flat(), 2, "{}", site.name);
    }
}

/// `examples/common/mod.rs`'s `terrain_height`, the generated heightfield
/// the bench checks meshes against.
fn terrain_height(x: f32, z: f32) -> f32 {
    let mut h = 0.0;
    let mut amp = 6.0;
    let mut freq = 0.03;
    for _ in 0..4 {
        h = ((x * freq).sin() * (z * freq * 1.3).cos()).mul_add(amp, h);
        amp *= 0.5;
        freq *= 2.1;
    }
    h
}

/// The bench's `heightfield_distance` (`examples/bench.rs`), in world units.
fn heightfield_distance(x: f32, y: f32, z: f32) -> f32 {
    let e = 0.05;
    let gx = (terrain_height(x + e, z) - terrain_height(x - e, z)) / (2.0 * e);
    let gz = (terrain_height(x, z + e) - terrain_height(x, z - e)) / (2.0 * e);
    (y - terrain_height(x, z)) / gz.mul_add(gz, gx.mul_add(gx, 1.0)).sqrt()
}

/// The numbers themselves: a vertex at each fly site's pinned position is
/// exactly as far off the terrain as the bench logged, past its 2.0 limit.
#[test]
fn pinned_vertices_are_as_far_off_as_logged() {
    use crate::{
        chunk::CHUNK_SIZE,
        field::LOD,
        num::{f32_from_i32, f32_from_u32},
    };
    for site in sites() {
        let Some(logged) = site.off else { continue };
        let lod = if site.seam.own() == MEDIUM {
            LOD::Medium
        } else {
            LOD::High
        };
        let voxel = lod.voxel_size();
        let at = |a: usize| {
            f32_from_i32(site.chunk[a]).mul_add(CHUNK_SIZE, f32_from_u32(site.today.lo[a]) * voxel)
        };
        let off = heightfield_distance(at(0), at(1), at(2)).abs() / voxel;
        assert!(
            (off - logged).abs() < 0.005,
            "{}: {off} voxels, logged {logged}",
            site.name
        );
        assert!(off > 2.0, "{}: {off}", site.name);
    }
}

/// The candidate fix the notes favour: non-resident neighbours stop
/// forcing a collapse. Only the first site has one, so at best it fixes
/// that site, and only if [8,-1,4] is unreadable rather than resident at
/// Medium; the other two are untouched.
#[test]
fn excluding_unreadable_neighbours_leaves_two_sites_collapsed() {
    let flat: Vec<usize> = sites()
        .iter()
        .map(|site| {
            seam_cell(&site.seam, site.id, |o| site.none.contains(&o))
                .as_ref()
                .map_or(0, Shared::flat)
        })
        .collect();
    assert_eq!(flat, [0, 2, 2]);

    // And at the first site, [8,-1,4] resident at Medium (the LOD its
    // Chebyshev distance gives it beside [8,0,4]) collapses exactly as
    // the substituted rank does: residency is not the variable there.
    let [first, ..] = sites();
    let resident = Seam::new(MEDIUM, &[([0, 0, 1], LOW), ([0, -1, 0], MEDIUM)]);
    assert_eq!(
        seam_cell(&resident, first.id, |_| false),
        seam_cell(&first.seam, first.id, |_| false)
    );
    // Only resident at Low does it go away.
    let low = Seam::new(MEDIUM, &[([0, 0, 1], LOW), ([0, -1, 0], LOW)]);
    assert_ne!(
        seam_cell(&low, first.id, |_| false).map(|c| c.flat()),
        Some(2)
    );
}

/// The property a fix must give: no logged site's vertex is pinned to a
/// chunk edge. Fails today on all three, which is the defect.
#[test]
#[ignore = "reproduces the LOD-seam collapse defect: two-axis collapses pin all three logged vertices to chunk edges; un-ignore with the fix"]
fn no_logged_site_is_pinned_to_an_edge() {
    let pinned: Vec<&str> = sites()
        .iter()
        .filter(|site| seam_cell(&site.seam, site.id, |_| false).is_some_and(|c| c.flat() >= 2))
        .map(|site| site.name)
        .collect();
    assert!(pinned.is_empty(), "pinned to a chunk edge: {pinned:?}");
}

/// The classifier fires on each logged site's vertex, as the GPU meshes it,
/// on a cell flat where the port's is. (A vertex at a lattice point lies on
/// the end of one collapsed segment and the start of the next, so either
/// may be named; both pin it the same way.)
#[test]
fn classifier_finds_each_logged_vertex() {
    for site in sites() {
        let local = Vec3::from_array(site.today.lo.map(crate::num::f32_from_u32));
        let cell = pinned_cell(site.seam.ranks(), local);
        let Some(cell) = cell else {
            panic!("{}: not classified", site.name);
        };
        assert!(cell.flat() >= 2, "{}: {cell:?}", site.name);
        for a in 0..3 {
            if site.today.step[a] == 0 {
                assert_eq!(cell.step[a], 0, "{}: {cell:?}", site.name);
                assert_eq!(cell.lo[a], site.today.lo[a], "{}: {cell:?}", site.name);
            }
        }
    }
}

/// And on nothing else: the same spot a hair off its face, the same spot
/// in a chunk with no seam, and an ordinary vertex inside the chunk.
#[test]
fn classifier_needs_the_pinned_position_exactly() {
    for site in sites() {
        let at = Vec3::from_array(site.today.lo.map(crate::num::f32_from_u32));
        let ranks = site.seam.ranks();
        // Off a flat axis by 0.01 voxels: a solved vertex, not a pinned one.
        let flat = site.today.step.iter().position(|&s| s == 0).unwrap();
        let mut nudged = at;
        nudged[flat] += if at[flat] > 0.0 { -0.01 } else { 0.01 };
        assert_eq!(pinned_cell(ranks, nudged), None, "{}: nudged", site.name);
        // The same position with every neighbour at this chunk's LOD.
        let flat_world = Seam::new(site.seam.own(), &[]).ranks();
        assert_eq!(pinned_cell(flat_world, at), None, "{}: no seam", site.name);
        // Inside the chunk.
        assert_eq!(
            pinned_cell(ranks, Vec3::splat(7.3)),
            None,
            "{}: interior",
            site.name
        );
    }
}
