//! A deterministic reproducer for the LOD-seam collapse behind
//! `fly/check:mesh_on_surface` (2.32 voxels off) and `hold`'s bad normal.
//!
//! The defect is in `surface_nets_pass3.wesl`, which no test can run, but
//! the part that decides it is pure integer logic on the 27 seam ranks:
//! `seam_depth_shift`, `seam_shift`, `seam_collapse` and the head of
//! `seam_cell`. This is a line-for-line port of those, fed the neighbour
//! LODs the bench logged at each failing vertex, so residency is pinned
//! exactly instead of left to whichever frame the probe catches.
//!
//! When two axes collapse, the shared cell is a segment on a chunk edge.
//! `relaxed_vertex` leaves step-0 axes where they are ("Flat (step 0) axes
//! stay put"), so the vertex lands on that edge however far the surface
//! is: that is the off-surface vertex and the bad normal. A port can
//! drift from the shader; keep it in step with pass3 by hand.

type V3 = [i32; 3];
type U3 = [u32; 3];

/// log2 of each chunk's voxels per edge around the meshing chunk, by
/// offset `(dx, dy, dz)` in -1..=1, as `seam_log2` reads them from the
/// ranks: 4 (Low), 5 (Medium), 6 (High).
struct Seam {
    log2: [i32; 27],
}

const LOW: i32 = 4;
const MEDIUM: i32 = 5;
const HIGH: i32 = 6;

impl Seam {
    /// Every neighbour at `own`, then `(offset, log2)` overrides.
    fn new(own: i32, others: &[(V3, i32)]) -> Self {
        let mut log2 = [own; 27];
        for &(o, l) in others {
            log2[index(o)] = l;
        }
        Self { log2 }
    }

    fn at(&self, o: V3) -> i32 {
        self.log2[index(o)]
    }

    fn own(&self) -> i32 {
        self.at([0; 3])
    }
}

fn index(o: V3) -> usize {
    let k = o[0]
        .wrapping_add(1)
        .wrapping_add(o[1].wrapping_add(1).wrapping_mul(3))
        .wrapping_add(o[2].wrapping_add(1).wrapping_mul(9));
    usize::try_from(k).unwrap()
}

fn add(a: V3, b: V3) -> V3 {
    [0, 1, 2].map(|i| a[i].wrapping_add(b[i]))
}

/// pass3 `below_offset`.
fn below_offset(i: u32) -> V3 {
    [i & 1, (i >> 1) & 1, (i >> 2) & 1].map(|b| b.cast_signed().wrapping_neg())
}

/// Whether below-offset `d` touches a cell starting at 0 on `zero`.
fn touches(d: V3, zero: [bool; 3]) -> bool {
    (0..3).all(|b| d[b] >= 0 || zero[b])
}

/// pass3 `seam_shift`.
fn seam_shift(seam: &Seam, o: V3, o_log2: i32, zero: [bool; 3]) -> i32 {
    (1..8)
        .map(below_offset)
        .filter(|&d| touches(d, zero))
        .map(|d| o_log2.wrapping_sub(seam.at(add(o, d))))
        .fold(0, i32::max)
}

/// pass3 `seam_depth_shift`.
fn seam_depth_shift(seam: &Seam, o: V3, o_log2: i32, id_o: U3) -> i32 {
    let mut s = 0;
    for d in (1..8).map(below_offset) {
        let sd = o_log2.wrapping_sub(seam.at(add(o, d)));
        if sd <= 0 {
            continue;
        }
        let reach = 1u32.wrapping_shl(sd.cast_unsigned());
        if (0..3).all(|a| id_o[a] < reach || d[a] == 0) {
            s = s.max(sd);
        }
    }
    s
}

/// pass3 `seam_collapse`, with `skip` dropping the below-neighbours a
/// candidate fix would no longer agree with.
fn seam_collapse(
    seam: &Seam,
    o: V3,
    o_log2: i32,
    zero: [bool; 3],
    s: i32,
    skip: impl Fn(V3) -> bool,
) -> [bool; 3] {
    let mut collapse = [false; 3];
    for d in (1..8).map(below_offset) {
        if !touches(d, zero) || skip(add(o, d)) {
            continue;
        }
        let l = seam.at(add(o, d));
        for b in 0..3 {
            if d[b] == 0 {
                continue;
            }
            let mut across = d;
            across[b] = 0;
            let seen = o_log2.wrapping_sub(l.min(seam.at(add(o, across))));
            if seen != s {
                collapse[b] = true;
            }
        }
    }
    collapse
}

/// The shared cell pass3's `seam_cell` builds for cell `id`, in the
/// meshing chunk's texels, or `None` where it keeps the cell's own.
#[derive(Debug, PartialEq, Eq)]
struct Shared {
    lo: U3,
    step: U3,
}

impl Shared {
    /// Axes the cell is flat on: the vertex cannot leave `lo` on them.
    fn flat(&self) -> usize {
        self.step.iter().filter(|&&s| s == 0).count()
    }
}

fn seam_cell(seam: &Seam, id: U3, skip: impl Fn(V3) -> bool) -> Option<Shared> {
    let own_log2 = seam.own();
    let n = 1u32.wrapping_shl(own_log2.cast_unsigned());
    let coarsest = seam.log2.iter().copied().fold(own_log2, i32::min);
    let reach = 1u32.wrapping_shl(own_log2.wrapping_sub(coarsest).cast_unsigned());
    if id.iter().all(|&c| c >= reach && c < n) {
        return None;
    }
    let o = id.map(|c| i32::from(c >= n));
    let o_log2 = seam.at(o);
    let up = own_log2.wrapping_sub(o_log2);
    let up_u = up.max(0).cast_unsigned();
    let down_u = up.wrapping_neg().max(0).cast_unsigned();
    let id_o: U3 = [0, 1, 2].map(|a| {
        if o[a] == 1 {
            0
        } else {
            id[a].wrapping_shr(up_u).wrapping_shl(down_u)
        }
    });
    let snap = |s: i32| {
        id_o.map(|c| {
            c.wrapping_shr(s.cast_unsigned())
                .wrapping_shl(s.cast_unsigned())
        })
    };
    let mut s = seam_depth_shift(seam, o, o_log2, id_o);
    let mut lo = snap(s);
    for _ in 0..3 {
        let s2 = seam_shift(seam, o, o_log2, lo.map(|c| c == 0));
        if s2 == s {
            break;
        }
        s = s2;
        lo = snap(s);
    }
    let collapse = seam_collapse(seam, o, o_log2, lo.map(|c| c == 0), s, skip);
    if s == 0 && up == 0 && !collapse.contains(&true) {
        return None;
    }
    let face_shift =
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]].map(|axis| own_log2.wrapping_sub(seam.at(axis)).max(0));
    let r = 1u32.wrapping_shl(s.cast_unsigned());
    let mut lo_t = [0u32; 3];
    let mut step = [0u32; 3];
    for a in 0..3 {
        if o[a] == 1 {
            lo_t[a] = n;
            step[a] = 1;
            if face_shift[a].wrapping_sub(own_log2) != s.wrapping_sub(o_log2) && !collapse[a] {
                return None;
            }
        } else {
            lo_t[a] = lo[a].wrapping_shl(up_u).wrapping_shr(down_u);
            step[a] = r.wrapping_shl(up_u).wrapping_shr(down_u);
        }
    }
    for a in 0..3 {
        if collapse[a] {
            step[a] = 0;
        }
    }
    Some(Shared { lo: lo_t, step })
}

/// A logged failing vertex: the meshing chunk's neighbourhood and the cell.
struct Site {
    name: &'static str,
    seam: Seam,
    id: U3,
    /// The shared cell pass3 builds today: its `lo` is the logged vertex.
    today: Shared,
    /// What the bench logged as `none` (no mesh in any arena) among the
    /// neighbours that matter here.
    none: &'static [V3],
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
