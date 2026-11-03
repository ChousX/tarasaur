//! The task-pool configuration tarasaur's edit path was measured with, for
//! an application that wants it.
//!
//! Nothing here is installed by [`TarasaurPlugin`](crate::TarasaurPlugin) and
//! no default changes: an application owns its task pools, the same way it
//! owns its camera's `Msaa`, and a library that reallocated threads behind
//! its back would fight any consumer running a pool policy of its own. This
//! module is the opt-in.
//!
//! # What it buys, and where that number comes from
//!
//! Bevy's default split gives the io and `async_compute` pools 25% of the
//! logical cores each (capped at 4) and leaves compute the rest. While
//! tarasaur is editing, the CSG and padding scopes are the critical path and
//! the other two pools sit idle, so compute is the pool that wants the cores.
//!
//! Measured on a **Steam Deck** (Van Gogh, 4 physical / 8 logical cores,
//! 15 W shared CPU+GPU rail) against the `dig` phase of `examples/bench.rs`,
//! three runs a side, rotated order, one binary and a flag as the only
//! difference:
//!
//! | compute threads | `dig` mean | spread |
//! |---|---|---|
//! | 6 (this module) | **29.52 ms** | 0.42% |
//! | 4 (Bevy's default) | **32.63 ms** | 0.94% |
//!
//! **+10.53% on the mean and +13.91% on p95**, against spreads under 1% and
//! with no overlap between the two groups.
//!
//! Three caveats worth reading before you expect that number:
//!
//! - **It is a `dig`-phase win, not a frame-rate win.** `dig` is the heavy
//!   continuous-editing phase and the only one on that hardware that badly
//!   misses 60 fps. Phases that are not editing have far less to gain, and
//!   whether they *lose* anything to the wider pool is still being measured.
//! - **It was measured on 4 physical cores.** On a machine with a lot of
//!   cores the default split already leaves compute plenty, and on one with
//!   very few there is nothing to redistribute. Measure on your own target.
//! - A `pool-*` glob over `bench_results/` returns +3.78% rather than
//!   +10.53%, because two unrelated run sets are both named `pool-A1`. The
//!   figure above is the Deck set alone, selected on its `commit=983e8ff`
//!   header field.
//!
//! # Use
//!
//! One line, replacing Bevy's own `TaskPoolPlugin`:
//!
//! ```no_run
//! # use bevy::prelude::*;
//! App::new().add_plugins(DefaultPlugins.set(tarasaur::task_pools::edit_task_pools()));
//! ```
//!
//! If you already build your own [`TaskPoolOptions`], take just the compute
//! policy and keep the rest of yours:
//!
//! ```no_run
//! # use bevy::prelude::*;
//! # let my_options = TaskPoolOptions::default();
//! let options = TaskPoolOptions {
//!     compute: tarasaur::task_pools::edit_compute_policy(),
//!     ..my_options
//! };
//! ```
//!
//! # What is promised, and what is not
//!
//! These functions return *the configuration tarasaur currently recommends
//! for editing*, not a fixed formula. The thread count is deliberately not
//! part of the contract: it is expected to change as it is measured on more
//! hardware, and a future version may widen only while an edit is in flight
//! rather than for the whole run. Call the function; do not hard-code what it
//! returns today.
//!
//! That is also why this is a **floor** rather than a pinned size. A floor
//! composes — it raises the minimum without capping what the pool may grow
//! to, so it neither fights an application that asked for more nor stops a
//! later policy from using the headroom. Bevy creates its pools once, at
//! `TaskPoolPlugin`, and they cannot be resized afterwards, so a dynamic
//! "widen while editing" policy can only ever mean *a pool with room in it,
//! plus tarasaur throttling its own submissions when idle*. The throttling
//! half belongs to tarasaur's systems; this half, the room, is what an
//! application has to grant. Neither call site below changes when that lands.

use std::num::NonZero;

use bevy::{
    app::{TaskPoolPlugin, TaskPoolThreadAssignmentPolicy},
    prelude::*,
};

/// How many compute threads the edit path wants: all but two logical cores,
/// and never fewer than one.
///
/// Two are left for the io and `async_compute` pools, which have real work to
/// do during chunk loading even though they idle mid-edit. The exact figure
/// is a measurement, not an invariant — see the module docs.
#[must_use]
pub fn edit_compute_threads() -> usize {
    std::thread::available_parallelism()
        .map_or(1, NonZero::get)
        .saturating_sub(2)
        .max(1)
}

/// Bevy's compute-pool policy with [`edit_compute_threads`] as its floor.
///
/// Only `min_threads` is set; `max_threads` and the pool's percentage stay
/// Bevy's, so this raises the minimum without capping the pool.
#[must_use]
pub fn edit_compute_policy() -> TaskPoolThreadAssignmentPolicy {
    TaskPoolThreadAssignmentPolicy {
        min_threads: edit_compute_threads(),
        ..TaskPoolOptions::default().compute
    }
}

/// A [`TaskPoolPlugin`] carrying [`edit_compute_policy`], to `set` on
/// `DefaultPlugins`. The io and `async_compute` pools keep Bevy's defaults.
#[must_use]
pub fn edit_task_pools() -> TaskPoolPlugin {
    TaskPoolPlugin {
        task_pool_options: TaskPoolOptions {
            compute: edit_compute_policy(),
            ..default()
        },
    }
}
