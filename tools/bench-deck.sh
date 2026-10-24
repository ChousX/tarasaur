#!/usr/bin/env bash
# Run the bench on a Steam Deck, from a checkout on the Deck itself.
#
#   tools/bench-deck.sh [label] [-- extra bench args]
#
# e.g. `tools/bench-deck.sh`, `tools/bench-deck.sh lod-a -- --msaa 1`.
# Works from Desktop Mode or over ssh (`ssh deck@steamdeck`). The bench runs
# headless; only `--renderdoc` opens a window, on the Deck's own screen (:0)
# when there is no display in the environment.
#
# Builds and runs inside a distrobox (preinstalled on SteamOS 3.5+), since
# SteamOS has no compiler or dev headers and its root is read-only. The first
# run creates an Ubuntu container with Rust, Bevy's build deps and the Deck's
# Vulkan driver, which takes a few minutes. Results land in bench_results/ as
# deck-<label>-<timestamp>.{frames,summary}.csv.
#
# Env: JOBS (cargo build jobs, default 4: a full-parallel Bevy build can run
# the Deck out of memory), BOX (container name, default tarasaur),
# DISPLAY (default :0 when nothing is set).
set -euo pipefail

cd "$(dirname "$0")/.."

box="${BOX:-tarasaur}"
label="$(git rev-parse --short HEAD 2>/dev/null || echo run)"
if [[ $# -gt 0 && "$1" != "--" ]]; then
    label="$1"
    shift
fi
if [[ "${1:-}" == "--" ]]; then
    shift
fi

if ! command -v distrobox >/dev/null; then
    echo "error: distrobox not found (it ships with SteamOS 3.5+; update SteamOS)" >&2
    exit 1
fi

# Over ssh there is no display; `--renderdoc`'s window uses the one on the
# Deck's screen. distrobox passes these through to the container.
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
    export DISPLAY=:0
    export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
fi

box_exists() {
    distrobox list --no-color 2>/dev/null | awk -F'|' '{gsub(/ /, "", $2); print $2}' | grep -qx "$box"
}

# A container whose first-run setup failed can't be entered; start over.
if box_exists && ! distrobox enter "$box" -- true >/dev/null 2>&1; then
    echo "+ distrobox '$box' is broken (setup failed earlier); recreating it" >&2
    distrobox rm --force "$box"
fi

if ! box_exists; then
    echo "+ creating distrobox '$box' (first run only)" >&2
    # Ubuntu: distrobox's best-tested image (an Arch one failed to set up
    # with SteamOS's distrobox).
    # Bevy's build and runtime deps, plus the Deck's Vulkan driver (mesa).
    distrobox create --yes --name "$box" \
        --image docker.io/library/ubuntu:24.04 \
        --additional-packages "build-essential curl git pkg-config libasound2-dev
            libudev-dev libwayland-dev libxkbcommon-dev libx11-dev libxcursor-dev
            libxrandr-dev libxi-dev libvulkan1 mesa-vulkan-drivers"
    if ! distrobox enter "$box" -- true; then
        echo "error: setting up distrobox '$box' failed; see: podman logs $box" >&2
        exit 1
    fi
fi

# The container shares $HOME, so rustup installs into ~/.cargo and ~/.rustup.
distrobox enter "$box" -- sh -c '[ -x "$HOME/.cargo/bin/cargo" ] ||
    curl --proto =https --tlsv1.2 -sSf https://sh.rustup.rs |
    sh -s -- -y --profile minimal --no-modify-path'

# Note the power state: a Deck on battery or with a TDP cap benches slower.
if [[ -r /sys/class/power_supply/ACAD/online ]]; then
    [[ "$(cat /sys/class/power_supply/ACAD/online)" == 1 ]] || echo "warning: on battery" >&2
fi

cmd=(cargo run --release -j "${JOBS:-4}" --example bench -- --label "deck-$label" "$@")
echo "+ ${cmd[*]}  (in distrobox '$box')" >&2
distrobox enter "$box" -- sh -c 'PATH="$HOME/.cargo/bin:$PATH" exec "$@"' sh "${cmd[@]}"
