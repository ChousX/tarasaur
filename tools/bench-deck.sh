#!/usr/bin/env bash
# Run the bench on a Steam Deck, from a checkout on the Deck itself.
#
#   tools/bench-deck.sh [label] [-- extra bench args]
#
# e.g. `tools/bench-deck.sh`, `tools/bench-deck.sh lod-a -- --msaa 1`.
# Works from Desktop Mode or over ssh (`ssh deck@steamdeck`): with no
# display in the environment it draws on the Deck's own screen (:0).
#
# Builds and runs inside a distrobox (preinstalled on SteamOS 3.5+), since
# SteamOS has no compiler or dev headers and its root is read-only. The first
# run creates an Arch container with Rust, Bevy's build deps and the Deck's
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

# Over ssh there is no display; use the one on the Deck's screen. distrobox
# passes these through to the container.
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
    export DISPLAY=:0
    export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
fi

if ! distrobox list --no-color 2>/dev/null | awk -F'|' '{gsub(/ /, "", $2); print $2}' | grep -qx "$box"; then
    echo "+ creating distrobox '$box' (first run only)" >&2
    # rustup: toolchain; alsa-lib, systemd-libs (udev), wayland, libxkbcommon,
    # libx*: Bevy's build and runtime deps; vulkan-radeon: the Deck's driver.
    distrobox create --yes --name "$box" \
        --image docker.io/library/archlinux:latest \
        --additional-packages "base-devel git rustup pkgconf alsa-lib systemd-libs
            wayland libxkbcommon libx11 libxcursor libxrandr libxi
            vulkan-icd-loader vulkan-radeon"
fi

# The container shares $HOME, so this toolchain lives in ~/.rustup.
distrobox enter "$box" -- sh -c 'rustup which cargo >/dev/null 2>&1 || rustup default stable'

# Note the power state: a Deck on battery or with a TDP cap benches slower.
if [[ -r /sys/class/power_supply/ACAD/online ]]; then
    [[ "$(cat /sys/class/power_supply/ACAD/online)" == 1 ]] || echo "warning: on battery" >&2
fi

cmd=(cargo run --release -j "${JOBS:-4}" --example bench -- --label "deck-$label" "$@")
echo "+ ${cmd[*]}  (in distrobox '$box')" >&2
distrobox enter "$box" -- "${cmd[@]}"
