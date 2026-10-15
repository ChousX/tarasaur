#!/usr/bin/env bash
# Turn a sandbox recording (F10) into a GIF.
#
#   tools/make-gif.sh captures/rec-<time> [out.gif] [width] [fps]
#
# Defaults: <dir>.gif, 800 px wide, 15 fps (the sandbox records at 15).
# Builds a palette from the whole clip first, so colors don't band.
set -euo pipefail

dir=${1:?usage: make-gif.sh <frames dir> [out.gif] [width] [fps]}
out=${2:-${dir%/}.gif}
width=${3:-800}
fps=${4:-15}

scale="scale=${width}:-1:flags=lanczos"
ffmpeg -loglevel error -y -framerate "$fps" -i "$dir/frame-%05d.png" \
    -vf "${scale},split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" \
    -loop 0 "$out"
echo "$out ($(du -h "$out" | cut -f1))"
