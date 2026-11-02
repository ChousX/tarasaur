#!/usr/bin/env bash
# Runs the bench (examples/bench.rs) on a Steam Deck and brings the results
# back. Run it from this checkout on an x86_64 host; the Deck itself needs
# nothing but ssh, rsync and its own Vulkan driver.
#
#   tools/bench-deck.sh [label] [-- bench args...]   # build, install, run, fetch
#   tools/bench-deck.sh install                      # build and install only
#   tools/bench-deck.sh package                      # build and lay out only
#
# e.g. `tools/bench-deck.sh`, `tools/bench-deck.sh arenas -- --small-arenas`.
#
# SteamOS has no compiler, its root is read-only and its /home runs full, so
# nothing is built there: the binary is built here and shipped with the glibc
# and C++ runtime it was linked against (SteamOS's is older), taking Vulkan,
# the GPU driver and Wayland from SteamOS. tools/deck/bench.sh says how it
# runs, and renders inside a headless gamescope so no display session is
# needed and nothing paces the frames.
#
# The bench loads no files at all (its shaders are embedded and its palette is
# generated), so the Deck gets no checkout: the commit only has to reach the
# result header, which `--commit` does. The tree must be clean so that header
# means something.
#
# Results land in bench_results/ as deck-<label>-<stamp>.{frames,summary}.csv,
# alongside toth's, for `bench --compare` and tools/bench-ab/.
#
#   DECK     ssh target of the Deck       (steamdeck)
#   REMOTE   its package directory        (tarasaur-bench)
#   RUNDIR   its scratch/results parent   (tarasaur-run)
#   LABEL    result label                 (deck-<short sha>)
set -euo pipefail
cd "$(dirname "$0")/.."
DECK=${DECK:-steamdeck}
REMOTE=${REMOTE:-tarasaur-bench}
RUNDIR=${RUNDIR:-tarasaur-run}
PKG=target/deck/tarasaur-bench

# cargo comes from the devenv shell; outside it, borrow one.
if ! command -v cargo >/dev/null; then
	command -v direnv >/dev/null || { echo "no cargo and no direnv: enter the devenv shell" >&2; exit 1; }
	exec direnv exec . "$0" "$@"
fi

# Builds the bench and lays out $PKG: the binary, the loader and C/C++ runtime
# it was linked against, and the launcher.
package() {
	[ "$(uname -m)" = x86_64 ] || { echo "package: needs an x86_64 host" >&2; exit 1; }
	cargo build --release --example bench
	local bin=target/release/examples/bench
	lib() { ldd "$bin" | awk -v l="$1" '$1 == l { print $3 }'; }
	rm -rf "$PKG"
	mkdir -p "$PKG/bin" "$PKG/lib"
	cp "$bin" "$PKG/bin/bench"
	# Every glibc .so from the one the bench links, so the bundled loader has
	# the matching set rather than a mix with SteamOS's.
	cp -L "$(dirname "$(lib libc.so.6)")"/*.so* "$PKG/lib/"
	# The compiler runtime, when the bench links it (it has no C++ today).
	local l
	for l in libstdc++.so.6 libgcc_s.so.1; do
		[ -n "$(lib "$l")" ] && cp -L "$(lib "$l")" "$PKG/lib/"
	done
	cp tools/deck/bench.sh "$PKG/bench"
	chmod +x "$PKG/bench"
	audit
	du -sh "$PKG" >&2
}

# Everything the bench links that we aren't bundling has to already be on the
# Deck, where it comes from /usr/lib. Checking here turns "built fine, won't
# start" into a build-time error naming the library.
audit() {
	local missing=() soname
	while read -r soname; do
		[ -e "$PKG/lib/$soname" ] && continue
		missing+=("$soname")
	# linux-vdso is mapped by the kernel and has no file anywhere.
	done < <(ldd "$PKG/bin/bench" |
		awk '$1 ~ /\.so/ && $1 !~ /^\// && $1 !~ /^linux-vdso/ {print $1}')
	[ ${#missing[@]} -eq 0 ] && return 0
	echo "+ checking ${#missing[@]} system libraries on $DECK: ${missing[*]}" >&2
	local absent
	absent=$(ssh "$DECK" "for s in ${missing[*]}; do
		[ -e /usr/lib/\$s ] || echo \$s; done")
	[ -z "$absent" ] && return 0
	echo "error: $DECK has no $absent (and we aren't bundling it)" >&2
	exit 1
}

install_pkg() {
	rsync -a --delete "$PKG/" "$DECK:$REMOTE/"
}

case ${1:-} in
	package)
		package
		;;
	install)
		package
		install_pkg
		;;
	*)
		label=${LABEL:-deck-$(git rev-parse --short HEAD)}
		if [[ $# -gt 0 && "$1" != "--" ]]; then
			label=$1
			shift
		fi
		[[ "${1:-}" == "--" ]] && shift
		[ -z "$(git status --porcelain --untracked-files=no)" ] ||
			{ echo "commit first: the result header records this commit" >&2; exit 1; }
		package
		install_pkg
		# The scratch world goes on the Deck's own disk, not the build host's
		# path baked into the binary.
		ssh "$DECK" "$REMOTE/bench --label $(printf '%q' "$label") \
			--commit $(git rev-parse --short HEAD) \
			--out \$HOME/$RUNDIR/results --world-dir \$HOME/$RUNDIR/world \
			$(printf '%q ' "$@")"
		mkdir -p bench_results
		rsync -a "$DECK:$RUNDIR/results/" bench_results/
		echo "results: bench_results/$label-*.summary.csv"
		;;
esac
