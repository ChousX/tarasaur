#!/usr/bin/env bash
# Runs the bench (examples/bench.rs) on a Steam Deck and brings the results
# back. Run it from this checkout on an x86_64 host; the Deck itself needs
# nothing but ssh, rsync and its own Vulkan driver.
#
#   tools/bench-deck.sh [label] [-- bench args...]   # build, install, run, fetch
#   tools/bench-deck.sh install                      # build and install only
#   tools/bench-deck.sh package                      # build and lay out only
#
# The run happens in a transient systemd user unit on the Deck, not under the
# ssh that starts it. Tailscale's ssh server reaps the whole session when the
# connection goes, `setsid` and `nohup` included -- a detached run died three
# seconds in, taking its own exit status with it -- and a run long enough to
# matter is exactly the one a dropped connection kills.
#
# The run then says when it is finished, rather than being asked: its last act
# is to open a TCP connection back to this host over the tailnet, and we block
# on one `nc -l` until it arrives. Polling instead was both unreliable and
# self-defeating -- every check is an ssh login on the machine under test, and
# it lands in the `other` CPU column of whichever phase is running, which is
# the column the disturbance verdict reads. POLL remains as the fallback for
# when the rendezvous cannot be set up (no `tailscale ip`, no `nc`, no
# `socat` on the Deck).
#
# e.g. `tools/bench-deck.sh`, `tools/bench-deck.sh arenas -- --small-arenas`.
#
# SteamOS has no compiler, its root is read-only and its /home runs full, so
# nothing is built there: the binary is built here and shipped with the glibc
# and C++ runtime it was linked against (SteamOS's is older), taking Vulkan,
# the GPU driver and Wayland from SteamOS. tools/deck/bench.sh says how it
# runs: inside a nested gamescope at a fixed 1280x800, which is how a game
# runs there, with nothing pacing the frames.
#
# The bench loads no files at all (its shaders are embedded and its palette is
# generated), so the Deck gets no checkout: the commit only has to reach the
# result header, which `--commit` does. The tree must be clean so that header
# means something.
#
# Results land in bench_results/ as deck-<label>-<stamp>.{frames,summary}.csv,
# alongside toth's, for `bench --compare` and tools/bench-ab/. The package
# carries tools/bench-ab as ab/, so a run there can have the sampler and the
# verdicts alongside it:
#
#   ssh <deck> 'BENCH_AB_DIR=$HOME/r/ab BENCH_RESULTS=$HOME/r/results \
#     BENCH_PIDFILE_OWNER=cmd tarasaur-bench/ab/run.sh tarasaur-bench/bench <label>'
#
# BENCH_PIDFILE_OWNER=cmd matters: the launcher wraps the bench in gamescope,
# which forks, so the launcher records the measured pid rather than run.sh.
#
#   DECK     ssh target of the Deck       (steamdeck)
#   REMOTE   its package directory        (tarasaur-bench)
#   RUNDIR   its scratch/results parent   (tarasaur-run)
#   LABEL    result label                 (deck-<short sha>)
#   POLL     seconds between completion checks, when falling back (30)
#   NOTIFY_PORT  the port the Deck reports completion on (45987)
#   BENCH_BIN   package this prebuilt bench instead of building one. The
#               caller owns the build, which is what lets tools/bench-deck-ab.sh
#               lay out two variants without either one's cargo build
#               clobbering the other's binary in target/.
#   BENCH_PKG   where to lay the package out (target/deck/tarasaur-bench).
#               Two variants need two directories, since install rsyncs
#               --delete and would otherwise overwrite the first.
set -euo pipefail
cd "$(dirname "$0")/.."
DECK=${DECK:-steamdeck}
REMOTE=${REMOTE:-tarasaur-bench}
RUNDIR=${RUNDIR:-tarasaur-run}
PKG=${BENCH_PKG:-target/deck/tarasaur-bench}

# cargo comes from the devenv shell; outside it, borrow one.
if ! command -v cargo >/dev/null; then
	command -v direnv >/dev/null || { echo "no cargo and no direnv: enter the devenv shell" >&2; exit 1; }
	exec direnv exec . "$0" "$@"
fi

# Builds the bench and lays out $PKG: the binary, the loader and C/C++ runtime
# it was linked against, and the launcher.
package() {
	[ "$(uname -m)" = x86_64 ] || { echo "package: needs an x86_64 host" >&2; exit 1; }
	local bin
	if [ -n "${BENCH_BIN:-}" ]; then
		bin=$(realpath "$BENCH_BIN")
		[ -x "$bin" ] || { echo "package: no such bench binary: $BENCH_BIN" >&2; exit 1; }
	else
		cargo build --release --example bench
		bin=target/release/examples/bench
	fi
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
	# The A/B harness travels with the package: install rsyncs --delete, so
	# anything merely copied alongside gets wiped by the next install. It is
	# pure bash and awk, and the sampler has to run on the machine under test.
	mkdir -p "$PKG/ab"
	cp tools/bench-ab/*.sh tools/bench-ab/*.awk "$PKG/ab/"
	chmod +x "$PKG/ab"/*.sh
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

# Runs a command on the Deck in a transient systemd user unit and waits for
# it, so the run outlives the ssh that started it (see the header). Its output
# goes to a log on the Deck and is printed once it finishes, rather than
# streamed, because holding the connection open is the thing we're avoiding.
run_detached() {
	local unit=tarasaur-bench-$$
	# An absolute path, resolved here: the command is quoted with %q on its way
	# to `bash -lc`, so a `$HOME` in it would arrive as four literal characters
	# and the unit would write its log relative to wherever systemd put it.
	local home
	home=$(ssh "$DECK" 'echo $HOME')
	local log="$home/$RUNDIR/$unit.log"
	# Does the Deck have a way to call back, and do we have a way to listen?
	local port=${NOTIFY_PORT:-45987} ip='' send=''
	if command -v nc >/dev/null; then
		ip=$(tailscale ip -4 2>/dev/null | head -1)
		[ -n "$ip" ] && send=$(ssh "$DECK" 'command -v socat || command -v nc' |
			head -1)
	fi
	# `; :` so a failed callback can't change the unit's own exit status, and
	# the log is flushed before it fires -- it is what we read on waking.
	local tail=''
	[ -n "$send" ] && tail="; echo \"$unit done rc=\$?\" | \
		$send - TCP:$ip:$port 2>/dev/null; :"
	ssh "$DECK" "mkdir -p $home/$RUNDIR && systemd-run --user --unit=$unit \
		--collect --service-type=exec \
		bash -lc $(printf '%q' "{ $1 ; } > $log 2>&1$tail")" >&2
	if [ -n "$tail" ]; then
		echo "+ $unit started on $DECK; waiting for it to report on $ip:$port" >&2
		# One blocking listen, no traffic to the Deck at all until it answers.
		nc -l "$ip" "$port" >&2
	else
		echo "+ $unit started on $DECK; no callback available, polling every ${POLL:-30}s" >&2
		while ssh "$DECK" "systemctl --user is-active $unit >/dev/null 2>&1"; do
			sleep "${POLL:-30}"
		done
	fi
	ssh "$DECK" "cat $log; rm -f $log"
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
		run_detached "$REMOTE/bench --label $(printf '%q' "$label") \
			--commit $(git rev-parse --short HEAD) \
			--out \$HOME/$RUNDIR/results --world-dir \$HOME/$RUNDIR/world \
			$(printf '%q ' "$@")"
		mkdir -p bench_results
		rsync -a "$DECK:$RUNDIR/results/" bench_results/
		echo "results: bench_results/$label-*.summary.csv"
		;;
esac
