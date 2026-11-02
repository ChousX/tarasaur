#!/usr/bin/env bash
# One command for an A/B measurement on the Steam Deck: build two variants,
# install them side by side, run them in a rotated order with the sampler on
# the Deck, reject and rerun a disturbed run, and print the same per-phase
# table tools/bench-ab/ab.sh prints locally.
#
#   tools/bench-deck-ab.sh <name> <A> <B> [pairs=3] [-- bench args...]
#
# <A> and <B> are each either a git ref (anything `git rev-parse` resolves:
# a sha, a branch, `HEAD~1`) or a path to a prebuilt bench binary. A ref is
# built in a detached worktree with its own CARGO_TARGET_DIR, so neither
# variant's build clobbers the other's or the working tree -- which is the
# part that made doing this by hand slow.
#
# A_ARGS and B_ARGS add bench flags to one variant only. Two of the things
# worth measuring on this hardware -- MSAA and the compute pool floor -- are
# flags rather than code, and a flag A/B is the better kind: the binary is
# byte-identical, so there is no build difference left to confound the result.
# Naming the same ref for both builds it once and installs it once.
#
# e.g.
#   tools/bench-deck-ab.sh cells HEAD~1 HEAD           # two builds
#   A_ARGS='--msaa 4' B_ARGS='--msaa 1' \
#     tools/bench-deck-ab.sh msaa HEAD HEAD            # one build, two flags
#   tools/bench-deck-ab.sh span target/a/bench target/b/bench
#
# Args given after `--` go to both variants; A_ARGS/B_ARGS are appended per
# variant on top of them.
#
# Why the defaults are what they are:
#
# - **Three pairs, not two.** The effects being chased here are 1-3%, the same
#   order as the run-to-run spread, and the spread is only meaningful with
#   three samples a variant. deck-improvements.md's method notes make this the
#   minimum and both published results used it.
# - **The full bench, not --quick.** `dig` is the phase every Deck conclusion
#   rests on and --quick stops at `visibility`. ab.sh defaults the other way
#   because locally the cheap phases are the interesting ones.
# - **Rotated order.** Pair 1 runs A then B, pair 2 B then A, pair 3 A then B.
#   Strict alternation already cancels monotonic drift; flipping the order
#   within the pair also stops either variant from always being the one that
#   runs on a warmer machine. Variant grouping is by argument position in
#   table.awk, so the run order is free.
# - **The binaries must differ.** Checked by md5 before anything is installed.
#   Comparing a variant against itself is the failure that looks exactly like
#   a null result, and it is an easy one to cause with a ref that didn't
#   rebuild.
#
# Never polls the Deck. Each run is a transient systemd user unit that reports
# its own exit status over TCP when it finishes (the mechanism bench-deck.sh's
# header explains), and the only ssh logins are before a run starts and after
# it has reported -- an ssh login during a run lands in the `other` CPU column
# that the disturbance verdict reads, and has rejected a run before.
#
# A run that grades DISTURBED (run.sh exit 3) is rerun, up to RETRIES times;
# the last attempt is kept either way and flagged in the log. A PRECONDITION
# failure (exit 4 -- on battery, a lowered TDP, a pinned clock) stops the
# experiment, because a retry cannot fix it.
#
#   DECK         ssh target of the Deck            (steamdeck)
#   RUNDIR       its scratch/results parent        (tarasaur-run)
#   NOTIFY_PORT  the port a run reports completion on (45987)
#   RUN_TIMEOUT  seconds to wait for a run to report (900)
#   RETRIES      attempts at a disturbed run       (3)
#   A_ARGS       extra bench flags for variant A only
#   B_ARGS       extra bench flags for variant B only
#   KEEP_BUILDS  don't delete the per-variant target dirs afterwards
set -euo pipefail
cd "$(dirname "$0")/.."
repo=$PWD
DECK=${DECK:-steamdeck}
RUNDIR=${RUNDIR:-tarasaur-run}
NOTIFY_PORT=${NOTIFY_PORT:-45987}
RUN_TIMEOUT=${RUN_TIMEOUT:-900}
RETRIES=${RETRIES:-3}

[ $# -ge 3 ] || { sed -n '2,8p' "$0" >&2; exit 2; }
name=$1; Aspec=$2; Bspec=$3; pairs=${4:-3}
shift $(( $# >= 4 ? 4 : 3 ))
[ "${1:-}" = "--" ] && shift
args=("$@")

work=$repo/target/deck-ab/$name
mkdir -p "$work/runs"
log=$work/log.txt
: >"$log"
say() { echo "$@" | tee -a "$log" >&2; }

# One ssh each for the facts we need about the Deck, taken before any run.
home=$(ssh "$DECK" 'echo $HOME')
send=$(ssh "$DECK" 'command -v socat || command -v nc' | head -1)
ip=$(tailscale ip -4 2>/dev/null | head -1)
[ -n "$send" ] && [ -n "$ip" ] && command -v nc >/dev/null || {
	echo "error: no completion callback available (deck socat/nc: ${send:-none}," \
		"tailscale ip: ${ip:-none}). Polling the Deck during a run spoils it," \
		"so this script will not fall back to it." >&2
	exit 1
}

# resolve <slot> <spec>: lays down a bench binary for <spec> and prints its
# path. A ref is built in its own worktree and target dir; a path is copied.
resolve() {
	# Separate statements on purpose: bash 5.3 declares every name in one
	# `local` before assigning any of them, so `out=...$slot` in the same
	# statement reads an unset `slot` and `set -u` kills the script.
	local slot=$1 spec=$2
	local out=$work/bench-$slot
	if [ -f "$spec" ] && [ -x "$spec" ]; then
		cp "$spec" "$out"
		say "+ $slot: prebuilt $spec"
	else
		local sha wt
		sha=$(git rev-parse --verify --quiet "$spec^{commit}") ||
			{ echo "error: $slot: '$spec' is neither a bench binary nor a git ref" >&2; exit 1; }
		wt=$work/src-$slot
		rm -rf "$wt"
		git worktree prune
		say "+ $slot: building $spec ($(git rev-parse --short "$sha"))"
		git worktree add --detach -f "$wt" "$sha" >&2
		# direnv runs in the main checkout, whose .envrc is the allowed one;
		# the worktree supplies only the manifest.
		CARGO_TARGET_DIR=$work/target-$slot \
			direnv exec "$repo" cargo build --release --example bench \
			--manifest-path "$wt/Cargo.toml" >&2
		cp "$work/target-$slot/release/examples/bench" "$out"
		git worktree remove --force "$wt"
		echo "$slot $spec $(git rev-parse --short "$sha")" >>"$work/refs.txt"
	fi
	echo "$out"
}

# deck_run <label> <slot>: starts one run and blocks until it reports. Prints
# run.sh's exit status on stdout; everything else goes to stderr.
deck_run() {
	local label=$1 slot=$2
	local unit=tarasaur-ab-$label
	local pkg; pkg=$(pkg_for "$slot")
	local dlog="$home/$RUNDIR/$unit.log"
	local cmd="BENCH_AB_DIR=$home/$RUNDIR/ab BENCH_RESULTS=$home/$RUNDIR/results"
	cmd="$cmd BENCH_PIDFILE_OWNER=cmd $home/tarasaur-bench-$pkg/ab/run.sh"
	cmd="$cmd $home/tarasaur-bench-$pkg/bench $label"
	# Provenance in the result header, per variant. bench-deck.sh gets this
	# from `git rev-parse HEAD` and so demands a clean tree; here each variant
	# is a committed ref or a hashed binary, so the stamp is exact whatever
	# the working tree is doing -- which is what makes iterating cheap.
	eval "local tag=\${${slot}_TAG:-}"
	[ -n "$tag" ] && cmd="$cmd --commit $(printf '%q' "$tag")"
	local a
	for a in ${args[@]+"${args[@]}"}; do cmd="$cmd $(printf '%q' "$a")"; done
	# Per-variant flags last, so they win over anything common.
	eval "local extra=\${${slot}_ARGS:-}"
	for a in $extra; do cmd="$cmd $(printf '%q' "$a")"; done
	# `; :` so a failed callback cannot change the unit's own exit status, and
	# the log is flushed before the callback fires.
	local wrapped="{ $cmd ; } > $dlog 2>&1; echo \"rc=\$?\" |"
	wrapped="$wrapped $send - TCP:$ip:$NOTIFY_PORT 2>/dev/null; :"
	# The listener goes up *before* the run starts. A run that fails in its
	# first second -- the launcher refusing on battery, a missing library --
	# fires its callback immediately, and a listener started afterwards would
	# miss it and then sit out the whole RUN_TIMEOUT for a failure already
	# known. `wait` blocks on it; nothing polls.
	local replyf=$work/$label.callback listener
	rm -f "$replyf"
	timeout "$RUN_TIMEOUT" nc -l "$ip" "$NOTIFY_PORT" >"$replyf" 2>/dev/null &
	listener=$!
	if ! ssh "$DECK" "mkdir -p $home/$RUNDIR && systemd-run --user --unit=$unit \
		--collect --service-type=exec bash -lc $(printf '%q' "$wrapped")" >&2; then
		kill "$listener" 2>/dev/null || true
		wait "$listener" 2>/dev/null || true
		echo "  $label: could not start the run on $DECK" >&2
		echo 98
		return 0
	fi
	echo "  $label started; waiting on $ip:$NOTIFY_PORT (no traffic to the Deck until it answers)" >&2
	wait "$listener" 2>/dev/null || true
	local reply rc=''
	reply=$(cat "$replyf" 2>/dev/null) || true
	rm -f "$replyf"
	case $reply in rc=*) rc=${reply#rc=}; rc=${rc//[^0-9]/};; esac
	if [ -z "$rc" ]; then
		# Only now, with nothing measuring, is it safe to ask the Deck.
		echo "  $label: no callback within ${RUN_TIMEOUT}s; asking the unit" >&2
		rc=$(ssh "$DECK" "systemctl --user show -p ExecMainStatus --value $unit \
			2>/dev/null; systemctl --user reset-failed $unit 2>/dev/null; :" |
			head -1 | tr -dc 0-9)
		[ -n "$rc" ] || rc=99
	fi
	ssh "$DECK" "cat $dlog 2>/dev/null; rm -f $dlog" >>"$work/$label.deck.log" 2>&1 || true
	# Bring back the results and the sampler's verdict for this run.
	rsync -a "$DECK:$RUNDIR/results/" "$repo/bench_results/" >&2 || true
	rsync -a "$DECK:$RUNDIR/ab/runs/" "$work/runs/" >&2 || true
	echo "$rc"
}

verdict() {
	local f=$work/runs/$1.sys.summary
	[ -f "$f" ] && sed -n 's/^verdict: //p' "$f" || echo "(no sys.summary)"
}

# Phases the sampler caught too few times to grade, and any memory reclaim.
# Neither fails a run, and both are worth seeing: a phase that goes unjudged
# run after run is a hole in the measurement rather than a clean result, and
# that is exactly how 04609e8's silently-unchanged sampler interval hid.
notes() {
	local f=$work/runs/$1.sys.summary
	[ -f "$f" ] && sed -n 's/^notes: /    /p' "$f" || true
}

# run_one <label> <slot>: deck_run, plus the reject-and-rerun policy.
run_one() {
	local label=$1 slot=$2 try rc n
	for try in $(seq 1 "$RETRIES"); do
		rc=$(deck_run "$label" "$slot")
		say "$label try $try rc=$rc $(verdict "$label")"
		n=$(notes "$label"); [ -n "$n" ] && say "$n"
		[ "$rc" = 4 ] && { say "$label: precondition failure, a retry won't fix it"; return 4; }
		case $rc in
			98) say "$label: the run would not start on $DECK"; return 5;;
			99) say "$label: no exit status came back; the run is unaccounted for"; return 5;;
		esac
		[ "$rc" = 3 ] || return 0
		sleep 10
	done
	say "$label: still disturbed after $RETRIES attempts; keeping the last, flagged above"
	return 0
}

# resolve() runs in a command substitution, so it cannot set a variable in
# this shell; it records a built ref in refs.txt and we read the tag back here.
# A prebuilt binary has no ref, so it is identified by its hash.
tag_for() { # tag_for <slot> <binary>
	if [ -f "$work/refs.txt" ] && grep -q "^$1 " "$work/refs.txt"; then
		awk -v s="$1" '$1 == s { t = $3 } END { print t }' "$work/refs.txt"
	else
		echo "bin-$(md5sum "$2" | cut -c1-8)"
	fi
}
A=$(resolve A "$Aspec")
if [ "$Aspec" = "$Bspec" ]; then
	B=$work/bench-B
	cp "$A" "$B"
	say "+ B: same as A ($Bspec), built once"
else
	B=$(resolve B "$Bspec")
fi
ma=$(md5sum "$A" | cut -d' ' -f1)
mb=$(md5sum "$B" | cut -d' ' -f1)
A_TAG=$(tag_for A "$A")
if [ "$Aspec" = "$Bspec" ]; then B_TAG=$A_TAG; else B_TAG=$(tag_for B "$B"); fi
export A_TAG B_TAG
say "+ A $A_TAG md5 $ma${A_ARGS:+  flags: $A_ARGS}"
say "+ B $B_TAG md5 $mb${B_ARGS:+  flags: $B_ARGS}"

# Identical binaries are a measurement only when the flags differ. Comparing
# a variant against itself is the failure that looks exactly like a null
# result, and a ref that quietly didn't rebuild causes it.
pkgB=B
if [ "$ma" = "$mb" ]; then
	[ "${A_ARGS:-}" = "${B_ARGS:-}" ] && {
		echo "error: A and B are the same binary with the same flags. That is a" \
			"null result by construction, not a measurement -- give different" \
			"refs, or different A_ARGS/B_ARGS." >&2
		exit 1
	}
	# One build, one package, one install: the flags are the whole difference.
	pkgB=A
	say "+ identical binaries, flags differ: no build difference to confound this"
fi
pkg_for() { case $1 in B) echo "$pkgB";; *) echo A;; esac; }

installed=''
for slot in A B; do
	pkg=$(pkg_for "$slot")
	case " $installed " in *" $pkg "*) continue;; esac
	installed="$installed $pkg"
	eval "bin=\$$pkg"
	say "+ installing $pkg as tarasaur-bench-$pkg on $DECK"
	BENCH_BIN=$bin BENCH_PKG=$work/pkg-$pkg REMOTE=tarasaur-bench-$pkg \
		tools/bench-deck.sh install >&2
done

say "+ $pairs pairs, rotated, full bench${args[0]+ with ${args[*]}}"
fail=0
for i in $(seq 1 "$pairs"); do
	if [ $((i % 2)) = 1 ]; then order="A B"; else order="B A"; fi
	for slot in $order; do
		run_one "$name-$slot$i" "$slot" || { fail=$?; break 2; }
	done
done
case $fail in
	4) say "+ stopped: the Deck was not in a fit state to measure"; exit 4;;
	5) say "+ stopped: a run could not be started or did not report"; exit 5;;
esac

sums() {
	local i
	for i in $(seq 1 "$pairs"); do
		ls -t "$repo/bench_results/$name-$1$i"-*.summary.csv 2>/dev/null | head -1
	done
}
{
	echo
	echo "== $name: A=$Aspec [$A_TAG md5 ${ma:0:8}]${A_ARGS:+ $A_ARGS}"
	echo "          B=$Bspec [$B_TAG md5 ${mb:0:8}]${B_ARGS:+ $B_ARGS}"
	if [ ${#args[@]} -eq 0 ]; then
		echo "   pairs=$pairs rotated, full bench, no common args"
	else
		echo "   pairs=$pairs rotated, common args: ${args[*]}"
	fi
	awk -F, -v pairs="$pairs" -f "$repo/tools/bench-ab/table.awk" $(sums A) $(sums B)
	echo "-- system (mean over runs: other cpu% / bench gpu% / other gpu% / GPU MHz / RAM GiB)"
	# dig and sculpt are in this list where ab.sh's stops at `edit`: on the
	# Deck the GPU clock during `dig` is half the evidence. A CPU saving on
	# the edit path should move the frame time *and* hand clock back, and the
	# clock column is the check that the mechanism is the one we think it is.
	for g in A B; do
		for i in $(seq 1 "$pairs"); do cat "$work/runs/$name-$g$i.sys.summary" 2>/dev/null; done |
			awk -v g="$g" '$1 ~ /^(hold|spin|fly|edit|sculpt|dig)$/ {
					k=$1; n[k]++; oc[k]+=$5; bg[k]+=$8; og[k]+=$9; mhz[k]+=$10; ram[k]+=$12 }
				END { for (k in n) printf "  %s %-7s other cpu %5.1f  bench gpu %5.1f  other gpu %4.1f  MHz %5.0f  RAM %5.2f\n",
					g, k, oc[k]/n[k], bg[k]/n[k], og[k]/n[k], mhz[k]/n[k], ram[k]/n[k] }' |
			sort -k2
	done
	echo "-- verdicts"
	sed -n "s/^$name-/  /p" "$log"
	# No screenshot diffs: the shots are on the Deck and `bench --diff` would
	# have to run there. ab.sh keeps them because its binaries are local.
} | tee -a "$log"

[ -n "${KEEP_BUILDS:-}" ] || rm -rf "$work"/target-A "$work"/target-B
echo "log: $log" >&2
