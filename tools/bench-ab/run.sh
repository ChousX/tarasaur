#!/usr/bin/env bash
# One bench run with the system sampler alongside (tools/bench-ab).
#
#   tools/bench-ab/run.sh <bench-binary> <label> [bench args]
#
# Keeps the session from idling, timestamps the bench's output, and writes
# to $BENCH_AB_DIR (default target/bench-ab)/runs/<label>.*: bench.log,
# sys.log and sys.summary, a per-phase table of system CPU, GPU and RAM, the
# bench's share of each, the busiest other processes and a verdict.
#
# Exit status: the bench's; 3 when the run isn't a measurement (other load
# disturbed a measured phase, the GPU thermally clamped, or the frame rate was
# capped by a compositor instead of by our own cost); 4 when the machine wasn't
# in a fit state to start, which a retry won't fix.
#
#   BENCH_RESULTS    where the bench writes its CSVs (repo/bench_results)
#   BENCH_PIDFILE_OWNER=cmd
#                    the command records the measured pid itself, via the
#                    BENCH_PIDFILE we pass it. Needed when it wraps the bench
#                    in something that forks, as the Deck launcher's gamescope
#                    does -- otherwise we would time the wrapper.
#   DISTURB_CPU_PCT  other-process CPU that spoils a phase (15)
#   DISTURB_CPU_PCT_DIG
#                    the same for dig, which has the tightest floor and
#                    carries the most weight (10)
#   DISTURB_GPU_PCT  other-process GPU that spoils a phase (5)
#   DISTURB_MIN_SAMPLES
#                    samples a phase needs before it can be graded at all (3)
#   THROTTLE_DEGC    GPU temperature that spoils a phase (92)
set -u
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
work=${BENCH_AB_DIR:-$repo/target/bench-ab}
bin=$(realpath "$1"); label=$2; shift 2
results=${BENCH_RESULTS:-$repo/bench_results}
mkdir -p "$work/runs" "$results"
base=$work/runs/$label
pidfile=$base.pid
: >"$pidfile"

"$here/sysmon.sh" "$base.sys.log" "$pidfile" &
mon=$!
sleep 1.5 # idle samples before the bench starts

cd "$repo" || exit 1
# A run launched straight from here is presented by whatever desktop
# compositor is up; the Deck launcher overrides this with what it wrapped the
# bench in. Either way the header says, rather than leaving it unknowable.
export TARASAUR_PRESENT=${TARASAUR_PRESENT:-host}
if [[ ${BENCH_PIDFILE_OWNER:-} == cmd ]]; then
  launch=(env "BENCH_PIDFILE=$pidfile" "$bin" --label "$label" --out "$results" "$@")
else
  launch=(bash -c 'echo $$ >"$0"; exec "$@"' "$pidfile" "$bin" --label "$label" --out "$results" "$@")
fi
# Taking the sleep lock as well needs interactive polkit auth on SteamOS,
# which an ssh session can't give; idle alone is granted and is what matters.
inhibit=(systemd-inhibit --what=idle --who=tarasaur-bench --why="benchmark run")
"${inhibit[@]}" true 2>/dev/null || {
  echo "note: no idle lock available; the screen may blank mid-run" >&2
  inhibit=()
}
( "${inhibit[@]}" "${launch[@]}" 2>&1
  echo "EXIT $?" ) |
  while IFS= read -r line; do
    printf '%s %s\n' "$(date +%s%3N)" "$(sed 's/\x1b\[[0-9;]*m//g' <<<"$line")"
  done >"$base.bench.log"
wait "$mon"
status=$(awk '$2=="EXIT"{print $3}' "$base.bench.log" | tail -1)

awk -v NCPU="$(nproc)" -f "$here/summary.awk" "$base.bench.log" "$base.sys.log" >"$base.sys.summary"
echo "top other processes (cpu s over run):" >>"$base.sys.summary"
join -1 1 -2 1 <(sort -k1,1 "$base.sys.log.procs_start") <(sort -k1,1 "$base.sys.log.procs_end") |
  awk -v b="$(<"$pidfile")" '$1!=b {d=($5-$3)/100; if(d>0.05) printf "  %7.2f  %s\n", d, $2}' |
  sort -rn | head -8 >>"$base.sys.summary"

# Was the machine fit to start? A lowered TDP, a pinned clock or a run on
# battery makes the numbers incomparable, and retrying won't change it. Only
# checked where the files exist, so a desktop is unaffected.
pre=$(awk '/^# invariants/ {
    for (i = 2; i <= NF; i++) { split($i, kv, "="); v[kv[1]] = kv[2] }
    if (v["mains"] == 0) bad = bad " on-battery"
    if (v["power_cap"] != 0 && v["power_cap"] != 15000000) bad = bad " power_cap=" v["power_cap"]
    if (v["dpm"] != 0 && v["dpm"] != "auto") bad = bad " dpm=" v["dpm"]
    print (bad == "" ? "ok" : "PRECONDITION:" bad)
    exit
  }' "$base.sys.log")
[[ -n $pre ]] && echo "preconditions: $pre" >>"$base.sys.summary"

verdict_notes=$(awk \
  -v dcpu="${DISTURB_CPU_PCT:-15}" -v ddig="${DISTURB_CPU_PCT_DIG:-10}" \
  -v dgpu="${DISTURB_GPU_PCT:-5}" -v minn="${DISTURB_MIN_SAMPLES:-3}" \
  -v tmax="${THROTTLE_DEGC:-92}" '
  # Every measured phase, not just the steady ones: a dig disturbed by
  # another process used to grade clean, and dig is the phase we care most
  # about. load and shots are startup and noisy by nature.
  $1 ~ /^(hold|spin|fly|edit|sculpt|box|query|reslot|visibility|dig)$/ {
    # A phase the sampler caught once or twice is not evidence either way.
    # SteamOS idles at 8-11% of this CPU, so a lone sample sits near any
    # useful threshold about as often as not: nine of nine runs were once
    # rejected for a one-sample `spin`. Say the phase went unjudged instead
    # of throwing the run away over it.
    if ($2 + 0 < minn) { thin = thin " " $1 "(n=" $2 ")"; next }
    # dig gets a tighter bound, because its floor is the steadiest and
    # every conclusion rests on it. Measured over six Deck runs at this
    # interval, other-process CPU per phase was: dig 7.6-8.0, box 7.1-7.4,
    # query 7.0-8.0, fly 7.5-8.0, edit 8.0-8.3, sculpt 7.4-8.9, reslot
    # 7.6-8.7, visibility 7.7-8.8, hold 9.7-10.2, spin 9.2-11.3. So 15 clears
    # every floor and dig can afford 10: two points over its worst, and low
    # enough to catch the run that once measured 2.8 points above its floor,
    # came out 5% slow, and graded clean.
    #
    # These are floors, not thresholds, and they move with the sampling
    # interval and with how the run is launched -- a transient unit reads
    # ~2 points higher than plain ssh. Re-measure before trusting them, which
    # is the mistake that produced the previous value here.
    lim = ($1 == "dig" ? ddig : dcpu)
    if ($5 > lim || $9 > dgpu) bad = bad " " $1 "(cpu " $5 " gpu " $9 ")"
  }
  # Thermal clamping only. At 15 W this hardware is *meant* to be power
  # limited -- the CPU and GPU share one rail, so a dip in GPU clock during
  # the CPU-bound dig is correct behaviour, not a spoiled run. Rejecting on
  # power would reject every honest run and quietly redefine the target
  # against hardware that does not exist.
  $1 ~ /^(hold|spin|fly|edit|sculpt|visibility|dig)$/ && $19 + 0 >= tmax {
    hot = hot " " $1 "(" $19 "C)"
  }
  /kcompactd|kswapd/ { note = note " " $2 "=" $1 "s" }
  # Two lines: what fails the run, then what is merely worth knowing. They
  # were one string, so a run whose only remark was a reclaim note or an
  # unjudged phase did not start with "clean" and exited 3 -- a note could
  # fail a run. Notes are information; only DISTURBED and THROTTLED decide.
  END {
    print (bad == "" ? "" : "DISTURBED:" bad) (hot == "" ? "" : " THROTTLED:" hot)
    print (thin == "" ? "" : "unjudged:" thin) (note == "" ? "" : " reclaim:" note)
  }' "$base.sys.summary")
verdict=$(sed -n 1p <<<"$verdict_notes")
notes=$(sed -n 2p <<<"$verdict_notes")
notes=${notes# }
[[ -n $notes ]] && echo "notes: $notes" >>"$base.sys.summary"

# A compositor capping us at the display refresh would read as "exactly at the
# 60 fps target" -- the one failure that looks like success. See pool-ab.md:
# this silently invalidated ~29% of runs on a machine that beats the target
# comfortably, detected only by eye.
#
# Being *slower* than 60 fps is a measurement, not a spoiled run -- on the
# Deck most phases may well be -- so this can't be a one-sided threshold on
# p50, or every honest slow run gets thrown away. A cap looks different from
# slow: frames wait for vblank, so p50 sits just above one refresh period and
# the distribution is pinned there with almost no spread above it. So: p50 in
# a narrow band around the period AND a p95 that hasn't moved.
#
# REFRESH_MS is the period to watch for (16.67 = 60 Hz).
summary=$(ls -t "$results/$label"-*.summary.csv 2>/dev/null | head -1)
if [[ -n $summary ]]; then
  capped=$(awk -F, -v r="${REFRESH_MS:-16.67}" '
    $2 == "frame_ms_p50" { p50[$1] = $3 + 0 }
    $2 == "frame_ms_p95" { p95[$1] = $3 + 0 }
    $1 == "load" && $2 == "timed_out" && $3 + 0 != 0 { to = 1 }
    /^# / { quick = ($0 ~ /quick=true/) }
    $2 == "frames" { ran[$1] = 1 }
    END {
      for (ph in p50) {
        if (p50[ph] >= r - 0.3 && p50[ph] <= r + 0.5 && p95[ph] < r + 1.3)
          bad = bad " " ph "(p50 " p50[ph] " p95 " p95[ph] ")"
      }
      if (bad != "") printf "CAPPED:%s", bad
      if (to) printf " LOAD_TIMEOUT"
      # Results are written after every phase, so a run that died partway
      # leaves a summary that reads like a valid shorter one -- and the bench
      # can exit 0 doing it. The last phase has to be there.
      last = quick ? "visibility" : "dig"
      if (!(last in ran)) printf " TRUNCATED:no-%s", last
    }' "$summary")
  [[ -n $capped ]] && verdict="$verdict $capped"
else
  echo "note: no summary csv under $results for $label" >>"$base.sys.summary"
fi

verdict=${verdict# }
[[ -z $verdict ]] && verdict=clean
echo "verdict: $verdict" >>"$base.sys.summary"
cat "$base.sys.summary"
grep -E "^[0-9]+ (summary|frames):" "$base.bench.log" | sed 's/^[0-9]* //'
echo "bench exit: $status"
[[ $status != 0 ]] && exit "${status:-1}"
[[ $pre == PRECONDITION* ]] && exit 4
[[ $verdict != clean* ]] && exit 3
exit 0
