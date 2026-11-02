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
#   DISTURB_GPU_PCT  other-process GPU that spoils a phase (5)
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
if [[ ${BENCH_PIDFILE_OWNER:-} == cmd ]]; then
  launch=(env "BENCH_PIDFILE=$pidfile" "$bin" --label "$label" --out "$results" "$@")
else
  launch=(bash -c 'echo $$ >"$0"; exec "$@"' "$pidfile" "$bin" --label "$label" --out "$results" "$@")
fi
( systemd-inhibit --what=idle:sleep --who=tarasaur-bench --why="benchmark run" \
    "${launch[@]}" 2>&1
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

verdict=$(awk \
  -v dcpu="${DISTURB_CPU_PCT:-15}" -v dgpu="${DISTURB_GPU_PCT:-5}" \
  -v tmax="${THROTTLE_DEGC:-92}" '
  $1 ~ /^(hold|spin|fly|edit|visibility)$/ {
    if ($5 > dcpu || $9 > dgpu) bad = bad " " $1 "(cpu " $5 " gpu " $9 ")"
  }
  # Thermal clamping only. At 15 W this hardware is *meant* to be power
  # limited -- the CPU and GPU share one rail, so a dip in GPU clock during
  # the CPU-bound dig is correct behaviour, not a spoiled run. Rejecting on
  # power would reject every honest run and quietly redefine the target
  # against hardware that does not exist.
  $1 ~ /^(hold|spin|fly|edit|visibility|dig)$/ && $19 + 0 >= tmax {
    hot = hot " " $1 "(" $19 "C)"
  }
  /kcompactd|kswapd/ { note = note " " $2 "=" $1 "s" }
  END {
    print (bad == "" ? "" : "DISTURBED:" bad) (hot == "" ? "" : " THROTTLED:" hot) \
      (note == "" ? "" : " [reclaim:" note "]")
  }' "$base.sys.summary")

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
    END {
      for (ph in p50) {
        if (p50[ph] >= r - 0.3 && p50[ph] <= r + 0.5 && p95[ph] < r + 1.3)
          bad = bad " " ph "(p50 " p50[ph] " p95 " p95[ph] ")"
      }
      if (bad != "") printf "CAPPED:%s", bad
      if (to) printf " LOAD_TIMEOUT"
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
