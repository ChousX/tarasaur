#!/usr/bin/env bash
# One bench run with the system sampler alongside (tools/bench-ab).
#
#   tools/bench-ab/run.sh <bench-binary> <label> [bench args]
#
# Keeps the session from idling, timestamps the bench's output, and writes
# to $BENCH_AB_DIR (default target/bench-ab)/runs/<label>.*: bench.log,
# sys.log and sys.summary, a per-phase table of system CPU, GPU and RAM, the
# bench's share of each, the busiest other processes and a verdict. Exit
# status: the bench's, or 3 when other load disturbed a measured phase
# (others over 15% CPU or 5% GPU).
set -u
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
work=${BENCH_AB_DIR:-$repo/target/bench-ab}
bin=$(realpath "$1"); label=$2; shift 2
mkdir -p "$work/runs"
base=$work/runs/$label
pidfile=$base.pid
: >"$pidfile"

"$here/sysmon.sh" "$base.sys.log" "$pidfile" &
mon=$!
sleep 1.5 # idle samples before the bench starts

cd "$repo" || exit 1
( systemd-inhibit --what=idle:sleep --who=tarasaur-bench --why="benchmark run" \
    bash -c 'echo $$ >"$0"; exec "$@"' "$pidfile" "$bin" --label "$label" "$@" 2>&1
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

verdict=$(awk '
  $1 ~ /^(hold|spin|fly|edit|visibility)$/ && ($5 > 15 || $9 > 5) { bad = bad " " $1 "(cpu " $5 " gpu " $9 ")" }
  /kcompactd|kswapd/ { note = note " " $2 "=" $1 "s" }
  END { print (bad == "" ? "clean" : "DISTURBED:" bad) (note == "" ? "" : " [reclaim:" note "]") }' "$base.sys.summary")
echo "verdict: $verdict" >>"$base.sys.summary"
cat "$base.sys.summary"
grep -E "^[0-9]+ (summary|frames):" "$base.bench.log" | sed 's/^[0-9]* //'
echo "bench exit: $status"
[[ $status != 0 ]] && exit "${status:-1}"
[[ $verdict != clean* ]] && exit 3
exit 0
