#!/usr/bin/env bash
# Alternating A/B bench runs (tools/bench-ab), each with the system sampler,
# rerunning a disturbed run up to twice, then a per-phase comparison (mean,
# spread between same-variant runs, change, flagged when it beats twice the
# spread and 1%), system load per variant, and screenshot diffs.
#
#   tools/bench-ab/ab.sh <name> <bench-A> <bench-B> [pairs=2] [-- bench args]
#
# Bench args default to --quick. Writes $BENCH_AB_DIR (default
# target/bench-ab)/exp/<name>.txt. Copy each variant's binary out of target/
# before building the next.
set -u
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
work=${BENCH_AB_DIR:-$repo/target/bench-ab}
name=$1; A=$(realpath "$2"); B=$(realpath "$3"); pairs=${4:-2}
shift $(( $# >= 4 ? 4 : 3 ))
[[ ${1:-} == -- ]] && shift
args=("$@"); [[ ${#args[@]} -eq 0 ]] && args=(--quick)
mkdir -p "$work/exp"
out=$work/exp/$name.txt
: >"$out"

run() { # run <bin> <label>: retries a disturbed run up to twice
  local bin=$1 label=$2 try rc
  for try in 1 2 3; do
    "$here/run.sh" "$bin" "$label" "${args[@]}" >/dev/null 2>&1
    rc=$?
    local v; v=$(grep '^verdict:' "$work/runs/$label.sys.summary" | cut -c10-)
    echo "$label try $try rc=$rc $v" | tee -a "$out"
    [[ $rc -ne 3 ]] && return $rc
    sleep 5
  done
  return 0 # still disturbed: keep it, flagged in the log
}

for i in $(seq 1 "$pairs"); do
  run "$A" "$name-A$i" || echo "A$i FAILED rc=$?" | tee -a "$out"
  run "$B" "$name-B$i" || echo "B$i FAILED rc=$?" | tee -a "$out"
done

cd "$repo"
sums() { for i in $(seq 1 "$pairs"); do ls -t bench_results/"$name-$1$i"-*.summary.csv 2>/dev/null | head -1; done; }
{
  echo; echo "== $name: A=$(basename "$A") B=$(basename "$B") pairs=$pairs args=${args[*]}"
  awk -F, -v pairs="$pairs" -f "$here/table.awk" $(sums A) $(sums B)
  echo "-- system (mean over runs: other cpu% / bench gpu% / other gpu% / GPU MHz / RAM GiB)"
  for g in A B; do
    for i in $(seq 1 "$pairs"); do cat "$work/runs/$name-$g$i.sys.summary"; done |
      awk -v g="$g" '$1 ~ /^(hold|spin|fly|edit)$/ { k=$1; n[k]++; oc[k]+=$5; bg[k]+=$8; og[k]+=$9; mhz[k]+=$10; ram[k]+=$12 }
        END { for (k in n) printf "  %s %-5s other cpu %5.1f  bench gpu %5.1f  other gpu %4.1f  MHz %5.0f  RAM %5.2f\n", g, k, oc[k]/n[k], bg[k]/n[k], og[k]/n[k], mhz[k]/n[k], ram[k]/n[k] }' | sort -k2
  done
  echo "-- screenshots (A1 vs B1; A1 vs A2 = noise floor)"
  mkdir -p "$work/exp/$name-heat"
  for v in 0 1 2 3; do
    a1=bench_results/shots/$name-A1-view$v.png
    b1=bench_results/shots/$name-B1-view$v.png
    a2=bench_results/shots/$name-A2-view$v.png
    printf "  view%s  A1/B1 %s" "$v" "$("$A" --diff "$a1" "$b1" "$work/exp/$name-heat/view$v.png" 2>&1 | tail -1)"
    [[ -f $a2 ]] && printf "   A1/A2 %s" "$("$A" --diff "$a1" "$a2" 2>&1 | tail -1)"
    echo
  done
} | tee -a "$out"
