#!/usr/bin/env bash
# System sampler for tools/bench-ab: every $INTERVAL s (default 0.5), one
# line of whole-system CPU and RAM, GPU clock, every DRM client's GPU busy
# time (fdinfo drm-engine-*), and the bench process's CPU and RSS.
#
#   sysmon.sh <out.log> <bench-pid-file>
#
# Stops once the pid in <bench-pid-file> has appeared and exited. Also
# snapshots every process's CPU time at start and end, to name the busiest
# other processes.
out=$1
pidfile=$2
INTERVAL=${INTERVAL:-0.5}
GPU_DEVFREQ=${GPU_DEVFREQ:-$(ls -d /sys/class/devfreq/*gpu* 2>/dev/null | head -1)}

scan_clients() { grep -l '^drm-engine-' /proc/[0-9]*/fdinfo/* 2>/dev/null; }

snap_procs() {
  for s in /proc/[0-9]*/stat; do
    { read -r line <"$s"; } 2>/dev/null || continue
    # comm can contain spaces: split around the last ')'.
    pid=${line%% *}
    comm=${line#*(}; comm=${comm%)*}; comm=${comm// /_}
    rest=${line##*) }
    # shellcheck disable=SC2086
    set -- $rest
    echo "$pid $comm $((${12} + ${13}))"
  done
}

clients=$(scan_clients)
last_scan=$SECONDS
bench=""
: >"$out"
snap_procs >"$out.procs_start"

while :; do
  [[ -z $bench && -s $pidfile ]] && bench=$(<"$pidfile")
  if [[ -n $bench && ! -d /proc/$bench ]]; then break; fi
  # Pick up new GPU clients (the bench) every 2 s.
  if (( SECONDS - last_scan >= 2 )); then clients=$(scan_clients); last_scan=$SECONDS; fi

  now=$(date +%s%3N)
  read -r _ u n s i w q sq st _ </proc/stat
  mt=$(awk '/^MemTotal:/{print $2}' /proc/meminfo)
  ma=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
  freq=0
  [[ -n $GPU_DEVFREQ ]] && freq=$(<"$GPU_DEVFREQ/cur_freq")
  bcpu=0; brss=0
  if [[ -n $bench && -r /proc/$bench/stat ]]; then
    line=$(</proc/$bench/stat); rest=${line##*) }
    # shellcheck disable=SC2086
    set -- $rest
    bcpu=$((${12} + ${13}))
    brss=$(awk '/^VmRSS:/{print $2}' /proc/"$bench"/status 2>/dev/null)
  fi
  gpu=""
  for f in $clients; do
    p=${f#/proc/}; p=${p%%/*}
    vals=$(awk '/^drm-client-id:/{c=$2} /^drm-engine-/{g+=$2} /^drm-resident-memory:/{m=$2}
      END{if(c!="")print c":"g+0":"m+0}' "$f" 2>/dev/null)
    [[ -n $vals ]] && gpu+="$vals:$p,"
  done
  echo "$now cpu $u $n $s $i $w $q $sq $st mem $mt $ma freq $freq bench ${bench:-0} $bcpu ${brss:-0} gpu $gpu" >>"$out"
  sleep "$INTERVAL"
done
snap_procs >"$out.procs_end"
