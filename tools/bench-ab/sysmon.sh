#!/usr/bin/env bash
# System sampler for tools/bench-ab: every $INTERVAL s, one line of
# whole-system CPU and RAM, GPU clock, every DRM client's GPU busy time
# (fdinfo drm-engine-*), the bench process's CPU and RSS, and on an
# integrated GPU the SoC power rail, GPU temperature and CPU clock.
#
#   sysmon.sh <out.log> <bench-pid-file>
#
# Stops once the pid in <bench-pid-file> has appeared and exited. Also
# snapshots every process's CPU time at start and end, to name the busiest
# other processes, and writes one `# invariants` header line recording the
# machine state a run's numbers are only meaningful against.
#
# Fields are read by key (see summary.awk), so new ones can be added, but
# `gpu` must stay last: its list can be empty, which would swallow whatever
# followed it.
#
# Env: INTERVAL (default 0.25 on <= 8 cores, else 0.5), GPU_DEVFREQ,
# GPU_HWMON (both probed if unset).
#
# The slow default used to be 1.0 on <= 8 cores, on the grounds that the
# sampler forks a few processes per tick and on 4 cores that is itself load.
# It is, and it counts as "other" in its own measurement -- but the Deck's
# short phases are 1.6 s (query) to 2.6 s (spin), so one tick a second gave
# them one or two samples, and `run.sh` was condemning whole runs on a single
# reading of a machine whose own idle floor is 8-11% of the CPU. Samples the
# verdict can stand on are worth more than the tick's own cost; it is small
# and it shows up in the `other` column, where it can be checked.
out=$1
pidfile=$2
ncpu=$(nproc 2>/dev/null || echo 1)
INTERVAL=${INTERVAL:-$([[ $ncpu -le 8 ]] && echo 1.0 || echo 0.5)}
GPU_DEVFREQ=${GPU_DEVFREQ:-$(ls -d /sys/class/devfreq/*gpu* 2>/dev/null | head -1)}

# An integrated GPU reports clock, power and temperature through amdgpu's
# hwmon instead of devfreq, which the Deck doesn't populate at all. The hwmon
# number isn't stable across boots, so glob it; only trust it when the DRM
# device really is amdgpu.
if [[ -z ${GPU_HWMON:-} ]]; then
  for dev in /sys/class/drm/card[0-9]*/device; do
    [[ $(readlink -f "$dev/driver" 2>/dev/null) == */amdgpu ]] || continue
    for h in "$dev"/hwmon/hwmon*; do
      [[ -r $h/freq1_input || -r $h/power1_average ]] && { GPU_HWMON=$h; break 2; }
    done
  done
fi
# freq1_input is Hz, like devfreq's cur_freq, so summary.awk's /1e6 holds.
GPU_FREQ_FILE=""
[[ -n $GPU_DEVFREQ && -r $GPU_DEVFREQ/cur_freq ]] && GPU_FREQ_FILE=$GPU_DEVFREQ/cur_freq
[[ -z $GPU_FREQ_FILE && -n ${GPU_HWMON:-} && -r $GPU_HWMON/freq1_input ]] &&
  GPU_FREQ_FILE=$GPU_HWMON/freq1_input
PW_FILE=""; TEMP_FILE=""; PW_CAP=0
if [[ -n ${GPU_HWMON:-} ]]; then
  [[ -r $GPU_HWMON/power1_average ]] && PW_FILE=$GPU_HWMON/power1_average
  [[ -r $GPU_HWMON/temp1_input ]] && TEMP_FILE=$GPU_HWMON/temp1_input
  [[ -r $GPU_HWMON/power1_cap ]] && PW_CAP=$(<"$GPU_HWMON/power1_cap")
fi
CPU_FREQ_FILE=/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
[[ -r $CPU_FREQ_FILE ]] || CPU_FREQ_FILE=""

read_or() { # read_or <file> -- 0 when there's nothing to read
  [[ -n $1 && -r $1 ]] && { read -r v <"$1" 2>/dev/null && echo "$v"; } || echo 0
}

# 1 on mains, 0 on battery -- and 1 on a machine with no battery at all,
# which is every desktop: there, no mains supply reports `online` either.
on_mains() {
  local p type any_battery=0
  for p in /sys/class/power_supply/*; do
    [[ -r $p/type ]] || continue
    read -r type <"$p/type"
    if [[ $type == Battery ]]; then
      any_battery=1
    elif [[ $(read_or "$p/online") == 1 ]]; then
      echo 1; return
    fi
  done
  [[ $any_battery == 1 ]] && echo 0 || echo 1
}

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
# What the run's numbers are only comparable against. summary.awk skips it.
{
  printf '# invariants interval=%s ncpu=%s' "$INTERVAL" "$ncpu"
  printf ' governor=%s' "$(read_or /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
  printf ' epp=%s' "$(read_or /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference)"
  printf ' dpm=%s' "$(read_or "${GPU_HWMON%/hwmon/*}/power_dpm_force_performance_level")"
  printf ' power_cap=%s' "$PW_CAP"
  printf ' mains=%s' "$(on_mains)"
  bat=$(cat /sys/class/power_supply/BAT*/capacity 2>/dev/null | head -1)
  printf ' battery=%s' "${bat:-none}"
  printf ' session=%s' "${XDG_SESSION_TYPE:-none}"
  printf ' gpu_freq=%s gpu_hwmon=%s\n' "${GPU_FREQ_FILE:-none}" "${GPU_HWMON:-none}"
} >>"$out"
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
  freq=$(read_or "$GPU_FREQ_FILE")
  pw=$(read_or "$PW_FILE")
  temp=$(read_or "$TEMP_FILE")
  cfreq=$(read_or "$CPU_FREQ_FILE")
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
  echo "$now cpu $u $n $s $i $w $q $sq $st mem $mt $ma freq $freq" \
    "power $pw $PW_CAP temp $temp cpufreq $cfreq" \
    "bench ${bench:-0} $bcpu ${brss:-0} gpu $gpu" >>"$out"
  sleep "$INTERVAL"
done
snap_procs >"$out.procs_end"
