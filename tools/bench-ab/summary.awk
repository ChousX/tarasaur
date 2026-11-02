# Per-phase system summary for tools/bench-ab. First file: the timestamped
# bench log (phase ends from "[bench] <phase> done"); second: sysmon.sh's log.
# Columns: system CPU % (of all cores), the bench's share, others = system -
# bench; GPU busy % summed over DRM clients, the bench's, others'; GPU MHz;
# system RAM used (GiB), bench RSS (GiB), bench GPU resident (GiB); then the
# SoC power rail (W and % of its cap), GPU temperature and CPU clock, which
# only an integrated GPU reports (0 elsewhere).
#
# Sample lines are read by key, not by field number: sysmon.sh's trailing
# `gpu` list can be empty, so anything positional past it reads the wrong
# field the moment a key is added. A missing key leaves its value 0, which is
# what archived logs from before it existed do.
FNR == NR {
  if (match($0, /\[bench\] [a-z_]+ done:/)) {
    s = substr($0, RSTART + 8, RLENGTH - 8); sub(/ done:/, "", s)
    np++; pname[np] = s; pend[np] = $1
  }
  next
}
# sysmon.sh's invariants header, and anything else that isn't a sample.
$1 !~ /^[0-9]+$/ { next }
{
  t = $1
  u = n = s = i = w = q = sq = st = 0
  mt = ma = freq = pw = pwcap = temp = cfreq = bpid = bcpu = brss = 0
  gs = ""
  for (k = 2; k <= NF; k++) {
    if      ($k == "cpu")     { u = $(k+1); n = $(k+2); s = $(k+3); i = $(k+4)
                                w = $(k+5); q = $(k+6); sq = $(k+7); st = $(k+8); k += 8 }
    else if ($k == "mem")     { mt = $(k+1); ma = $(k+2); k += 2 }
    else if ($k == "freq")    { freq = $(k+1); k += 1 }
    else if ($k == "power")   { pw = $(k+1); pwcap = $(k+2); k += 2 }
    else if ($k == "temp")    { temp = $(k+1); k += 1 }
    else if ($k == "cpufreq") { cfreq = $(k+1); k += 1 }
    else if ($k == "bench")   { bpid = $(k+1); bcpu = $(k+2); brss = $(k+3); k += 3 }
    else if ($k == "gpu")     { gs = (k < NF) ? $(k+1) : ""; k += 1 }
  }
  busy = u + n + s + q + sq + st; tot = busy + i + w
  n_cl = split(gs, cl, ",")
  gb = 0; go = 0; gm = 0
  # One client appears once per open fd (the bench holds several), and the
  # entries are identical, so count each drm-client-id once per sample:
  # otherwise resident memory comes out multiplied by the fd count.
  delete seen
  for (k = 1; k <= n_cl; k++) {
    if (cl[k] == "") continue
    split(cl[k], f, ":") # id:ns:resident_kib:pid
    id = f[1]
    if (id in seen) continue
    seen[id] = 1
    if (id in last_g) d = f[2] - last_g[id]; else d = 0
    last_g[id] = f[2]
    if (d < 0) d = 0
    if (f[4] == bpid) { gb += d; gm += f[3] } else go += d
  }
  if (have) {
    dt = t - lt; dtot = tot - ltot
    if (dt > 0 && dtot > 0) {
      ph = "idle"
      if (bpid != 0) {
        ph = "post"
        for (k = 1; k <= np; k++) if (t <= pend[k]) { ph = pname[k]; break }
      }
      if (!(ph in cnt)) { order[++no] = ph }
      cnt[ph]++
      sc[ph] += 100 * (busy - lbusy) / dtot
      ncpu = NCPU + 0
      bc[ph] += 100 * (bcpu - lbcpu) / (dt / 10) / ncpu  # ticks are 10 ms
      ggb[ph] += 100 * gb / (dt * 1e6); ggo[ph] += 100 * go / (dt * 1e6)
      fq[ph] += freq / 1e6
      mu[ph] += (mt - ma) / 1048576
      if ((mt - ma) / 1048576 > mumax[ph]) mumax[ph] = (mt - ma) / 1048576
      if (brss / 1048576 > rmax[ph]) rmax[ph] = brss / 1048576
      if (gm / 1048576 > gmmax[ph]) gmmax[ph] = gm / 1048576
      wt[ph] += pw / 1e6                                      # uW -> W
      if (pwcap > 0) cap[ph] += 100 * pw / pwcap
      tp[ph] += temp / 1000                                   # mdegC -> degC
      if (temp / 1000 > tpmax[ph]) tpmax[ph] = temp / 1000
      cf[ph] += cfreq / 1000                                  # kHz -> MHz
    }
  }
  have = 1; lt = t; ltot = tot; lbusy = busy; lbcpu = bcpu
}
END {
  printf "%-10s %4s %6s %6s %6s | %6s %6s %6s %6s | %6s %6s %6s %6s | %6s %6s %6s %6s %7s\n",
    "phase", "n", "cpu%", "bench", "other", "gpu%", "bench", "other", "MHz",
    "ramGiB", "ramMax", "rssMax", "gpuMem", "W", "%cap", "degC", "degCmx", "cpuMHz"
  for (k = 1; k <= no; k++) {
    p = order[k]; c = cnt[p]
    printf "%-10s %4d %6.1f %6.1f %6.1f | %6.1f %6.1f %6.1f %6.0f | %6.2f %6.2f %6.2f %6.2f | %6.2f %6.0f %6.1f %6.1f %7.0f\n",
      p, c, sc[p]/c, bc[p]/c, sc[p]/c - bc[p]/c, (ggb[p]+ggo[p])/c, ggb[p]/c, ggo[p]/c,
      fq[p]/c, mu[p]/c, mumax[p], rmax[p], gmmax[p],
      wt[p]/c, cap[p]/c, tp[p]/c, tpmax[p], cf[p]/c
  }
}
