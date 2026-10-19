# Per-phase system summary for tools/bench-ab. First file: the timestamped
# bench log (phase ends from "[bench] <phase> done"); second: sysmon.sh's log.
# Columns: system CPU % (of all cores), the bench's share, others = system -
# bench; GPU busy % summed over DRM clients, the bench's, others'; GPU MHz;
# system RAM used (GiB), bench RSS (GiB), bench GPU resident (GiB).
FNR == NR {
  if (match($0, /\[bench\] [a-z_]+ done:/)) {
    s = substr($0, RSTART + 8, RLENGTH - 8); sub(/ done:/, "", s)
    np++; pname[np] = s; pend[np] = $1
  }
  next
}
{
  t = $1
  busy = $3 + $4 + $5 + $8 + $9 + $10; tot = busy + $6 + $7
  mt = $12; ma = $13; freq = $15; bpid = $17; bcpu = $18; brss = $19
  gs = $21; n = split(gs, cl, ",")
  gb = 0; go = 0; gm = 0
  for (k = 1; k <= n; k++) {
    if (cl[k] == "") continue
    split(cl[k], f, ":") # id:ns:resident_kib:pid
    id = f[1]
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
    }
  }
  have = 1; lt = t; ltot = tot; lbusy = busy; lbcpu = bcpu
}
END {
  printf "%-10s %4s %6s %6s %6s | %6s %6s %6s %6s | %6s %6s %6s %6s\n",
    "phase", "n", "cpu%", "bench", "other", "gpu%", "bench", "other", "MHz",
    "ramGiB", "ramMax", "rssMax", "gpuMem"
  for (k = 1; k <= no; k++) {
    p = order[k]; c = cnt[p]
    printf "%-10s %4d %6.1f %6.1f %6.1f | %6.1f %6.1f %6.1f %6.0f | %6.2f %6.2f %6.2f %6.2f\n",
      p, c, sc[p]/c, bc[p]/c, sc[p]/c - bc[p]/c, (ggb[p]+ggo[p])/c, ggb[p]/c, ggo[p]/c,
      fq[p]/c, mu[p]/c, mumax[p], rmax[p], gmmax[p]
  }
}
