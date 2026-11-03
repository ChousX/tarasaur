# The per-phase A/B comparison table (tools/bench-ab). Shared by ab.sh, which
# compares two local builds, and tools/bench-deck-ab.sh, which compares two
# builds on the Deck -- the prompt for the Deck driver asked for "the same
# per-phase table ab.sh prints locally", so it is one file rather than two
# copies that drift.
#
#   awk -F, -v pairs=<n> -f table.awk <A summaries...> <B summaries...>
#
# Variant is decided by argument position, not by filename: the first `pairs`
# files are A and the rest B. That is what lets a caller run the variants in
# any order -- A/B/A, B/A/B, a rotation -- and still group them correctly.
#
# Spread is max-min as a percentage of the mean across same-variant runs, and
# a change is flagged only when it beats 1% *and* twice the larger spread.
# Reporting the spread is not decoration: the effects being chased here are
# 1-3%, which is the same order as the spread, and pool-ab.md's rejected runs
# were caught by it.
FNR == 1 { f++; grp = (f <= pairs) ? "A" : "B" }
$1 ~ /^(load|hold|spin|fly|edit|box|query|reslot|visibility|sculpt|dig)$/ && $2 ~ /^(seconds|frame_ms_mean|frame_ms_p95|frame_ms_p99|gpu_raster_ms_mean|gpu_mesh_ms_total)$/ {
  if ($1 == "load" && $2 != "seconds") next
  if ($1 != "load" && $2 == "seconds") next
  k = $1 " " $2; if (!(k in seen)) { seen[k] = 1; keys[++nk] = k }
  n[k, grp]++; s[k, grp] += $3
  if (!((k, grp) in mn) || $3 < mn[k, grp]) mn[k, grp] = $3
  if (!((k, grp) in mx) || $3 > mx[k, grp]) mx[k, grp] = $3
}
$2 ~ /^check:/ && $3 == 0 { fail[$1 "/" $2 " (" grp ")"] = 1 }
END {
  printf "%-30s %9s %7s %9s %7s %8s\n", "metric", "A mean", "A sprd", "B mean", "B sprd", "B vs A"
  for (j = 1; j <= nk; j++) {
    k = keys[j]
    if (!n[k, "A"] || !n[k, "B"]) continue
    a = s[k, "A"] / n[k, "A"]; b = s[k, "B"] / n[k, "B"]
    sa = (a != 0) ? 100 * (mx[k, "A"] - mn[k, "A"]) / a : 0
    sb = (b != 0) ? 100 * (mx[k, "B"] - mn[k, "B"]) / b : 0
    d = (a != 0) ? 100 * (b - a) / a : 0
    flag = ""
    if (d < -1 && -d > 2 * (sa > sb ? sa : sb)) flag = " better"
    if (d > 1 && d > 2 * (sa > sb ? sa : sb)) flag = " WORSE"
    printf "%-30s %9.3f %6.1f%% %9.3f %6.1f%% %+7.1f%%%s\n", k, a, sa, b, sb, d, flag
  }
  for (x in fail) print "FAILED CHECK " x
}
