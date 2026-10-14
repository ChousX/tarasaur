# Summarises a RenderDoc capture: GPU time per pass (debug marker) and the
# slowest individual actions. Runs inside qrenderdoc's Python:
#
#   RDC=capture.rdc RDC_OUT=summary.txt qrenderdoc --python tools/renderdoc_summary.py
#
# The script exits the process when done, so the UI never opens.

import os
import traceback

import renderdoc as rd


def walk(actions, path, sf, out):
    for a in actions:
        name = a.GetName(sf)
        if a.children:
            walk(a.children, path + [name], sf, out)
        out.append((a, name, path))


def summarise(ctrl, write):
    sf = ctrl.GetStructuredFile()
    actions = []
    walk(ctrl.GetRootActions(), [], sf, actions)

    durations = {}
    if rd.GPUCounter.EventGPUDuration in ctrl.EnumerateCounters():
        for r in ctrl.FetchCounters([rd.GPUCounter.EventGPUDuration]):
            durations[r.eventId] = r.value.d * 1000.0  # seconds -> ms
    else:
        write("EventGPUDuration counter not supported on this replay device")

    draw = rd.ActionFlags.Drawcall
    dispatch = rd.ActionFlags.Dispatch
    leaves = [(a, n, p) for a, n, p in actions if a.flags & (draw | dispatch)]
    draws = [x for x in leaves if x[0].flags & draw]
    total = sum(durations.get(a.eventId, 0.0) for a, _, _ in leaves)

    write(f"actions: {len(leaves)} ({len(draws)} draws, {len(leaves) - len(draws)} dispatches)")
    write(f"summed GPU duration of actions: {total:.3f} ms")
    indices = sum(a.numIndices * max(a.numInstances, 1) for a, _, _ in draws)
    write(f"indices drawn: {indices} (~{indices // 3} triangles)")
    write("")

    # Per pass: the outermost marker, then the innermost, so both a frame's
    # big phases and individual passes show up.
    for depth_name, key in (("outermost marker", lambda p: p[0] if p else "<none>"),
                            ("innermost marker", lambda p: p[-1] if p else "<none>")):
        per = {}
        for a, _, p in leaves:
            ms, n = per.get(key(p), (0.0, 0))
            per[key(p)] = (ms + durations.get(a.eventId, 0.0), n + 1)
        write(f"GPU ms by {depth_name}:")
        for name, (ms, n) in sorted(per.items(), key=lambda kv: -kv[1][0])[:25]:
            write(f"  {ms:9.3f} ms  {n:6d} actions  {name}")
        write("")

    write("slowest actions:")
    for a, n, p in sorted(leaves, key=lambda x: -durations.get(x[0].eventId, 0.0))[:25]:
        where = " > ".join(p[-2:])
        write(f"  {durations.get(a.eventId, 0.0):9.3f} ms  eid {a.eventId:6d}  "
              f"idx {a.numIndices:8d}  {n}  [{where}]")


def main():
    path = os.environ["RDC"]
    out = open(os.environ.get("RDC_OUT", "/dev/stdout"), "w")

    def write(line):
        out.write(line + "\n")
        out.flush()

    try:
        cap = rd.OpenCaptureFile()
        if cap.OpenFile(path, "", None) != rd.ResultCode.Succeeded:
            write(f"couldn't open {path}")
            return
        if not cap.LocalReplaySupport():
            write("capture can't be replayed on this machine")
            return
        result, ctrl = cap.OpenCapture(rd.ReplayOptions(), None)
        if result != rd.ResultCode.Succeeded:
            write(f"couldn't replay: {result}")
            return
        summarise(ctrl, write)
        ctrl.Shutdown()
        cap.Shutdown()
    except Exception:
        write(traceback.format_exc())
    finally:
        out.close()


main()
os._exit(0)
