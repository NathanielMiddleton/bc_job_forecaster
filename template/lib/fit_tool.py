#!/usr/bin/env python3
"""
fit_tool.py - tune a job's requirements until it fits sooner.

Three ways to use it:

  fit_tool.py                      terminal UI (arrow keys adjust sliders, estimate updates live)
  fit_tool.py --web                local web page with sliders. The port is chosen automatically
                                   (a free one on this host) and printed; on a login node run it,
                                   then from your laptop: ssh -L <port>:localhost:<port> login.cluster
                                   and open http://localhost:<port>.  --port N pins a specific port.
  fit_tool.py --once --gpus 2 --gpu-type a100 --time 4:00:00
                                   print the estimate for one set of requirements and exit

Add --demo to any of these to use a synthetic cluster (no Slurm needed).

The estimate is "earliest moment the resources are physically free", based on running jobs'
end times and pending jobs Slurm has already scheduled. Priority, fairshare and QOS limits are
not modelled, so real start times can be later; the *relative* effect of changing requirements
is what the tool is for.
"""

import argparse
import sys
from dataclasses import asdict

from job_forecaster import Request, sbatch_snippet, explain, parse_duration, parse_mem_mb, fmt_mem, fmt_dur, fmt_time, fmt_gpus, INF

WALL_STEPS = [15 * 60, 30 * 60, 3600, 2 * 3600, 4 * 3600, 6 * 3600, 8 * 3600, 12 * 3600,
              24 * 3600, 36 * 3600, 48 * 3600, 72 * 3600, 5 * 86400, 7 * 86400]


# ----------------------------------------------------------------------------
# shared state
# ----------------------------------------------------------------------------

from job_forecaster_web import Cluster  # snapshot cache + evaluate(), shared with the web app


def request_from_args(a, lim):
    gpu_type = (a.gpu_type or "any").lower()
    if a.gpu_type and gpu_type not in lim["gpu_types"] + ["any"]:
        print(f"warning: no node has GPU type '{a.gpu_type}'; cluster has: {', '.join(lim['gpu_types']) or 'none'}",
              file=sys.stderr)
    return Request(nodes=a.nodes, cpus=a.cpus, mem=parse_mem_mb(a.mem), gpus=a.gpus, gpu_type=gpu_type,
                   features=[f for f in (a.features or "").split(",") if f],
                   partition=a.partition or "", wall=parse_duration(a.time) or 3600)


def print_result(res, cluster=None, req=None, show_explain=False):
    print(f"request : {res['describe']}")
    print(f"start   : {res['eta']}")
    if res["nodes"]:
        print(f"on      : {','.join(res['nodes'])}")
    if show_explain and cluster is not None:
        now = cluster.snap.now
        with cluster.lock:
            rows = explain(cluster.snap, req, {"nodes": res["nodes"], "start": res["start"]}, cluster.tl)
        print(f"\neligible nodes: {res['eligible']}   (why these nodes, from scontrol show node):")
        for r in rows:
            print(f"\n  {r['node']}  state={r['state']}")
            print(f"    free now : {r['cpu_free_now']}/{r['cpu_total']} cpu, "
                  f"{fmt_mem(r['mem_free_now'])}/{fmt_mem(r['mem_total'])} mem, "
                  f"gpu {fmt_gpus(r['gpu_free_now']) if any(r['gpu_free_now'].values()) else '0'} of {fmt_gpus(r['gpu_total'])}")
            if res["start"] and res["start"] > now + 1:
                print(f"    free at start: {r['cpu_free_at_start']} cpu, gpu {fmt_gpus(r['gpu_free_at_start'])}")
            for j in r["running"]:
                print(f"    running  : job {j['id']} ({j['user']}) {j['cpus']} cpu, gpu {fmt_gpus(j['gpus'])}, "
                      f"ends {fmt_time(j['end'], now) if j['end'] != INF else 'unlimited'}")
            for j in r["scheduled"]:
                print(f"    scheduled: job {j['id']} ({j['user']}) gpu {fmt_gpus(j['gpus'])}, "
                      f"{fmt_time(j['start'], now)} - {fmt_time(j['end'], now)}")
            if r["raw"]:
                keys = ("CPUAlloc", "CPUTot", "AllocMem", "RealMemory", "Gres", "GresUsed", "AllocTRES", "State")
                bits = [t for t in r["raw"].split() if t.split("=")[0] in keys]
                print("    scontrol : " + " ".join(bits))
    if res["reason"]:
        print(f"why not : {res['reason']}  ({res['eligible']} eligible node(s))")
    if res["suggestions"]:
        print("\nstart sooner if you:")
        for s in res["suggestions"]:
            print(f"  - {s['label']:<42} -> {s['eta']}  (saves {s['gain']})")
    print("\n" + res["sbatch"])
    for w in res["warnings"]:
        print(f"\nnote: {w}")


# ----------------------------------------------------------------------------
# terminal UI
# ----------------------------------------------------------------------------

def run_tui(cluster, req):
    import curses

    lim = cluster.limits()
    gpu_types = ["any"] + lim["gpu_types"]
    partitions = [""] + lim["partitions"]
    feats = lim["features"]
    wall_idx = min(range(len(WALL_STEPS)), key=lambda i: abs(WALL_STEPS[i] - req.wall))
    req.wall = WALL_STEPS[wall_idx]

    fields = ["nodes", "cpus", "mem", "gpus", "gpu_type", "features", "partition", "wall"]
    labels = {"nodes": "Nodes", "cpus": "CPUs / node", "mem": "Memory / node", "gpus": "GPUs / node",
              "gpu_type": "GPU type", "features": "Features", "partition": "Partition", "wall": "Walltime"}
    state = {"field": 0, "feat_idx": 0, "wall_idx": wall_idx, "result": None, "dirty": True, "msg": ""}

    def value_str(f):
        if f == "mem":
            return fmt_mem(req.mem)
        if f == "wall":
            return fmt_dur(req.wall)
        if f == "gpu_type":
            return req.gpu_type
        if f == "partition":
            return req.partition or "any"
        if f == "features":
            if not feats:
                return "(cluster has none)"
            parts = []
            for i, ft in enumerate(feats):
                on = "x" if ft in req.features else " "
                cur = ">" if (fields[state["field"]] == "features" and i == state["feat_idx"]) else " "
                parts.append(f"{cur}[{on}]{ft}")
            return " ".join(parts)
        return str(getattr(req, f))

    def adjust(f, d):
        if f == "nodes":
            req.nodes = max(1, min(lim["max_nodes"], req.nodes + d))
        elif f == "cpus":
            step = 1 if req.cpus < 8 or (d < 0 and req.cpus <= 8) else 4
            req.cpus = max(1, min(lim["max_cpus"], req.cpus + d * step))
        elif f == "mem":
            gb = req.mem // 1024
            step = 1 if gb < 16 or (d < 0 and gb <= 16) else (8 if gb < 128 else 64)
            req.mem = max(1024, min(lim["max_mem_gb"] * 1024, (gb + d * step) * 1024))
        elif f == "gpus":
            req.gpus = max(0, min(lim["max_gpus"], req.gpus + d))
        elif f == "gpu_type":
            i = (gpu_types.index(req.gpu_type) if req.gpu_type in gpu_types else 0) + d
            req.gpu_type = gpu_types[i % len(gpu_types)]
        elif f == "partition":
            i = (partitions.index(req.partition) if req.partition in partitions else 0) + d
            req.partition = partitions[i % len(partitions)]
        elif f == "wall":
            state["wall_idx"] = max(0, min(len(WALL_STEPS) - 1, state["wall_idx"] + d))
            req.wall = WALL_STEPS[state["wall_idx"]]
        elif f == "features" and feats:
            state["feat_idx"] = (state["feat_idx"] + d) % len(feats)
            return
        state["dirty"] = True

    def toggle_feature():
        if not feats:
            return
        ft = feats[state["feat_idx"]]
        if ft in req.features:
            req.features.remove(ft)
        else:
            req.features.append(ft)
        state["dirty"] = True

    def apply_suggestion(k):
        res = state["result"]
        if res and k < len(res["suggestions"]):
            r = res["suggestions"][k]["request"]
            for key, val in r.items():
                setattr(req, key, val)
            state["wall_idx"] = min(range(len(WALL_STEPS)), key=lambda i: abs(WALL_STEPS[i] - req.wall))
            state["dirty"] = True

    def draw(scr):
        scr.erase()
        h, w = scr.getmaxyx()
        y = 0

        def put(line, attr=0):
            nonlocal y
            if y < h - 1:
                scr.addnstr(y, 0, line, w - 1, attr)
                y += 1

        src = "DEMO data" if cluster.demo else "Slurm"
        put(f" fit_tool  [{src}, snapshot {fmt_time(cluster.snap.now)}]   "
            "up/down: field   left/right: change   space: toggle feature   1-8: apply suggestion   "
            "r: refresh   s: print sbatch & quit   q: quit", curses.A_DIM)
        put("")
        for i, f in enumerate(fields):
            sel = i == state["field"]
            put(f"  {'>' if sel else ' '} {labels[f]:<14} {value_str(f)}",
                curses.A_REVERSE if sel else 0)
        put("")
        res = state["result"]
        if res:
            put(f"  Estimated start: {res['eta']}", curses.A_BOLD)
            if res["nodes"]:
                put("  would land on:")
                for d in res["node_detail"][:8]:
                    put(f"    {d}")
                if len(res["nodes"]) > 8:
                    put(f"    … and {len(res['nodes']) - 8} more")
            if res["reason"]:
                put(f"  cannot fit:      {res['reason']}  ({res['eligible']} eligible nodes)")
            put("")
            if res["suggestions"]:
                put("  Start sooner if you (press number to apply):")
                for k, s in enumerate(res["suggestions"], 1):
                    put(f"    {k}. {s['label']:<42} {s['eta']:<28} saves {s['gain']}")
            else:
                put("  No single change to these requirements starts the job earlier." if res["start"]
                    else "  No simple change makes this request fit.")
            put("")
            # mini timeline of the relevant resource over the next 24h
            key = None
            if req.gpus:
                key = ("gpus", req.gpu_type)
            put("  Free " + (f"{req.gpu_type} GPUs" if req.gpus else "CPUs") + " over the next 48h (need "
                + str(req.nodes * (req.gpus or req.cpus)) + "):")
            width = max(20, min(w - 12, 96))
            now = res["now"]
            horizon = 48 * 3600
            curve = res["curve"]
            cells = []
            for i in range(width):
                t = now + horizon * i / width
                val = 0
                for row in curve:
                    if row["t"] <= t:
                        val = (sum(row["gpus"].values()) if key[1] == "any" else row["gpus"].get(key[1], 0)) \
                            if key else row["cpus"]
                cells.append(val)
            mx = max(cells + [1])
            bars = " ▁▂▃▄▅▆▇█"
            line = "".join(bars[min(8, int(8 * v / mx))] for v in cells)
            put("  " + line)
            if res["start"] is not None and res["start"] - now < horizon:
                a = int((res["start"] - now) / horizon * width)
                b = min(width, max(a + 1, int((res["start"] + req.wall - now) / horizon * width)))
                put("  " + " " * a + "^" * (b - a) + "  your job")
            put("  now" + " " * (width - 10) + "+48h")
            for wmsg in res["warnings"]:
                put("  note: " + wmsg, curses.A_DIM)
        if state["msg"]:
            put("  " + state["msg"], curses.A_DIM)
        scr.refresh()

    def loop(scr):
        curses.curs_set(0)
        scr.timeout(200)
        while True:
            if state["dirty"]:
                state["result"] = cluster.evaluate(req)
                state["dirty"] = False
            draw(scr)
            k = scr.getch()
            if k == -1:
                continue
            f = fields[state["field"]]
            if k in (ord("q"), 27):
                return None
            if k == ord("s"):
                return sbatch_snippet(req)
            if k == ord("r"):
                cluster.refresh(force=True)
                state["dirty"] = True
                state["msg"] = "refreshed"
            elif k in (curses.KEY_UP, ord("k")):
                state["field"] = (state["field"] - 1) % len(fields)
            elif k in (curses.KEY_DOWN, ord("j"), 9):
                state["field"] = (state["field"] + 1) % len(fields)
            elif k in (curses.KEY_LEFT, ord("h")):
                adjust(f, -1)
            elif k in (curses.KEY_RIGHT, ord("l")):
                adjust(f, +1)
            elif k in (ord(" "), 10, 13) and f == "features":
                toggle_feature()
            elif ord("1") <= k <= ord("8"):
                apply_suggestion(k - ord("1"))

    out = curses.wrapper(loop)
    if out:
        print(out)


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--web", action="store_true", help="serve the browser UI")
    mode.add_argument("--once", action="store_true", help="print one estimate and exit")
    ap.add_argument("--explain", action="store_true", help="with --once: show per-node evidence for the estimate")
    ap.add_argument("--port", type=int, default=None,
                    help="port for --web (default: a free port chosen automatically)")
    ap.add_argument("--host", default="127.0.0.1", help="bind address for --web (default localhost only)")
    ap.add_argument("--demo", action="store_true", help="synthetic cluster, no Slurm needed")
    ap.add_argument("-N", "--nodes", type=int, default=1)
    ap.add_argument("-c", "--cpus", type=int, default=4, help="CPUs per node")
    ap.add_argument("--mem", default="16G", help="memory per node, e.g. 64G")
    ap.add_argument("-G", "--gpus", type=int, default=0, help="GPUs per node")
    ap.add_argument("--gpu-type", default=None, help="e.g. a100 (default: any)")
    ap.add_argument("-C", "--features", default="", help="comma-separated node features")
    ap.add_argument("-p", "--partition", default="")
    ap.add_argument("-t", "--time", default="1:00:00", help="walltime, Slurm format")
    args = ap.parse_args()

    try:
        cluster = Cluster(demo=args.demo)
        cluster.refresh(force=True)
    except Exception as e:  # noqa
        sys.exit(f"could not read Slurm state: {e}\n(try --demo to test without Slurm)")
    req = request_from_args(args, cluster.limits())

    if args.once:
        print_result(cluster.evaluate(req), cluster, req, show_explain=args.explain)
    elif args.web:
        from job_forecaster_web import serve
        serve(cluster, port=args.port, host=args.host, initial=asdict(req))
    else:
        try:
            import curses  # noqa
        except ImportError:
            sys.exit("curses is not available in this Python; use --web or --once")
        if not sys.stdout.isatty():
            sys.exit("not a terminal; use --web or --once")
        run_tui(cluster, req)


if __name__ == "__main__":
    main()
