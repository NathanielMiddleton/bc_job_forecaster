#!/usr/bin/env python3
"""
cluster_view.py - one-screen view of a Slurm cluster:

  * total / used / available CPUs, memory, GPUs (per type)
  * current job allocations (running jobs, when they end)
  * future allocations (pending jobs Slurm has already given a start time)
  * gaps: free space on nodes right now and how long it stays free,
          plus a cluster-wide free-capacity timeline

Usage:
  cluster_view.py                      whole cluster
  cluster_view.py -p gpu               one partition
  cluster_view.py -a                   job tables for all users (default: only yours)
  cluster_view.py -u alice             job tables for one other user
  cluster_view.py --gaps-only          just the gap tables
  cluster_view.py --min-gpus 1         only show gaps with >= 1 free GPU
  cluster_view.py --hours 72           timeline horizon (default 48)
  cluster_view.py --json               machine-readable snapshot
  cluster_view.py --demo               synthetic data, no Slurm needed
"""

import argparse
import getpass
import sys
from job_forecaster import (collect, build_timelines, capacity_curve, node_gaps, fmt_mem, fmt_gpus, GPU_TYPE_DISPLAY,
                      fmt_dur, fmt_time, INF)


def table(headers, rows, max_width=None):
    if not rows:
        return "  (none)\n"
    cols = len(headers)
    widths = [len(h) for h in headers]
    srows = [[str(c) for c in r] for r in rows]
    for r in srows:
        for i in range(cols):
            widths[i] = max(widths[i], len(r[i]))
    if max_width:
        widths = [min(w, max_width) for w in widths]
    fmt = "  " + "  ".join("{:<%d}" % w for w in widths)
    out = [fmt.format(*headers), "  " + "  ".join("-" * w for w in widths)]
    for r in srows:
        out.append(fmt.format(*[c[:w] for c, w in zip(r, widths)]))
    return "\n".join(out) + "\n"


def pct(a, b):
    return f"{100 * a / b:.0f}%" if b else "-"


def section(title):
    print(f"\n== {title} " + "=" * max(0, 70 - len(title)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-p", "--partition", default=None)
    ap.add_argument("-u", "--user", default=None, help="show this user's jobs (default: you)")
    ap.add_argument("-a", "--all", action="store_true", help="show all users' jobs")
    ap.add_argument("--hours", type=float, default=48, help="timeline horizon")
    ap.add_argument("--min-gpus", type=int, default=0, help="only list gaps with at least this many free GPUs")
    ap.add_argument("--min-cpus", type=int, default=1, help="only list gaps with at least this many free CPUs")
    ap.add_argument("--gaps-only", action="store_true")
    ap.add_argument("--all-nodes", action="store_true", help="include unusable (down/drain) nodes in gap table")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--demo", action="store_true", help="use a synthetic cluster (no Slurm needed)")
    args = ap.parse_args()

    try:
        snap = collect(demo=args.demo)
    except Exception as e:  # noqa
        sys.exit(f"could not read Slurm state: {e}\n(try --demo to test without Slurm)")

    if args.json:
        print(snap.to_json())
        return

    now = snap.now
    part = args.partition
    tl = build_timelines(snap)
    user = None if args.all else (args.user or getpass.getuser())
    if args.demo and not args.all and not args.user:
        user = "alice"   # demo data has no real users

    def mine(j):
        return user is None or j.user == user
    who = f" [{user}]" if user else " [all users]"

    for w in snap.warnings:
        print(f"note: {w}")
    print(f"snapshot: {fmt_time(now)}" + (f"   partition: {part}" if part else ""))

    if not args.gaps_only:
        # ---------------------------------------------------------------- totals
        t = snap.totals(part)
        section("Resources")
        rows = [
            ["Nodes", t["nodes"], f'{t["nodes"] - t["nodes_usable"]} down/drained', t["nodes_usable"], ""],
            ["CPUs", t["cpus"], t["cpus_alloc"], t["cpus"] - t["cpus_alloc"], pct(t["cpus_alloc"], t["cpus"])],
            ["Memory", fmt_mem(t["mem"]), fmt_mem(t["mem_alloc"]), fmt_mem(t["mem"] - t["mem_alloc"]),
             pct(t["mem_alloc"], t["mem"])],
        ]
        for g, tot in sorted(t["gpus"].items()):
            used = t["gpus_used"].get(g, 0)
            rows.append([f"GPU {GPU_TYPE_DISPLAY.get(g, g)}", tot, used, tot - used, pct(used, tot)])
        print(table(["", "total", "used", "available", "util"], rows))

        # ------------------------------------------------------ current allocs
        section("Current allocations (running)" + who)
        run = [j for j in snap.running if mine(j) and (not part or j.partition == part)]
        run.sort(key=lambda j: j.end if j.end is not None else INF)
        rows = [[j.id, j.user, j.name, j.partition, len(j.nodes),
                 ",".join(j.nodes[:3]) + (f",+{len(j.nodes) - 3}" if len(j.nodes) > 3 else ""),
                 j.cpus, fmt_mem(j.mem), fmt_gpus(j.gpus),
                 fmt_time(j.end, now) if j.end != INF else "unlimited",
                 fmt_dur(j.end - now) if j.end not in (None, INF) else "-"]
                for j in run]
        print(table(["job", "user", "name", "part", "N", "nodes", "cpus", "mem", "gpus", "ends", "in"],
                    rows, max_width=28))

        # ------------------------------------------------------- future allocs
        section("Future allocations (pending with a Slurm start estimate)" + who)
        pend = [j for j in snap.pending if mine(j) and (not part or j.partition == part)]
        sched = sorted([j for j in pend if j.start is not None], key=lambda j: j.start)
        rows = [[j.id, j.user, j.name, j.partition, j.num_nodes,
                 ",".join(j.nodes[:3]) + (f",+{len(j.nodes) - 3}" if len(j.nodes) > 3 else "") or "(unassigned)",
                 j.cpus, fmt_mem(j.mem), fmt_gpus(j.gpus), fmt_time(j.start, now),
                 fmt_dur(j.start - now), fmt_dur(j.time_limit)]
                for j in sched]
        print(table(["job", "user", "name", "part", "N", "planned nodes", "cpus", "mem", "gpus",
                     "starts", "in", "limit"], rows, max_width=28))
        unsched = [j for j in pend if j.start is None]
        if unsched:
            print(f"  + {len(unsched)} pending job(s) with no start estimate yet "
                  f"(reasons: {', '.join(sorted({j.reason or '?' for j in unsched}))})")
        if user:
            others = sum(1 for j in snap.jobs if not mine(j) and (not part or j.partition == part))
            print(f"  ({others} job(s) from other users hidden; -a to show all)")

    # ------------------------------------------------------------------ gaps
    section("Gaps: free space on nodes right now")
    gaps = node_gaps(snap, tl, partition=part, min_cpus=args.min_cpus, min_gpus=args.min_gpus,
                     include_unusable=args.all_nodes)
    gap_count = len(gaps)
    rows = [[g["node"], g["kind"], g["cpu"], fmt_mem(g["mem"]), fmt_gpus(g["gpus"]) if g["gpus_total"] else "-",
             g["stays_free"], g["next_change"], ",".join(g["features"])[:30], "" if g["usable"] else g["state"]]
            for g in gaps]
    print(table(["node", "", "free cpu", "free mem", "free gpu", "stays free", "next change", "features",
                 "state"], rows, max_width=34))
    print(f"  {gap_count} node(s) with free space now "
          f"(filters: >= {args.min_cpus} cpu, >= {args.min_gpus} gpu)")

    # ------------------------------------------------- cluster timeline
    section(f"Gaps: cluster-wide free capacity over the next {args.hours:g}h")
    curve = capacity_curve(snap, tl, horizon_s=args.hours * 3600, partition=part)
    rows = []
    last = None
    for row in curve:
        key = (row["cpus"], row["mem"], tuple(sorted(row["gpus"].items())), row["idle_nodes"])
        if key == last:
            continue
        last = key
        rows.append([fmt_time(row["t"], now), "now" if row["t"] == now else "+" + fmt_dur(row["t"] - now),
                     row["cpus"], fmt_mem(row["mem"]), fmt_gpus(row["gpus"]), row["idle_nodes"]])
        if len(rows) >= 40:
            rows.append(["...", "", "", "", "", ""])
            break
    print(table(["time", "offset", "free cpu", "free mem", "free gpu", "idle nodes"], rows))
    print("  Free capacity rises as running jobs end and drops as scheduled jobs start.\n"
          "  Pending jobs without a Slurm start estimate are not included, so treat this as a best case.\n"
          "  To see when YOUR job would fit and how to make it fit sooner: fit_tool.py")


if __name__ == "__main__":
    main()
