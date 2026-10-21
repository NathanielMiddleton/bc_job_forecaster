#!/usr/bin/env python3
"""
qos_view.py - QOS limits as they apply to each partition, with current usage.

  qos_view.py              every partition that has a QOS: group limits, usage by everyone, free room,
                           per-user limits, your usage, your room
  qos_view.py -p braintv   one partition
  qos_view.py -u alice     another user's usage column
  qos_view.py --json
  qos_view.py --demo

Usage = allocated TRES of running jobs in the QOS, as Slurm counts GrpTRES / MaxTRESPU.
"""
import argparse
import getpass
import json
import sys
from job_forecaster import collect, qos_rows, fmt_time


def table(headers, rows):
    if not rows:
        return "  (none)\n"
    widths = [len(h) for h in headers]
    srows = [[str(c) for c in r] for r in rows]
    for r in srows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))
    fmt = "  " + "  ".join("{:<%d}" % w for w in widths)
    return "\n".join([fmt.format(*headers), "  " + "  ".join("-" * w for w in widths)] + [fmt.format(*r) for r in srows]) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-p", "--partition")
    ap.add_argument("-u", "--user", help="whose usage to show in the 'your' columns (default: you)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    user = args.user or ("alice" if args.demo else getpass.getuser())
    try:
        snap = collect(demo=args.demo)
    except Exception as e:  # noqa
        sys.exit(f"could not read Slurm state: {e}\n(try --demo to test without Slurm)")
    rows = qos_rows(snap, user=user, partition=args.partition)
    if args.json:
        print(json.dumps(rows, indent=1))
        return
    for w in snap.warnings:
        print(f"note: {w}")
    print(f"snapshot: {fmt_time(snap.now)}   user: {user}")
    if not snap.qos:
        print("  no QOS information (sacctmgr unavailable or no QOS defined)")
        return
    for r in rows:
        extra = f"   MaxJobsPU {r['max_jobs_pu']}" if r["max_jobs_pu"] else ""
        print(f"\n== {r['partition']}" + (f" -> QOS {r['qos']}" if r["qos"] != r["partition"] else "")
              + f"   priority {r['priority']}   MaxWall {r['max_wall']}   running jobs {r['jobs_running']}"
              + f" (yours {r['my_jobs_running']})" + extra)
        print(table(["tres", "group limit", "in use (all)", "free", "per-user limit", "your use", "your room"],
                    [[t["label"], t["text"]["grp_limit"],
                      t["text"]["grp_used"] + (f" ({t['grp_pct']}%)" if t["grp_pct"] is not None else ""),
                      t["text"]["grp_free"], t["text"]["pu_limit"], t["text"]["my_used"], t["text"]["my_free"]]
                     for t in r["tres"]]))
    print("  A request larger than 'free' or 'your room' waits for jobs in that QOS to end, even when nodes are idle.")


if __name__ == "__main__":
    main()
