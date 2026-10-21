#!/usr/bin/env python3
"""
pending_jobs.py - pending jobs, what they asked for, and when they are expected to start.

  pending_jobs.py                  your pending jobs, soonest expected start first
  pending_jobs.py -a               all users
  pending_jobs.py -u alice         one other user
  pending_jobs.py -p gpu           one partition
  pending_jobs.py --estimate       for jobs Slurm has not given a start time yet, add this tool's
                                   own best-case estimate (marked with ~)
  pending_jobs.py --json           machine-readable
  pending_jobs.py --demo           synthetic data, no Slurm needed

"start" is Slurm's own backfill estimate (squeue --start). Jobs still waiting on priority usually
have none; --estimate fills those in from free-resource timelines, ignoring priority, so it is a
best case.
"""

import argparse
import getpass
import json
import sys
from job_forecaster import collect, build_timelines, pending_rows, fmt_time


def table(headers, rows):
    if not rows:
        return "  (none)\n"
    widths = [len(h) for h in headers]
    srows = [[str(c) for c in r] for r in rows]
    for r in srows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))
    fmt = "  " + "  ".join("{:<%d}" % w for w in widths)
    out = [fmt.format(*headers), "  " + "  ".join("-" * w for w in widths)]
    out += [fmt.format(*r) for r in srows]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-u", "--user", help="show this user's jobs (default: you)")
    ap.add_argument("-a", "--all", action="store_true", help="all users")
    ap.add_argument("-p", "--partition")
    ap.add_argument("--estimate", action="store_true", help="add best-case estimate where Slurm has none")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    user = None if args.all else (args.user or getpass.getuser())
    if args.demo and not args.all and not args.user:
        user = "alice"   # demo data has no real users

    try:
        snap = collect(demo=args.demo)
    except Exception as e:  # noqa
        sys.exit(f"could not read Slurm state: {e}\n(try --demo to test without Slurm)")
    now = snap.now
    rows = pending_rows(snap, build_timelines(snap) if args.estimate else None,
                        user=user, partition=args.partition, with_estimate=args.estimate)

    if args.json:
        print(json.dumps(rows, indent=1))
        return

    for w in snap.warnings:
        print(f"note: {w}")
    print(f"snapshot: {fmt_time(now)}   {len(rows)} pending job(s)"
          + (f"   user: {user}" if user else "   all users") + (f"   partition: {args.partition}" if args.partition else ""))
    out = [[r["job"], r["user"], r["name"][:20], r["partition"], r["nodes"], r["cpus"], r["mem"], r["gpus"],
            r["features"] or "-", r["limit"], r["start_text"], r["start_in"], r["reason"] or "-",
            ",".join(r["planned_nodes"][:3]) + (f",+{len(r['planned_nodes']) - 3}" if len(r["planned_nodes"]) > 3 else "")]
           for r in rows]
    print(table(["job", "user", "name", "part", "N", "cpus", "mem", "gpus", "features", "limit",
                 "start", "in", "reason", "planned nodes"], out))
    n_none = sum(1 for r in rows if r["start"] is None)
    if n_none:
        print(f"  {n_none} job(s) have no Slurm start estimate" +
              ("" if args.estimate else "; run with --estimate for a best-case guess"))
    if user:
        print("  showing your jobs only; -a for all users")
    if args.estimate:
        print("  ~ = this tool's estimate (resources free, priority ignored); others are Slurm's own.")


if __name__ == "__main__":
    main()
