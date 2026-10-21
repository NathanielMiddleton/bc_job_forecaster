#!/usr/bin/env python3
"""
job_forecaster.py - shared library for cluster_view.py and fit_tool.py

Collects cluster state from Slurm (scontrol show node / scontrol show job),
builds a per-node resource timeline, and estimates when a job with given
requirements could start.

Only the Python standard library is used, so this runs on a bare login node.

Limitations of the estimate (be honest with users about these):
  * Slurm's actual decision also depends on priority, fairshare, QOS limits,
    reservations and licenses. The estimate here is "earliest time the
    resources are physically free", i.e. a best case.
  * Per-node resource usage of a multi-node job is approximated by dividing
    the job's total TRES evenly across its nodes (scontrol -o does not expose
    the per-node split without -d).
  * Pending jobs are only projected into the future if Slurm has already
    assigned them an expected StartTime and SchedNodeList (backfill estimates).
"""

import datetime as dt
import json
import math
import os
import random
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field, asdict

INF = float("inf")
# GPU type names are compared lower-case everywhere (Slurm lower-cases them in TRES even when
# gres.conf says H200). This map remembers the spelling the nodes use, for #SBATCH output.
GPU_TYPE_DISPLAY = {}
BAD_NODE_STATES = {
    "DOWN", "DRAIN", "DRAINED", "DRAINING", "FAIL", "FAILING",
    "NOT_RESPONDING", "INVALID", "INVALID_REG", "MAINT", "FUTURE", "UNKNOWN",
}
ACTIVE_JOB_STATES = {"RUNNING", "COMPLETING", "CONFIGURING", "PENDING", "SUSPENDED"}

# ----------------------------------------------------------------------------
# parsing helpers
# ----------------------------------------------------------------------------

_KV_SPLIT = re.compile(r"\s+(?=[A-Za-z_][\w:/.-]*=)")


def parse_kv_line(line):
    """Parse one 'Key=Value Key=Value' line as printed by scontrol -o."""
    out = {}
    for tok in _KV_SPLIT.split(line.strip()):
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def parse_mem_mb(s):
    """'16G' -> 16384, '512M' -> 512, '2T' -> 2097152, '0' -> 0, junk -> 0."""
    if s is None:
        return 0
    s = str(s).strip().rstrip("nc")  # MinMemoryNode may carry n/c suffix
    m = re.match(r"^([\d.]+)\s*([KMGTP]?)", s, re.I)
    if not m:
        return 0
    val = float(m.group(1))
    unit = m.group(2).upper()
    mult = {"": 1, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 ** 2, "P": 1024 ** 3}[unit]
    return int(val * mult)


def parse_duration(s):
    """Slurm time string -> seconds, or None for UNLIMITED/Unknown/N/A."""
    if s is None:
        return None
    s = s.strip()
    if s.upper() in ("UNLIMITED", "UNKNOWN", "N/A", "NONE", "(NULL)", "INVALID", ""):
        return None
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    parts = s.split(":")
    try:
        parts = [int(p) for p in parts]
    except ValueError:
        return None
    if len(parts) == 3:
        h, m, sec = parts
    elif len(parts) == 2:
        h, m, sec = 0, parts[0], parts[1]
        if days:  # d-hh:mm
            h, m, sec = parts[0], parts[1], 0
    else:
        h, m, sec = 0, parts[0], 0
        if days:  # d-hh
            h, m, sec = parts[0], 0, 0
    return days * 86400 + h * 3600 + m * 60 + sec


def parse_datetime(s):
    """'2026-10-20T10:00:00' -> epoch seconds (local time), else None."""
    if not s or s.strip().upper() in ("UNKNOWN", "N/A", "NONE", "(NULL)"):
        return None
    try:
        return dt.datetime.strptime(s.strip(), "%Y-%m-%dT%H:%M:%S").timestamp()
    except ValueError:
        return None


def parse_gres(s):
    """
    'gpu:a100:4(S:0-1),gpu:v100:2,tmpfs:100G' -> {'a100': 4, 'v100': 2}
    'gpu:4' -> {'gpu': 4}      '(null)' -> {}
    """
    out = {}
    if not s or s.strip() in ("(null)", "N/A"):
        return out
    s = re.sub(r"\([^)]*\)", "", s)  # strip (IDX:..) / (S:..)
    for item in s.split(","):
        parts = item.strip().split(":")
        if not parts or parts[0] != "gpu":
            continue
        if len(parts) == 2:
            typ, cnt = "gpu", parts[1]
        elif len(parts) >= 3:
            typ, cnt = parts[1], parts[2]
        else:
            continue
        try:
            key = typ.lower()
            GPU_TYPE_DISPLAY.setdefault(key, typ)
            out[key] = out.get(key, 0) + int(float(cnt))
        except ValueError:
            pass
    return out


def parse_tres(s):
    """'cpu=4,mem=16G,node=1,gres/gpu=2,gres/gpu:a100=2' -> dict of strings."""
    out = {}
    if not s or s.strip() in ("(null)", "N/A"):
        return out
    for item in s.split(","):
        if "=" in item:
            k, v = item.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def job_gpus_from_tres(tres):
    """Typed GPU counts if present, otherwise generic 'gpu' bucket."""
    typed = {k.split(":", 1)[1].lower(): int(float(v)) for k, v in tres.items()
             if k.startswith("gres/gpu:")}
    if typed:
        return typed
    if "gres/gpu" in tres:
        return {"gpu": int(float(tres["gres/gpu"]))}
    return {}


def fmt_mem(mb):
    if mb >= 1024 * 1024:
        return f"{mb / 1024 / 1024:.1f}T"
    if mb >= 1024:
        return f"{mb / 1024:.0f}G" if mb % 1024 == 0 else f"{mb / 1024:.1f}G"
    return f"{int(mb)}M"


def fmt_gpus(g):
    return ",".join(f"{v}x{GPU_TYPE_DISPLAY.get(k, k)}" for k, v in sorted(g.items()) if v) or "-"


def fmt_dur(sec):
    if sec is None or sec == INF:
        return "unlimited"
    sec = int(round(sec))
    if sec < 0:
        sec = 0
    d, r = divmod(sec, 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    if d:
        return f"{d}d{h:02d}h{m:02d}m"
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s" if s and m < 5 else f"{m}m"


def fmt_time(ts, now=None):
    if ts is None or ts == INF:
        return "never"
    t = dt.datetime.fromtimestamp(ts)
    s = t.strftime("%a %H:%M") if (now is None or ts - now < 6 * 86400) else t.strftime("%Y-%m-%d %H:%M")
    return s


def fmt_eta(ts, now):
    if ts is None or ts == INF:
        return "no fit found"
    d = ts - now
    if d <= 1:
        return "now"
    return f"in {fmt_dur(d)} ({fmt_time(ts, now)})"


# ----------------------------------------------------------------------------
# data model
# ----------------------------------------------------------------------------

@dataclass
class Node:
    name: str
    cpus: int
    cpus_alloc: int
    mem: int            # MB
    mem_alloc: int      # MB
    gpus: dict          # type -> count
    gpus_used: dict     # type -> count
    features: list
    partitions: list
    state: str
    raw: str = ""        # the scontrol line this came from (for --explain)

    @property
    def usable(self):
        flags = set(re.split(r"[+*~#$@^!%-]", self.state.upper()))
        return not (flags & BAD_NODE_STATES)

    @property
    def gpu_total(self):
        return sum(self.gpus.values())

    @property
    def gpu_used(self):
        return sum(self.gpus_used.values())


@dataclass
class Job:
    id: str
    name: str
    user: str
    partition: str
    state: str
    nodes: list          # allocated (running) or scheduled (pending) node names
    num_nodes: int
    cpus: int            # total
    mem: int             # total MB
    gpus: dict           # total, type -> count
    features: str
    start: float         # epoch or None
    end: float           # epoch, INF for unlimited, None if unknown
    time_limit: float    # seconds or None
    reason: str = ""

    def per_node(self):
        n = max(1, self.num_nodes or len(self.nodes) or 1)
        return (math.ceil(self.cpus / n), math.ceil(self.mem / n),
                {k: math.ceil(v / n) for k, v in self.gpus.items()})


@dataclass
class Snapshot:
    now: float
    nodes: list
    jobs: list
    source: str = "slurm"
    warnings: list = field(default_factory=list)

    # --- derived -----------------------------------------------------------
    @property
    def running(self):
        return [j for j in self.jobs if j.state in ("RUNNING", "COMPLETING", "CONFIGURING")]

    @property
    def pending(self):
        return [j for j in self.jobs if j.state == "PENDING"]

    @property
    def gpu_types(self):
        t = set()
        for n in self.nodes:
            t.update(n.gpus.keys())
        return sorted(t)

    @property
    def feature_list(self):
        f = set()
        for n in self.nodes:
            f.update(n.features)
        return sorted(f)

    @property
    def partition_list(self):
        p = set()
        for n in self.nodes:
            p.update(n.partitions)
        return sorted(p)

    def totals(self, partition=None):
        tot = {"nodes": 0, "nodes_usable": 0, "cpus": 0, "cpus_alloc": 0,
               "mem": 0, "mem_alloc": 0, "gpus": {}, "gpus_used": {}}
        for n in self.nodes:
            if partition and partition not in n.partitions:
                continue
            tot["nodes"] += 1
            tot["nodes_usable"] += 1 if n.usable else 0
            tot["cpus"] += n.cpus
            tot["cpus_alloc"] += n.cpus_alloc
            tot["mem"] += n.mem
            tot["mem_alloc"] += n.mem_alloc
            for k, v in n.gpus.items():
                tot["gpus"][k] = tot["gpus"].get(k, 0) + v
            for k, v in n.gpus_used.items():
                tot["gpus_used"][k] = tot["gpus_used"].get(k, 0) + v
        return tot

    def to_json(self):
        return json.dumps({
            "now": self.now, "source": self.source, "warnings": self.warnings,
            "nodes": [asdict(n) for n in self.nodes],
            "jobs": [asdict(j) for j in self.jobs],
        }, default=lambda o: None if o == INF else o)


# ----------------------------------------------------------------------------
# collection from Slurm
# ----------------------------------------------------------------------------

def scontrol_path():
    """scontrol binary: $JOB_FORECASTER_SCONTROL, then PATH, then common install locations.
    Under Open OnDemand the PUN's PATH can be minimal, hence the fallbacks."""
    env = os.environ.get("JOB_FORECASTER_SCONTROL")
    if env:
        return env
    found = shutil.which("scontrol")
    if found:
        return found
    for cand in ("/usr/bin/scontrol", "/usr/local/bin/scontrol", "/opt/slurm/bin/scontrol",
                 "/cm/shared/apps/slurm/current/bin/scontrol"):
        if os.path.exists(cand):
            return cand
    return "scontrol"


def _run(cmd):
    if cmd[0] == "scontrol":
        cmd = [scontrol_path()] + cmd[1:]
    return subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60).stdout


_hostlist_cache = {}


def expand_hostlist(expr):
    if not expr or expr in ("(null)", "None", "N/A"):
        return []
    if "[" not in expr and "," not in expr:
        return [expr]
    if expr not in _hostlist_cache:
        try:
            _hostlist_cache[expr] = _run(["scontrol", "show", "hostnames", expr]).split()
        except Exception:
            _hostlist_cache[expr] = [expr]
    return _hostlist_cache[expr]


def collect_slurm():
    now = time.time()
    warnings = []

    nodes = []
    for line in _run(["scontrol", "show", "node", "-o"]).splitlines():
        kv = parse_kv_line(line)
        if "NodeName" not in kv:
            continue
        gpus = parse_gres(kv.get("Gres", ""))
        gpus_used = parse_gres(kv.get("GresUsed", ""))
        if gpus and not any(gpus_used.values()):
            # Some versions/configs report usage only in AllocTRES=...,gres/gpu=4,gres/gpu:h200=4
            alloc = job_gpus_from_tres(parse_tres(kv.get("AllocTRES", "")))
            if alloc:
                if list(alloc) == ["gpu"] and len(gpus) == 1:      # untyped count, single-type node
                    alloc = {next(iter(gpus)): alloc["gpu"]}
                gpus_used = {k: v for k, v in alloc.items() if k in gpus}
        if not gpus and kv.get("CfgTRES"):
            gpus = job_gpus_from_tres(parse_tres(kv["CfgTRES"]))
        nodes.append(Node(
            name=kv["NodeName"],
            cpus=int(kv.get("CPUTot", kv.get("CPUEfctv", 0)) or 0),
            cpus_alloc=int(kv.get("CPUAlloc", 0) or 0),
            mem=parse_mem_mb(kv.get("RealMemory", "0")),
            mem_alloc=parse_mem_mb(kv.get("AllocMem", "0")),
            gpus=gpus,
            gpus_used=gpus_used,
            features=[f for f in kv.get("AvailableFeatures", "").split(",")
                      if f and f != "(null)"],
            partitions=[p for p in kv.get("Partitions", "").split(",") if p],
            state=kv.get("State", "UNKNOWN"),
            raw=line.strip(),
        ))

    jobs = []
    for line in _run(["scontrol", "show", "job", "-o"]).splitlines():
        kv = parse_kv_line(line)
        if "JobId" not in kv:
            continue
        state = kv.get("JobState", "").split()[0] if kv.get("JobState") else ""
        if state not in ACTIVE_JOB_STATES:
            continue
        tres = parse_tres(kv.get("TRES") or kv.get("ReqTRES") or "")
        start = parse_datetime(kv.get("StartTime"))
        limit = parse_duration(kv.get("TimeLimit"))
        end = parse_datetime(kv.get("EndTime"))
        if state in ("RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED"):
            if end is None:
                end = INF if limit is None or start is None else start + limit
            node_list = expand_hostlist(kv.get("NodeList"))
        else:
            if start is not None and start < now:   # stale backfill estimate
                start = now
            end = None if start is None else (INF if limit is None else start + limit)
            node_list = expand_hostlist(kv.get("SchedNodeList"))
        num_nodes = kv.get("NumNodes", "1").split("-")[0]
        try:
            num_nodes = int(num_nodes)
        except ValueError:
            num_nodes = max(1, len(node_list))
        gpus = job_gpus_from_tres(tres)
        if not gpus:
            gpus = parse_gres((kv.get("TresPerNode") or kv.get("Gres") or "").replace("gres/", ""))
            gpus = {k: v * num_nodes for k, v in gpus.items()}
        mem = parse_mem_mb(tres.get("mem"))
        if not mem:
            if kv.get("MinMemoryNode"):
                mem = parse_mem_mb(kv["MinMemoryNode"]) * num_nodes
            elif kv.get("MinMemoryCPU"):
                mem = parse_mem_mb(kv["MinMemoryCPU"]) * int(float(tres.get("cpu", kv.get("NumCPUs", 1))))
        jobs.append(Job(
            id=kv["JobId"] if not kv.get("ArrayJobId") else f'{kv["ArrayJobId"]}_{kv.get("ArrayTaskId","")}',
            name=kv.get("JobName", ""),
            user=kv.get("UserId", "?").split("(")[0],
            partition=kv.get("Partition", "").split(",")[0],
            state=state,
            nodes=node_list,
            num_nodes=num_nodes,
            cpus=int(float(tres.get("cpu", kv.get("NumCPUs", 0)) or 0)),
            mem=mem,
            gpus=gpus,
            features=kv.get("Features", "") if kv.get("Features") not in ("(null)", None) else "",
            start=start, end=end, time_limit=limit,
            reason=kv.get("Reason", ""),
        ))

    gpu_jobs_running = any(j.gpus for j in jobs if j.state == "RUNNING")
    if gpu_jobs_running and not any(n.gpu_used for n in nodes):
        warnings.append("GPU jobs are running but no node reports used GPUs (GresUsed/AllocTRES): "
                        "free-GPU counts are probably wrong. Run fit_tool.py --once --explain and check "
                        "the raw node line.")
    pend_unplaced = sum(1 for j in jobs if j.state == "PENDING" and (j.start is None or not j.nodes))
    if pend_unplaced:
        warnings.append(f"{pend_unplaced} pending job(s) have no Slurm start estimate yet; "
                        "they are listed but not projected onto the timeline.")
    return Snapshot(now=now, nodes=nodes, jobs=jobs, source="slurm", warnings=warnings)


# ----------------------------------------------------------------------------
# synthetic cluster for testing without Slurm  (--demo)
# ----------------------------------------------------------------------------

def collect_demo(seed=7):
    rnd = random.Random(seed)
    now = time.time()
    nodes, jobs = [], []
    specs = [("cpu", 24, 64, 256 * 1024, {}, ["cpu", "ib"], ["batch"]),
             ("v100", 8, 40, 384 * 1024, {"v100": 4}, ["gpu", "v100", "ib"], ["gpu", "batch"]),
             ("a100", 8, 64, 512 * 1024, {"a100": 4}, ["gpu", "a100", "ib", "nvlink"], ["gpu"]),
             ("h200", 4, 96, 1536 * 1024, {"h200": 8}, ["gpu", "h200", "nvlink"], ["gpu"])]
    for prefix, count, cpus, mem, gpus, feats, parts in specs:
        for i in range(1, count + 1):
            state = "IDLE"
            if rnd.random() < 0.06:
                state = "DRAIN"
            nodes.append(Node(f"{prefix}{i:03d}", cpus, 0, mem, 0, dict(gpus), {k: 0 for k in gpus},
                              feats, parts, state))
    jid = 10000
    users = ["alice", "bob", "chen", "dmitri", "eva", "farah"]
    for n in nodes:
        if not n.usable or rnd.random() < 0.18:
            continue
        # fill node with 1-2 running jobs
        for _ in range(rnd.choice([1, 1, 2])):
            free_c = n.cpus - n.cpus_alloc
            if free_c <= 0:
                break
            c = rnd.choice([c for c in (2, 4, 8, 16, 32, free_c) if c <= free_c])
            g = {}
            for t, tot in n.gpus.items():
                fg = tot - n.gpus_used.get(t, 0)
                if fg > 0:
                    g[t] = rnd.choice([x for x in (1, 2, 4, 8, fg) if x <= fg])
            m = min(n.mem - n.mem_alloc, c * rnd.choice([2, 4, 8]) * 1024)
            limit = rnd.choice([3600, 7200, 4 * 3600, 8 * 3600, 24 * 3600, 48 * 3600])
            elapsed = rnd.uniform(0.05, 0.95) * limit
            jid += 1
            jobs.append(Job(str(jid), rnd.choice(["train", "sim", "pre", "eval", "mdrun"]),
                            rnd.choice(users), n.partitions[0], "RUNNING", [n.name], 1, c, m, g,
                            "", now - elapsed, now - elapsed + limit, limit))
            n.cpus_alloc += c
            n.mem_alloc += m
            for t, v in g.items():
                n.gpus_used[t] = n.gpus_used.get(t, 0) + v
    # a few multi-node running jobs on a100
    a = [n for n in nodes if n.name.startswith("a100") and n.usable and n.cpus_alloc == 0][:4]
    if len(a) >= 2:
        jid += 1
        jobs.append(Job(str(jid), "llm-pretrain", "chen", "gpu", "RUNNING", [n.name for n in a[:2]], 2,
                        128, 2 * 400 * 1024, {"a100": 8}, "nvlink", now - 3 * 3600, now + 21 * 3600, 24 * 3600))
        for n in a[:2]:
            n.cpus_alloc, n.mem_alloc, n.gpus_used = 64, 400 * 1024, {"a100": 4}
    # pending jobs: some scheduled, some not
    for k in range(14):
        jid += 1
        gt = rnd.choice(["v100", "a100", "h200", None])
        nn = rnd.choice([1, 1, 1, 2])
        limit = rnd.choice([3600, 4 * 3600, 12 * 3600, 24 * 3600])
        sched = rnd.random() < 0.6
        cand = [n for n in nodes if n.usable and (gt in n.gpus if gt else not n.gpus)]
        sel = rnd.sample(cand, min(nn, len(cand))) if (sched and cand) else []
        start = now + rnd.uniform(0.5, 30) * 3600 if sched else None
        jobs.append(Job(str(jid), rnd.choice(["sweep", "finetune", "md", "cfd"]), rnd.choice(users),
                        "gpu" if gt else "batch", "PENDING", [n.name for n in sel], nn,
                        8 * nn, 64 * 1024 * nn, {gt: 2 * nn} if gt else {}, gt or "",
                        start, None if start is None else start + limit, limit,
                        "Resources" if sched else "Priority"))
    snap = Snapshot(now, nodes, jobs, source="demo")
    snap.warnings.append("DEMO MODE: synthetic cluster, not real Slurm data.")
    return snap


def collect(demo=False):
    return collect_demo() if demo else collect_slurm()


# ----------------------------------------------------------------------------
# timeline: per-node free resources over time
# ----------------------------------------------------------------------------

class NodeTimeline:
    """Piecewise-constant free resources for one node, from now to INF."""

    def __init__(self, node, now):
        self.node = node
        self.now = now
        self.events = []        # (time, dcpu, dmem, dgpus dict, label)

    def add(self, t, dcpu, dmem, dgpus, label):
        if t is None:
            return
        if t <= self.now:            # already due (COMPLETING, overrunning): release imminently
            t = self.now + 1
        self.events.append((t, dcpu, dmem, dgpus, label))

    def segments(self):
        """-> list of (t_start, t_end, free_cpu, free_mem, free_gpus{type}, label_of_change)."""
        n = self.node
        cpu = n.cpus - n.cpus_alloc
        mem = n.mem - n.mem_alloc
        gpus = {k: v - n.gpus_used.get(k, 0) for k, v in n.gpus.items()}
        segs = []
        t = self.now
        label = "now"
        for ev in sorted(e for e in self.events if e[0] > self.now) + [(INF, 0, 0, {}, "")]:
            segs.append((t, ev[0], cpu, mem, dict(gpus), label))
            if ev[0] == INF:
                break
            cpu = max(0, min(n.cpus, cpu + ev[1]))
            mem = max(0, min(n.mem, mem + ev[2]))
            for k, v in ev[3].items():
                if k in gpus:
                    gpus[k] = max(0, min(n.gpus[k], gpus[k] + v))
            t, label = ev[0], ev[4]
        # merge consecutive identical segments
        merged = []
        for s in segs:
            if merged and merged[-1][2:5] == s[2:5]:
                p = merged[-1]
                merged[-1] = (p[0], s[1], p[2], p[3], p[4], p[5])
            else:
                merged.append(s)
        return merged


def _gpus_on_node(g, node, releasing):
    """
    Translate a job's per-node GPU dict onto the node's GPU types. Jobs submitted with an untyped
    --gres=gpu:N show up in TRES as plain 'gpu'; on a node whose GRES is typed (h200) that count
    must land in the h200 bucket or it is never released/reserved.
    """
    if "gpu" not in g or not node.gpus:
        return dict(g)
    out = {k: v for k, v in g.items() if k != "gpu"}
    n = g["gpu"]
    if len(node.gpus) == 1:
        t = next(iter(node.gpus))
        out[t] = out.get(t, 0) + n
        return out
    # multiple types on one node: releases follow what is currently in use, reservations take
    # whatever is free, both in the node's declared type order
    order = sorted(node.gpus, key=lambda t: -(node.gpus_used.get(t, 0) if releasing
                                              else node.gpus[t] - node.gpus_used.get(t, 0)))
    for t in order:
        cap = node.gpus_used.get(t, 0) if releasing else node.gpus[t] - node.gpus_used.get(t, 0)
        take = min(n, cap) if cap > 0 else 0
        if take:
            out[t] = out.get(t, 0) + take
            n -= take
    if n > 0:                      # whatever is left goes to the first type
        t = next(iter(node.gpus))
        out[t] = out.get(t, 0) + n
    return out


def build_timelines(snap):
    tl = {n.name: NodeTimeline(n, snap.now) for n in snap.nodes}
    for j in snap.running:
        c, m, g = j.per_node()
        for name in j.nodes:
            if name in tl and j.end is not None and j.end != INF:
                gg = _gpus_on_node(g, tl[name].node, releasing=True)
                tl[name].add(j.end, +c, +m, {k: +v for k, v in gg.items()}, f"job {j.id} ({j.user}) ends")
    for j in snap.pending:
        if j.start is None or not j.nodes:
            continue
        c, m, g = j.per_node()
        for name in j.nodes:
            if name in tl:
                gg = _gpus_on_node(g, tl[name].node, releasing=False)
                tl[name].add(j.start, -c, -m, {k: -v for k, v in gg.items()}, f"job {j.id} ({j.user}) starts")
                if j.end is not None and j.end != INF:
                    tl[name].add(j.end, +c, +m, {k: +v for k, v in gg.items()}, f"job {j.id} ({j.user}) ends")
    return tl


# ----------------------------------------------------------------------------
# fit estimation
# ----------------------------------------------------------------------------

@dataclass
class Request:
    nodes: int = 1
    cpus: int = 4           # per node
    mem: int = 16 * 1024    # per node, MB
    gpus: int = 0           # per node
    gpu_type: str = "any"   # 'any' or a type name
    features: list = field(default_factory=list)
    partition: str = ""     # '' = any
    wall: int = 3600        # seconds

    def __post_init__(self):
        self.gpu_type = (self.gpu_type or "any").lower()

    def describe(self):
        g = f", {self.gpus}x{GPU_TYPE_DISPLAY.get(self.gpu_type.lower(), self.gpu_type)} gpu" if self.gpus else ""
        f = f", features {','.join(self.features)}" if self.features else ""
        p = f", partition {self.partition}" if self.partition else ""
        return (f"{self.nodes} node(s) x [{self.cpus} cpu, {fmt_mem(self.mem)}{g}]"
                f" for {fmt_dur(self.wall)}{f}{p}")


def _gpu_count(gpus, gpu_type):
    """Count of GPUs of gpu_type in a {type: n} dict, case-insensitive; 'any' sums all."""
    if gpu_type == "any":
        return sum(gpus.values())
    return gpus.get(gpu_type.lower(), 0)


def _node_eligible(n, req):
    if not n.usable:
        return False
    if req.partition and req.partition not in n.partitions:
        return False
    if any(f not in n.features for f in req.features):
        return False
    if n.cpus < req.cpus or n.mem < req.mem:
        return False
    if req.gpus and _gpu_count(n.gpus, req.gpu_type) < req.gpus:
        return False
    return True


def _seg_fits(seg, req):
    _, _, cpu, mem, gpus, _ = seg
    if cpu < req.cpus or mem < req.mem:
        return False
    if req.gpus and _gpu_count(gpus, req.gpu_type) < req.gpus:
        return False
    return True


def node_start_windows(segs, req):
    """Intervals [s, e] of feasible *start* times on this node (e may be INF)."""
    wins = []
    run_start = None
    for seg in segs + [(INF, INF, 0, 0, {}, "")]:
        if _seg_fits(seg, req):
            if run_start is None:
                run_start = seg[0]
        else:
            if run_start is not None:
                free_until = seg[0]
                if free_until == INF:
                    wins.append((run_start, INF))
                elif free_until - run_start >= req.wall:
                    wins.append((run_start, free_until - req.wall))
                run_start = None
    return wins


def _why_no_window(snap, segs, node, req):
    """Human explanation of why a node never offers a long-enough window."""
    last = segs[-1]
    longest = 0
    run_start = None
    for seg in segs + [(INF, INF, 0, 0, {}, "")]:
        if seg[0] != INF and _seg_fits(seg, req):
            run_start = seg[0] if run_start is None else run_start
        elif run_start is not None:
            longest = max(longest, seg[0] - run_start)
            run_start = None
    if not _seg_fits(last, req):
        short = []
        if last[2] < req.cpus:
            short.append(f"{last[2]} cpu")
        if last[3] < req.mem:
            short.append(f"{fmt_mem(last[3])} mem")
        if req.gpus and _gpu_count(last[4], req.gpu_type) < req.gpus:
            short.append(f"{_gpu_count(last[4], req.gpu_type)} {req.gpu_type} gpu")
        holders = [f"job {j.id} ({j.user})" for j in snap.running if node.name in j.nodes and j.end == INF]
        why = f"never has enough ({', '.join(short)} at best)"
        if holders:
            why += "; held by " + ", ".join(holders) + " with no time limit"
        else:
            why += "; a job on it may be reporting resources this tool cannot match - run --explain"
        return why
    return f"longest free window is {fmt_dur(longest)}, shorter than the {fmt_dur(req.wall)} walltime"


def estimate(snap, req, timelines=None):
    """
    -> dict(start=epoch|None, nodes=[names], eligible=int, reason=str)
    """
    timelines = timelines or build_timelines(snap)
    per_node = {}
    eligible = 0
    for n in snap.nodes:
        if not _node_eligible(n, req):
            continue
        eligible += 1
        wins = node_start_windows(timelines[n.name].segments(), req)
        if wins:
            per_node[n.name] = wins
    if eligible < req.nodes:
        return {"start": None, "nodes": [], "eligible": eligible,
                "reason": f"only {eligible} node(s) in the cluster can ever satisfy this per-node request"}
    if len(per_node) < req.nodes:
        blocked = []
        for n in snap.nodes:
            if not _node_eligible(n, req) or n.name in per_node:
                continue
            blocked.append(f"{n.name}: {_why_no_window(snap, timelines[n.name].segments(), n, req)}")
        return {"start": None, "nodes": [], "eligible": eligible, "blocked": blocked,
                "reason": (f"only {len(per_node)} of the {eligible} eligible node(s) ever have a free window "
                           f"of {fmt_dur(req.wall)} for this request. " + " | ".join(blocked))}
    candidates = sorted({s for wins in per_node.values() for s, _ in wins})
    for t in candidates:
        ok = [name for name, wins in per_node.items() if any(s <= t <= e for s, e in wins)]
        if len(ok) >= req.nodes:
            return {"start": t, "nodes": sorted(ok)[:req.nodes], "eligible": eligible, "reason": ""}
    return {"start": None, "nodes": [], "eligible": eligible,
            "reason": "no common window for the requested number of nodes"}


def explain(snap, req, result, timelines=None):
    """Per-node evidence behind an estimate: what is free now and which jobs sit on those nodes."""
    timelines = timelines or build_timelines(snap)
    names = result["nodes"] or [n.name for n in snap.nodes if _node_eligible(n, req)][:req.nodes]
    by_name = {n.name: n for n in snap.nodes}
    out = []
    for name in names:
        n = by_name[name]
        segs = timelines[name].segments()
        seg_at_start = next((s for s in segs if s[0] <= (result["start"] or snap.now) < s[1]), segs[0])
        out.append({
            "node": name, "state": n.state,
            "cpu_free_now": n.cpus - n.cpus_alloc, "cpu_total": n.cpus,
            "mem_free_now": n.mem - n.mem_alloc, "mem_total": n.mem,
            "gpu_free_now": {k: v - n.gpus_used.get(k, 0) for k, v in n.gpus.items()},
            "gpu_total": dict(n.gpus),
            "gpu_free_at_start": seg_at_start[4], "cpu_free_at_start": seg_at_start[2],
            "running": [{"id": j.id, "user": j.user, "gpus": j.per_node()[2], "cpus": j.per_node()[0],
                         "end": j.end} for j in snap.running if name in j.nodes],
            "scheduled": [{"id": j.id, "user": j.user, "gpus": j.per_node()[2], "start": j.start, "end": j.end}
                          for j in snap.pending if name in j.nodes],
            "raw": n.raw,
        })
    return out


def suggestions(snap, req, base, timelines=None, max_items=8):
    """Single-dimension tweaks that would start the job earlier."""
    timelines = timelines or build_timelines(snap)
    base_start = base["start"] if base["start"] is not None else INF
    out = []

    def consider(label, **changes):
        r = Request(**{**asdict(req), **changes})
        e = estimate(snap, r, timelines)
        s = e["start"] if e["start"] is not None else INF
        if s < base_start - 60:
            out.append({"label": label, "start": s, "gain": base_start - s, "request": asdict(r)})

    if req.wall > 900:
        for frac, name in ((0.5, "halve"), (0.25, "quarter")):
            w = max(900, int(req.wall * frac))
            consider(f"{name} walltime to {fmt_dur(w)}", wall=w)
    if req.gpus > 1:
        consider(f"{req.gpus - 1} GPU(s) per node instead of {req.gpus}", gpus=req.gpus - 1)
    if req.gpus and req.gpu_type != "any":
        consider("accept any GPU type", gpu_type="any")
        for t in snap.gpu_types:
            if t != req.gpu_type:
                consider(f"use {t} instead of {req.gpu_type}", gpu_type=t)
    if req.cpus > 1:
        consider(f"{max(1, req.cpus // 2)} CPUs per node instead of {req.cpus}", cpus=max(1, req.cpus // 2))
    if req.mem > 1024:
        consider(f"{fmt_mem(req.mem // 2)} per node instead of {fmt_mem(req.mem)}", mem=req.mem // 2)
    if req.nodes > 1:
        consider(f"{req.nodes - 1} node(s) instead of {req.nodes}", nodes=req.nodes - 1)
    if req.features:
        consider("drop feature constraints", features=[])
    if req.partition:
        consider("any partition", partition="")
    out.sort(key=lambda x: x["start"])
    return out[:max_items]


def sbatch_snippet(req):
    lines = ["#!/bin/bash", f"#SBATCH --nodes={req.nodes}", "#SBATCH --ntasks-per-node=1",
             f"#SBATCH --cpus-per-task={req.cpus}",
             f"#SBATCH --mem={req.mem}M", f"#SBATCH --time={fmt_slurm_time(req.wall)}"]
    if req.gpus:
        t = "" if req.gpu_type == "any" else GPU_TYPE_DISPLAY.get(req.gpu_type.lower(), req.gpu_type) + ":"
        lines.append(f"#SBATCH --gpus-per-node={t}{req.gpus}")
    if req.features:
        lines.append(f"#SBATCH --constraint={'&'.join(req.features)}")
    if req.partition:
        lines.append(f"#SBATCH --partition={req.partition}")
    return "\n".join(lines)


def fmt_slurm_time(sec):
    d, r = divmod(int(sec), 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    return f"{d}-{h:02d}:{m:02d}:{s:02d}" if d else f"{h:02d}:{m:02d}:{s:02d}"


# ----------------------------------------------------------------------------
# aggregate free-capacity curve (for charts / gap summaries)
# ----------------------------------------------------------------------------

def capacity_curve(snap, timelines, horizon_s=48 * 3600, partition=None, step=None):
    """
    -> list of dict(t, cpus, mem, gpus{type}, idle_nodes) at every change point
       within the horizon (plus the horizon end).
    """
    segs_by_node = {}
    points = {snap.now, snap.now + horizon_s}
    for n in snap.nodes:
        if not n.usable or (partition and partition not in n.partitions):
            continue
        s = timelines[n.name].segments()
        segs_by_node[n.name] = s
        for seg in s:
            if snap.now <= seg[0] <= snap.now + horizon_s:
                points.add(seg[0])
    curve = []
    for t in sorted(points):
        row = {"t": t, "cpus": 0, "mem": 0, "gpus": {}, "idle_nodes": 0}
        for name, segs in segs_by_node.items():
            n = timelines[name].node
            for seg in segs:
                if seg[0] <= t < seg[1]:
                    row["cpus"] += seg[2]
                    row["mem"] += seg[3]
                    for k, v in seg[4].items():
                        row["gpus"][k] = row["gpus"].get(k, 0) + v
                    if seg[2] == n.cpus and all(seg[4].get(k, 0) == v for k, v in n.gpus.items()):
                        row["idle_nodes"] += 1
                    break
        curve.append(row)
    return curve


# ----------------------------------------------------------------------------
# shared report rows (used by the CLI tools and the web app)
# ----------------------------------------------------------------------------

def node_gaps(snap, timelines, partition=None, min_cpus=1, min_gpus=0, include_unusable=False):
    """Free space per node right now and how long it is guaranteed to stay free."""
    now = snap.now
    rows = []
    for n in sorted(snap.nodes, key=lambda n: n.name):
        if partition and partition not in n.partitions:
            continue
        if not n.usable and not include_unusable:
            continue
        segs = timelines[n.name].segments()
        s0 = segs[0]
        fcpu, fmem, fg = s0[2], s0[3], s0[4]
        fg_total = sum(fg.values())
        if fcpu < min_cpus or fg_total < min_gpus:
            continue
        until, nxt = INF, ""
        for seg in segs[1:]:
            if seg[2] < fcpu or seg[3] < fmem or sum(seg[4].values()) < fg_total:
                until, nxt = seg[0], seg[5]
                break
        rows.append({
            "node": n.name,
            "kind": "idle" if fcpu == n.cpus and fg_total == sum(n.gpus.values()) else "partial",
            "cpu": fcpu, "cpu_total": n.cpus, "mem": fmem, "mem_total": n.mem,
            "gpus": fg, "gpus_total": dict(n.gpus),
            "until": None if until == INF else until,
            "stays_free": "open-ended" if until == INF else fmt_dur(until - now),
            "next_change": nxt, "features": list(n.features), "partitions": list(n.partitions),
            "state": n.state, "usable": n.usable,
        })
    return rows


def pending_estimate(snap, timelines, j):
    """Best-case start for a pending job Slurm has not scheduled yet (priority ignored)."""
    c, m, g = j.per_node()
    gtype, gcount = "any", 0
    if g:
        gtype, gcount = max(g.items(), key=lambda kv: kv[1])
        if gtype == "gpu":
            gtype = "any"
    feats = [f for f in j.features.replace("&", ",").split(",") if f and "|" not in f] if j.features else []
    req = Request(nodes=max(1, j.num_nodes), cpus=max(1, c), mem=m, gpus=gcount, gpu_type=gtype,
                  features=feats, partition=j.partition, wall=j.time_limit or 3600)
    return estimate(snap, req, timelines)["start"]


def pending_rows(snap, timelines=None, user=None, partition=None, with_estimate=False):
    """Pending jobs with requirements and expected start, soonest first."""
    now = snap.now
    out = []
    for j in snap.pending:
        if (user and j.user != user) or (partition and j.partition != partition):
            continue
        start, kind = j.start, "slurm"
        if start is None and with_estimate:
            start, kind = pending_estimate(snap, timelines or build_timelines(snap), j), "estimate"
        out.append({
            "job": j.id, "user": j.user, "name": j.name, "partition": j.partition,
            "nodes": j.num_nodes, "cpus": j.cpus, "mem": fmt_mem(j.mem), "mem_mb": j.mem,
            "gpus": fmt_gpus(j.gpus), "gpus_raw": j.gpus, "features": j.features or "",
            "limit": fmt_dur(j.time_limit), "time_limit_s": j.time_limit,
            "start": None if start in (None, INF) else start,
            "start_source": None if start in (None, INF) else kind,
            "start_text": ("no estimate" if start is None else "never fits" if start == INF
                           else ("~" if kind == "estimate" else "") + fmt_time(start, now)),
            "start_in": "" if start in (None, INF) else fmt_dur(start - now),
            "reason": j.reason, "planned_nodes": list(j.nodes),
        })
    out.sort(key=lambda r: (r["start"] is None, r["start"] or 0, r["job"]))
    return out
