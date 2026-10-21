#!/usr/bin/env python3
"""
job_forecaster_web.py - the browser UI as a WSGI application.

Used two ways:
  * Open OnDemand: passenger_wsgi.py imports `application` and Passenger serves it as the
    logged-in user under /pun/sys/<app>/ (no port forwarding, no extra auth).
  * fit_tool.py --web: the same application on a local port via wsgiref.

Routes (all GET, relative to the mount point):
  /                 single page: "Fit my job", "Cluster", "Pending jobs"
  /api/limits       slider ranges, GPU types, features, partitions, initial request
  /api/estimate     earliest start for ?nodes=&cpus=&mem=&gpus=&gpu_type=&features=&partition=&wall=
  /api/planned      pending queue + running jobs finishing soonest (?all=1 for every user)
  /api/cluster      totals, free-capacity curve, per-node gaps (?partition=&min_gpus=&unusable=1)
  /api/pending      pending jobs with requirements and start (?all=1&estimate=1&partition=)
  /healthz          "ok"
"""

import getpass
import json
import os
import threading
import time
import urllib.parse
from dataclasses import asdict

from job_forecaster import (Request, collect, build_timelines, estimate, suggestions, sbatch_snippet, explain,
                      capacity_curve, node_gaps, pending_rows, GPU_TYPE_DISPLAY,
                      fmt_mem, fmt_dur, fmt_time, fmt_eta, fmt_gpus, INF)

SNAPSHOT_TTL = int(os.environ.get("JOB_FORECASTER_TTL", "30"))   # seconds between scontrol re-reads
DEMO = os.environ.get("JOB_FORECASTER_DEMO", "") not in ("", "0", "false")


def current_user(demo):
    if demo:
        return "alice"
    return os.environ.get("JOB_FORECASTER_USER") or getpass.getuser()


class Cluster:
    """Snapshot + timelines, refreshed lazily and shared by all requests in this process."""

    def __init__(self, demo=False):
        self.demo = demo
        self.lock = threading.Lock()
        self.snap = None
        self.tl = None
        self.loaded_at = 0
        self.error = None

    def refresh(self, force=False):
        with self.lock:
            if not force and self.snap is not None and time.time() - self.loaded_at < SNAPSHOT_TTL:
                return
            try:
                self.snap = collect(demo=self.demo)
                self.tl = build_timelines(self.snap)
                self.loaded_at = time.time()
                self.error = None
            except Exception as e:  # noqa
                self.error = str(e)
                if self.snap is None:
                    raise

    def limits(self):
        self.refresh()
        s = self.snap
        usable = [n for n in s.nodes if n.usable] or s.nodes
        return {
            "max_nodes": max(1, min(64, len(usable))),
            "max_cpus": max(n.cpus for n in usable) if usable else 1,
            "max_mem_gb": max(1, max(n.mem for n in usable) // 1024) if usable else 1,
            "max_gpus": max([max(n.gpus.values()) for n in usable if n.gpus] or [0]),
            "gpu_types": s.gpu_types,
            "gpu_type_names": {t: GPU_TYPE_DISPLAY.get(t, t) for t in s.gpu_types},
            "features": s.feature_list,
            "partitions": s.partition_list,
        }

    def evaluate(self, req, horizon_h=48):
        self.refresh()
        with self.lock:
            snap, tl = self.snap, self.tl
        base = estimate(snap, req, tl)
        sugg = suggestions(snap, req, base, tl)
        curve = capacity_curve(snap, tl, horizon_s=horizon_h * 3600, partition=req.partition or None)
        detail = []
        if base["nodes"]:
            for r in explain(snap, req, base, tl):
                g = r["gpu_free_at_start"] if req.gpus else None
                detail.append(f"{r['node']} ({r['cpu_free_at_start']} cpu"
                              + (f", {fmt_gpus(g)} gpu free" if g is not None else " free")
                              + (f", {len(r['running'])} job(s) running" if r["running"] else "") + ")")
        warnings = list(snap.warnings)
        if self.error:
            warnings.append(f"Slurm could not be re-read ({self.error}); showing data from "
                            f"{fmt_time(self.loaded_at)}.")
        return {
            "now": snap.now, "request": asdict(req), "describe": req.describe(),
            "start": base["start"], "eta": fmt_eta(base["start"], snap.now),
            "wait_s": None if base["start"] is None else base["start"] - snap.now,
            "nodes": base["nodes"], "node_detail": detail, "eligible": base["eligible"], "reason": base["reason"],
            "suggestions": [{**x, "eta": fmt_eta(x["start"], snap.now), "gain": fmt_dur(x["gain"])} for x in sugg],
            "sbatch": sbatch_snippet(req),
            "curve": [{**r, "mem_gb": r["mem"] // 1024} for r in curve],
            "warnings": warnings,
        }

    def planned_work(self, user=None):
        self.refresh()
        with self.lock:
            snap = self.snap
        now = snap.now
        pend = sorted([j for j in snap.pending if not user or j.user == user],
                      key=lambda j: (j.start is None, j.start or 0))
        run = sorted([j for j in snap.running if j.end not in (None, INF) and (not user or j.user == user)],
                     key=lambda j: j.end)
        return {
            "pending": [{"id": j.id, "user": j.user, "name": j.name, "partition": j.partition,
                         "nodes": j.num_nodes, "cpus": j.cpus, "mem": fmt_mem(j.mem),
                         "gpus": fmt_gpus(j.gpus), "limit": fmt_dur(j.time_limit),
                         "start": None if j.start is None else fmt_time(j.start, now),
                         "start_in": None if j.start is None else fmt_dur(j.start - now),
                         "reason": j.reason, "planned": ",".join(j.nodes)} for j in pend],
            "ending": [{"id": j.id, "user": j.user, "name": j.name, "partition": j.partition,
                        "nodes": ",".join(j.nodes[:4]) + ("…" if len(j.nodes) > 4 else ""),
                        "cpus": j.cpus, "mem": fmt_mem(j.mem), "gpus": fmt_gpus(j.gpus),
                        "end": fmt_time(j.end, now), "end_in": fmt_dur(j.end - now)} for j in run[:25]],
            "totals": snap.totals(), "user": user,
        }

    def cluster_report(self, partition=None, min_gpus=0, unusable=False, horizon_h=48):
        self.refresh()
        with self.lock:
            snap, tl = self.snap, self.tl
        now = snap.now
        curve, last = [], None
        for row in capacity_curve(snap, tl, horizon_s=horizon_h * 3600, partition=partition):
            key = (row["cpus"], row["mem"], tuple(sorted(row["gpus"].items())), row["idle_nodes"])
            if key == last:
                continue
            last = key
            curve.append({"time": fmt_time(row["t"], now), "offset": "now" if row["t"] == now else "+" + fmt_dur(row["t"] - now),
                          "cpus": row["cpus"], "mem_gb": row["mem"] // 1024, "gpus_text": fmt_gpus(row["gpus"]),
                          "idle_nodes": row["idle_nodes"]})
            if len(curve) >= 60:
                break
        gaps = node_gaps(snap, tl, partition=partition, min_gpus=min_gpus, include_unusable=unusable)
        for g in gaps:
            g["gpus_text"] = fmt_gpus(g["gpus"]) if g["gpus_total"] else "-"
        return {"now": now, "totals": snap.totals(partition), "gpu_names": dict(GPU_TYPE_DISPLAY),
                "curve": curve, "gaps": gaps, "warnings": snap.warnings}

    def pending_report(self, user=None, partition=None, with_estimate=False):
        self.refresh()
        with self.lock:
            snap, tl = self.snap, self.tl
        return {"now": snap.now, "user": user,
                "rows": pending_rows(snap, tl, user=user, partition=partition or None, with_estimate=with_estimate),
                "warnings": snap.warnings}


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Job Forecaster</title>
<style>
:root{color-scheme:light dark;
  --bg:#f1f3f6;--panel:#ffffff;--ink:#1b2430;--muted:#66717f;--line:#d9dee6;
  --free:#1f7a74;--free-soft:#d6ecea;--job:#c9742a;--job-soft:#f8e3cf;--focus:#2b5cb8;
  --font:"Avenir Next","Segoe UI",system-ui,-apple-system,"Helvetica Neue",sans-serif;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace}
@media (prefers-color-scheme:dark){:root{
  --bg:#12171e;--panel:#1a2029;--ink:#e4e9f0;--muted:#8d98a6;--line:#2c3540;
  --free:#5fcfc6;--free-soft:#163a38;--job:#f0a35a;--job-soft:#4a2f16;--focus:#7fa6f0}}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 var(--font)}
main{max-width:1180px;margin:0 auto;padding:18px 20px 60px}
.tabs{display:flex;gap:4px;margin:0 0 22px;border-bottom:1px solid var(--line)}
.tabs button{font:inherit;font-weight:500;background:none;border:0;border-bottom:2px solid transparent;color:var(--muted);padding:8px 14px;cursor:pointer;margin-bottom:-1px}
.tabs button[aria-selected=true]{color:var(--ink);border-bottom-color:var(--free)}
.note select,.note button{font:inherit;font-size:13px;padding:3px 6px;border-radius:6px;border:1px solid var(--line);background:var(--panel);color:var(--ink)}
.scroll.tall{max-height:460px;overflow:auto}
#view-cluster td,#view-pending td,#view-cluster th,#view-pending th{white-space:nowrap}
h1{font-size:30px;font-weight:600;letter-spacing:-.01em;margin:0 0 4px}
h2{font-size:17px;font-weight:600;margin:0 0 12px}
.sub{color:var(--muted);margin:0 0 24px;max-width:70ch}
.grid{display:grid;grid-template-columns:minmax(300px,380px) 1fr;gap:20px;align-items:start}
@media (max-width:820px){.grid{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px 20px}
.ctl{display:grid;grid-template-columns:1fr auto;gap:2px 12px;align-items:baseline;padding:10px 0;border-top:1px solid var(--line)}
.ctl:first-of-type{border-top:0;padding-top:0}
.ctl label{font-weight:500}
.ctl output{font-variant-numeric:tabular-nums;color:var(--muted)}
.ctl input[type=range]{grid-column:1/-1;width:100%;accent-color:var(--free);margin:4px 0 0}
.ctl select{grid-column:1/-1;margin-top:6px;font:inherit;padding:6px 8px;border-radius:6px;border:1px solid var(--line);background:var(--bg);color:var(--ink)}
.chips{grid-column:1/-1;display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}
.chips label{font-weight:400;border:1px solid var(--line);border-radius:999px;padding:2px 10px;cursor:pointer;user-select:none}
.chips input{display:none}
.chips input:checked+span{color:var(--free);font-weight:600}
.chips label:has(input:checked){border-color:var(--free);background:var(--free-soft)}
:focus-visible{outline:2px solid var(--focus);outline-offset:2px}
.eta{font-size:44px;font-weight:600;letter-spacing:-.02em;line-height:1.1;margin:2px 0 2px}
.eta.never{color:var(--job)}
.when{color:var(--muted)}
.on{margin-top:6px;font-family:var(--mono);font-size:13px;color:var(--muted);word-break:break-all}
svg{width:100%;height:auto;display:block;margin-top:14px}
.legend{display:flex;gap:18px;color:var(--muted);font-size:13px;margin-top:4px}
.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-1px;margin-right:6px}
ol.sugg{margin:8px 0 0;padding:0;list-style:none}
ol.sugg li{display:flex;justify-content:space-between;gap:12px;padding:8px 0;border-top:1px solid var(--line)}
ol.sugg li:first-child{border-top:0}
ol.sugg button{font:inherit;background:none;border:0;color:var(--focus);cursor:pointer;padding:0;text-align:left}
ol.sugg button:hover{text-decoration:underline}
.gain{white-space:nowrap;color:var(--free);font-weight:600;font-variant-numeric:tabular-nums}
.gain small{display:block;font-weight:400;color:var(--muted)}
pre{font:13px/1.45 var(--mono);background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:12px 14px;overflow-x:auto;margin:10px 0 0}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:20px}
@media (max-width:820px){.row2{grid-template-columns:1fr}}
table{width:100%;border-collapse:collapse;font-size:13.5px;font-variant-numeric:tabular-nums}
th{text-align:left;font-weight:500;color:var(--muted);padding:4px 8px 6px 0;border-bottom:1px solid var(--line)}
td{padding:5px 8px 5px 0;border-bottom:1px solid var(--line);vertical-align:top}
tr:last-child td{border-bottom:0}
.scroll{overflow-x:auto}
.note{color:var(--muted);font-size:13px;margin-top:14px}
.stale{color:var(--job)}
@media (prefers-reduced-motion:no-preference){.eta{transition:color .2s}}
</style>
</head>
<body>
<main>
<nav class="tabs" role="tablist">
  <button role="tab" data-view="fit" aria-selected="true">Fit my job</button>
  <button role="tab" data-view="cluster">Cluster</button>
  <button role="tab" data-view="pending">Pending jobs</button>
</nav>

<section id="view-fit" class="view">
<h1>Will my job fit?</h1>
<p class="sub">Move the sliders and watch the earliest start time. The estimate counts free nodes, running jobs' end times and already-scheduled pending jobs; it does not know your priority, so it is a best case, but the <em>difference</em> between two settings is what matters.</p>

<div class="grid">
  <section class="panel" aria-label="Job requirements">
    <h2>Your job</h2>
    <div class="ctl"><label for="nodes">Nodes</label><output id="nodes-o"></output>
      <input type="range" id="nodes" min="1" value="1"></div>
    <div class="ctl"><label for="cpus">CPUs per node</label><output id="cpus-o"></output>
      <input type="range" id="cpus" min="1" value="4"></div>
    <div class="ctl"><label for="mem">Memory per node</label><output id="mem-o"></output>
      <input type="range" id="mem" min="1" value="16"></div>
    <div class="ctl"><label for="gpus">GPUs per node</label><output id="gpus-o"></output>
      <input type="range" id="gpus" min="0" value="0"></div>
    <div class="ctl"><label for="gpu_type">GPU type</label><span></span>
      <select id="gpu_type"></select></div>
    <div class="ctl"><label>Node features</label><span></span>
      <div class="chips" id="features"></div></div>
    <div class="ctl"><label for="partition">Partition</label><span></span>
      <select id="partition"></select></div>
    <div class="ctl"><label for="wall">Walltime</label><output id="wall-o"></output>
      <input type="range" id="wall" min="0" value="2"></div>
  </section>

  <section class="panel" aria-live="polite">
    <h2>Earliest start</h2>
    <div class="eta" id="eta">…</div>
    <div class="when" id="when"></div>
    <div class="on" id="on"></div>
    <svg id="chart" viewBox="0 0 760 190" role="img" aria-label="Free capacity over the next 48 hours"></svg>
    <div class="legend"><span><i style="background:var(--free-soft);border:1px solid var(--free)"></i>free <b id="legend-res"></b></span>
      <span><i style="background:var(--job-soft);border:1px solid var(--job)"></i>your job</span>
      <span id="need"></span></div>
    <h2 style="margin-top:22px">Start sooner if you…</h2>
    <ol class="sugg" id="sugg"></ol>
    <h2 style="margin-top:22px">Submit it like this</h2>
    <pre id="sbatch"></pre>
    <p class="note" id="warn"></p>
  </section>
</div>

<p class="note" style="margin:22px 0 -8px"><label><input type="checkbox" id="allusers"> Show every user's jobs in the two tables below (default: only yours)</label></p>
<div class="row2">
  <section class="panel">
    <h2>Planned work (pending queue)</h2>
    <div class="scroll"><table id="pending"></table></div>
  </section>
  <section class="panel">
    <h2>Running jobs finishing soonest</h2>
    <div class="scroll"><table id="ending"></table></div>
  </section>
</div>
<p class="note" id="status"></p>
</section>

<section id="view-cluster" class="view" hidden>
<h1>Cluster status</h1>
<p class="sub">What is free now, which jobs hold the rest, and where free space opens up. Totals and gaps are cluster-wide; pick a partition to narrow them.</p>
<p class="note"><label>Partition <select id="c-part"><option value="">all</option></select></label>
   &nbsp; <label>Gaps with at least <select id="c-mingpu"><option value="0">any</option><option value="1">1</option><option value="2">2</option><option value="4">4</option><option value="8">8</option></select> free GPUs</label>
   &nbsp; <label><input type="checkbox" id="c-unusable"> include down/drained nodes</label>
   &nbsp; <button type="button" id="c-reload">Refresh</button> <span id="c-stamp"></span></p>
<div class="row2">
  <section class="panel"><h2>Resources</h2><div class="scroll"><table id="c-totals"></table></div></section>
  <section class="panel"><h2>Free capacity over the next 48 h</h2><div class="scroll tall"><table id="c-curve"></table></div>
    <p class="note">Rises as running jobs end, drops as scheduled jobs start. Pending jobs without a Slurm start estimate are not included.</p></section>
</div>
<section class="panel" style="margin-top:20px"><h2>Gaps: free space on nodes right now</h2><div class="scroll"><table id="c-gaps"></table></div></section>
</section>

<section id="view-pending" class="view" hidden>
<h1>Pending jobs</h1>
<p class="sub">What each waiting job asked for and when Slurm expects to start it. Jobs held on priority usually have no Slurm estimate; the ~ figures are this tool's best case, ignoring priority.</p>
<p class="note"><label><input type="checkbox" id="p-all"> all users</label>
   &nbsp; <label><input type="checkbox" id="p-est" checked> estimate missing start times (~)</label>
   &nbsp; <label>Partition <select id="p-part"><option value="">all</option></select></label>
   &nbsp; <button type="button" id="p-reload">Refresh</button> <span id="p-stamp"></span></p>
<section class="panel"><div class="scroll"><table id="p-table"></table></div></section>
</section>
</main>

<script>
const WALL=[900,1800,3600,7200,14400,21600,28800,43200,86400,129600,172800,259200,432000,604800];
const API=location.pathname.replace(/\/+$/,'')+'/api';
const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function dur(s){s=Math.max(0,Math.round(s));const d=Math.floor(s/86400),h=Math.floor(s%86400/3600),m=Math.floor(s%3600/60);
  if(d)return `${d}d${String(h).padStart(2,'0')}h`;if(h)return `${h}h${String(m).padStart(2,'0')}m`;return `${m}m`;}
function mem(gb){return gb>=1024?(gb/1024).toFixed(1)+'T':gb+'G'}
let lim=null,timer=null,ctrl=null;

function params(){
  const feats=[...document.querySelectorAll('#features input:checked')].map(i=>i.value);
  return {nodes:+$('nodes').value,cpus:+$('cpus').value,mem:(+$('mem').value)*1024,gpus:+$('gpus').value,
    gpu_type:$('gpu_type').value,features:feats.join(','),partition:$('partition').value,wall:WALL[+$('wall').value]};
}
function showValues(){
  const p=params();
  $('nodes-o').textContent=p.nodes;$('cpus-o').textContent=p.cpus;$('mem-o').textContent=mem(p.mem/1024);
  $('gpus-o').textContent=p.gpus||'none';$('wall-o').textContent=dur(p.wall);
  $('gpu_type').disabled=!p.gpus;
}
function schedule(){showValues();clearTimeout(timer);timer=setTimeout(update,120);}
async function update(){
  if(ctrl)ctrl.abort();ctrl=new AbortController();
  const q=new URLSearchParams(params());
  let r;try{r=await(await fetch(API+'/estimate?'+q,{signal:ctrl.signal})).json();}catch(e){if(e.name!=='AbortError')$('status').textContent='Lost contact with the server — reload the page.';return;}
  render(r);
}
function render(r){
  const e=$('eta');
  if(r.start===null){e.textContent='does not fit';e.classList.add('never');$('when').textContent=r.reason+(r.eligible?` (${r.eligible} eligible nodes)`:'');$('on').textContent='';}
  else{e.classList.remove('never');e.textContent=r.wait_s<=1?'now':'in '+dur(r.wait_s);
    $('when').textContent=new Date(r.start*1000).toLocaleString([], {weekday:'short',hour:'2-digit',minute:'2-digit',month:'short',day:'numeric'});
    $('on').textContent='on '+(r.node_detail.length?r.node_detail.join('; '):r.nodes.join(', '));}
  const s=$('sugg');s.innerHTML='';
  if(!r.suggestions.length){s.innerHTML=`<li><span class="when">${r.start===null?'No single change makes this request fit — try several at once.':'Nothing simple starts it sooner; these requirements already fit as early as the cluster allows.'}</span></li>`;}
  for(const x of r.suggestions){const li=document.createElement('li');
    li.innerHTML=`<button type="button">${esc(x.label)}</button><span class="gain">${esc(x.eta.replace(/\s*\(.*\)/,''))}<small>saves ${esc(x.gain)}</small></span>`;
    li.querySelector('button').onclick=()=>applyRequest(x.request);s.appendChild(li);}
  $('sbatch').textContent=r.sbatch;
  $('warn').textContent=r.warnings.join(' ');
  drawChart(r);
  $('status').textContent='Snapshot '+new Date(r.now*1000).toLocaleTimeString()+' — refreshes every 30 s while you move a slider.';
}
function applyRequest(q){
  $('nodes').value=q.nodes;$('cpus').value=q.cpus;$('mem').value=Math.max(1,Math.round(q.mem/1024));$('gpus').value=q.gpus;
  $('gpu_type').value=q.gpu_type;$('partition').value=q.partition;
  let best=0;WALL.forEach((w,i)=>{if(Math.abs(w-q.wall)<Math.abs(WALL[best]-q.wall))best=i});$('wall').value=best;
  document.querySelectorAll('#features input').forEach(i=>i.checked=q.features.includes(i.value));
  schedule();
}
function drawChart(r){
  const p=r.request,W=760,H=190,L=44,R=10,T=14,B=28,hor=48*3600,now=r.now;
  const pick=row=>p.gpus?(p.gpu_type==='any'?Object.values(row.gpus).reduce((a,b)=>a+b,0):(row.gpus[p.gpu_type]||0)):row.cpus;
  const need=p.nodes*(p.gpus||p.cpus);
  const pts=r.curve.map(row=>[row.t,pick(row)]);
  const mx=Math.max(need,...pts.map(q=>q[1]),1);
  const x=t=>L+(W-L-R)*Math.min(1,Math.max(0,(t-now)/hor)),y=v=>T+(H-T-B)*(1-v/mx);
  let d=`M${x(now)},${y(pts[0]?pts[0][1]:0)}`;
  for(let i=0;i<pts.length;i++){const v=pts[i][1],t2=i+1<pts.length?pts[i+1][0]:now+hor;d+=` H${x(t2)}`;if(i+1<pts.length)d+=` V${y(pts[i+1][1])}`;}
  const area=d+` V${y(0)} H${x(now)} Z`;
  let job='';
  if(r.start!==null&&r.start-now<hor){const a=x(r.start),b=x(Math.min(now+hor,r.start+p.wall));
    job=`<rect x="${a}" y="${T}" width="${Math.max(2,b-a)}" height="${H-T-B}" fill="var(--job-soft)" stroke="var(--job)" stroke-width="1.2" rx="2"/>`;}
  let ticks='';
  for(const h of [0,6,12,18,24,30,36,42,48]){const xx=x(now+h*3600);ticks+=`<line x1="${xx}" y1="${T}" x2="${xx}" y2="${H-B}" stroke="var(--line)" stroke-dasharray="2 4"/><text x="${xx}" y="${H-8}" text-anchor="middle" font-size="11" fill="var(--muted)">${h?'+'+h+'h':'now'}</text>`;}
  const yt=[0,Math.round(mx/2),mx].map(v=>`<text x="${L-6}" y="${y(v)+4}" text-anchor="end" font-size="11" fill="var(--muted)">${v}</text>`).join('');
  $('chart').innerHTML=`${ticks}${yt}<path d="${area}" fill="var(--free-soft)"/><path d="${d}" fill="none" stroke="var(--free)" stroke-width="1.8"/>
    <line x1="${L}" y1="${y(need)}" x2="${W-R}" y2="${y(need)}" stroke="var(--job)" stroke-width="1" stroke-dasharray="5 4"/>${job}`;
  $('legend-res').textContent=p.gpus?(p.gpu_type==='any'?'GPUs (any type)':p.gpu_type+' GPUs'):'CPUs';
  $('need').textContent=`dashed line: ${need} ${p.gpus?'GPUs':'CPUs'} you need in total`;
}
function tableRows(id,cols,rows,empty){
  const t=$(id);if(!rows.length){t.innerHTML=`<tr><td class="when">${empty}</td></tr>`;return;}
  t.innerHTML='<tr>'+cols.map(c=>`<th>${c[0]}</th>`).join('')+'</tr>'+rows.map(r=>'<tr>'+cols.map(c=>`<td>${esc(c[1](r))}</td>`).join('')+'</tr>').join('');
}
async function loadPlanned(){
  const d=await(await fetch(API+'/planned'+($('allusers').checked?'?all=1':''))).json();
  const empty=d.user?`none of yours (${d.user})`:'none';
  tableRows('pending',[['job',r=>r.id],['user',r=>r.user],['part',r=>r.partition],['N',r=>r.nodes],['cpu',r=>r.cpus],['mem',r=>r.mem],['gpu',r=>r.gpus],['limit',r=>r.limit],
    ['expected start',r=>r.start?`${r.start} (${r.start_in})`:'no estimate ('+r.reason+')']],d.pending,'Pending: '+empty+'.');
  tableRows('ending',[['job',r=>r.id],['user',r=>r.user],['nodes',r=>r.nodes],['cpu',r=>r.cpus],['mem',r=>r.mem],['gpu',r=>r.gpus],['ends',r=>`${r.end} (${r.end_in})`]],d.ending,'Running: '+empty+'.');
}
function showView(v){
  document.querySelectorAll('.view').forEach(s=>s.hidden=(s.id!=='view-'+v));
  document.querySelectorAll('.tabs button').forEach(b=>b.setAttribute('aria-selected',b.dataset.view===v));
  if(v==='cluster'&&!$('c-totals').innerHTML)loadCluster();
  if(v==='pending'&&!$('p-table').innerHTML)loadPending();
  if(location.hash!=='#'+v)history.replaceState(null,'','#'+v);
}
async function loadCluster(){
  $('c-stamp').textContent='loading…';
  const q=new URLSearchParams({partition:$('c-part').value,min_gpus:$('c-mingpu').value,unusable:$('c-unusable').checked?1:''});
  const d=await(await fetch(API+'/cluster?'+q)).json();
  const t=d.totals,rows=[['Nodes',t.nodes,(t.nodes-t.nodes_usable)+' down/drained',t.nodes_usable,''],
    ['CPUs',t.cpus,t.cpus_alloc,t.cpus-t.cpus_alloc,pct(t.cpus_alloc,t.cpus)],
    ['Memory',mem(t.mem/1024),mem(t.mem_alloc/1024),mem((t.mem-t.mem_alloc)/1024),pct(t.mem_alloc,t.mem)]];
  for(const g of Object.keys(t.gpus).sort()){const u=t.gpus_used[g]||0;rows.push(['GPU '+(d.gpu_names[g]||g),t.gpus[g],u,t.gpus[g]-u,pct(u,t.gpus[g])]);}
  $('c-totals').innerHTML='<tr><th></th><th>total</th><th>used</th><th>available</th><th>util</th></tr>'+rows.map(r=>'<tr>'+r.map(c=>`<td>${esc(c)}</td>`).join('')+'</tr>').join('');
  tableRows('c-curve',[['time',r=>r.time],['offset',r=>r.offset],['free cpu',r=>r.cpus],['free mem',r=>mem(r.mem_gb)],['free gpu',r=>r.gpus_text],['idle nodes',r=>r.idle_nodes]],d.curve,'No data.');
  tableRows('c-gaps',[['node',r=>r.node],['',r=>r.kind],['free cpu',r=>r.cpu+'/'+r.cpu_total],['free mem',r=>mem(Math.round(r.mem/1024))],['free gpu',r=>r.gpus_text],
    ['stays free',r=>r.stays_free],['next change',r=>r.next_change],['features',r=>r.features.join(',')],['state',r=>r.usable?'':r.state]],d.gaps,'No node matches these filters.');
  $('c-stamp').textContent=`${d.gaps.length} node(s) with free space · snapshot ${new Date(d.now*1000).toLocaleTimeString()}`;
}
async function loadPending(){
  $('p-stamp').textContent='loading…';
  const q=new URLSearchParams({all:$('p-all').checked?1:'',estimate:$('p-est').checked?1:'',partition:$('p-part').value});
  const d=await(await fetch(API+'/pending?'+q)).json();
  tableRows('p-table',[['job',r=>r.job],['user',r=>r.user],['name',r=>r.name],['part',r=>r.partition],['N',r=>r.nodes],['cpus',r=>r.cpus],['mem',r=>r.mem],['gpus',r=>r.gpus],
    ['features',r=>r.features||'-'],['limit',r=>r.limit],['start',r=>r.start_text],['in',r=>r.start_in],['reason',r=>r.reason||'-'],['planned nodes',r=>r.planned_nodes.join(',')]],
    d.rows,d.user?`No pending jobs for ${d.user}.`:'Queue is empty.');
  $('p-stamp').textContent=`${d.rows.length} job(s)${d.user?' for '+d.user:''} · snapshot ${new Date(d.now*1000).toLocaleTimeString()}`;
}
function pct(a,b){return b?Math.round(100*a/b)+'%':'-'}
async function init(){
  document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>showView(b.dataset.view));
  ['c-part','c-mingpu','c-unusable'].forEach(id=>$(id).addEventListener('change',loadCluster));
  $('c-reload').onclick=loadCluster;$('p-reload').onclick=loadPending;
  ['p-all','p-est','p-part'].forEach(id=>$(id).addEventListener('change',loadPending));
  lim=await(await fetch(API+'/limits')).json();
  $('nodes').max=lim.max_nodes;$('cpus').max=lim.max_cpus;$('mem').max=lim.max_mem_gb;$('gpus').max=lim.max_gpus;$('wall').max=WALL.length-1;
  $('gpu_type').innerHTML=['any',...lim.gpu_types].map(t=>`<option value="${esc(t)}">${esc(lim.gpu_type_names[t]||t)}</option>`).join('');
  $('partition').innerHTML=['',...lim.partitions].map(p=>`<option value="${esc(p)}">${p||'any'}</option>`).join('');
  for(const id of ['c-part','p-part'])$(id).innerHTML='<option value="">all</option>'+lim.partitions.map(p=>`<option>${esc(p)}</option>`).join('');
  $('features').innerHTML=lim.features.length?lim.features.map(f=>`<label><input type="checkbox" value="${esc(f)}"><span>${esc(f)}</span></label>`).join(''):'<span class="when">none defined</span>';
  const init=lim.initial;if(init)applyRequest(init);
  document.querySelectorAll('#view-fit input,#view-fit select').forEach(el=>{if(el.id!=='allusers')el.addEventListener('input',schedule)});
  $('allusers').addEventListener('change',loadPlanned);
  showValues();update();loadPlanned();setInterval(loadPlanned,60000);
  const v=location.hash.replace('#','');if(['cluster','pending'].includes(v))showView(v);
}
init();
</script>
</body>
</html>
"""


def make_app(cluster, initial=None):
    """Build the WSGI callable around one Cluster instance."""
    initial = initial or asdict(Request())

    def respond(start_response, status, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        start_response(status, [("Content-Type", ctype + "; charset=utf-8"),
                                ("Content-Length", str(len(data))), ("Cache-Control", "no-store")])
        return [data]

    def application(environ, start_response):
        # Route on the path *suffix*: behind OnDemand's Batch Connect proxy the app sees
        # /node/<host>/<port>/api/limits; behind Passenger it sees /api/limits. Both must work.
        raw = environ.get("PATH_INFO", "") or "/"
        if "/api/" in raw:
            path = "/api/" + raw.split("/api/", 1)[1].rstrip("/")
        elif raw.rstrip("/").endswith("/healthz"):
            path = "/healthz"
        else:
            path = "/"
        q = {k: v[0] for k, v in urllib.parse.parse_qs(environ.get("QUERY_STRING", "")).items()}
        me = current_user(cluster.demo)
        try:
            if path in ("", "/"):
                return respond(start_response, "200 OK", HTML, "text/html")
            if path == "/healthz":
                return respond(start_response, "200 OK", "ok", "text/plain")
            if path == "/api/limits":
                return respond(start_response, "200 OK", json.dumps({**cluster.limits(), "initial": initial}))
            if path == "/api/planned":
                return respond(start_response, "200 OK",
                               json.dumps(cluster.planned_work(None if q.get("all") else me)))
            if path == "/api/cluster":
                return respond(start_response, "200 OK", json.dumps(cluster.cluster_report(
                    partition=q.get("partition") or None, min_gpus=int(q.get("min_gpus", 0) or 0),
                    unusable=bool(q.get("unusable")))))
            if path == "/api/pending":
                return respond(start_response, "200 OK", json.dumps(cluster.pending_report(
                    user=None if q.get("all") else me, partition=q.get("partition") or None,
                    with_estimate=bool(q.get("estimate")))))
            if path == "/api/estimate":
                r = Request(nodes=int(q.get("nodes", 1)), cpus=int(q.get("cpus", 1)),
                            mem=int(float(q.get("mem", 1024))), gpus=int(q.get("gpus", 0)),
                            gpu_type=q.get("gpu_type", "any") or "any",
                            features=[f for f in q.get("features", "").split(",") if f],
                            partition=q.get("partition", ""), wall=int(float(q.get("wall", 3600))))
                return respond(start_response, "200 OK",
                               json.dumps(cluster.evaluate(r), default=lambda o: None if o == INF else o))
            return respond(start_response, "404 Not Found", "not found", "text/plain")
        except Exception as e:  # noqa
            return respond(start_response, "500 Internal Server Error",
                           f"job_forecaster error: {e}\n(if this says scontrol was not found, set JOB_FORECASTER_SCONTROL "
                           f"in passenger_wsgi.py)", "text/plain")

    return application


def serve(cluster, port=None, host="127.0.0.1", initial=None):
    """Run the app on a local port (fit_tool.py --web). port=None picks a free one."""
    import sys
    from socketserver import ThreadingMixIn
    from wsgiref.simple_server import WSGIServer, WSGIRequestHandler, make_server

    class Threaded(ThreadingMixIn, WSGIServer):
        daemon_threads = True

    class Quiet(WSGIRequestHandler):
        def log_message(self, *a):
            pass

    try:
        srv = make_server(host, port or 0, make_app(cluster, initial), server_class=Threaded, handler_class=Quiet)
    except OSError as e:
        sys.exit(f"cannot listen on {host}:{port}: {e.strerror}  (omit --port to pick a free one)")
    port = srv.server_address[1]
    shown = "localhost" if host in ("127.0.0.1", "localhost") else host
    print(f"Job Forecaster web UI: http://{shown}:{port}")
    if shown == "localhost":
        print(f"  remote? from your machine:  ssh -L {port}:localhost:{port} <this-login-node>")
    print("  Ctrl-C to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


# module-level application for Passenger (imported by passenger_wsgi.py)
application = make_app(Cluster(demo=DEMO))
