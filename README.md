# Job Forecaster — Open OnDemand Batch Connect app

An **Interactive App** for Open OnDemand. Launching it submits a 1-core, 1 GB Slurm job that runs a
small web server; OnDemand proxies the browser to it and shows a **Connect** button on
*My Interactive Sessions*. The page has three views:

| view | what it answers |
|---|---|
| **Fit my job** | Sliders for nodes, CPUs, memory, GPUs + type, node features, partition, walltime. Earliest start, the nodes it would land on, a 48 h free-capacity chart with your job's window on it, one-click "start sooner if you…" suggestions, and the matching `#SBATCH` header. Below: your pending and soon-finishing jobs (toggle to all users). |
| **Cluster** | Total / used / available CPUs, memory, GPUs per type; cluster-wide free-capacity timeline; per-node gaps (what is free on each node right now and for how long). |
| **Pending jobs** | Every pending job's requirements and Slurm's expected start. Yours by default; toggle to all users; optional best-case estimate for jobs Slurm hasn't scheduled. |

The job runs as the user who launched it, so `scontrol` sees what they would see and "my jobs" means
theirs. Standard library only; Python ≥ 3.7 on the compute nodes.

## Directory structure

```
bc_job_forecaster/                       ← app root; the directory name is the app id
├── manifest.yml                name, menu placement (Interactive Apps ▸ Servers), role: batch_connect
├── form.yml                    the launch form: cluster, account, partition, hours, python module, refresh
├── submit.yml.erb              how the job is submitted: basic template + sbatch native args (1 core, 1 GB)
├── view.html.erb               the Connect button shown once the job is running
├── icon.png                    dashboard icon (manifest.yml's fa:// icon wins if both are present)
├── template/                   staged into each session's working directory
│   ├── before.sh.erb           picks a free port on the compute node
│   ├── script.sh.erb           loads the (optional) python module and starts the server in the foreground
│   ├── after.sh.erb            waits until the port answers, then OnDemand shows Connect
│   └── lib/                    the application and the CLI tools (copied with every session)
│       ├── job_forecaster.py         scontrol parsing, per-node timelines, fit estimator, shared report rows
│       ├── job_forecaster_web.py     WSGI application + the single-page UI (HTML/CSS/JS inlined)
│       ├── fit_tool.py         `--web` is what the job runs; also a terminal UI and `--once --explain`
│       ├── cluster_view.py     CLI: totals, allocations, gaps
│       └── pending_jobs.py     CLI: pending jobs + start times
├── .gitignore
└── README.md
```

There is deliberately no `passenger_wsgi.py`: Batch Connect apps are launched as jobs, not run inside
the per-user NGINX.

## Install

### 1. Set the cluster

Edit `form.yml` and set `cluster:` to the name of a file in `/etc/ood/config/clusters.d/`
(`hpc` for `hpc.yml`). That cluster config must have a `batch_connect: basic:` section (it does on any
site already running Jupyter or RStudio through OnDemand).

### 2. Check the proxy is enabled

The Connect button opens `/node/<host>/<port>/`. In `/etc/ood/config/ood_portal.yml` you need
`host_regex` to match your compute node names and `node_uri: "/node"` (and usually
`rnode_uri: "/rnode"`), then regenerate the portal config
(`sudo /opt/ood/ood-portal-generator/sbin/update_ood_portal` and restart Apache). Again, any site
with Jupyter on OnDemand already has this.

### 3. Try it in your sandbox

```bash
mkdir -p ~/ondemand/dev/bc_job_forecaster
tar xzf bc_job_forecaster.tar.gz -C ~/ondemand/dev/bc_job_forecaster --strip-components=1
ls ~/ondemand/dev/bc_job_forecaster        # manifest.yml must be directly here, not in a subfolder
```

`--strip-components=1` matters: the tarball contains a top-level `bc_job_forecaster/` folder, and OnDemand looks
for `manifest.yml` directly inside the app directory. If the dashboard shows the folder name with
*"Manifest is missing, add a title and description to fix this"*, the files are nested one level too deep:
`cd` into the app directory and `mv job_forecaster/* job_forecaster/.gitignore . && rmdir job_forecaster`. The directory name
itself is only the app id; call it whatever you like.

Dashboard → **Develop → My Sandbox Apps (Development)** → *Job Forecaster* → **Launch**. Fill the form,
submit, wait for the Connect button on *My Interactive Sessions*.

### 4. Deploy for everyone

```bash
sudo mkdir -p /var/www/ood/apps/sys/bc_job_forecaster
sudo tar xzf bc_job_forecaster.tar.gz -C /var/www/ood/apps/sys/bc_job_forecaster --strip-components=1
sudo chown -R root:root /var/www/ood/apps/sys/bc_job_forecaster
sudo chmod -R a+rX /var/www/ood/apps/sys/bc_job_forecaster
```

It appears under **Interactive Apps → Servers → Job Forecaster** after users restart their web server
(Help menu) or log in again.

## Form fields

| field | notes |
|---|---|
| Account (`bc_account`) | standard OnDemand field; remove from `form.yml` if your site doesn't use accounts |
| Partition (`bc_queue`) | pick something that starts a tiny job fast (a debug/interactive partition). To hard-wire one instead of asking, delete `bc_queue` from `form.yml` and uncomment `queue_name` in `submit.yml.erb`. |
| Hours | the server stays up this long; the page keeps re-reading Slurm, so leaving it running costs one core |
| Python module | blank → system `python3` on the compute node. Set e.g. `python/3.11` if the system one is < 3.7. |
| Seconds between re-reads of scontrol | becomes `JOB_FORECASTER_TTL`; each open session issues `scontrol show node -o` + `scontrol show job -o` at most this often |

## Per-site knobs

* **Job size**: `submit.yml.erb → script.native`. 1 core / 1 GB is enough for thousands of nodes and jobs.
* **scontrol not on PATH on compute nodes**: set `JOB_FORECASTER_SCONTROL` in `template/script.sh.erb`.
* **Startup wait**: `template/after.sh.erb` waits up to 90 s for the port; raise it on slow nodes.
* **Modules**: `script.sh.erb` does `module purge` + `module load <python_module>` only when a module was
  given. Adjust if your site needs other modules for `scontrol`.
* **Listen address**: the server binds `0.0.0.0` on the compute node so OnDemand's web host can reach
  it. It is read-only and shows what `squeue`/`sinfo` would show anyone, but if your policy requires it,
  change `--host 0.0.0.0` to `--host "${host}"` in `script.sh.erb` to bind a single interface.

## Command-line tools

The same code runs from a shell (login node, or OnDemand *Clusters → Shell*):

```bash
template/lib/cluster_view.py                      # totals, your running/pending jobs, gaps  (-a all, -p partition)
template/lib/cluster_view.py --gaps-only --min-gpus 1
template/lib/pending_jobs.py                      # your pending jobs + start  (-a all, --estimate, --json)
template/lib/fit_tool.py                          # terminal UI with the sliders
template/lib/fit_tool.py --once -N 2 -G 4 --gpu-type h200 -t 4:00:00 --explain
template/lib/fit_tool.py --web                    # the same page on a login-node port, without OnDemand
```

Copy `template/lib/` somewhere on `$PATH` if you want them generally available; the five files only
need each other. `--demo` on any of them uses a synthetic cluster.

## How the estimate works

1. Free CPUs / memory / GPUs per node *right now* come from `scontrol show node`: `CPUAlloc`, `AllocMem`,
   `GresUsed` — or `AllocTRES` when `GresUsed` is absent (Slurm 23.x/24.x prints
   `AllocTRES=…,gres/gpu:h200=4` and no `GresUsed` line). GPU type names match case-insensitively,
   because Slurm lower-cases them in TRES even when `gres.conf` says `H200`.
2. Running jobs return their share of each node at their `EndTime` (jobs already past it, e.g. `COMPLETING`,
   release one second from now). A job submitted with an untyped `--gres=gpu:N` shows up in TRES as plain
   `gpu`; its count is mapped onto the node's real GPU type so the release lands in the right bucket.
3. Pending jobs that Slurm's backfill scheduler has given a `StartTime` and `SchedNodeList` take their share
   on those nodes from then, for their time limit.
4. That gives a free-resource timeline per node; the estimator finds every window long enough for the
   requested walltime on each eligible node, then the earliest instant at which `--nodes` of them overlap.
5. Suggestions are single-dimension tweaks (shorter walltime, one GPU fewer, another GPU type, half the memory,
   one node fewer, drop a constraint) re-evaluated the same way.

### What it does not model

* **Priority, fairshare, QOS, account limits.** The estimate means "the hardware is free": a best case.
  Use it for the *relative* effect of changing a request.
* **Pending jobs with no Slurm start estimate** are listed, not projected; the page says how many.
* **Reservations and licenses.**
* The per-node split of a multi-node job is approximated as total ÷ node count (present usage is exact).

If an estimate looks wrong, run `template/lib/fit_tool.py --once … --explain`: it prints, per node, what is
free according to `scontrol`, the jobs on it, and the raw `CPUAlloc/AllocMem/Gres/GresUsed/AllocTRES`
fields. Every view also warns if GPU jobs are running but no node reports used GPUs.

## Troubleshooting

Each session's files are in `~/ondemand/data/sys/dashboard/batch_connect/sys/bc_job_forecaster/output/<id>/`
(`dev/` instead of `sys/` for sandbox apps): `output.log` has everything the scripts printed.

| symptom | fix |
|---|---|
| App not listed, or listed as the folder name with *Manifest is missing* | `manifest.yml` must sit directly in the app directory (see Install); `form.yml`/`manifest.yml` must be valid YAML; restart web server (Help menu). |
| Submit fails: unknown cluster | `cluster:` in `form.yml` must match a file name in `clusters.d/`. |
| Job runs but no Connect button / "Timed out waiting for Job Forecaster" | `output.log` shows the Python traceback. Usually: Python < 3.7 (set a module in the form), or `scontrol` not on PATH on compute nodes (set `JOB_FORECASTER_SCONTROL`). |
| Connect button gives 404/502 | `/node` proxy not configured, or `host_regex` doesn't match the compute node's hostname. |
| Page loads, tables empty, 500 on `…/api/…` | open `…/api/limits` directly; the error text names the cause. |
| "does not fit … only 2 of the 3 eligible nodes ever have a free window" | The message names the blocking node and why: usually a running job with `TimeLimit=UNLIMITED` holding it. `--explain` shows the jobs on it. |
| All GPUs look free while jobs run | your `scontrol` output puts usage somewhere unparsed; `--explain` shows the raw line. |
| Job won't start promptly | pick a debug/interactive partition in the form or hard-wire `queue_name`. |

## Security notes

* The server has no authentication of its own. OnDemand authenticates the user and proxies to the job;
  the port is also reachable from inside the cluster network. The app only reads Slurm state (two
  read-only `scontrol` calls) and never submits or cancels jobs; it shows the same information
  `squeue`/`sinfo` show every user.
* All endpoints are `GET`; the page loads no external resources.
