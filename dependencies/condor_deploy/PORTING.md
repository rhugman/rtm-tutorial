# Porting `condor_deploy` to another PEST++ project

`condor_deploy` runs a `pestpp-ies` (or any pestpp-`*`) PANTHER master locally and
submits its worker *agents* as HTCondor jobs. It assumes **no shared filesystem**:
the whole Python environment travels to each slot as a `conda-pack` tarball and the
model + PEST interface travels as a zip. That is what makes it portable to an
arbitrary HTCondor pool.

The package is pure standard library (no hard third-party deps) and is fully
config-driven. Porting to a new project is: **install it, satisfy the template
contract, make one `configure()` call, and reproduce three small integration
seams.** No fork, no template rewrite.

---

## 1. Submit-node prerequisites

The machine you launch from (the HTCondor *submit* node) needs:

- `condor_submit` / `condor_q` / `condor_status` / `condor_rm` on `PATH`.
- `conda` or `mamba` **and** `conda-pack` — only for the one-time worker-env build
  (`condor-deploy --build-env`). Not needed if you supply a pre-built `env_zip`.
- The pestpp binary and any model binaries for the **worker** platform, placed
  inside the template dir (see §3). These are shipped to the slots — the submit
  node itself does not run them (except the local master process).

Worker slots need nothing pre-installed: the env tarball + template zip carry
everything.

---

## 2. Install

```bash
pip install -e /path/to/condor_deploy          # editable, from a checkout
# or vendor it under your project's dependencies/ and pip install -e it
```

Console script `condor-deploy` and `python -m condor_deploy` both work.

---

## 3. The template contract

`condor_deploy` zips your **template dir** and ships it to every worker. The dir
must be a self-contained, runnable PEST interface:

- `pest.pst` (or your control file name — pass via `pst_name`).
- `forward_run.py` and all model inputs / templates / instruction files.
- **The pestpp binary** (e.g. `pestpp-ies`) for the *worker* OS/arch.
- **All model/solver binaries** the forward run invokes (e.g. `mf6`), for the
  *worker* OS/arch.

Zips do not preserve execute bits, so binaries are `chmod +x`ed on the slot at
startup: the pestpp exe always, plus anything you list in `worker_chmod_exes`
(§4). The forward-model command line itself lives in `pest.pst` and is run by the
pestpp agent — `condor_deploy` does not need to know it.

Keep the template *clean* (no run outputs): it is both zipped for workers and
copied to form the master dir. Use `zip_exclude` to drop bulky/derived files.

---

## 4. The one-time `configure()` call

Call this once, early, before creating any pool (mirror it from your driver's
top-level / `__main__`). Everything project-specific lives here:

```python
from condor_deploy import configure

configure(
    # --- worker environment (conda-pack) ---
    worker_env_packages=["pyarrow", "pyemu", "numpy", "pandas"],  # conda solve
    worker_pip_packages=["pypestutils"],                          # pip after solve
    worker_pip_editable=["./dependencies/flopy"],                 # bundled + `pip install -e`
    worker_python_version="3.13",

    # --- cluster / model specifics (NEW in 0.2.0; default to neutral) ---
    platform_requirements='( (OpSys == "LINUX") )',  # "" = no restriction (pool defaults)
    worker_chmod_exes=["mf6"],                        # model binaries to chmod on the slot

    # --- what to leave out of the worker zip ---
    zip_exclude=["*.hds", "*.cbc", "*.grb", "*.rec", "*.log", "__pycache__", "figures"],
)
```

Notes:

- **`worker_pip_editable`** is for source trees you develop against (vendored
  `flopy`/`pyemu` forks). `conda-pack` can't pack editable installs, so the source
  is bundled into the template zip under `_editable_pkgs/` and `pip install -e`d on
  each worker at startup. Keep such packages *out* of `worker_env_packages`.
- **`platform_requirements`** is the raw HTCondor `Requirements` expression. Empty
  string ⇒ the line is omitted and the pool's own defaults apply. Set
  `LINUX`/`MacOS`/`WINDOWS` (or a richer ClassAd) to match the OS your worker
  binaries were built for. A per-run `condor_requirements=...` on the pool / a
  `--requirements` CLI flag overrides this default for a single run.
- **`worker_chmod_exes`** lists model binaries (besides pestpp) that must be made
  executable on the slot. MODFLOW users: `["mf6"]`.

---

## 5. The three integration seams

Copy these three snippets into your driver (they total ~20 lines). This is the
entire wiring — `run_pestpp()` then routes to HTCondor automatically whenever
`condor_submit` is on `PATH`, with a transparent local fallback otherwise.

### Seam 1 — import + auto-detect probe (module top)

```python
# Optional HTCondor deployment — import guarded so the script runs without it.
try:
    from condor_deploy import submit_condor_workers_from_env as _condor_submit_from_env
    import subprocess as _subprocess
    try:
        _subprocess.run(["condor_submit", "--version"], capture_output=True, timeout=20)
        _CONDOR_AVAILABLE = True            # submit node
    except (FileNotFoundError, OSError):
        _CONDOR_AVAILABLE = False
except ImportError:
    _CONDOR_AVAILABLE = False
    _condor_submit_from_env = None
```

### Seam 2 — dispatch inside your run function

`t_d` = template dir, `m_d` = master dir. Route to HTCondor when available;
otherwise fall back to local PANTHER workers.

```python
def run_pestpp(t_d, m_d, casename="pest", num_workers=50,
               condor_kwargs=None, force_local=False):
    # ... write/adjust the .pst here ...
    if _CONDOR_AVAILABLE and not force_local:
        _condor_submit_from_env(
            template_dir=t_d,
            pst_name=f"{casename}.pst",
            master_dir=m_d,
            n_workers=num_workers,
            **(condor_kwargs or {}),     # env_zip, master_port, memory_mb, disk_mb, ...
        )
    else:
        import pyemu
        pyemu.os_utils.start_workers(
            t_d, "pestpp-ies", f"{casename}.pst",
            num_workers=num_workers, worker_root=".",
            master_dir=m_d, port=4004, cleanup=False,
        )
```

`condor_kwargs` is forwarded verbatim to `CondorWorkerPool` — see its constructor
for the full set (`master_port`, `master_host`, `pestpp_exe`, `cpus_per_worker`,
`memory_mb`, `disk_mb`, `condor_requirements`, `timeout_h`, `env_zip`, `dry_run`,
`force_local`).

### Seam 3 — `configure()` + shared kwargs (driver `__main__`)

```python
if _CONDOR_AVAILABLE:
    configure(...)                       # the §4 block

NUM_WORKERS = 60 if _CONDOR_AVAILABLE else 10
CONDOR_KWDS = dict(master_port=4004, memory_mb=1800, disk_mb=13000)

# First stage: no env_zip → build it once. Then reuse for later stages:
run_pestpp("pst_template", "master_prior_mc", num_workers=NUM_WORKERS,
           condor_kwargs=CONDOR_KWDS)
CONDOR_KWDS["env_zip"] = "worker_env.tar.gz"   # reuse the built tarball
CONDOR_KWDS["master_port"] += 1                # bump port for the next master
run_pestpp("pst_template", "master_hm", num_workers=NUM_WORKERS,
           condor_kwargs=CONDOR_KWDS)
```

Two idioms worth keeping: build the env-zip **once** then reuse it by setting
`env_zip`; and **increment `master_port`** between sequential masters to avoid
collisions.

---

## 6. Port checklist

- [ ] `condor_deploy` installed on the submit node (`pip install -e`).
- [ ] `conda`/`mamba` + `conda-pack` available on the submit node.
- [ ] Template dir is clean, self-contained, and contains the **worker-platform**
      pestpp + model binaries.
- [ ] One `configure()` call sets `worker_env_packages` / pip / editable,
      `platform_requirements`, `worker_chmod_exes`.
- [ ] Three seams wired into the driver.
- [ ] `hello_world` smoke test passes on the target cluster (§7).
- [ ] Worker env built once: `condor-deploy --build-env --env-zip worker_env.tar.gz`.
- [ ] First real run does a `--dry-run` (inspect `_condor_staging_*/`), then live.

---

## 7. Smoke-test the cluster first (`examples/hello_world/`)

Before wiring a real model, prove env-transfer + panther connectivity on the new
pool with the no-simulation example:

```bash
cd examples/hello_world
cp /path/to/pestpp-ies .
python build_hello_pst.py                       # build template
python build_hello_pst.py --local               # local sanity (no HTCondor)
python -m condor_deploy --build-env --env-zip hello_worker_env.tar.gz --env-name _hello_worker
python build_hello_pst.py --condor-dry-run --env-zip hello_worker_env.tar.gz --n-workers 5
python build_hello_pst.py --condor      --env-zip hello_worker_env.tar.gz --n-workers 5
```

Green here ⇒ the env build, transfer, `conda-unpack`, and master↔agent handshake
all work on the cluster. Only then wire your real model.

---

## 8. Cluster-specific gotchas

| Concern | Where to set it |
|---|---|
| Worker OS (Linux vs Mac vs Windows) | `platform_requirements` in `configure()`, or per-run `condor_requirements` |
| Model binary names to `chmod` | `worker_chmod_exes` |
| Slot resources | `condor_kwargs`: `memory_mb`, `disk_mb`, `cpus_per_worker` |
| Master↔worker networking | `master_host` (defaults to `$HOSTNAME`/`gethostname()`), `master_port` |
| Firewall blocks the master port | pick a `master_port` the pool can reach; bump per stage |
| Huge model outputs bloating the zip | add patterns to `zip_exclude` |
| Non-`ies` tool | `condor_kwargs["pestpp_exe"]` (e.g. `pestpp-da`); the master is started as `<exe> <pst> /h :<port>` |

Binaries must match the **worker** platform, not the submit node. If your submit
node is macOS but workers are Linux, ship Linux binaries in the template and set
`platform_requirements` to `LINUX`.

---

## 9. What changed in 0.2.0 (why the port is now clean)

Before 0.2.0 the submit template hardcoded `requirements = ((OpSys == "MacOS"))`
and the worker script hardcoded `chmod +x ... mf6`, and the `condor_requirements`
constructor arg / `--requirements` CLI flag were silently ignored (no
`{requirements}` placeholder existed). 0.2.0 lifts all of that into config with
neutral defaults:

- `platform_requirements` (default `""` → no restriction), now actually rendered;
  `condor_requirements` / `--requirements` override it per run.
- `worker_chmod_exes` (default `[]`).
- `worker_python_version` (default `"3.13"`).

Existing MODFLOW/wombat projects preserve their old behavior by setting
`platform_requirements='( (OpSys == "MacOS") )'` and `worker_chmod_exes=["mf6"]`
in their `configure()` call.
