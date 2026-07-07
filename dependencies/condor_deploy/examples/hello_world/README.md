# hello_world example

Minimal end-to-end verification of the `condor_deploy` package.
The "forward model" is `hello_worker.py` — it does no simulation, just confirms
that the conda-pack environment was transferred and activated correctly on each
worker slot.

Run locally first (Step 2) to confirm the PEST wiring is correct before touching
HTCondor.

---

## Files

| File | Purpose |
|---|---|
| `hello_worker.py` | Forward model: checks all `WORKER_ENV_PACKAGES` are importable, runs numpy/pandas smoke tests, writes `hello_result.txt` + `hello_obs.dat`. |
| `build_hello_pst.py` | Builds `hello_template/`; optionally runs a local test or submits to HTCondor. |

`hello_template/` is created by `build_hello_pst.py` — do not commit it.

---

## Step 1 — install condor_deploy

From the package root (`dependencies/condor_deploy/`):

```bash
uv pip install -e .
# or into your existing conda env:
pip install -e .
```

---

## Step 2 — build the template

```bash
cd examples/hello_world

# Copy pestpp-ies binary here (or ensure it is on PATH)
cp ../../bin/mac/pestpp-ies .

python build_hello_pst.py
# or: uv run build_hello_pst.py
```

Creates `hello_template/` with: `hello.pst`, `forward_run.py`, `hello_worker.py`,
`dummy.tpl`, `dummy.dat`, `hello_obs.ins`, `pestpp-ies`.

---

## Step 3 — local sanity check (no HTCondor needed)

Confirms the PEST problem structure is correct. Uses `pyemu.os_utils.start_workers`
exactly as a real `run_pestpp()` call does.

`noptmax=-1` runs all 10 realisations (no iteration) — both workers get work
to do, which confirms the multi-worker plumbing as well as the env.

```bash
python build_hello_pst.py --local
```

Check the result — with `noptmax=-1` pestpp-ies writes the collected obs
ensemble `hello.0.obs.csv` to the master directory:

```bash
# plain CSV — no pyemu needed to inspect
cat hello_master/hello.0.obs.csv
# obs_n_failed column → all 0.0
# obs_smoke column    → all 1.0
```

> **Note:** `hello_result.txt` is written on each worker's local scratch
> directory and is **not** transferred back to the master — it is only
> visible in the per-worker subdirectories (local: `worker_N/hello_result.txt`,
> HTCondor: check per-job `.out` logs).

---

## Step 4 — build the worker conda-pack env (once per cluster)

```bash
python -m condor_deploy --build-env \
    --env-zip hello_worker_env.tar.gz \
    --env-name _hello_worker \
    --python-version 3.13
```

Creates a minimal relocatable conda env (~300–500 MB). Reuse this zip for
subsequent runs unless `hello_worker.py` dependencies change.

---

## Step 5 — dry-run (inspect submit files without submitting)

```bash
python build_hello_pst.py --condor-dry-run \
    --env-zip hello_worker_env.tar.gz \
    --n-workers 5
```

Inspect `_condor_staging_<timestamp>/`:
- `condor_workers.sub` — HTCondor submit description
- `worker.sh` — unpacks env + template, runs pestpp-ies agent
- `hello_template.zip` — zipped template sent to workers
- `log/` — where condor writes per-job stdout/stderr

---

## Step 6 — live HTCondor run

```bash
python build_hello_pst.py --condor \
    --env-zip hello_worker_env.tar.gz \
    --n-workers 5
```

What `CondorWorkerPool.submit_and_wait` does internally:
1. Zips `hello_template/` → `_condor_staging_*/hello_template.zip`
2. Copies `hello_template/` → `hello_master/`
3. Writes `worker.sh` and `condor_workers.sub`
4. Starts `pestpp-ies hello.pst /h :4004` in `hello_master/`
5. `condor_submit condor_workers.sub` → N HTCondor jobs
6. Each worker: unpacks env, unpacks template, runs as panther agent
7. Blocks until master exits (noptmax=0 → one forward-model pass per real)

---

## Step 7 — check results

PEST only sees what is written to `hello_obs.dat` and read back via `hello_obs.ins`.
With `noptmax=-1`, pestpp-ies runs all realisations and collects them into
`hello.0.obs.csv` on the master (plain CSV — no pyemu needed to inspect):

```bash
cat hello_master/hello.0.obs.csv
# obs_n_failed → all 0.0
# obs_smoke    → all 1.0
```

Par ensemble is in `hello_master/hello.0.par.csv`.
```

Per-job stdout (HTCondor) contains the full `hello_worker.py` output including
the human-readable `hello_result.txt` content — `hello_result.txt` itself is
written on each worker's local scratch and is **not** transferred back to the
master:

```bash
# HTCondor: per-job output logs
ls _condor_staging_*/log/
cat _condor_staging_*/log/worker_*.out

# Local run: check individual worker dirs
cat worker_0/hello_result.txt
```

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `conda-unpack: command not found` | `conda-pack` missing from `worker_env_packages` config |
| `ModuleNotFoundError: pyemu` | `conda-unpack` step failed; check `worker_N.err` |
| Workers connect then immediately exit | `hello_worker.py` exited non-zero; check `worker_N.out` for `FAILED` |
| Master exits before any worker connects | Port blocked by firewall, or increase startup wait |
| `hello_result.txt` not found | It lives on the worker scratch, not the master — check worker dirs or HTCondor job logs |

