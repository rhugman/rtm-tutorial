# condor_deploy

HTCondor deployment helper for pestpp-ies panther runs.

## Installation

```bash
# editable install into your project env
uv pip install -e .
# or
pip install -e .
```

## Quick start

```python
from condor_deploy import configure, CondorWorkerPool

# Optional: override defaults before creating a pool
configure(
    zip_exclude=["*.hds", "*.cbc", "big_raster.tif"],
    worker_env_packages=["pyemu", "flopy", "numpy", "pandas", "conda-pack"],
)

pool = CondorWorkerPool(
    template_dir="pst_template",
    env_zip="worker_env.tar.gz",
    master_dir="master",
    pst_name="pest.pst",
    n_workers=50,
)
pool.submit_and_wait(block=True)
```

Integration with `workflow.py` — `run_pestpp()` auto-routes to HTCondor when
`condor_submit` is on PATH; pass `condor_kwargs=dict(...)` to forward settings.

## CLI

```bash
# build worker conda-pack env
condor-deploy --build-env --env-zip worker_env.tar.gz

# dry-run (write submit files, no submission)
condor-deploy --template pst_template --env-zip worker_env.tar.gz --dry-run

# live run
condor-deploy --template pst_template --env-zip worker_env.tar.gz --n-workers 50
```

Or: `python -m condor_deploy --help`

## Configuration

Project-wide defaults are set via `configure()`:

```python
from condor_deploy import configure
configure(
    zip_exclude=[...],              # files excluded from worker zip
    worker_env_packages=[...],      # conda packages in auto-built worker env
    worker_pip_packages=[...],      # pip packages installed after the conda solve
    worker_pip_editable=[...],      # source trees bundled + `pip install -e` on each worker
    platform_requirements='( (OpSys == "LINUX") )',  # HTCondor Requirements (empty = pool defaults)
    worker_chmod_exes=["mf6"],      # model binaries in the template to chmod +x on each worker
    worker_python_version="3.13",   # python version for the auto-built worker env
    sub_template="...",             # override the .sub file template string
    worker_sh="...",                # override the worker.sh template string
)
```

> **Portability note.** `platform_requirements` and `worker_chmod_exes` default to
> neutral values (no platform restriction, no extra binaries) so the package is
> cluster- and model-agnostic out of the box. Set them once per project — see
> [`PORTING.md`](PORTING.md) for the full port contract.

## Examples

See [`examples/hello_world/`](examples/hello_world/README.md) for a minimal
end-to-end test that verifies environment transfer and worker connectivity
without needing a real simulation model.

## Package layout

```
src/condor_deploy/
├── __init__.py        public API
├── __main__.py        CLI
├── config.py          CondorDeployConfig dataclass + configure() / get_config()
├── templates.py       default SUB_TEMPLATE and WORKER_SH strings
├── pool.py            CondorWorkerPool
├── env_builder.py     build_worker_env_zip()
├── local.py           run_local() — local fallback
├── entry_points.py    submit_condor_workers_from_env()
└── _utils.py          timestamp helpers
```

