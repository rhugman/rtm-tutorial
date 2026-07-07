"""
build_hello_pst.py
==================
Builds a minimal PEST++ problem for testing condor_deploy end-to-end.

The "model" is hello_worker.py which:
  - confirms all WORKER_ENV_PACKAGES are importable
  - runs basic numpy/pandas smoke tests
  - writes hello_result.txt + hello_obs.dat (read by PEST)

noptmax=-1 so pestpp-ies runs the forward model once per realisation
(no iteration/optimisation) — the goal is exercising multiple workers
and verifying the env is intact on each slot, not any calibration.
A small par ensemble (N_REALS realisations) is generated so that
multiple workers actually get work to do.

Usage
-----
Run from this directory (examples/hello_world/):

    # Step 1 — build the template only
    python build_hello_pst.py

    # or with uv from the package root:
    uv run examples/hello_world/build_hello_pst.py

    # Step 2 — build + local 2-worker sanity check (no HTCondor needed)
    python build_hello_pst.py --local

    # Step 3 — dry-run condor deploy (writes submit files, no submission)
    python build_hello_pst.py --condor-dry-run --env-zip /path/to/worker_env.tar.gz

    # Step 4 — live HTCondor run
    python build_hello_pst.py --condor --env-zip /path/to/worker_env.tar.gz --n-workers 5
"""
import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).parent
TEMPLATE_DIR = HERE / "hello_template"
PST_NAME = "hello.pst"
N_REALS = 10   # realisations — enough to keep 2 workers busy

# ---------------------------------------------------------------------------
# PEST file contents
# ---------------------------------------------------------------------------

TPL = "ptf ~\ndummy_mult = ~   dummy    ~\n"

INS = (
    "pif @\n"
    "l1 !obs_n_ok!\n"
    "l1 !obs_n_failed!\n"
    "l1 !obs_smoke!\n"
)

PST = """\
pcf
* control data
restart estimation
     1     3     1     0     1
     1 1 single point
     10.0 -3.0 0.3 0.03 10
      10.0 10.0 0.001
      0.1
      -1 0.005 4 4 0.005 4
     1     1   1
* singular value decomposition
       1
    5000   1.0000e-08
       0
* parameter groups
    pargp  relative  0.001  0.0  switch  2.0  parabolic
* parameter data
   dummy   none  relative  1.0  0.001  100.0  pargp  1.0  0.0  1
* observation groups
    obsgrp
* observation data
   obs_n_ok      6.0  1.0  obsgrp
   obs_n_failed  0.0  1.0  obsgrp
   obs_smoke     1.0  1.0  obsgrp
* model command line
    python forward_run.py
* model input/output
   dummy.tpl   dummy.dat
   hello_obs.ins   hello_obs.dat
* prior information
* pestpp options
   ++ies_num_reals({n_reals})
""".format(n_reals=N_REALS)

FORWARD_RUN = (
    "import subprocess, sys\n"
    "r = subprocess.run([sys.executable, 'hello_worker.py'], check=False)\n"
    "sys.exit(r.returncode)\n"
)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_template(template_dir: Path, pestpp_exe: str = "pestpp-ies") -> Path:
    """Create hello_template/ with all PEST++ files."""
    if template_dir.exists():
        shutil.rmtree(template_dir)
    template_dir.mkdir(parents=True)

    (template_dir / "dummy.tpl").write_text(TPL)
    (template_dir / "hello_obs.ins").write_text(INS)
    (template_dir / PST_NAME).write_text(PST)
    (template_dir / "forward_run.py").write_text(FORWARD_RUN)
    (template_dir / "dummy.dat").write_text("dummy_mult = 1.0\n")

    hw = HERE / "hello_worker.py"
    if hw.exists():
        shutil.copy(hw, template_dir / "hello_worker.py")
    else:
        print(f"WARNING: hello_worker.py not found at {hw}")

    # Copy pestpp-ies binary into template if it exists locally
    for exe_name in [pestpp_exe, "pestpp-ies"]:
        for search in [HERE, HERE.parent.parent]:
            exe = search / exe_name
            if exe.exists():
                shutil.copy(exe, template_dir / exe_name)
                (template_dir / exe_name).chmod(0o755)
                break

    print(f"Template built → {template_dir}")
    print(f"Files: {sorted(p.name for p in template_dir.iterdir())}")

    # Generate a small prior par ensemble so noptmax=-1 distributes work
    # across multiple workers.  Requires pyemu; skipped gracefully if absent.
    _generate_par_ensemble(template_dir)

    return template_dir


def _generate_par_ensemble(template_dir: Path) -> None:
    """Draw N_REALS realisations from the prior and write hello.par.csv."""
    try:
        import numpy as np
        import pyemu
    except ImportError:
        print("NOTE: pyemu not available — par ensemble not generated; "
              "workers will all run the base parameter values.")
        return

    pst = pyemu.Pst(str(template_dir / PST_NAME))
    pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst, num_reals=N_REALS)
    par_csv = template_dir / "hello.par.csv"
    pe.to_csv(str(par_csv))
    # Tell the PST to use this ensemble
    pst.pestpp_options["ies_par_en"] = par_csv.name
    pst.pestpp_options["save_binary"] = False
    pst.write(str(template_dir / PST_NAME), version=2)
    print(f"Par ensemble ({N_REALS} reals) → {par_csv.name}")


# ---------------------------------------------------------------------------
# Local test (no HTCondor)
# ---------------------------------------------------------------------------

def run_local_test(template_dir: Path, n_workers: int = 2,
                   pestpp_exe: str = "pestpp-ies") -> None:
    """
    Quick sanity check using pyemu.os_utils.start_workers.

    noptmax=-1 runs all N_REALS realisations without iterating — enough
    work to exercise both workers.
    """
    try:
        import pyemu
    except ImportError:
        print("pyemu not importable — skipping local test")
        return

    master_dir = template_dir.parent / "hello_master"
    if master_dir.exists():
        shutil.rmtree(master_dir)

    print(f"\nRunning local test: {n_workers} workers, port 9999 …")
    pyemu.os_utils.start_workers(
        str(template_dir),
        pestpp_exe,
        PST_NAME,
        num_workers=n_workers,
        worker_root=str(template_dir.parent),
        master_dir=str(master_dir),
        port=9999,
        cleanup=False,
    )
    check_results(master_dir, local=True)


def check_results(master_dir: Path,
                  local: bool = False) -> None:
    result = master_dir / "hello.0.obs.csv"
    if result.exists():
        print(f"\n✓ {result.name} written — {N_REALS} realisations collected by master.")
        try:
            import pandas as pd
            df = pd.read_csv(result, index_col=0)
            print(df[["obs_n_ok", "obs_n_failed", "obs_smoke"]].to_string())
        except Exception:
            print(result.read_text()[:500])
    else:
        if local:
            print(f"NOTE: {result} not found — check worker dirs for hello_result.txt")
            for w in sorted(master_dir.parent.glob("worker_*/hello_result.txt")):
                print(f"\n--- {w} ---")
                print(w.read_text())
        else:
            raise RuntimeError(f"Expected result {result} not found. "
                               f"Check HTCondor job logs and worker "
                               f"scratch dirs.")


# ---------------------------------------------------------------------------
# Condor deploy (dry-run or live)
# ---------------------------------------------------------------------------

def run_condor(template_dir: Path, env_zip: str, n_workers: int = 5,
               dry_run: bool = False, **condor_kwargs) -> None:
    """
    Deploy to HTCondor via condor_deploy.CondorWorkerPool.

    Pass dry_run=True to write submit files without actually submitting.
    Any extra keyword arguments are forwarded to CondorWorkerPool.
    """
    from condor_deploy import CondorWorkerPool
    master_dir = template_dir.parent / "hello_master"
    pool = CondorWorkerPool(
        template_dir=template_dir,
        master_dir=master_dir,
        env_zip=env_zip,
        pst_name=PST_NAME,
        n_workers=n_workers,
        dry_run=dry_run,
        **condor_kwargs,
    )
    pool.submit_and_wait(block=not dry_run)
    if not dry_run:
        check_results(master_dir, local=False)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Build + optionally run the hello-world condor_deploy example.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--template-dir", default=str(TEMPLATE_DIR), dest="template_dir",
                   help=f"Template output directory (default: {TEMPLATE_DIR})")
    p.add_argument("--pestpp-exe", default="pestpp-ies", dest="pestpp_exe",
                   help="pestpp-ies binary name (default: pestpp-ies)")
    p.add_argument("--local", action="store_true",
                   help="Run a local test with pyemu.os_utils.start_workers after building.")
    p.add_argument("--n-workers", type=int, default=2, dest="n_workers",
                   help=f"Workers for --local or --condor (default: 2; {N_REALS} reals split across workers)")
    p.add_argument("--condor", action="store_true",
                   help="Submit a live HTCondor run after building.")
    p.add_argument("--condor-dry-run", action="store_true", dest="condor_dry_run",
                   help="Write HTCondor submit files without submitting (implies --condor).")
    p.add_argument("--env-zip", default=None, dest="env_zip",
                   help="Path to conda-pack worker env tar.gz (required for --condor / --condor-dry-run).")
    args = p.parse_args()

    td = build_template(Path(args.template_dir), pestpp_exe=args.pestpp_exe)

    if args.local:
        run_local_test(td, n_workers=args.n_workers, pestpp_exe=args.pestpp_exe)

    if args.condor or args.condor_dry_run:
        if not args.env_zip:
            print("ERROR: --env-zip is required for --condor / --condor-dry-run", file=sys.stderr)
            sys.exit(1)
        run_condor(td, env_zip=args.env_zip, n_workers=args.n_workers,
                   dry_run=args.condor_dry_run)

