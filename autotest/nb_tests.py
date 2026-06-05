#!/usr/bin/env python3
"""Notebook regression runner for the rtm-tutorial curriculum.

Maintainer / nightly tool, *not* part of the tutorial itself. It executes the
tutorial notebooks end to end (the mothership pattern: ``jupyter nbconvert
--execute``), reports per-notebook timings, collects failures and exits
non-zero if any notebook fails.

Run order
---------
1. All ``part0_*`` notebooks first, in any order. ``part0_02_intro_to_mf6rtm``
   (the pyrite-column model) is the fast canary -- it runs in seconds and is
   placed first so an obviously broken environment fails quickly.
2. Then every ``part1_*`` notebook in strict numeric order. part1 builds a
   chain of workspaces (each notebook consumes the one the previous wrote), so
   the order is load-bearing, not cosmetic.

Cost
----
``--fast`` runs only the part0 set (CI-able subset: pyrite column + background
notebooks, minutes). The full part1 sequence requires the ``rtm_gmdsi``
environment (the vendored ``pyemu``/``mf6rtm``/``flopy`` trees and the
committed ``bin/`` binaries) and a live forward run in several notebooks at
~6 min per model run -- budget roughly 1-2 hours wall time on a laptop.

Usage
-----
    python autotest/nb_tests.py            # full run (part0 then part1)
    python autotest/nb_tests.py --fast     # part0 only (the CI-able subset)
    python autotest/nb_tests.py --no-clear # keep executed outputs in place

By default each notebook is executed *in place* and then has its outputs
cleared again, so a passing run leaves the tree as clean as it found it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Repo root is one level up from this file (autotest/).
REPO_ROOT = Path(__file__).resolve().parent.parent
TUTORIALS = REPO_ROOT / "tutorials"

# Per-notebook execution timeout, in seconds. The expensive part1 notebooks
# each carry at most one live ~6 min forward run plus PEST++/pyemu work; 1800 s
# (30 min) leaves comfortable headroom without hanging a stuck kernel forever.
CELL_TIMEOUT = 1800

# The canary: fast, standalone, pure-API pyrite column. Listed first so a
# broken environment is caught in seconds rather than after a long part1 climb.
CANARY = "part0_02_intro_to_mf6rtm"


def discover_notebooks():
    """Return (part0, part1) lists of notebook paths in run order.

    part0 notebooks run first with the canary at the front (otherwise
    directory order). part1 notebooks run in strict numeric directory order.
    """
    def single_nb(part_dir):
        nbs = sorted(part_dir.glob("*.ipynb"))
        nbs = [n for n in nbs if ".ipynb_checkpoints" not in str(n)]
        if len(nbs) != 1:
            raise RuntimeError(
                f"expected exactly one notebook in {part_dir}, found {len(nbs)}"
            )
        return nbs[0]

    part0_dirs = sorted(d for d in TUTORIALS.glob("part0_*") if d.is_dir())
    part1_dirs = sorted(d for d in TUTORIALS.glob("part1_*") if d.is_dir())

    part0 = [single_nb(d) for d in part0_dirs]
    part1 = [single_nb(d) for d in part1_dirs]

    # float the canary to the front of part0
    part0.sort(key=lambda p: (CANARY not in str(p), str(p)))
    return part0, part1


def run_notebook(nb_path, clear=True):
    """Execute one notebook in place; return (ok, seconds, message).

    Uses ``jupyter nbconvert --execute --inplace`` from the notebook's own
    directory so relative paths (``../../data``, the notebook's workspace)
    resolve the way they do for a reader. On success, outputs are cleared
    again unless ``clear`` is False.
    """
    rel = nb_path.relative_to(REPO_ROOT)
    t0 = time.time()
    cmd = [
        "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        f"--ExecutePreprocessor.timeout={CELL_TIMEOUT}",
        "--ExecutePreprocessor.kernel_name=python3",
        nb_path.name,
    ]
    proc = subprocess.run(
        cmd,
        cwd=nb_path.parent,
        capture_output=True,
        text=True,
    )
    dt = time.time() - t0

    if proc.returncode != 0:
        # nbconvert puts the traceback on stderr; keep the tail for the report.
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        msg = "\n".join(tail[-25:]) if tail else "(no output captured)"
        return False, dt, msg

    if clear:
        subprocess.run(
            ["jupyter", "nbconvert", "--clear-output", "--inplace", nb_path.name],
            cwd=nb_path.parent,
            capture_output=True,
            text=True,
        )
    return True, dt, ""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fast", action="store_true",
                        help="run only the part0 notebooks (the CI-able subset)")
    parser.add_argument("--no-clear", action="store_true",
                        help="leave executed outputs in the notebooks")
    args = parser.parse_args(argv)

    part0, part1 = discover_notebooks()
    if args.fast:
        notebooks = part0
        print(f"[nb_tests] --fast: running {len(part0)} part0 notebook(s) only")
    else:
        notebooks = part0 + part1
        print(f"[nb_tests] running {len(part0)} part0 + {len(part1)} part1 "
              f"notebook(s) ({len(notebooks)} total)")
    print(f"[nb_tests] per-notebook timeout: {CELL_TIMEOUT}s; repo root: {REPO_ROOT}")
    print()

    results = []   # (rel_path, ok, seconds, message)
    failures = []
    for nb in notebooks:
        rel = nb.relative_to(REPO_ROOT)
        print(f"[nb_tests] running {rel} ...", flush=True)
        ok, dt, msg = run_notebook(nb, clear=not args.no_clear)
        results.append((rel, ok, dt, msg))
        status = "PASS" if ok else "FAIL"
        print(f"[nb_tests]   {status} {rel}  ({dt:6.1f}s)", flush=True)
        if not ok:
            failures.append((rel, msg))
        print(flush=True)

    # --- summary -------------------------------------------------------------
    print("=" * 72)
    print("[nb_tests] timings")
    for rel, ok, dt, _ in results:
        print(f"    {'PASS' if ok else 'FAIL'}  {dt:7.1f}s  {rel}")
    total = sum(dt for _, _, dt, _ in results)
    print(f"    {'':4}  {total:7.1f}s  TOTAL")
    print("=" * 72)

    if failures:
        print(f"[nb_tests] {len(failures)} FAILURE(S):")
        for rel, msg in failures:
            print(f"\n----- {rel} -----")
            print(msg)
        return 1

    print(f"[nb_tests] all {len(results)} notebook(s) passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
