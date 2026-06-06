#!/usr/bin/env python3
"""Launch the full prior Monte Carlo (maintainer script).

Copies the part1_02 template to a staging dir, sets ``noptmax=-1`` (run the
prior ensemble, no upgrades) and ``save_binary``, and runs PESTPP-IES with
parallel workers into a repo-root master directory. ~201 forward runs at
~7 min each: budget ~2-2.5 h with 12 workers.

Run from the repo root, inside the ``rtm_gmdsi`` env (the forward run needs
``python``/``mf6rtm`` on PATH):

    python etc/run_prior_mc.py [--workers N] [--master master_priormc_v2]

Afterwards: re-bake part1_03/04/05/07, then
``python etc/make_prebaked.py prior-mc --source <master>``.
"""
import argparse
import shutil
import sys
from pathlib import Path

import psutil

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "dependencies" / "pyemu"))
import pyemu  # noqa: E402

TEMPLATE = REPO_ROOT / "tutorials" / "part1_02_pstfrom_setup" / "pst_template"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int,
                    default=max(2, psutil.cpu_count(logical=False) - 4))
    ap.add_argument("--master", default="master_priormc_v2")
    args = ap.parse_args()

    staging = REPO_ROOT / "pst_template_priormc_stage"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(TEMPLATE, staging)

    pst = pyemu.Pst(str(staging / "pest.pst"))
    pst.control_data.noptmax = -1
    pst.pestpp_options["save_binary"] = True
    pst.write(str(staging / "pest.pst"), version=2)
    print(f"staged {staging} (noptmax=-1, {pst.npar_adj} adj pars, "
          f"{pst.pestpp_options['ies_num_reals']} reals)")

    master = REPO_ROOT / args.master
    if master.exists():
        shutil.rmtree(master)
    print(f"launching {args.workers} workers -> {master}")
    pyemu.os_utils.start_workers(
        str(staging), "pestpp-ies", "pest.pst",
        num_workers=args.workers,
        worker_root=str(REPO_ROOT),
        master_dir=str(master),
    )
    shutil.rmtree(staging)
    print("prior MC complete")


if __name__ == "__main__":
    main()
