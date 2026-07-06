#!/usr/bin/env python3
"""Stage and launch the DSIVC training sweep (maintainer script).

The part1_08 optimization needs an emulator that knows how the forecast
responds to the DECISION VARIABLES — and coverage of decision space is
designed, not inherited. This script runs the full model over the posterior
parameter fields (the part1_06 best-phi ensemble) with the supply-well
operation re-sampled across its decision bounds. The decision is the
*vertical distribution* of extraction across the three supply screens plus
the switch-on timing:

* ``dv-rate-ly1`` in [0, 2] — multiplier on screen welopt-ly1 (base -1300)
* ``dv-rate-ly3`` in [0, 2] — multiplier on screen welopt-ly3 (base -300)
* ``dv-rate-ly5`` in [0, 2] — multiplier on screen welopt-ly5 (base -300)
  (0 fully closes a screen; 2 doubles it — the screens move independently,
  which is the whole point: it lets the emulator learn how redistributing
  pumping *vertically* trades total volume against peak sulfate)
* ``dv-switch-day`` in [308, 500] — the well activates at the first stress
  period whose start is >= this day (SP granularity; the continuous dvar is
  a relaxation the emulator smooths over)

Plumbing: a tiny ``dv.dat`` template carries the four dvars; a self-contained
pre-processor appended to ``forward_run.py`` rewrites the
``gwf.welopt_stress_period_data_*.txt`` files from pristine ``.orig`` copies
before every model call (idempotent across re-runs in a worker), keying each
screen's multiplier off the well name in the last column.

One dv draw per posterior field (~160 runs x ~7 min => ~2.5 h on 12 workers).

    python etc/run_dsivc_sweep.py --stage-only
    python etc/run_dsivc_sweep.py --workers 12

Afterwards: ``python etc/make_prebaked.py dsivc-sweep --source master_sweep``.
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import psutil

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "dependencies" / "pyemu"))
import pyemu  # noqa: E402

TEMPLATE = REPO_ROOT / "tutorials" / "part1_02_pstfrom_setup" / "pst_template"
FULL_MODEL_IES = REPO_ROOT / "prebaked" / "full_model_ies"
SWEEP_SEED = 20260607


def resolve_post_par():
    """Resolve the shipped posterior parameter ensemble.

    ``make_prebaked.py full-model-ies`` copies only the best-phi iteration's
    parameter ensemble as ``hm.<best_iter>.par.jcb``. The best iteration is not
    fixed — it can move when the full-model IES is re-baked under a different
    truth/weighting — so we resolve it dynamically rather than assume iter 1.
    """
    cands = sorted(FULL_MODEL_IES.glob("hm.*.par.jcb"))
    if not cands:
        raise FileNotFoundError(
            f"no posterior parameter ensemble (hm.*.par.jcb) in {FULL_MODEL_IES}; "
            "run run_full_model_ies.py then 'make_prebaked.py full-model-ies' first")
    if len(cands) > 1:
        print(f"  note: {len(cands)} hm.*.par.jcb present; using {cands[-1].name}")
        return cands[-1]
    return cands[0]

# supply stress periods (1-based) and their start days
SUPPLY_SP_STARTS = {22: 308, 23: 336, 24: 364, 25: 392, 26: 420, 27: 455,
                    28: 490, 29: 518, 30: 546, 31: 574, 32: 609, 33: 644,
                    34: 672, 35: 700}

PREPROC = '''

# function added for the DSIVC training sweep: rewrite the supply-well rates
# from the decision variables in dv.dat. Each screen has its own multiplier,
# keyed off the well name in the last column of each row (welopt-ly1/ly3/ly5 ->
# dv-rate-ly1/ly3/ly5), so pumping can be redistributed vertically. Reads
# pristine .orig copies so the transform is idempotent across repeated forward
# runs in this directory.
def apply_wellopt_dvars():
    import shutil as _sh
    from pathlib import Path as _P
    starts = {22: 308, 23: 336, 24: 364, 25: 392, 26: 420, 27: 455, 28: 490,
              29: 518, 30: 546, 31: 574, 32: 609, 33: 644, 34: 672, 35: 700}
    vals = {}
    for ln in _P("dv.dat").read_text().split("\\n"):
        parts = ln.split()
        if len(parts) == 2:
            vals[parts[0]] = float(parts[1])
    switch = vals["dv-switch-day"]
    for sp, start in starts.items():
        f = _P(f"gwf.welopt_stress_period_data_{sp}.txt")
        orig = _P(str(f) + ".orig")
        if not orig.exists():
            _sh.copy(f, orig)
        out = []
        for row in orig.read_text().split("\\n"):
            parts = row.split()
            if len(parts) > 3:
                screen = parts[-1]                       # e.g. welopt-ly1
                mult = vals["dv-rate-" + screen.split("-")[-1]]
                factor = mult if start >= switch else 0.0
                parts[2] = "{0:.8E}".format(float(parts[2]) * factor)
                out.append("  " + " ".join(parts))
            elif row.strip():
                out.append(row)
        f.write_text("\\n".join(out) + "\\n")
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int,
                    default=max(2, psutil.cpu_count(logical=False) - 4))
    ap.add_argument("--master", default="master_sweep")
    ap.add_argument("--stage-only", action="store_true")
    args = ap.parse_args()

    staging = REPO_ROOT / "pst_template_sweep_stage"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(TEMPLATE, staging)
    # the template carries residue from its own in-notebook test runs; a stale
    # pest.rns with a different parameter count makes RunStorage::reset() fail
    for patt in ("pest.rns", "*.rec", "pest.[0-9].*", "pest.obs+noise.jcb",
                 "pest.phi.*", "test.*", "*.rei", "frun.out", "panther.*"):
        for f in staging.glob(patt):
            f.unlink()

    # --- dv template + parameters: three per-screen rate multipliers + switch ---
    rate_dvs = ["dv-rate-ly1", "dv-rate-ly3", "dv-rate-ly5"]
    all_dvs = rate_dvs + ["dv-switch-day"]
    (staging / "dv.dat.tpl").write_text(
        "ptf ~\n" + "".join(f"{d}   ~   {d}    ~\n" for d in all_dvs))
    (staging / "dv.dat").write_text(
        "".join(f"{d}   1.0\n" for d in rate_dvs) + "dv-switch-day  308.0\n")

    pst = pyemu.Pst(str(staging / "pest.pst"))
    pst.add_parameters(str(staging / "dv.dat.tpl"), pst_path=".")
    par = pst.parameter_data
    for d in rate_dvs:
        par.loc[d, ["parval1", "parlbnd", "parubnd"]] = [1.0, 0.0, 2.0]
    par.loc["dv-switch-day", ["parval1", "parlbnd", "parubnd"]] = [308.0, 308.0, 500.0]
    par.loc[all_dvs, "pargp"] = "decvars"
    par.loc[all_dvs, "partrans"] = "none"

    # --- pre-processor into forward_run.py, called right before mf6rtm ---
    fr = (staging / "forward_run.py").read_text()
    anchor = "    pyemu.os_utils.run(r'mf6rtm')"
    assert anchor in fr, "mf6rtm anchor not found in forward_run.py"
    fr = fr.replace(anchor, "    apply_wellopt_dvars()\n" + anchor, 1)
    fr = fr.replace("def main():", PREPROC + "\ndef main():", 1)
    (staging / "forward_run.py").write_text(fr)

    # --- sweep ensemble: posterior fields x one dv draw each ---
    post_par = resolve_post_par()
    print(f"posterior parameter fields: {post_par.name}")
    pe = pyemu.ParameterEnsemble.from_binary(
        pst=pst, filename=str(post_par))
    df = pe._df
    rng = np.random.default_rng(SWEEP_SEED)
    n = df.shape[0]
    for d in rate_dvs:
        df[d] = rng.uniform(0.0, 2.0, n)
    df["dv-switch-day"] = rng.uniform(308.0, 500.0, n)
    # the base row is the control: canonical operation (all screens at 1.0)
    if "base" in df.index:
        df.loc["base", rate_dvs] = 1.0
        df.loc["base", "dv-switch-day"] = 308.0
    pyemu.ParameterEnsemble(pst=pst, df=df).to_binary(
        str(staging / "sweep_pe.jcb"))
    print(f"sweep ensemble: {df.shape[0]} runs "
          f"(posterior fields x dv draws; base = canonical operation)")

    pst.control_data.noptmax = -1
    pst.pestpp_options["ies_par_en"] = "sweep_pe.jcb"
    pst.pestpp_options["ies_num_reals"] = df.shape[0]
    pst.pestpp_options["save_binary"] = True
    pst.write(str(staging / "pest.pst"), version=2)
    print(f"staged: noptmax=-1, {pst.npar_adj} adj pars (incl 4 dvars)")

    if args.stage_only:
        print(f"stage-only: inspect {staging}")
        return

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
    print("DSIVC training sweep complete")


if __name__ == "__main__":
    main()
