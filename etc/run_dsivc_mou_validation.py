#!/usr/bin/env python3
"""Full-model validation of the DSIVC-recommended plan (maintainer script).

part1_08 uses DSIVC to search a four-dimensional supply-well decision space --
three per-screen rate multipliers (the *vertical distribution* of pumping) plus
the switch-on day -- and hands back a recommended plan. That answer is the
emulator's; the discipline of the series is *never trust an emulator you have
not tested*. This script spends one ~6-min full-model run to test it: the
``base`` posterior parameter field, run through the real reactive-transport
model **at the recommended decision variables** (see ``OPT_PLAN`` below, filled
from the MOU archive) -- a decision that field was never run at in the training
sweep (its sweep run was the canonical 1/1/1, 308).

The result is thinned to the curated obs set and shipped to
``prebaked/dsivc_sweep/`` as ``mou_validation_obs.csv`` (+ a provenance
sidecar), where the notebook loads it and compares the genuine full-model peak
SO4 against the DSIVC-conditioned forecast distribution at that plan.

    python etc/run_dsivc_mou_validation.py                # run it
    python etc/run_dsivc_mou_validation.py --stage-only   # inspect the staged dir

Mirrors the staging/plumbing of ``run_dsivc_sweep.py`` (same dv preprocessor).
"""
import argparse
import json
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "dependencies" / "pyemu"))
import pyemu  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "etc"))
from make_prebaked import resolve_curated_obs, forecast_obsnames, git_sha  # noqa: E402

TEMPLATE = REPO_ROOT / "tutorials" / "part1_02_pstfrom_setup" / "pst_template"
SWEEP_MASTER = REPO_ROOT / "master_sweep"          # carries the base field + dv pars
OUT_DIR = REPO_ROOT / "prebaked" / "dsivc_sweep"

# the DSIVC-recommended plan from part1_08, read off the MOU archive. Three
# per-screen rate multipliers + the switch-on day. UPDATE these after the
# notebook's MOU search reveals the recommended plan (see the archive table).
OPT_PLAN = {
    "dv-rate-ly1": 2.0,
    "dv-rate-ly3": 2.0,
    "dv-rate-ly5": 2.0,
    "dv-switch-day": 308.0,
}
RATE_DVS = ["dv-rate-ly1", "dv-rate-ly3", "dv-rate-ly5"]
ALL_DVS = RATE_DVS + ["dv-switch-day"]
BASE_REAL = "base"

SO4_MW = 96.06
SUPPLY_SCREENS = ["welopt-ly1", "welopt-ly3", "welopt-ly5"]

# same dv preprocessor as run_dsivc_sweep.py: rewrite the supply-well rates from
# dv.dat before every model call, from pristine .orig copies (idempotent). Each
# screen scales by its own multiplier, keyed off the well name in the last column.
PREPROC = '''

# function added for the DSIVC validation run: rewrite the supply-well rates
# from the decision variables in dv.dat. Each screen has its own multiplier,
# keyed off the well name in the last column (welopt-ly1/ly3/ly5 -> dv-rate-
# ly1/ly3/ly5). Reads pristine .orig copies so the transform is idempotent.
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
                screen = parts[-1]
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
    ap.add_argument("--stage-only", action="store_true")
    ap.add_argument("--master", default="master_mou_validation")
    args = ap.parse_args()

    if not (SWEEP_MASTER / "pest.0.par.jcb").exists():
        raise FileNotFoundError(
            f"{SWEEP_MASTER}/pest.0.par.jcb not found; run run_dsivc_sweep.py first "
            "(it carries the base posterior field + the dv parameters)")

    staging = REPO_ROOT / "pst_template_val_stage"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(TEMPLATE, staging)
    for patt in ("pest.rns", "*.rec", "pest.[0-9].*", "pest.obs+noise.jcb",
                 "pest.phi.*", "test.*", "*.rei", "frun.out", "panther.*"):
        for f in staging.glob(patt):
            f.unlink()

    # --- dv template + parameters (identical to run_dsivc_sweep.py) ---
    (staging / "dv.dat.tpl").write_text(
        "ptf ~\n" + "".join(f"{d}   ~   {d}    ~\n" for d in ALL_DVS))
    (staging / "dv.dat").write_text(
        "".join(f"{d}   {OPT_PLAN[d]}\n" for d in ALL_DVS))

    pst = pyemu.Pst(str(staging / "pest.pst"))
    pst.add_parameters(str(staging / "dv.dat.tpl"), pst_path=".")
    par = pst.parameter_data
    for d in RATE_DVS:
        par.loc[d, ["parval1", "parlbnd", "parubnd"]] = [OPT_PLAN[d], 0.0, 2.0]
    par.loc["dv-switch-day", ["parval1", "parlbnd", "parubnd"]] = [
        OPT_PLAN["dv-switch-day"], 308.0, 500.0]
    par.loc[ALL_DVS, "pargp"] = "decvars"
    par.loc[ALL_DVS, "partrans"] = "none"

    # --- dv preprocessor into forward_run.py, called right before mf6rtm ---
    fr = (staging / "forward_run.py").read_text()
    anchor = "    pyemu.os_utils.run(r'mf6rtm')"
    assert anchor in fr, "mf6rtm anchor not found in forward_run.py"
    fr = fr.replace(anchor, "    apply_wellopt_dvars()\n" + anchor, 1)
    fr = fr.replace("def main():", PREPROC + "\ndef main():", 1)
    (staging / "forward_run.py").write_text(fr)

    # --- one-realisation ensemble: the base field at the OPTIMAL decvars ---
    sweep_pst = pyemu.Pst(str(SWEEP_MASTER / "pest.pst"))
    pe = pyemu.ParameterEnsemble.from_binary(
        pst=sweep_pst, filename=str(SWEEP_MASTER / "pest.0.par.jcb"))
    if BASE_REAL not in pe._df.index:
        raise KeyError(f"'{BASE_REAL}' realisation not in {SWEEP_MASTER}/pest.0.par.jcb")
    row = pe._df.loc[[BASE_REAL], pst.par_names].copy()   # align to staged pst pars
    for d in ALL_DVS:
        row.loc[BASE_REAL, d] = OPT_PLAN[d]
    pyemu.ParameterEnsemble(pst=pst, df=row).to_binary(str(staging / "val_pe.jcb"))

    pst.control_data.noptmax = -1
    pst.pestpp_options["ies_par_en"] = "val_pe.jcb"
    pst.pestpp_options["ies_num_reals"] = 1
    pst.pestpp_options["save_binary"] = True
    pst.write(str(staging / "pest.pst"), version=2)
    _split = ", ".join(f"{d.split('-')[-1]}={OPT_PLAN[d]:.2f}" for d in RATE_DVS)
    print(f"staged validation run: base field @ (screens {_split}, "
          f"switch {OPT_PLAN['dv-switch-day']:.0f}) -> {staging}")

    if args.stage_only:
        print("stage-only: inspect the staged dir")
        return

    master = REPO_ROOT / args.master
    if master.exists():
        shutil.rmtree(master)
    print(f"running the full model once (~6 min) -> {master}")
    # pestpp-ies noptmax=-1 evaluates the (single-realisation) ensemble and writes
    # pest.0.obs.jcb, then raises "too few active realizations" trying to proceed
    # past a 1-member ensemble. That error is after the run we care about, so we
    # tolerate it as long as the obs file landed.
    try:
        pyemu.os_utils.start_workers(
            str(staging), "pestpp-ies", "pest.pst",
            num_workers=1, worker_root=str(REPO_ROOT), master_dir=str(master))
    except Exception as e:
        if not (master / "pest.0.obs.jcb").exists():
            raise
        print(f"  (tolerated post-run pestpp-ies message: {str(e)[:80]}...)")
    if not (master / "pest.0.obs.jcb").exists():
        raise RuntimeError(f"full-model run produced no obs in {master}")
    shutil.rmtree(staging)

    # --- thin to the curated obs set and ship to prebaked ---
    mpst = pyemu.Pst(str(master / "pest.pst"))
    mpst.try_parse_name_metadata()
    oe = pyemu.ObservationEnsemble.from_binary(
        pst=mpst, filename=str(master / "pest.0.obs.jcb"))
    keep = resolve_curated_obs(mpst, None)
    cols = [c for c in keep if c in oe._df.columns]
    val = oe._df.loc[:, cols].iloc[0]          # single realisation -> Series
    val.name = "obsval"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    val.to_csv(OUT_DIR / "mou_validation_obs.csv")

    # peak SO4 over the supply-well screens / supply window, for the sidecar
    fc = forecast_obsnames(mpst)
    fc = [c for c in fc if c in oe._df.columns]
    peak_mol = float(oe._df.loc[:, fc].iloc[0].max())
    peak_mgl = peak_mol * 1000.0 * SO4_MW

    prov = {
        "artifact": "DSIVC optimum full-model validation",
        "produced_by": "etc/run_dsivc_mou_validation.py",
        "git_sha": git_sha(),
        "generated_on": str(date.today()),
        "base_field": BASE_REAL,
        "decvars": OPT_PLAN,
        "n_obs": len(cols),
        "peak_so4_molL": peak_mol,
        "peak_so4_mgL": round(peak_mgl, 2),
        "notes": ("base posterior field run through the full reactive-transport "
                  "model at the DSIVC-recommended plan (per-screen vertical rate "
                  "split + switch-on) -- a held-out decision (the field's sweep "
                  "run was the canonical 1/1/1, 308). Compared in part1_08 "
                  "against the DSIVC-conditioned forecast distribution there."),
    }
    with open(OUT_DIR / "mou_validation.json", "w") as f:
        json.dump(prov, f, indent=2)
    print(f"full-model peak SO4 at the optimum: {peak_mgl:.1f} mg/L")
    print(f"wrote {OUT_DIR/'mou_validation_obs.csv'} ({len(cols)} obs) + sidecar")


if __name__ == "__main__":
    main()
