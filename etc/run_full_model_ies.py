#!/usr/bin/env python3
"""Stage and launch the full-model PESTPP-IES history match (maintainer script).

Conditions the full DIZON model on the canonical synthetic truth, with the
canonical noise model — the apples-to-apples counterpart of part1_05's
emulator conditioning, consumed by part1_06 and (its posterior fields) by the
part1_08 DSIVC training sweep.

What it stages
--------------
* template: the part1_02 ``pst_template`` (15,025 pars incl. ``pyr-lograte``)
* truth: the part1_03 pick (75th percentile of the prior peak; pass
  ``--truth`` to override) — its obs values become the conditioning targets,
  and its realisation is DROPPED from the prior parameter ensemble
* weights: part1_03's staged canonical noise model (species sigmas in stored
  units, weight = 1/sigma) — measurement error only; the full model needs no
  emulator-error allowance
* phi: balanced per site:species group (equal share), the same per-species
  rebalancing part1_03/part1_05 use to stop a long series over-counting
* noise: correlated within each site:species series (one draw per realisation
  per series, scaled per-time by each obs's sigma), concentrations truncated
  at zero (physical bounds)
* solver: default IES, ``noptmax=3``, no multimodal options (deliberate; see
  part1_05's evidence beat)

Usage (repo root, rtm_gmdsi env; ~200 reals x 4 evals x ~7 min => plan ~9 h
with 12 workers)::

    python etc/run_full_model_ies.py --stage-only      # inspect first
    python etc/run_full_model_ies.py --workers 12      # stage + launch

Afterwards: ``python etc/make_prebaked.py full-model-ies --source master_hm_v2``
(implement the stub), re-bake part1_06.
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "dependencies" / "pyemu"))
import pyemu  # noqa: E402

TEMPLATE = REPO_ROOT / "tutorials" / "part1_02_pstfrom_setup" / "pst_template"
STAGED_13 = (REPO_ROOT / "tutorials" / "part1_03_obs_weights_and_truth"
             / "obs_template" / "pest.pst")
PRIOR_MASTER = REPO_ROOT / "master_priormc"
TRUTH_PCTILE = 0.75
NOISE_SEED = 20260606


def pick_truth(oe_df, obs_data):
    od = obs_data.copy()
    od["time"] = od["time"].astype(float)
    m = (od.obsid.astype(str).isin(["welopt-ly1", "welopt-ly3", "welopt-ly5"])
         & (od.variable == "so4") & (od.time >= 308) & (od.time <= 728))
    cols = [c for c in od.loc[m].obsnme if c in oe_df.columns]
    pk = (oe_df[cols] * 96060.0).max(axis=1).drop(index="base", errors="ignore")
    target = pk.quantile(TRUTH_PCTILE)
    real = (pk - target).abs().idxmin()
    return real, float(pk.loc[real])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int,
                    default=max(2, psutil.cpu_count(logical=False) - 4))
    ap.add_argument("--master", default="master_hm_v2")
    ap.add_argument("--truth", default=None,
                    help="realisation name; default = the 87.5th-pctile pick")
    ap.add_argument("--stage-only", action="store_true")
    args = ap.parse_args()

    staging = REPO_ROOT / "pst_template_hm_stage"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(TEMPLATE, staging)

    pst = pyemu.Pst(str(staging / "pest.pst"))
    pst.try_parse_name_metadata()

    # --- canonical weights/sigmas from part1_03's staged control file ---
    p3 = pyemu.Pst(str(STAGED_13))
    w3 = p3.observation_data
    pst.observation_data["weight"] = 0.0
    common = pst.observation_data.index.intersection(w3.index)
    pst.observation_data.loc[common, "weight"] = w3.loc[common, "weight"]
    pst.observation_data.loc[common, "standard_deviation"] = \
        w3.loc[common, "standard_deviation"]
    nz = pst.observation_data.loc[pst.observation_data.weight > 0]
    print(f"weights: {len(nz)} conditioning obs from part1_03's staged canon")

    # --- balance phi per site:species group (equal share), as part1_03 does ---
    od = pst.observation_data
    od.loc[nz.index, "obgnme"] = (od.loc[nz.index, "obsid"].astype(str) + ":"
                                  + od.loc[nz.index, "variable"].astype(str))
    wgroups = od.loc[od.weight > 0, "obgnme"].unique()
    phi_factors = pd.Series(1.0 / len(wgroups), index=wgroups)
    phi_factors.to_csv(staging / "ies_phi_factors.csv", header=False)
    pst.pestpp_options["ies_phi_factor_file"] = "ies_phi_factors.csv"
    print(f"phi factors: {len(wgroups)} site:species groups, equal share")

    # --- truth values: pick from the SAME artifact part1_03 uses (the thinned
    # prebaked ensemble), so the pick matches the notebooks by construction;
    # values are then taken from the full master ensemble (all obs) ---
    prebaked = REPO_ROOT / "prebaked"
    if (prebaked / "prior_mc_obs_ensemble.jcb").exists():
        bpst = pyemu.Pst(str(prebaked / "pest.pst"))
        boe = pyemu.ObservationEnsemble.from_binary(
            pst=bpst, filename=str(prebaked / "prior_mc_obs_ensemble.jcb"))._df
        pick_df, pick_obs = boe, bpst.observation_data
    else:
        pick_df, pick_obs = None, None
    ppst = pyemu.Pst(str(PRIOR_MASTER / "pest.pst"))
    oe = pyemu.ObservationEnsemble.from_binary(
        pst=ppst, filename=str(PRIOR_MASTER / "pest.0.obs.jcb"))._df
    if pick_df is None:
        pick_df, pick_obs = oe, ppst.observation_data
    if args.truth is None:
        truth_real, truth_pk = pick_truth(pick_df, pick_obs)
    else:
        # explicit pick (use part1_03's baked pick -- it is the canon; this
        # script's automatic pick can disagree on base-row/quantile handling)
        truth_real = args.truth
        od = pick_obs.copy(); od["time"] = od["time"].astype(float)
        m = (od.obsid.astype(str).isin(["welopt-ly1", "welopt-ly3", "welopt-ly5"])
             & (od.variable == "so4") & (od.time >= 308) & (od.time <= 728))
        cols = [c for c in od.loc[m].obsnme if c in pick_df.columns]
        truth_pk = float((pick_df.loc[truth_real, cols] * 96060.0).max())
    print(f"truth: realisation {truth_real} (peak ~{truth_pk:.1f} mg/L)")
    tvals = oe.loc[truth_real]
    settable = [o for o in pst.observation_data.index if o in tvals.index]
    pst.observation_data.loc[settable, "obsval"] = tvals[settable].values

    # --- prior par ensemble, truth dropped ---
    pe = pyemu.ParameterEnsemble.from_binary(
        pst=pst, filename=str(staging / "prior_pe.jcb"))
    if truth_real in pe._df.index:
        pe._df = pe._df.drop(index=truth_real)
        print(f"dropped truth realisation {truth_real} from the prior ensemble "
              f"({pe.shape[0]} remain)")
    pe.to_binary(str(staging / "prior_pe_hm.jcb"))

    # --- correlated-within-series noise ensemble (measurement error only) ---
    nzobs = pst.observation_data.loc[pst.observation_data.weight > 0]
    noise = pyemu.ObservationEnsemble.from_gaussian_draw(
        pst, num_reals=pe.shape[0])
    rng = np.random.default_rng(NOISE_SEED)
    ABS_SPECIES = {"ph", "tmp"}   # absolute-sigma species; the rest are concentrations
    series = nzobs.obsid.astype(str) + ":" + nzobs.variable.astype(str)
    for grp in series.unique():
        g = nzobs.loc[series == grp]
        z = rng.standard_normal(noise.shape[0])
        vals = (g.obsval.values[None, :]
                + z[:, None] * g.standard_deviation.values[None, :])
        if grp.split(":")[-1] not in ABS_SPECIES:   # concentrations stay >= 0
            np.clip(vals, 0.0, None, out=vals)
        noise.loc[:, g.index] = vals
    noise._df = noise._df.dropna(axis=1)
    noise.to_binary(str(staging / "noise_hm.jcb"))

    # --- solver settings: default IES, deliberately no multimodal ---
    pst.control_data.noptmax = 3
    pst.pestpp_options["ies_par_en"] = "prior_pe_hm.jcb"
    pst.pestpp_options["ies_observation_ensemble"] = "noise_hm.jcb"
    pst.pestpp_options["ies_num_reals"] = pe.shape[0]
    pst.pestpp_options["save_binary"] = True
    pst.pestpp_options.pop("ies_multimodal_alpha", None)
    pst.write(str(staging / "pest.pst"), version=2)
    print(f"staged: noptmax=3, {pst.npar_adj} adj pars, "
          f"{(pst.observation_data.weight > 0).sum()} nnz obs, "
          f"{pe.shape[0]} reals")

    if args.stage_only:
        print(f"stage-only: inspect {staging}, then rerun without --stage-only")
        return

    master = REPO_ROOT / args.master
    if master.exists():
        shutil.rmtree(master)
    print(f"launching {args.workers} workers -> {master} "
          f"(~4 x {pe.shape[0]} runs x ~7 min)")
    pyemu.os_utils.start_workers(
        str(staging), "pestpp-ies", "pest.pst",
        num_workers=args.workers,
        worker_root=str(REPO_ROOT),
        master_dir=str(master),
    )
    shutil.rmtree(staging)
    print("full-model IES complete")


if __name__ == "__main__":
    main()
