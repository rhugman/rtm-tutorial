#!/usr/bin/env python3
"""Write the tracked, thinned ``prebaked/`` artifacts the tutorials read.

Maintainer script. **It does not run any model.** It reads the results of
expensive runs that already exist in the repo-root legacy master directories
(``master_priormc/``, ``master_hm/``) and thins them down to the small,
curated obs set the notebooks actually consume -- so the tutorials can ship
the answers to a ~20-hour ensemble without shipping a 700 MB obs ensemble or
asking the reader to wait.

What this implements *now*
--------------------------
The first row of the ``prebaked/`` inventory in ``docs/REDESIGN.md``: the
**prior Monte Carlo obs ensemble**, thinned to the curated obs (conditioning
species in the history window) plus the named forecast group, written as a
binary ``.jcb`` next to a thinned ``pest.pst`` and a provenance ``.json``
sidecar. ``part1_03``/``part1_04``/``part1_05`` resolve
``prebaked/prior_mc_obs_ensemble.jcb`` + ``prebaked/pest.pst`` in preference to
a live ``master_priormc/`` run.

What is stubbed (clear TODOs below)
-----------------------------------
* ``thin_full_model_ies`` -- the full-model PESTPP-IES results (``master_hm``)
  for ``part1_06`` and ``part1_08``.
* ``build_dsivc_training_sweep`` -- the DSIVC training sweep for ``part1_08``.

Provenance
----------
Each artifact gets a ``.json`` sidecar recording the source directory, the
current git SHA, the curated obs count and the realisation count. The
``hardware``, ``walltime_hours`` and ``run_date`` fields are written as
``null``/``"TODO"`` for the maintainer to fill from their run log -- this
script cannot know them.

Usage
-----
    # default: thin the prior MC, curated obs taken from the part1_03 staged
    # control file if present, else from the built-in fallback rule
    python etc/make_prebaked.py prior-mc

    # supply the curated obs set explicitly (one obsname per line, or a CSV
    # with an 'obsnme' column)
    python etc/make_prebaked.py prior-mc --obs-csv curated_obs.csv

    # override source / output locations
    python etc/make_prebaked.py prior-mc --source master_priormc --out prebaked

Pure python + pyemu. Requires the ``rtm_gmdsi`` environment (vendored pyemu).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pyemu

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- canonical numbers (mirror CONTEXT.md; do not drift) --------------------
HISTORY_END = 252      # decision date: last day usable for conditioning
SUPPLY_START = 308     # supply well switches on
SUPPLY_END = 728       # end of the supply period
COND_SPECIES = ["so4", "o0", "no3", "ph", "tmp"]   # conditioning species (O2 == 'o0')
HELDBACK_SPECIES = ["ca", "na", "fe", "fe2"]       # held-back cations (the dataworth question; mg/k not simulated)
SUPPLY_SCREENS = ["welopt-ly1", "welopt-ly3", "welopt-ly5"]   # ALL supply-well screens; the forecast is the max over every one
FORECAST_SPECIES = "so4"
N_REALS_TARGET = 201   # realisations drawn and kept (a handful fail, as ensembles do)


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #
def git_sha():
    """Return the current HEAD SHA, or 'unknown' if git is unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def write_provenance(out_json, *, artifact, source_dir, n_obs, n_reals, notes=""):
    """Write a provenance sidecar. Maintainer fills the TODO fields by hand."""
    payload = {
        "artifact": artifact,
        "produced_by": "etc/make_prebaked.py",
        "source_dir": str(source_dir),
        "git_sha": git_sha(),
        "thinned_to_n_obs": int(n_obs),
        "n_realisations": int(n_reals),
        "generated_on": date.today().isoformat(),
        # --- maintainer to fill from the run log; this script cannot know them ---
        "hardware": "TODO: e.g. MacBook Pro M2, 16 GB",
        "walltime_hours": None,   # TODO: wall time of the underlying ensemble run
        "run_date": "TODO: date the underlying ensemble was actually run",
        "cost_note": "~6 min per forward model run; full prior MC is ~201 runs",
        "notes": notes,
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"  wrote provenance -> {out_json}")


# --------------------------------------------------------------------------- #
# curated obs resolution
# --------------------------------------------------------------------------- #
def forecast_obsnames(pst):
    """Obsnames of the forecast group: supply-well SO4 over the supply period.

    Prefer the named ``forecast`` obs group if the control file carries one
    (part1_02 tags it across ALL supply-well screens — the canonical forecast
    is the max over every screen, so dropping any screen silently corrupts
    every downstream peak). Fall back to a name-based selection over all
    supply screens otherwise.
    """
    od = pst.observation_data.copy()
    if "forecast" in set(od.obgnme):
        return od.loc[od.obgnme == "forecast"].obsnme.tolist()
    od["time"] = od["time"].astype(float)
    m = (
        (od.variable == FORECAST_SPECIES)
        & (od.obsid.astype(str).isin(SUPPLY_SCREENS))
        & (od.time >= SUPPLY_START)
        & (od.time <= SUPPLY_END)
    )
    return od.loc[m].obsnme.tolist()


def curated_obs_from_part1_03(pst):
    """Curated obs from a part1_03-staged control file, if one is on disk.

    ``part1_03`` writes a staged ``pest.pst`` into its workspace with the
    conditioning weights set (non-zero only for conditioning species at field
    sites in the history window). That weighting *is* the curated conditioning
    set. We take the non-zero-weight obs and add the forecast group.

    Returns the obsname list, or ``None`` if the staged file is absent.
    """
    staged = (REPO_ROOT / "tutorials" / "part1_03_obs_weights_and_truth"
              / "obs_template" / "pest.pst")
    if not staged.exists():
        return None
    print(f"  curated obs source: part1_03 staged control file {staged}")
    p3 = pyemu.Pst(str(staged))
    p3.try_parse_name_metadata()
    od = p3.observation_data
    cond = od.loc[od.weight > 0].obsnme.tolist()
    # the held-back cations ride along at the same sites/window: part1_07's
    # dataworth question ("would measuring these have helped?") needs their
    # series in the shipped ensemble even though nothing conditions on them
    od2 = od.copy()
    od2["time"] = od2["time"].astype(float)
    cond_sites = set(od.loc[od.weight > 0, "obsid"].astype(str))
    # fortnightly stride: a lab campaign, not a sensor -- and it keeps the
    # shipped artifact under git-friendly size
    hb = od2.loc[od2.variable.isin(HELDBACK_SPECIES)
                 & od2.obsid.astype(str).isin(cond_sites)
                 & (od2.time <= HISTORY_END)
                 & (od2.time % 14 < 0.5)].obsnme.tolist()
    keep = sorted(set(cond).union(hb).union(forecast_obsnames(p3)))
    return keep


def curated_obs_from_csv(csv_path):
    """Curated obs from an explicit CSV/text file of obsnames."""
    import pandas as pd
    csv_path = Path(csv_path)
    text = csv_path.read_text()
    first = text.splitlines()[0].strip().lower() if text.strip() else ""
    if "obsnme" in first or "," in first:
        df = pd.read_csv(csv_path)
        col = "obsnme" if "obsnme" in df.columns else df.columns[0]
        names = df[col].astype(str).tolist()
    else:
        names = [ln.strip() for ln in text.splitlines() if ln.strip()]
    print(f"  curated obs source: --obs-csv {csv_path} ({len(names)} names)")
    return names


def curated_obs_fallback(pst):
    """Built-in fallback curated set, mirroring the part1_03 thinning rule.

    Used only when neither ``--obs-csv`` nor a part1_03 staged control file is
    available, so the script is runnable standalone. Keeps:

    * conditioning species (SO4, O2 as 'o0', NO3, pH, Tmp) at the model output
      times in the history window (time <= 252) with a finite value; plus
    * the forecast group (supply-well SO4 over the supply period).

    This does *not* apply the field-site restriction part1_03 uses (that needs
    ``data/obs_chem_cleaned.csv``); it is deliberately a slight over-set so the
    fallback never drops an obs the notebook expects.
    """
    od = pst.observation_data.copy()
    od["time"] = od["time"].astype(float)
    cond = od.loc[
        (od.oname == "conc")
        & (od.variable.isin(COND_SPECIES))
        & (od.time <= HISTORY_END)
        & (od.obsval < 1e30)
    ].obsnme.tolist()
    keep = sorted(set(cond).union(forecast_obsnames(pst)))
    print(f"  curated obs source: built-in fallback rule "
          f"(conditioning species <= day {HISTORY_END} + forecast group)")
    return keep


def resolve_curated_obs(pst, obs_csv):
    """Resolve the curated obs list in the canonical precedence order."""
    if obs_csv is not None:
        return curated_obs_from_csv(obs_csv)
    from_p3 = curated_obs_from_part1_03(pst)
    if from_p3 is not None:
        return from_p3
    return curated_obs_fallback(pst)


# --------------------------------------------------------------------------- #
# prior MC (implemented)
# --------------------------------------------------------------------------- #
def thin_prior_mc(source_dir, out_dir, obs_csv=None):
    """Thin the prior MC obs ensemble to the curated obs set and write prebaked.

    Reads ``<source>/pest.pst`` and the prior obs ensemble (``pest.0.obs.jcb``),
    keeps only the curated obs columns, and writes:

    * ``<out>/prior_mc_obs_ensemble.jcb`` -- thinned obs ensemble (binary)
    * ``<out>/prior_mc_obs_ensemble.csv`` -- same, as CSV (human-inspectable)
    * ``<out>/pest.pst``                  -- control file dropped to curated obs
    * ``<out>/prior_mc_obs_ensemble.json`` -- provenance sidecar

    Notebooks load it via
    ``ObservationEnsemble.from_binary(pst=Pst('prebaked/pest.pst'),
    filename='prebaked/prior_mc_obs_ensemble.jcb')``.
    """
    source_dir = Path(source_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pst_path = source_dir / "pest.pst"
    if not pst_path.exists():
        raise FileNotFoundError(f"no control file at {pst_path}")
    print(f"reading control file {pst_path}")
    pst = pyemu.Pst(str(pst_path))
    pst.try_parse_name_metadata()

    # locate the prior obs ensemble (iteration-0 obs ensemble from PESTPP-IES)
    oe_jcb = source_dir / "pest.0.obs.jcb"
    oe_csv = source_dir / "pest.0.obs.csv"
    if oe_jcb.exists():
        print(f"reading prior obs ensemble {oe_jcb}")
        oe = pyemu.ObservationEnsemble.from_binary(pst=pst, filename=str(oe_jcb))
    elif oe_csv.exists():
        print(f"reading prior obs ensemble {oe_csv}")
        oe = pyemu.ObservationEnsemble.from_csv(pst=pst, filename=str(oe_csv))
    else:
        raise FileNotFoundError(
            f"no prior obs ensemble (pest.0.obs.jcb/.csv) in {source_dir}")
    print(f"  full obs ensemble: {oe.shape[0]} reals x {oe.shape[1]} obs")

    # resolve and validate the curated obs set
    requested = resolve_curated_obs(pst, obs_csv)
    available = set(oe.columns)
    keep = [o for o in requested if o in available]
    missing = sorted(set(requested) - available)
    if missing:
        print(f"  WARNING: {len(missing)} curated obs absent from the ensemble "
              f"(skipped); first few: {missing[:5]}")
    if not keep:
        raise RuntimeError("curated obs set is empty after intersecting with the ensemble")
    print(f"  thinning {oe.shape[1]} -> {len(keep)} curated obs")

    thinned = oe.loc[:, keep].copy()
    if thinned.shape[0] != N_REALS_TARGET:
        print(f"  note: {thinned.shape[0]} realisations on disk "
              f"(target draw was {N_REALS_TARGET}; some runs fail, as expected)")

    # write the thinned ensemble (binary + csv)
    jcb_out = out_dir / "prior_mc_obs_ensemble.jcb"
    csv_out = out_dir / "prior_mc_obs_ensemble.csv"
    thinned.to_binary(str(jcb_out))
    thinned.to_csv(str(csv_out))
    print(f"  wrote thinned ensemble -> {jcb_out}")
    print(f"  wrote thinned ensemble -> {csv_out}")

    # write a control file dropped to the curated obs so the artifact is
    # self-describing (the notebook builds an ObservationEnsemble from it). We
    # subset observation_data directly rather than using drop_observations
    # (which is instruction-file based): this is a *read-only* descriptor of the
    # thinned ensemble, never used to drive a model run, so the tpl/ins plumbing
    # is irrelevant and is cleared.
    pst_out = out_dir / "pest.pst"
    pst_thin = pyemu.Pst(str(pst_path))
    pst_thin.try_parse_name_metadata()
    keep_set = set(keep)
    pst_thin.observation_data = pst_thin.observation_data.loc[
        pst_thin.observation_data.obsnme.isin(keep_set)].copy()
    # this artifact is not runnable; drop the model i/o plumbing so writing it
    # does not depend on tpl/ins files that are not shipped in prebaked/
    pst_thin.model_input_data = pst_thin.model_input_data.iloc[0:0].copy()
    pst_thin.model_output_data = pst_thin.model_output_data.iloc[0:0].copy()
    pst_thin.model_command = []
    # version=1: a single self-contained .pst, no external *_data.csv companions
    # cluttering prebaked/ (the artifact is a read-only obs descriptor).
    pst_thin.write(str(pst_out), version=1)
    print(f"  wrote thinned control file -> {pst_out} ({pst_thin.nobs} obs)")

    write_provenance(
        out_dir / "prior_mc_obs_ensemble.json",
        artifact="prior_mc_obs_ensemble",
        source_dir=source_dir,
        n_obs=len(keep),
        n_reals=thinned.shape[0],
        notes=("Prior Monte Carlo obs ensemble thinned to the curated "
               "conditioning + forecast obs. Consumed by part1_03, part1_04, "
               "part1_05."),
    )
    print("prior-mc: done")


# --------------------------------------------------------------------------- #
# stubs -- full-model IES and DSIVC sweep (not yet implementable end to end)
# --------------------------------------------------------------------------- #
def thin_full_model_ies(source_dir, out_dir, obs_csv=None):
    """STUB: thin the full-model PESTPP-IES results (master_hm) for 1_06/1_08.

    TODO (maintainer):
      * Read ``<source>/pest.pst`` and, for each IES iteration, the par and obs
        ensembles (``pest.<i>.par.jcb`` / ``pest.<i>.obs.jcb``) plus the
        ``obs+noise`` ensemble.
      * Thin the obs ensembles to the same curated obs set used for the prior
        MC (reuse ``resolve_curated_obs``), and keep the par ensembles whole
        (they are small relative to the obs ensembles and part1_08 needs the
        posterior parameter fields for the DSIVC sweep).
      * Write per-iteration thinned ensembles into ``prebaked/full_model_ies/``
        with a provenance sidecar (source dir, git SHA, noptmax, n iterations).
      * Decide which iteration is "the posterior" and record it in provenance
        so part1_06/part1_08 do not have to guess.
    """
    raise NotImplementedError(
        "full-model IES thinning (master_hm) is not implemented yet -- see the "
        "TODO in thin_full_model_ies(). Implement once part1_06 fixes the "
        "curated obs set."
    )


def build_dsivc_training_sweep(source_dir, out_dir, obs_csv=None):
    """STUB: build the DSIVC training sweep for part1_08.

    TODO (maintainer):
      * Take the posterior parameter fields from the thinned full-model IES
        (``thin_full_model_ies`` output) -- these are the fields the decision is
        made against.
      * Re-sample the two decision variables (supply-well rate multiplier and
        switch-on day) across their bounds, independently of the prior, so the
        decision space is *covered by design* (see CONTEXT.md 'decision
        problem'): roughly ~200 (posterior field x dvar) combinations.
      * Run the emulator (NOT the full model) over that design to produce the
        training table DSIVC consumes, and write it to ``prebaked/dsivc_sweep/``
        with a provenance sidecar (n runs, dvar bounds, source posterior).
    """
    raise NotImplementedError(
        "DSIVC training sweep is not implemented yet -- see the TODO in "
        "build_dsivc_training_sweep(). Requires the full-model IES posterior "
        "fields and pyemu feat_dsivc."
    )


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="artifact", required=True)

    p = sub.add_parser("prior-mc", help="thin the prior MC obs ensemble (implemented)")
    p.add_argument("--source", default=str(REPO_ROOT / "master_priormc"),
                   help="legacy master dir holding pest.pst + ies output (default: master_priormc)")
    p.add_argument("--out", default=str(REPO_ROOT / "prebaked"),
                   help="output dir for prebaked artifacts (default: prebaked/)")
    p.add_argument("--obs-csv", default=None,
                   help="explicit curated obs list (CSV with 'obsnme' col, or one name per line)")

    p = sub.add_parser("full-model-ies", help="thin master_hm IES results (STUB)")
    p.add_argument("--source", default=str(REPO_ROOT / "master_hm"))
    p.add_argument("--out", default=str(REPO_ROOT / "prebaked" / "full_model_ies"))
    p.add_argument("--obs-csv", default=None)

    p = sub.add_parser("dsivc-sweep", help="build the DSIVC training sweep (STUB)")
    p.add_argument("--source", default=str(REPO_ROOT / "master_hm"))
    p.add_argument("--out", default=str(REPO_ROOT / "prebaked" / "dsivc_sweep"))
    p.add_argument("--obs-csv", default=None)

    args = parser.parse_args(argv)

    if args.artifact == "prior-mc":
        thin_prior_mc(args.source, args.out, args.obs_csv)
    elif args.artifact == "full-model-ies":
        thin_full_model_ies(args.source, args.out, args.obs_csv)
    elif args.artifact == "dsivc-sweep":
        build_dsivc_training_sweep(args.source, args.out, args.obs_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
