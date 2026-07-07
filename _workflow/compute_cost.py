"""Exact treatment-cost model command for the DSIVC outer forward run (Section 7 / part1_08).

DSIVC wraps pestpp-mou around the DSI emulator; f_treat is the outer decision variable. The cost
objective is DETERMINISTIC in f_treat (cost = c_unit * V_inj * -ln(1 - f_treat)), so it must be
computed EXACTLY here, NOT pushed through the DSI/Gaussian conditioning (which corrupts a
decvar-deterministic quantity by tens of percent -- the hard-won lesson from the earlier design).

Wired as a SECOND model command in the DSIVC template (after the nested `pestpp-ies dsi.pst /e`):
reads the outer decvar value that pestpp-mou wrote, computes cost, and writes it as the `cost` obs.
V_inj is fixed (continuous constant operation), materialized to `v_inj.dat` at DSIVC prepare time so
this command needs no model load.

    python compute_cost.py        # cwd = the DSIVC run dir

DRAFT: the exact on-disk name/shape of the decvar file that pestpp-mou writes for the outer problem
(here assumed `dsivc_pars.csv`, one row per decvar) must be confirmed against the first
DSIVC.prepare_pestpp output; adjust `_read_f_treat` if it differs.
"""
from pathlib import Path
import numpy as np
import pandas as pd

C_UNIT = 2.75e-4                       # keep in sync with workflow.C_UNIT (retune post-rebake)
DECVAR_FILE = "dsivc_pars.csv"        # outer decvar file written by the mou template (confirm name)
VINJ_FILE = "v_inj.dat"               # single float, written at DSIVC prepare time
COST_OBS_FILE = "cost_obs.csv"        # one-row obs CSV read by the cost instruction file


def _read_f_treat(ws):
    """Read the injected f_treat outer decvar. Handles the two plausible csv_tpl_from_parnames
    layouts (name-indexed single column, or a one-row wide frame)."""
    df = pd.read_csv(Path(ws) / DECVAR_FILE)
    cols = {c.lower(): c for c in df.columns}
    if "f_treat" in cols:                              # wide: a column named f_treat
        return float(df[cols["f_treat"]].iloc[0])
    df = df.set_index(df.columns[0])                   # long: first col = parname, second = value
    idx = {str(i).lower(): i for i in df.index}
    return float(df.loc[idx["f_treat"]].iloc[0])


def main(ws="."):
    ws = Path(ws)
    f = _read_f_treat(ws)
    if not (0.0 <= f < 1.0):
        raise ValueError(f"f_treat out of range [0,1): {f}")
    v_inj = float(np.loadtxt(ws / VINJ_FILE))
    cost = C_UNIT * v_inj * (-np.log(1.0 - f))         # convex; -> inf as f_treat -> 1
    pd.DataFrame({"item": ["cost"], "value": [cost]}).to_csv(ws / COST_OBS_FILE, index=False)
    print(f"  [compute_cost] f_treat={f:.4f}  V_inj={v_inj:.3e}  cost={cost:.6g}")
    return cost


if __name__ == "__main__":
    main()
