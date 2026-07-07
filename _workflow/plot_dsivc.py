"""DSIVC optimization result figures (stage 7).

Two reads of the same 1-D f_treat sweep, straight from the LIVE pestpp-mou output in the master dir
(safe to run while the MOU is still going -- the per-generation populations + the running archive both
grow in place):

* the cost-vs-recovered-SO4 Pareto FRONT (the trade-off a manager chooses along); and
* the RISK BAND the parameter uncertainty puts on the recovered-SO4 forecast at each treatment level
  -- the ADR-0003 payoff (the decision is 1-D and easy; the uncertainty on it is the story).

Every evaluated candidate is one nested-conditioned DSI posterior summarized to stack-stats, so
``fore_peak_so4_stat:5%/mean/95%`` per member ARE that posterior's percentiles.

    python _workflow/plot_dsivc.py [master_dir]      # default: the DSIVC template/master
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from wf_style import apply_style, savefig, C, ROLE, LBL

STAGE = "07_dsivc"
P5, PMEAN, P95 = "fore_peak_so4_stat:5%", "fore_peak_so4_stat:mean", "fore_peak_so4_stat:95%"
_KEEP = ["cost", P5, PMEAN, P95]


def _join(dv_file, ob_file):
    """f_treat (dv_pop) + cost/percentile stack-stats (obs_pop), joined on real_name."""
    dv = pd.read_csv(dv_file).set_index("real_name")
    ob = pd.read_csv(ob_file).set_index("real_name")
    keep = [c for c in _KEEP if c in ob.columns]
    return dv[["f_treat"]].join(ob[keep])


def load_population(md):
    """Every EVALUATED candidate across all generations (the dense f_treat cloud), from the per-gen
    full populations dsivc.<g>.{dv,obs}_pop.csv."""
    md = Path(md)
    frames = []
    for dvf in sorted(md.glob("dsivc.*.dv_pop.csv")):
        if "archive" in dvf.name:
            continue
        g = dvf.name.split(".")[1]
        obf = md / f"dsivc.{g}.obs_pop.csv"
        if obf.exists():
            frames.append(_join(dvf, obf).assign(gen=int(g)))
    if not frames:
        return None
    return pd.concat(frames).dropna(subset=_KEEP).sort_values("f_treat").reset_index(drop=True)


def load_archive(md):
    """The running non-dominated archive = the Pareto front."""
    md = Path(md)
    dvf, obf = md / "dsivc.archive.dv_pop.csv", md / "dsivc.archive.obs_pop.csv"
    if not (dvf.exists() and obf.exists()):
        return None
    return _join(dvf, obf).dropna(subset=[c for c in _KEEP if c]).sort_values("cost").reset_index(drop=True)


def _truth_peak(md):
    for p in (Path(md).parent / "_truth" / "truth_meta.txt", Path(__file__).parent / "_truth" / "truth_meta.txt"):
        if p.exists():
            for line in p.read_text().splitlines():
                if "peak" in line and "=" in line:
                    return float(line.split("=")[1])
    return None


def fig_dsivc_front(master_dir):
    apply_style()
    pop = load_population(master_dir)
    arc = load_archive(master_dir)
    if pop is None or pop.empty:
        raise FileNotFoundError(f"no dsivc.<g>.dv_pop.csv / obs_pop.csv in {master_dir}")
    tpk = _truth_peak(master_dir)

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 5.2))

    # (a) the cost-vs-SO4 trade-off: every evaluated candidate (cloud, coloured by f_treat) + the
    #     non-dominated archive as the front line.
    sc = a0.scatter(pop["cost"], pop[P95], c=pop["f_treat"], cmap="viridis", s=32,
                    edgecolor="none", alpha=0.7, zorder=4)
    if arc is not None and not arc.empty:
        a0.plot(arc["cost"], arc[P95], color=C["black"], lw=1.8, marker="o", ms=5,
                zorder=6, label=f"Pareto front (n={len(arc)})")
        a0.legend(fontsize=9, loc="upper right")
    cb = fig.colorbar(sc, ax=a0)
    cb.set_label("$f_{treat}$ (treatment fraction)")
    a0.set_xlabel(LBL["cost"])
    a0.set_ylabel("P95 peak recovered SO$_4$ (mg/L)")
    a0.set_title(f"Cost vs SO$_4$ trade-off ({len(pop)} evaluated)", fontsize=12)

    # (b) the 1-D parametric view: recovered-SO4 vs f_treat with the parameter-uncertainty risk band,
    #     plus the (deterministic) cost on a twin axis.
    s = pop.sort_values("f_treat")
    a1.fill_between(s["f_treat"], s[P5], s[P95], color=ROLE["posterior"], alpha=0.22,
                    label="P5–P95 risk band")
    a1.plot(s["f_treat"], s[PMEAN], color=ROLE["posterior"], lw=2, label="posterior mean")
    if tpk is not None:
        a1.axhline(tpk, color=ROLE["truth"], ls="--", lw=1.4, label=f"truth ({tpk:.0f})")
    a1.set_xlabel(LBL["ftreat"])
    a1.set_ylabel("peak recovered SO$_4$ (mg/L)")
    a1.set_title("Recovered-SO$_4$ risk band vs treatment", fontsize=12)
    a2 = a1.twinx()
    a2.grid(False)
    a2.plot(s["f_treat"], s["cost"], color=ROLE["forecast"], lw=2, ls=":", label="cost")
    a2.set_ylabel(LBL["cost"], color=ROLE["forecast"])
    a2.tick_params(axis="y", colors=ROLE["forecast"])
    h1, l1 = a1.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    a1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")

    fig.suptitle("DSIVC optimization — cost vs recovered-SO$_4$ with the parameter-uncertainty risk band",
                 fontweight="bold")
    return savefig(fig, "dsivc_front", STAGE)


if __name__ == "__main__":
    md = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "_s7_dsivc_master")
    if not (Path(md) / "dsivc.archive.obs_pop.csv").exists():
        md = str(Path(__file__).parent / "_s7_dsivc_template")   # serial test outputs land in the template
    print(f"[plot_dsivc] reading MOU output from {md}")
    print(f"[plot_dsivc] wrote {fig_dsivc_front(md)}")
