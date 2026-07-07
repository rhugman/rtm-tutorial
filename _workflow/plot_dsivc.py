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
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from wf_style import apply_style, savefig, C, ROLE, LBL

FT_CMAP = "viridis"
FT_NORM = mpl.colors.Normalize(vmin=0.0, vmax=1.0)   # shared f_treat colorscale (sweep cloud + Pareto)


def _grad_line(ax, x, y, c, lw=1.8, zorder=6):
    """A polyline whose segments are coloured by f_treat on the shared FT_CMAP/FT_NORM scale."""
    x, y, c = np.asarray(x, float), np.asarray(y, float), np.asarray(c, float)
    if x.size < 2:
        return None
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=FT_CMAP, norm=FT_NORM, zorder=zorder)
    lc.set_array(0.5 * (c[:-1] + c[1:]))             # segment colour = mean f_treat of its endpoints
    lc.set_linewidth(lw)
    ax.add_collection(lc)
    return lc

STAGE = "07_dsivc"
P5, PMEAN, P95 = "fore_peak_so4_stat:5%", "fore_peak_so4_stat:mean", "fore_peak_so4_stat:95%"
PMIN, PMAX = "fore_peak_so4_stat:min", "fore_peak_so4_stat:max"
_KEEP = ["cost", PMIN, P5, PMEAN, P95, PMAX]


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


def load_sweep(sweep_master, sweep_template):
    """Ground-truth full-model sweep: per-realization f_treat / cost / peak recovered-SO4 from the DSIVC
    sweep obs ensemble (the data the DSI emulator trains on) -- the reference the emulated front must match."""
    import pyemu
    pst = pyemu.Pst(str(Path(sweep_template) / "pest.pst"))
    pst.try_parse_name_metadata()
    oe = pyemu.ObservationEnsemble.from_binary(
        pst=pst, filename=str(Path(sweep_master) / "pest.0.obs.jcb"))
    df = pd.DataFrame(oe.values, index=oe.index.astype(str), columns=[c.lower() for c in oe.columns])
    o = pst.observation_data
    ft = [c.lower() for c in o.index[o.obgnme == "ftreat"]][0]
    ct = [c.lower() for c in o.index[o.obgnme == "cost"]][0]
    fc = [c.lower() for c in o.index[o.obgnme == "forecast"]]
    return pd.DataFrame({"f_treat": df[ft].values, "cost": df[ct].values,
                         "peak_so4": df[fc].max(axis=1).values}).sort_values("f_treat").reset_index(drop=True)


def fig_sweep_tradeoff(sweep_master, sweep_template):
    """Full-model (FOM) sweep -- the GROUND-TRUTH trade-off the DSIVC front must reproduce:
    (a) cost vs peak recovered-SO4 cloud coloured by f_treat (the decision view); and
    (b) peak recovered-SO4 vs f_treat (the real, param-scattered leverage)."""
    apply_style()
    s = load_sweep(sweep_master, sweep_template)
    tpk = _truth_peak(sweep_master)
    ff = np.linspace(0.0, float(s["f_treat"].max()), 200)

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 5.2))

    # (a) the ground-truth trade-off cloud: cost (x, deterministic in f_treat) vs peak recovered-SO4
    #     (y, param-scattered), coloured by the decision lever f_treat. Direct analog of the DSIVC front.
    sc = a0.scatter(s["cost"], s["peak_so4"], c=s["f_treat"], cmap="viridis", s=34,
                    edgecolor="none", alpha=0.8, zorder=4)
    if tpk is not None:
        a0.axhline(tpk, color=ROLE["truth"], ls="--", lw=1.4, label=f"truth ({tpk:.0f})")
        a0.legend(fontsize=9, loc="upper right")
    cb = fig.colorbar(sc, ax=a0)
    cb.set_label("$f_{treat}$ (treatment fraction)")
    a0.set_xlabel(LBL["cost"])
    a0.set_ylabel("peak recovered SO$_4$ (mg/L)")
    a0.set_title(f"Cost vs SO$_4$ trade-off ({len(s)} full-model runs)", fontsize=12)

    # (b) peak recovered-SO4 vs f_treat -- param-driven scatter + linear trend (the leverage)
    a1.scatter(s["f_treat"], s["peak_so4"], s=30, color=ROLE["prior"], edgecolor="none", alpha=0.75, zorder=4)
    z = np.polyfit(s["f_treat"], s["peak_so4"], 1)
    r = s["f_treat"].corr(s["peak_so4"])
    a1.plot(ff, np.polyval(z, ff), color=ROLE["emulated"], lw=2.2, label=f"trend (corr = {r:.2f})")
    if tpk is not None:
        a1.axhline(tpk, color=ROLE["truth"], ls="--", lw=1.4, label=f"truth ({tpk:.0f})")
    a1.set_xlabel(LBL["ftreat"])
    a1.set_ylabel("peak recovered SO$_4$ (mg/L)")
    a1.legend(fontsize=8, loc="upper right")
    a1.set_title("Peak recovered SO$_4$ vs $f_{treat}$ (full model)", fontsize=12)

    fig.suptitle("Full-model sweep — treatment cost and recovered-SO$_4$ leverage vs $f_{treat}$",
                 fontweight="bold")
    return savefig(fig, "sweep_tradeoff", STAGE)


def fig_sweep_vs_dsivc(sweep_master, sweep_template, dsivc_master):
    """The full-model sweep (ground truth) with the DSIVC emulator Pareto front overlaid, and the DSI
    posterior STACK RANGE (min-max recovered-SO4) drawn as a vertical bar at each Pareto-optimal point.
    (a) cost-vs-SO4: sweep cloud + Pareto front (P95) with stack min-max whiskers;
    (b) SO4-vs-f_treat: sweep scatter + Pareto members at their injected f_treat with stack range --
        the emulated posterior spread should track the ground-truth scatter at each treatment level."""
    apply_style()
    s = load_sweep(sweep_master, sweep_template)
    arc = load_archive(dsivc_master)
    if arc is None or arc.empty:
        raise FileNotFoundError(f"no dsivc.archive.*_pop.csv in {dsivc_master}")
    if PMIN not in arc.columns or PMAX not in arc.columns:
        raise KeyError(f"stack min/max not tracked in {dsivc_master} (need {PMIN}/{PMAX})")
    arc = arc.sort_values("f_treat").reset_index(drop=True)
    tpk = _truth_peak(dsivc_master) or _truth_peak(sweep_master)

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 5.2))

    # (a) cost vs SO4: ground-truth sweep cloud (faded) + the DSIVC Pareto front, BOTH on the shared
    #     f_treat colorscale (graded front line + f_treat-coloured markers), neutral min-max whiskers.
    sc = a0.scatter(s["cost"], s["peak_so4"], c=s["f_treat"], cmap=FT_CMAP, norm=FT_NORM, s=28,
                    edgecolor="none", alpha=0.5, zorder=3)
    cb = fig.colorbar(sc, ax=a0)
    cb.set_label("$f_{treat}$ (sweep + Pareto)")
    af = arc.sort_values("cost")
    a0.errorbar(af["cost"], af[P95], yerr=[(af[P95] - af[PMIN]).clip(lower=0),
                (af[PMAX] - af[P95]).clip(lower=0)], fmt="none", ecolor="0.45",
                elinewidth=0.9, capsize=2, alpha=0.6, zorder=5)
    _grad_line(a0, af["cost"].values, af[P95].values, af["f_treat"].values, lw=1.8, zorder=6)
    a0.scatter(af["cost"], af[P95], c=af["f_treat"], cmap=FT_CMAP, norm=FT_NORM, s=46,
               edgecolor="k", linewidth=0.4, zorder=7)
    hh = [Line2D([], [], color=plt.get_cmap(FT_CMAP)(0.5), marker="o", ls="-", lw=1.8, mec="k",
                 mew=0.4, label=f"DSIVC Pareto (P95 + stack min–max, n={len(af)})")]
    if tpk is not None:
        a0.axhline(tpk, color=ROLE["truth"], ls="--", lw=1.4)
        hh.append(Line2D([], [], color=ROLE["truth"], ls="--", lw=1.4, label=f"truth ({tpk:.0f})"))
    a0.set_xlabel(LBL["cost"])
    a0.set_ylabel("peak recovered SO$_4$ (mg/L)")
    a0.legend(handles=hh, fontsize=8, loc="upper right")
    a0.set_title("Cost vs SO$_4$: full-model sweep + DSIVC front", fontsize=12)

    # (b) SO4 vs f_treat: sweep scatter (neutral) + Pareto members at their injected f_treat, f_treat-
    #     coloured on the same scale, whiskered by the DSI stack range (min-max posterior spread).
    a1.scatter(s["f_treat"], s["peak_so4"], s=28, color="0.7", edgecolor="none",
               alpha=0.5, zorder=3)
    a1.errorbar(arc["f_treat"], arc[PMEAN], yerr=[(arc[PMEAN] - arc[PMIN]).clip(lower=0),
                (arc[PMAX] - arc[PMEAN]).clip(lower=0)], fmt="none", ecolor="0.45",
                elinewidth=0.9, capsize=2, alpha=0.7, zorder=4)
    a1.scatter(arc["f_treat"], arc[PMEAN], c=arc["f_treat"], cmap=FT_CMAP, norm=FT_NORM, s=44,
               edgecolor="k", linewidth=0.4, zorder=5)
    hh1 = [Line2D([], [], color="0.7", marker="o", ls="none", label="full-model sweep"),
           Line2D([], [], color=plt.get_cmap(FT_CMAP)(0.5), marker="o", ls="none", mec="k", mew=0.4,
                  label="DSIVC stack (mean, min–max)")]
    if tpk is not None:
        a1.axhline(tpk, color=ROLE["truth"], ls="--", lw=1.4)
        hh1.append(Line2D([], [], color=ROLE["truth"], ls="--", lw=1.4, label=f"truth ({tpk:.0f})"))
    a1.set_xlabel(LBL["ftreat"])
    a1.set_ylabel("peak recovered SO$_4$ (mg/L)")
    a1.legend(handles=hh1, fontsize=8, loc="upper right")
    a1.set_title("SO$_4$ vs $f_{treat}$: sweep scatter vs DSIVC stack range", fontsize=12)

    fig.suptitle("DSIVC emulator front vs the full-model sweep (ground truth)", fontweight="bold")
    return savefig(fig, "sweep_vs_dsivc", STAGE)


if __name__ == "__main__":
    base = Path(__file__).parent
    if "--sweep" in sys.argv:
        print(f"[plot_dsivc] wrote {fig_sweep_tradeoff(base / '_s7_sweep_master', base / '_s7_sweep_template')}")
        sys.exit(0)
    if "--overlay" in sys.argv:
        md = base / "_s7_dsivc_master"
        if not (md / "dsivc.archive.obs_pop.csv").exists():
            md = base / "_s7_dsivc_template"
        print(f"[plot_dsivc] overlay reading MOU output from {md}")
        print(f"[plot_dsivc] wrote {fig_sweep_vs_dsivc(base / '_s7_sweep_master', base / '_s7_sweep_template', md)}")
        sys.exit(0)
    md = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "_s7_dsivc_master")
    if not (Path(md) / "dsivc.archive.obs_pop.csv").exists():
        md = str(Path(__file__).parent / "_s7_dsivc_template")   # serial test outputs land in the template
    print(f"[plot_dsivc] reading MOU output from {md}")
    print(f"[plot_dsivc] wrote {fig_dsivc_front(md)}")
