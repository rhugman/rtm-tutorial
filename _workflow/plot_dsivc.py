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
    a1.set_xlabel(LBL["ftreat"])
    a1.set_ylabel("peak recovered SO$_4$ (mg/L)")
    a1.legend(fontsize=8, loc="upper right")
    a1.set_title("Peak recovered SO$_4$ vs $f_{treat}$ (full model)", fontsize=12)

    fig.suptitle("Full-model sweep — treatment cost and recovered-SO$_4$ leverage vs $f_{treat}$",
                 fontweight="bold")
    return savefig(fig, "sweep_tradeoff", STAGE)


def _draw_overlay(fig, a0, a1, cloud, arc, tpk, cloud_label="full-model sweep",
                  new_pts=None, ylim=None, xlim_cost=None):
    """Shared 2-panel overlay used by both the static sweep-vs-front figure and the GIF frames:
    the FOM ground-truth cloud (cols f_treat/cost/peak_so4) + the DSIVC Pareto (arc) on ONE f_treat
    colorscale. new_pts (cols f_treat/cost/peak_so4) are highlighted as red stars -- the FOM samples
    just added this iteration. Pass fixed ylim / xlim_cost to keep axes steady across an animation."""
    has_arc = arc is not None and len(arc)
    if has_arc:
        arc = arc.sort_values("f_treat").reset_index(drop=True)
        af = arc.sort_values("cost")

    # (a) cost vs SO4 -- the FOM cloud always; the f_treat-graded Pareto (line + markers + min-max
    #     whiskers) only when a front is supplied (arc=None -> the starting-point sweep alone)
    sc = a0.scatter(cloud["cost"], cloud["peak_so4"], c=cloud["f_treat"], cmap=FT_CMAP, norm=FT_NORM,
                    s=26, edgecolor="none", alpha=0.22, zorder=3)   # faded so the Pareto reads on top
    cb = fig.colorbar(sc, ax=a0)
    cb.set_label("$f_{treat}$ (cloud + Pareto)" if has_arc else "$f_{treat}$ (FOM sweep)")
    hh = []
    if has_arc:
        a0.errorbar(af["cost"], af[P95], yerr=[(af[P95] - af[PMIN]).clip(lower=0),
                    (af[PMAX] - af[P95]).clip(lower=0)], fmt="none", ecolor="0.25",   # stronger conf bars
                    elinewidth=1.6, capsize=3, capthick=1.4, alpha=0.9, zorder=5)
        _grad_line(a0, af["cost"].values, af[P95].values, af["f_treat"].values, lw=1.8, zorder=6)
        a0.scatter(af["cost"], af[P95], c=af["f_treat"], cmap=FT_CMAP, norm=FT_NORM, s=46,
                   edgecolor="k", linewidth=0.4, zorder=7)
        hh.append(Line2D([], [], color=plt.get_cmap(FT_CMAP)(0.5), marker="o", ls="-", lw=1.8, mec="k",
                         mew=0.4, label=f"DSIVC Pareto (P95 + min–max, n={len(af)})"))
    if new_pts is not None and len(new_pts):
        a0.scatter(new_pts["cost"], new_pts["peak_so4"], marker="*", s=130, c="red",
                   edgecolor="k", linewidth=0.5, zorder=9)
        hh.append(Line2D([], [], color="red", marker="*", ls="none", mec="k", mew=0.5,
                         label=f"new FOM samples (n={len(new_pts)})"))
    a0.set_xlabel(LBL["cost"])
    a0.set_ylabel("peak recovered SO$_4$ (mg/L)")
    if hh:
        a0.legend(handles=hh, fontsize=8, loc="upper right")
    a0.set_title("Cost vs SO$_4$: FOM cloud + DSIVC front" if has_arc
                 else "Cost vs SO$_4$: initial FOM training sweep", fontsize=12)

    # (b) SO4 vs f_treat -- neutral cloud always; f_treat-coloured Pareto (stack min-max) only if present
    a1.scatter(cloud["f_treat"], cloud["peak_so4"], s=26, color="0.72", edgecolor="none",
               alpha=0.32, zorder=3)   # faded to match panel (a); the DSIVC stack reads on top
    hh1 = [Line2D([], [], color="0.7", marker="o", ls="none", label=cloud_label)]
    if has_arc:
        a1.errorbar(arc["f_treat"], arc[PMEAN], yerr=[(arc[PMEAN] - arc[PMIN]).clip(lower=0),
                    (arc[PMAX] - arc[PMEAN]).clip(lower=0)], fmt="none", ecolor="0.25",   # stronger conf bars
                    elinewidth=1.6, capsize=3, capthick=1.4, alpha=0.9, zorder=4)
        a1.scatter(arc["f_treat"], arc[PMEAN], c=arc["f_treat"], cmap=FT_CMAP, norm=FT_NORM, s=44,
                   edgecolor="k", linewidth=0.4, zorder=5)
        hh1.append(Line2D([], [], color=plt.get_cmap(FT_CMAP)(0.5), marker="o", ls="none", mec="k",
                          mew=0.4, label="DSIVC stack (mean, min–max)"))
    if new_pts is not None and len(new_pts):
        a1.scatter(new_pts["f_treat"], new_pts["peak_so4"], marker="*", s=130, c="red",
                   edgecolor="k", linewidth=0.5, zorder=9)
        hh1.append(Line2D([], [], color="red", marker="*", ls="none", mec="k", mew=0.5,
                          label=f"new FOM samples (n={len(new_pts)})"))
    a1.set_xlabel(LBL["ftreat"])
    a1.set_ylabel("peak recovered SO$_4$ (mg/L)")
    a1.legend(handles=hh1, fontsize=8, loc="upper right")
    a1.set_title("SO$_4$ vs $f_{treat}$: cloud vs DSIVC stack range" if has_arc
                 else "SO$_4$ vs $f_{treat}$: initial FOM training sweep", fontsize=12)
    if ylim is not None:
        a0.set_ylim(*ylim)
        a1.set_ylim(*ylim)
    if xlim_cost is not None:
        a0.set_xlim(*xlim_cost)


def _check_stack(arc, where):
    if arc is None or arc.empty:
        raise FileNotFoundError(f"no Pareto front in {where}")
    if PMIN not in arc.columns or PMAX not in arc.columns:
        raise KeyError(f"stack min/max not tracked in {where} (need {PMIN}/{PMAX})")


def fig_sweep_vs_dsivc(sweep_master, sweep_template, dsivc_master):
    """The full-model sweep (ground truth) with the DSIVC emulator Pareto front overlaid, and the DSI
    posterior STACK RANGE (min-max recovered-SO4) drawn as a vertical bar at each Pareto-optimal point."""
    apply_style()
    s = load_sweep(sweep_master, sweep_template)
    arc = load_archive(dsivc_master)
    _check_stack(arc, dsivc_master)
    tpk = _truth_peak(dsivc_master) or _truth_peak(sweep_master)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 5.2))
    _draw_overlay(fig, a0, a1, s, arc, tpk)
    fig.suptitle("DSIVC emulator front vs the full-model sweep (ground truth)", fontweight="bold")
    return savefig(fig, "sweep_vs_dsivc", STAGE)


# ----- outer-loop movement frames (per iteration / per generation) --------------------------------

def _nondominated(df, xcol, ycol):
    """Non-dominated (minimize both) rows: sort by x asc, keep rows whose y strictly improves."""
    d = df.sort_values([xcol, ycol]).reset_index(drop=True)
    keep, best = [], np.inf
    for i in range(len(d)):
        y = float(d.iloc[i][ycol])
        if y < best - 1e-9:
            keep.append(i)
            best = y
    return d.iloc[keep].reset_index(drop=True)


def _infer_k_cost(arc):
    """cost = k*[-ln(1-f_treat)] is deterministic; recover k from the archive's (cost, f_treat)."""
    m = arc["f_treat"] < 0.98
    return float(np.median(arc.loc[m, "cost"] / np.maximum(-np.log(1.0 - arc.loc[m, "f_treat"]), 1e-9)))


def load_loop_cloud(iter_dir, k_cost):
    """The accumulated FOM ground-truth cloud saved per iteration (f_treat + peak recovered-SO4), index
    preserved (s* = base sweep, i{k}r* = iteration-k infill); cost is reconstructed from f_treat
    (deterministic) so panel (a) can place it."""
    df = pd.read_csv(Path(iter_dir) / "train_fom_cloud.csv", index_col=0)
    cl = df.rename(columns={"fore_peak_so4": "peak_so4"})[["f_treat", "peak_so4"]].copy()
    cl["cost"] = k_cost * -np.log(np.clip(1.0 - cl["f_treat"], 1e-9, None))
    return cl


def loop_front(iter_dir, gen=None):
    """The DSIVC Pareto front for one loop iteration: the final archive (gen=None) or the cumulative
    non-dominated set of all populations up to generation `gen`."""
    md = Path(iter_dir) / "master"
    if gen is None:
        return load_archive(str(md)), "final"
    pop = load_population(str(md))
    if pop is None:
        return None, f"gen{gen}"
    return _nondominated(pop[pop["gen"] <= gen], "cost", P95), f"gen{gen}"


def fig_loop_frame(iter_dir, gen=None, frame_idx=None, phase="front", backdrop=None, new_pts=None,
                   ylim=None, xlim_cost=None):
    """One movement-GIF frame. Two phases:
      * phase='front'   -- the DSIVC Pareto front at generation `gen` (cumulative non-dominated up to
        `gen`; None = final archive) against the FOM cloud it was TRAINED on;
      * phase='samples' -- the same final front + the NEW FOM sample points this iteration added,
        highlighted as red stars (the state just before the next DSIVC retrain).
    `backdrop`/`new_pts` (dfs with f_treat/cost/peak_so4) may be supplied by render_loop_frames (so an
    in-progress iteration's front can be drawn against the previous iteration's cloud); if omitted they
    are read from this iteration's own train_fom_cloud.csv. Pass ylim/xlim_cost to freeze axes."""
    apply_style()
    iter_dir = Path(iter_dir)
    it_num = int(iter_dir.name.replace("iter", ""))
    arc, gtag = loop_front(iter_dir, None if phase == "samples" else gen)
    _check_stack(arc, f"{iter_dir} {gtag}")
    if backdrop is None:                                  # self-contained fallback (complete iter only)
        full = load_loop_cloud(iter_dir, _infer_k_cost(arc))
        is_new = full.index.astype(str).str.startswith(f"i{it_num}r")
        backdrop, new_pts = full[~is_new], full[is_new]
    tpk = _truth_peak(str(iter_dir / "master")) or _truth_peak(str(iter_dir.parent))

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 5.2))
    if phase == "samples":
        _draw_overlay(fig, a0, a1, backdrop, arc, tpk, cloud_label=f"FOM cloud (n={len(backdrop)})",
                      new_pts=new_pts, ylim=ylim, xlim_cost=xlim_cost)
        nn = 0 if new_pts is None else len(new_pts)
        fig.suptitle(f"Outer loop — iter {it_num}: {nn} NEW FOM samples added at the Pareto picks "
                     f"→ retrain (train n={len(backdrop) + nn})", fontweight="bold")
        name = f"frame_{frame_idx:03d}" if frame_idx is not None else f"loop_iter{it_num:02d}_samples"
    else:
        _draw_overlay(fig, a0, a1, backdrop, arc, tpk, cloud_label=f"FOM cloud (n={len(backdrop)})",
                      ylim=ylim, xlim_cost=xlim_cost)
        fig.suptitle(f"Outer loop — iter {it_num}, {gtag}: DSIVC front vs FOM training cloud "
                     f"(n={len(backdrop)})", fontweight="bold")
        name = f"frame_{frame_idx:03d}" if frame_idx is not None else f"loop_iter{it_num:02d}_{gtag}"
    return savefig(fig, name, f"{STAGE}/frames")


def render_loop_frames(loop_dir=None, per_gen=True, freeze_axes=True):
    """Render every (iter, generation) frame under loop_dir to _figs/07_dsivc/frames/ in sequence for a
    movement GIF. freeze_axes keeps y (SO4) and x (cost) steady across all frames so motion is the front,
    not the axes. Returns the ordered list of frame paths; prints the ffmpeg/ImageMagick assembly line."""
    base = Path(__file__).parent
    loop_dir = Path(loop_dir or base / "_s7_loop")
    # every iter with a front (has populations); an in-progress iter (no train_fom_cloud.csv yet) still
    # gets its FRONT frames drawn against the cloud it was trained on (the previous iter's accumulation).
    iters = sorted(p for p in loop_dir.glob("iter*")
                   if (p / "master").exists() and list((p / "master").glob("dsivc.*.obs_pop.csv")))
    if not iters:
        raise FileNotFoundError(f"no iterNN/ with a front under {loop_dir}")
    kc = next((_infer_k_cost(a) for a in (loop_front(it, None)[0] for it in iters)
               if a is not None and not a.empty), None)                    # k_cost: cost=k*[-ln(1-f)]
    complete = {it: (it / "train_fom_cloud.csv").exists() for it in iters}
    if not complete[iters[0]]:
        raise FileNotFoundError("iter00 not complete yet -- need its cloud to seed the backdrop")

    def full_cloud(it):                       # accumulated FOM cloud AFTER a completed iteration
        return load_loop_cloud(it, kc)

    def backdrop_for(it):                      # the cloud iteration `it`'s emulator was TRAINED on
        k = int(it.name.replace("iter", ""))
        if k == 0:
            c0 = full_cloud(iters[0])
            return c0[~c0.index.astype(str).str.startswith("i0r")]         # base sweep (strip iter-0 infill)
        prev = loop_dir / f"iter{k - 1:02d}"
        return full_cloud(prev) if complete.get(prev) else None

    # freeze axes over every backdrop + front we will draw
    ylim = xlim = None
    if freeze_axes:
        ymins, ymaxs, cmax = [], [], []
        for it in iters:
            a, _ = loop_front(it, None)
            bd = backdrop_for(it)
            if a is None or a.empty or bd is None:
                continue
            # range from the PHYSICAL cloud + front P95/mean, NOT the DSI stack min/max (extreme Gaussian
            # tails would blow the y-axis out and leave frames mostly empty).
            ymins.append(min(bd["peak_so4"].min(), a[P95].min(), a[PMEAN].min()))
            ymaxs.append(max(bd["peak_so4"].max(), a[P95].max(), a[PMEAN].max()))
            cmax.append(max(a["cost"].max(), bd["cost"].max()))
        if ymins:
            pad = 0.04 * (max(ymaxs) - min(ymins))
            ylim = (min(ymins) - pad, max(ymaxs) + pad)
            xlim = (-0.03 * max(cmax), 1.03 * max(cmax))

    frames, idx = [], 0
    # frame 0: the initial FOM training sweep ALONE (no DSIVC front) -- the starting point
    start_cloud = backdrop_for(iters[0])
    if start_cloud is not None:
        apply_style()
        fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 5.2))
        _draw_overlay(fig, a0, a1, start_cloud, None, None,
                      cloud_label=f"FOM training sweep (n={len(start_cloud)})", ylim=ylim, xlim_cost=xlim)
        fig.suptitle(f"Starting point — initial FOM training sweep (n={len(start_cloud)}), no DSIVC front yet",
                     fontweight="bold")
        frames.append(savefig(fig, f"frame_{idx:03d}", f"{STAGE}/frames"))
        idx += 1

    for it in iters:
        k = int(it.name.replace("iter", ""))
        bd = backdrop_for(it)
        if bd is None:
            print(f"  [frames] skip {it.name}: previous iteration not complete (no training cloud)")
            continue
        gens = sorted(int(p.name.split(".")[1]) for p in (it / "master").glob("dsivc.*.obs_pop.csv")
                      if "archive" not in p.name) if per_gen else [None]
        for g in gens:
            try:
                frames.append(fig_loop_frame(it, gen=g, frame_idx=idx, phase="front", backdrop=bd,
                                             ylim=ylim, xlim_cost=xlim))
                idx += 1
            except (FileNotFoundError, KeyError) as e:
                print(f"  [frames] skip {it.name} gen={g}: {e}")
        # samples frame only once THIS iter is complete (its infill FOM wave has landed)
        if complete[it]:
            full = full_cloud(it)
            new = full[full.index.astype(str).str.startswith(f"i{k}r")]
            try:
                frames.append(fig_loop_frame(it, frame_idx=idx, phase="samples", backdrop=bd,
                                             new_pts=new, ylim=ylim, xlim_cost=xlim))
                idx += 1
            except (FileNotFoundError, KeyError) as e:
                print(f"  [frames] skip {it.name} samples: {e}")
    fdir = frames[0].parent if frames else base / "_figs" / STAGE / "frames"
    if frames:
        try:
            from PIL import Image
            imgs = [Image.open(f).convert("RGB") for f in frames]
            gif = fdir / "loop.gif"
            imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=600, loop=0)
            print(f"[frames] {len(frames)} frames + GIF -> {gif}")
        except ImportError:
            print(f"[frames] {len(frames)} frames in {fdir} (Pillow absent; assemble the GIF manually)")
    return frames


# ----- screen-choice vs SO4 objective figures ----------------------------------------------------

SCREEN_DVS = [f"swin_l{L}" for L in (1, 2, 3, 5, 7)] + [f"swout_l{L}" for L in (1, 3, 5)]


def _screen_label(d):
    """swin_l3 -> 'inj L3', swout_l5 -> 'rec L5'."""
    well, lay = d.split("_l")
    return f"{'inj' if well == 'swin' else 'rec'} L{lay}"


def _map_screen_obs(obsdata):
    """{echo obsnme -> decvar name} for the obgnme='screen' echo obs in a sweep pst."""
    m = {}
    for o in obsdata.index:
        if obsdata.loc[o, "obgnme"] == "screen":
            ol = o.lower()
            hit = next((D for D in SCREEN_DVS if f"item:{D}".lower() in ol or ol.endswith(D.lower())), None)
            if hit:
                m[ol] = hit
    return m


def load_sweep_screens(sweep_master, sweep_template):
    """Per-realization f_treat + peak recovered-SO4 + the 8 screen toggle values, from the sweep."""
    import pyemu
    pst = pyemu.Pst(str(Path(sweep_template) / "pest.pst"))
    pst.try_parse_name_metadata()
    oe = pyemu.ObservationEnsemble.from_binary(pst=pst, filename=str(Path(sweep_master) / "pest.0.obs.jcb"))
    df = pd.DataFrame(oe.values, index=oe.index.astype(str), columns=[c.lower() for c in oe.columns])
    o = pst.observation_data
    fc = [c.lower() for c in o.index[o.obgnme == "forecast"]]
    ft = [c.lower() for c in o.index[o.obgnme == "ftreat"]][0]
    smap = _map_screen_obs(o)
    out = pd.DataFrame({"f_treat": df[ft].values, "peak_so4": df[fc].max(axis=1).values})
    for ol, D in smap.items():
        out[D] = df[ol].values
    return out


def load_archive_screens(dsivc_master):
    """DSIVC archive: f_treat + 8 screen decvars (dv_pop) joined to cost + SO4 stack-stats (obs_pop)."""
    dv = pd.read_csv(Path(dsivc_master) / "dsivc.archive.dv_pop.csv").set_index("real_name")
    ob = pd.read_csv(Path(dsivc_master) / "dsivc.archive.obs_pop.csv").set_index("real_name")
    keep = [c for c in ["cost", PMIN, PMEAN, P95, PMAX] if c in ob.columns]
    return dv.join(ob[keep])


def fig_screen_effect(sweep_master, sweep_template):
    """How each injection/recovery screen choice moves the recovered-SO4 objective, from the sweep
    (screens varied U[0,1], f_treat varied independently so it averages out of the ON/OFF contrast):
    (a) OFF->ON shift in mean peak-SO4 per screen (dumbbell, sorted; blue lowers SO4 = helps the
        minimize objective, red raises it), with the f_treat lever as scale reference;
    (b) the peak-SO4 distribution OFF vs ON per screen (is the shift real vs the spread?)."""
    apply_style()
    s = load_sweep_screens(sweep_master, sweep_template)
    screens = [c for c in SCREEN_DVS if c in s.columns]
    rows = []
    for c in screens:
        on = s.loc[s[c] >= 0.5, "peak_so4"]
        off = s.loc[s[c] < 0.5, "peak_so4"]
        rows.append(dict(screen=c, off=off.mean(), on=on.mean(), delta=on.mean() - off.mean(),
                         on_vals=on.values, off_vals=off.values))
    eff = pd.DataFrame(rows).sort_values("delta").reset_index(drop=True)
    # f_treat reference: same ON/OFF-at-0.5 contrast, for scale
    ft_delta = s.loc[s.f_treat >= 0.5, "peak_so4"].mean() - s.loc[s.f_treat < 0.5, "peak_so4"].mean()

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 5.8))
    y = np.arange(len(eff))
    good, bad = ROLE["emulated"], "crimson"
    for i, r in eff.iterrows():
        col = good if r["delta"] < 0 else bad
        a0.plot([r["off"], r["on"]], [i, i], color=col, lw=2.4, alpha=0.8, zorder=2)
        a0.scatter(r["off"], i, color="0.6", s=55, zorder=3)
        a0.scatter(r["on"], i, color=col, s=70, edgecolor="k", linewidth=0.4, zorder=4)
    a0.axvline(s["peak_so4"].mean(), color="0.5", ls=":", lw=1.1, zorder=1)
    a0.set_yticks(y)
    a0.set_yticklabels([_screen_label(c) for c in eff["screen"]])
    a0.set_xlabel("mean peak recovered SO$_4$ (mg/L)")
    a0.set_title("Screen OFF → ON shift in the SO$_4$ objective", fontsize=12)
    a0.legend(handles=[Line2D([], [], color="0.6", marker="o", ls="none", label="OFF (screen inactive)"),
                       Line2D([], [], color=good, marker="o", ls="none", mec="k", label="ON — lowers SO$_4$ (helps)"),
                       Line2D([], [], color=bad, marker="o", ls="none", mec="k", label="ON — raises SO$_4$"),
                       Line2D([], [], color="0.5", ls=":", label="overall mean")],
              fontsize=8, loc="lower right")

    # (b) OFF vs ON distributions per screen (paired boxes), same screen order
    for i, r in eff.iterrows():
        bp = a1.boxplot([r["off_vals"], r["on_vals"]], positions=[i - 0.18, i + 0.18], widths=0.32,
                        vert=False, patch_artist=True, showfliers=False, manage_ticks=False)
        col = good if r["delta"] < 0 else bad
        for patch, fc in zip(bp["boxes"], ("0.75", col)):
            patch.set_facecolor(fc)
            patch.set_alpha(0.65)
            patch.set_edgecolor("k")
            patch.set_linewidth(0.5)
        for med in bp["medians"]:
            med.set_color("k")
    a1.set_yticks(y)
    a1.set_yticklabels([_screen_label(c) for c in eff["screen"]])
    a1.set_xlabel("peak recovered SO$_4$ (mg/L)")
    a1.set_title("Peak SO$_4$ distribution: OFF (grey) vs ON", fontsize=12)

    fig.suptitle(f"Injection/recovery screen choice vs the recovered-SO$_4$ objective  "
                 f"(sweep n={len(s)}; f_treat ON−OFF ref = {ft_delta:+.1f} mg/L)", fontweight="bold")
    return savefig(fig, "screen_effect", STAGE)


def fig_screen_front(dsivc_master):
    """The optimizer's screen STRATEGY across the Pareto front: a heatmap of each screen's on/off state
    for the archive members sorted by the SO4 objective (P95, low → high), with the P95-SO4 and cost of
    each member on top -- shows which inj/recovery screens are opened/closed to reach each SO4 level."""
    apply_style()
    a = load_archive_screens(dsivc_master)
    screens = [c for c in SCREEN_DVS if c in a.columns]
    if not screens or P95 not in a.columns:
        raise KeyError(f"no screen decvars / P95 in {dsivc_master}")
    a = a.sort_values(P95).reset_index(drop=True)
    active = (a[screens].values.T >= 0.5).astype(float)          # (n_screens, n_members) on/off
    x = np.arange(len(a))

    fig, (top, hm) = plt.subplots(2, 1, figsize=(12, 6.4), height_ratios=[1, 2.2], sharex=True,
                                  gridspec_kw=dict(hspace=0.12))
    top.plot(x, a[P95], color=ROLE["emulated"], lw=2, marker="o", ms=3, label="P95 peak SO$_4$")
    top.set_ylabel("P95 SO$_4$ (mg/L)", color=ROLE["emulated"])
    top.tick_params(axis="y", colors=ROLE["emulated"])
    t2 = top.twinx()
    t2.plot(x, a["cost"], color=ROLE["forecast"], lw=1.6, ls=":", label="cost")
    t2.set_ylabel("cost", color=ROLE["forecast"])
    t2.tick_params(axis="y", colors=ROLE["forecast"])
    t2.grid(False)
    top.set_title("Pareto front — objective per member (P95 SO$_4$) + cost", fontsize=12)

    im = hm.imshow(active, aspect="auto", cmap=mpl.colors.ListedColormap(["white", ROLE["emulated"]]),
                   vmin=0, vmax=1, interpolation="nearest",
                   extent=[-0.5, len(a) - 0.5, len(screens) - 0.5, -0.5])
    hm.set_xticks(np.arange(0.5, len(a) - 0.5), minor=True)      # cell-boundary gridlines for readability
    hm.set_yticks(np.arange(0.5, len(screens) - 0.5), minor=True)
    hm.grid(which="minor", color="0.8", lw=0.5)
    hm.tick_params(which="minor", length=0)
    hm.set_yticks(range(len(screens)))
    hm.set_yticklabels([_screen_label(c) for c in screens])
    hm.axhline(4.5, color="k", lw=1.4)                           # divide injection (top) from recovery (bottom)
    hm.text(-0.06, 0.78, "inject", transform=hm.transAxes, rotation=90, va="center", ha="center", fontsize=9)
    hm.text(-0.06, 0.20, "recover", transform=hm.transAxes, rotation=90, va="center", ha="center", fontsize=9)
    hm.set_xlabel("Pareto member (sorted by P95 SO$_4$: low/expensive → high/cheap)")
    hm.set_title("Screen on/off across the front (green = active, white = off)", fontsize=12)

    fig.suptitle("Injection/recovery screen strategy along the DSIVC Pareto front", fontweight="bold")
    return savefig(fig, "screen_front", STAGE)


def fig_final_validation(vdir, out=None):
    """FINAL VALIDATION figure (post-loop): does the emulated (DSIVC) forecast hold up under the FULL
    model at the Pareto-optimal decvars? Reads val_summary.csv (per-point emu stack stats + full-model
    percentiles), val_fom_dist.csv (the full FOM forecast per point), val_front.csv (emulated backdrop).
      (a) emulated vs full-model P95 (the objective) along the cost front -- the decision-relevant check;
      (b) per optimum: full-model forecast DISTRIBUTION (violin) vs emulated P5-P95 stack (box)."""
    from matplotlib.patches import Rectangle, Patch
    apply_style()
    vdir = Path(vdir)
    summ = pd.read_csv(vdir / "val_summary.csv").set_index("point").sort_values("cost")
    dist = pd.read_csv(vdir / "val_fom_dist.csv")
    front = pd.read_csv(vdir / "val_front.csv") if (vdir / "val_front.csv").exists() else None
    pts = list(summ.index)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 5.4))

    # (a) emulated vs full-model P95 along the cost front
    if front is not None:
        a0.plot(front["cost"], front["emu_p95"], color=ROLE["emulated"], lw=1.4, alpha=0.5, zorder=2,
                label="emulated front (P95)")
    for _, r in summ.iterrows():
        a0.plot([r["cost"], r["cost"]], [r["emu_95%"], r["fom_95%"]], color="0.45", lw=1.0, zorder=4)
    a0.scatter(summ["cost"], summ["emu_95%"], s=60, color=ROLE["emulated"], edgecolor="k", lw=0.6,
               zorder=5, label="emulated P95 @ optimum")
    a0.scatter(summ["cost"], summ["fom_95%"], s=72, marker="D", color=ROLE["posterior"], edgecolor="k",
               lw=0.6, zorder=6, label="full-model P95 (validation)")
    a0.set_xlabel(LBL["cost"])
    a0.set_ylabel("P95 peak SO$_4$ (mg/L)")
    a0.set_title("(a) emulated vs full-model P95 along the front")
    a0.legend(fontsize=8, loc="best")

    # (b) per-point forecast distributions: full-model violin vs emulated stack box
    xs = np.arange(len(pts))
    data = [dist.loc[dist["point"] == p, "peak_so4"].values for p in pts]
    if any(len(d) for d in data):
        parts = a1.violinplot([d for d in data if len(d)],
                              positions=[xs[i] - 0.16 for i, d in enumerate(data) if len(d)],
                              widths=0.28, showextrema=False)
        for b in parts["bodies"]:
            b.set_facecolor(ROLE["posterior"]); b.set_alpha(0.35); b.set_edgecolor(ROLE["posterior"])
    a1.scatter(xs - 0.16, summ["fom_95%"].values, s=26, color=ROLE["posterior"], edgecolor="k",
               lw=0.5, zorder=6)
    for i, p in enumerate(pts):                                  # emulated stack: min-max whisker, P5-P95 box, mean, P95
        r = summ.loc[p]
        x = xs[i] + 0.16
        a1.plot([x, x], [r["emu_min"], r["emu_max"]], color=ROLE["emulated"], lw=1.0, zorder=5)
        a1.add_patch(Rectangle((x - 0.08, r["emu_5%"]), 0.16, r["emu_95%"] - r["emu_5%"],
                               facecolor=ROLE["emulated"], alpha=0.35, edgecolor=ROLE["emulated"], zorder=5))
        a1.plot([x - 0.08, x + 0.08], [r["emu_mean"]] * 2, color=ROLE["emulated"], lw=1.4, zorder=6)
        a1.scatter([x], [r["emu_95%"]], s=26, color=ROLE["emulated"], edgecolor="k", lw=0.5, zorder=7)
    a1.set_xticks(xs)
    a1.set_xticklabels([f"cost\n{summ.loc[p, 'cost']:.2f}" for p in pts], fontsize=8)
    a1.set_ylabel("peak SO$_4$ (mg/L)")
    a1.set_title("(b) full-model forecast (violin) vs emulated stack (box) per optimum")
    a1.legend(handles=[Patch(facecolor=ROLE["posterior"], alpha=0.35, label="full-model dist. (P95 = dot)"),
                       Patch(facecolor=ROLE["emulated"], alpha=0.35, label="emulated P5–P95 (P95 = dot, mean = bar)")],
              fontsize=8, loc="best")
    fig.suptitle("Final validation: DSIVC-optimal decvars run through the full FOM parameter ensemble",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if out is not None:
        fig.savefig(str(out)); plt.close(fig); print(f"  [fig] {out}"); return Path(out)
    return savefig(fig, "final_validation", STAGE)


if __name__ == "__main__":
    base = Path(__file__).parent
    if "--validate" in sys.argv:
        vd = base / "_s7_loop" / "validation"
        print(f"[plot_dsivc] wrote {fig_final_validation(vd)}")
        sys.exit(0)
    if "--screens" in sys.argv:
        print(f"[plot_dsivc] wrote {fig_screen_effect(base / '_s7_sweep_master', base / '_s7_sweep_template')}")
        md = base / "_s7_dsivc_master"
        if (md / "dsivc.archive.dv_pop.csv").exists():
            print(f"[plot_dsivc] wrote {fig_screen_front(md)}")
        else:
            print("[plot_dsivc] no DSIVC archive -> skipped fig_screen_front")
        sys.exit(0)
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
    if "--frames" in sys.argv:
        per_gen = "--per-iter" not in sys.argv          # default: one frame per generation
        render_loop_frames(per_gen=per_gen)
        sys.exit(0)
    md = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "_s7_dsivc_master")
    if not (Path(md) / "dsivc.archive.obs_pop.csv").exists():
        md = str(Path(__file__).parent / "_s7_dsivc_template")   # serial test outputs land in the template
    print(f"[plot_dsivc] reading MOU output from {md}")
    print(f"[plot_dsivc] wrote {fig_dsivc_front(md)}")
