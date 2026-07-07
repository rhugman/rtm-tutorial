"""Static model-setup diagnostics for the two-well ASR build (stage 02).

Everything here is available BEFORE the reactive run -- it reads the built model structure in
``_workflow/_s2_model`` (grid, boundary conditions, screens, base properties) plus the field
location inputs in ``data/``. Draws:

* PLAN layout + BCs  -- domain, voronoi mesh, CHD boundary, wellin/wellout cells, monitoring
  clusters (the conditioning-data locations), deleted wellopt shown greyed for context.
* XSECTION layout    -- the 12-layer grid along the well axis, well screens, monitoring depths,
  CHD at the lateral boundaries.
* PROPERTIES         -- base fields (K, porosity, pyrite m0). The base build is HOMOGENEOUS
  (heterogeneity is added later by PstFrom pilot points, slice 3), so these read as uniform;
  the figure is the ready template + a scalar summary, honestly labelled.

Figures -> gitignored _figs/02_build/ via wf_style.savefig.

    conda run -n rtm_gmdsi python _workflow/plot_setup.py
"""

import sys
import glob
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import flopy
from flopy.utils.gridintersect import GridIntersect

import workflow as w
from wf_style import apply_style, savefig, C

WS = Path(__file__).parent / "_s2_model"
DATA_D = Path(__file__).parent.parent / "data"
STAGE = "02_build"

WELLS = {"wellin": (1.0, 0.0), "wellout": (-99.0, 0.0)}
WELLOPT = (-40.0, 80.0)                 # deleted (ADR-0003); shown greyed for context
SCREENS = {"wellin": w.LAYERS_IN, "wellout": w.LAYERS_OUT}


def _load():
    sim = flopy.mf6.MFSimulation.load(sim_ws=str(WS), verbosity_level=0)
    return sim, sim.get_model("gwf")


def _monitoring():
    obs = pd.read_csv(DATA_D / "obs_loc.csv")
    obs["well"] = obs["obsid"].str.split("-").str[0]
    return obs


# -----------------------------------------------------------------------------
def fig_plan_layout(gwf):
    apply_style()
    mg = gwf.modelgrid
    fig, ax = plt.subplots(figsize=(11, 6.2))
    pmv = flopy.plot.PlotMapView(model=gwf, ax=ax, layer=0)
    pmv.plot_grid(lw=0.2, color="0.7", alpha=0.5)

    # boundary conditions
    pmv.plot_bc(package=gwf.get_package("chd"), color=C["sky"], alpha=0.35)
    pmv.plot_bc(package=gwf.get_package("welin"), color=C["red"])
    pmv.plot_bc(package=gwf.get_package("welout"), color=C["blue"])

    # wells
    for nm, (x, y) in WELLS.items():
        ax.scatter([x], [y], s=130, marker="v" if nm == "wellin" else "^",
                   facecolor=C["red"] if nm == "wellin" else C["blue"],
                   edgecolor="k", zorder=7)
        ax.annotate(nm, (x, y), textcoords="offset points", xytext=(6, 6),
                    fontsize=9, fontweight="bold", zorder=7)
    ax.scatter(*WELLOPT, s=90, marker="x", color=C["grey"], zorder=6)
    ax.annotate("wellopt (deleted)", WELLOPT, textcoords="offset points", xytext=(6, 4),
                fontsize=8, color=C["grey"], zorder=6)

    # monitoring clusters (conditioning data locations)
    obs = _monitoring()
    for wname, g in obs.groupby("well"):
        x, y = g.x.iloc[0], g.y.iloc[0]
        ax.scatter([x], [y], s=55, marker="s", facecolor="none",
                   edgecolor=C["green"], linewidths=1.6, zorder=8)
        ax.annotate(f"{wname}\n({g.layer.nunique()} depths)", (x, y),
                    textcoords="offset points", xytext=(3, -16), fontsize=7,
                    color=C["green"], zorder=8)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    handles = [
        Patch(fc=C["sky"], alpha=0.35, label="CHD boundary"),
        Patch(fc=C["red"], label="wellin cells"),
        Patch(fc=C["blue"], label="wellout cells"),
        Line2D([0], [0], marker="s", mfc="none", mec=C["green"], ls="",
               label="monitoring cluster"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.set_title("Two-well ASR -- plan layout & boundary conditions "
                 "(layer 1; monitoring = conditioning data)", fontweight="bold")
    # NB: the workflow's _fig_layout owns "setup_plan_layout" (CHD coloured by gradient head);
    # this standalone adds monitoring-cluster labels + the deleted-wellopt marker.
    return savefig(fig, "setup_plan_layout_annotated", STAGE)


# -----------------------------------------------------------------------------
def _section(mg):
    yc = mg.ycellcenters
    ysec = float(np.min(yc))
    x0 = float(np.min(mg.xcellcenters)) - 2.0
    x1 = float(np.max(mg.xcellcenters)) + 2.0
    return {"line": [(x0, ysec), (x1, ysec)]}, ysec, x0


def _screen_segments(mg, ix, nm, x0):
    cell = int(ix.intersect([WELLS[nm]], "point").cellids[0])
    botm = np.asarray(mg.botm)
    top = np.asarray(mg.top)
    px = WELLS[nm][0] - x0
    segs = []
    for L in SCREENS[nm]:
        ztop = float(top[cell]) if L == 0 else float(botm[L - 1, cell])
        zbot = float(botm[L, cell])
        segs.append((px, zbot, ztop))
    return px, segs


def fig_xsection_layout(gwf):
    apply_style()
    mg = gwf.modelgrid
    line, ysec, x0 = _section(mg)
    ix = GridIntersect(mg)
    botm = np.asarray(mg.botm)
    top = np.asarray(mg.top)

    fig, ax = plt.subplots(figsize=(12, 4.6))
    pxs = flopy.plot.PlotCrossSection(model=gwf, ax=ax, line=line, geographic_coords=False)
    pxs.plot_grid(lw=0.25, color="0.6", alpha=0.5)
    pxs.plot_bc(package=gwf.get_package("chd"), color=C["sky"], alpha=0.35)

    # well screens
    for nm in WELLS:
        col = C["red"] if nm == "wellin" else C["blue"]
        px, segs = _screen_segments(mg, ix, nm, x0)
        ax.plot([px, px], [min(s[1] for s in segs), max(s[2] for s in segs)],
                color=col, lw=1.0, alpha=0.6, zorder=6)
        for (pxx, zb, zt) in segs:
            ax.plot([pxx, pxx], [zb, zt], color=col, lw=7, solid_capstyle="butt", zorder=7)
        ax.annotate(nm, (px, max(s[2] for s in segs)), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=9, fontweight="bold",
                    color=col, zorder=8)

    # monitoring points at their screened depths
    obs = _monitoring()
    for _, r in obs.iterrows():
        cell = int(ix.intersect([(r.x, ysec)], "point").cellids[0])
        L = int(r.layer)
        ztop = float(top[cell]) if L == 0 else float(botm[L - 1, cell])
        zc = 0.5 * (ztop + float(botm[L, cell]))
        ax.scatter([r.x - x0], [zc], s=26, marker="s", facecolor="none",
                   edgecolor=C["green"], linewidths=1.2, zorder=9)

    ax.set_xlabel(f"distance along well axis (m); x0 = {x0:.0f} m")
    ax.set_ylabel("elevation (m)")
    handles = [
        Line2D([0], [0], color=C["red"], lw=7, label="wellin screens"),
        Line2D([0], [0], color=C["blue"], lw=7, label="wellout screens"),
        Line2D([0], [0], marker="s", mfc="none", mec=C["green"], ls="",
               label="monitoring point"),
        Patch(fc=C["sky"], alpha=0.35, label="CHD boundary"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, ncol=2)
    ax.set_title(f"Two-well ASR -- cross-section layout along the well axis "
                 f"({mg.nlay} layers)", fontweight="bold")
    return savefig(fig, "setup_xsection_layout", STAGE)


# -----------------------------------------------------------------------------
def _pyrite_array(mg):
    files = sorted(glob.glob(str(WS / "kinetic_phases.Pyrite.m0.layer*.txt")),
                   key=lambda p: int("".join(ch for ch in p.split("layer")[1] if ch.isdigit())))
    return np.array([np.loadtxt(f) for f in files])


def fig_properties(gwf):
    apply_style()
    mg = gwf.modelgrid
    line, ysec, x0 = _section(mg)
    npf = gwf.get_package("npf")
    k = float(np.asarray(npf.k.array).flat[0])
    k33 = float(np.asarray(npf.k33.array).flat[0])
    sy = float(np.asarray(gwf.get_package("sto").sy.array).flat[0])
    pyr = _pyrite_array(mg)

    fig, ax = plt.subplots(figsize=(12, 4.6))
    pxs = flopy.plot.PlotCrossSection(model=gwf, ax=ax, line=line, geographic_coords=False)
    im = pxs.plot_array(pyr, cmap="YlOrBr")
    pxs.plot_grid(lw=0.25, color="0.6", alpha=0.4)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.01)
    cb.set_label("pyrite m0 (mol/L pore water)")
    ax.set_xlabel(f"distance along well axis (m); x0 = {x0:.0f} m")
    ax.set_ylabel("elevation (m)")

    txt = ("BASE BUILD -- homogeneous\n"
           f"K  = {k:g} m/d   K33 = {k33:g} m/d\n"
           f"porosity (sy) = {sy:g}\n"
           f"pyrite m0 = {pyr.min():.4f} mol/L (all cells)\n"
           "heterogeneity added at PstFrom\n(pilot points, slice 3)")
    ax.text(0.012, 0.03, txt, transform=ax.transAxes, fontsize=8, va="bottom",
            ha="left", bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))
    ax.set_title("Two-well ASR -- base aquifer properties (homogeneous; template for "
                 "post-PstFrom fields)", fontweight="bold")
    return savefig(fig, "setup_properties", STAGE)


def main():
    sim, gwf = _load()
    p1 = fig_plan_layout(gwf)
    p2 = fig_xsection_layout(gwf)
    p3 = fig_properties(gwf)
    print(f"[plot_setup] wrote:\n  {p1}\n  {p2}\n  {p3}")


if __name__ == "__main__":
    main()
