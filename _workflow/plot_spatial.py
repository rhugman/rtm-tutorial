"""Spatial SO4 diagnostics for the two-well ASR build (stage 02).

Reads the built + run model in ``_workflow/_s2_model`` (``S.ucn`` sulfur, ``gwf.hds`` heads)
and draws, at several times:

* PLAN view  -- SO4 (mg/L) filled on one layer, head contour lines over it, well locations.
* XSECTION view -- SO4 (mg/L) along the well axis (y approx 0), head contour lines, and the
  well SCREENS drawn as thick bars over their screened layers.

SO4 mg/L = S(mol/m3) * 96.06. Figures go to the gitignored _figs/02_build/ via wf_style.savefig.

    conda run -n rtm_gmdsi python _workflow/plot_spatial.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import flopy
from flopy.utils.gridintersect import GridIntersect

from wf_style import apply_style, savefig, C, LBL

WS = Path(__file__).parent / "_s2_model"
STAGE = "02_build"

SO4_GMOL = 96.06
LAYERS_IN = [1, 2, 3, 5, 7]     # wellin screens (0-based)
LAYERS_OUT = [1, 3, 5]          # wellout screens (0-based)
WELLS = {"wellin": (1.0, 0.0), "wellout": (-99.0, 0.0)}
SCREENS = {"wellin": LAYERS_IN, "wellout": LAYERS_OUT}

# times (days) to snapshot -- set adaptively from the run in main() (end-conditioning + spread
# to the end of the extended window). Placeholder defaults here.
TIMES = [252.0, 700.0, 1100.0, 1458.0]
PLAN_LAYER = 3                  # 0-based; a main wellin screen
CMAP = "viridis"
MASK = [1e30, -1e30, -999.0, 1e-30]


def _load(ws=WS):
    ws = Path(ws)
    sim = flopy.mf6.MFSimulation.load(sim_ws=str(ws), verbosity_level=0)
    gwf = sim.get_model("gwf")
    hds = gwf.output.head()
    ucn = flopy.utils.HeadFile(str(ws / "S.ucn"), text="CONCENTRATION")
    return gwf, hds, ucn


def _so4(ucn, t):
    """SO4 (mg/L) full array (nlay, ncpl) at time t, masked."""
    a = ucn.get_data(totim=t).astype(float)
    a = np.ma.masked_where((a > 1e29) | (a < -1e29), a)
    return a * SO4_GMOL


def _head(hds, t):
    h = hds.get_data(totim=t).astype(float)
    return np.ma.masked_where((h > 1e29) | (h < -1e29), h)


def _so4_levels(ucn):
    """Shared vmin/vmax + contour levels across all times (last time sets the max)."""
    vmax = 0.0
    for t in TIMES:
        vmax = max(vmax, float(np.ma.max(_so4(ucn, t))))
    vmax = np.ceil(vmax / 5.0) * 5.0
    return 0.0, vmax


def _head_levels(hds):
    lo, hi = 1e30, -1e30
    for t in TIMES:
        h = _head(hds, t)
        lo, hi = min(lo, float(np.ma.min(h))), max(hi, float(np.ma.max(h)))
    span = max(abs(lo), abs(hi))
    return np.linspace(-span, span, 11)


def plot_plan(gwf, hds, ucn, layer=PLAN_LAYER, name="spatial_plan_so4", stage=STAGE,
              title_tag=""):
    apply_style()
    mg = gwf.modelgrid
    vmin, vmax = _so4_levels(ucn)
    hlev = _head_levels(hds)

    fig, axes = plt.subplots(1, len(TIMES), figsize=(4.4 * len(TIMES), 4.6), sharey=True)
    im = None
    for ax, t in zip(axes, TIMES):
        pmv = flopy.plot.PlotMapView(model=gwf, ax=ax, layer=layer)
        im = pmv.plot_array(_so4(ucn, t), cmap=CMAP, vmin=vmin, vmax=vmax)
        cs = pmv.contour_array(_head(hds, t), levels=hlev,
                               colors="k", linewidths=0.7, alpha=0.8)
        ax.clabel(cs, fmt="%.0f", fontsize=7, inline=True)
        pmv.plot_grid(lw=0.15, color="0.7", alpha=0.4)
        for nm, (x, y) in WELLS.items():
            ax.scatter([x], [y], s=90, marker="v" if nm == "wellin" else "^",
                       facecolor=C["red"] if nm == "wellin" else C["blue"],
                       edgecolor="k", zorder=6)
            ax.annotate(nm, (x, y), textcoords="offset points", xytext=(4, 6),
                        fontsize=8, fontweight="bold", zorder=6)
        ax.set_title(f"t = {t:.0f} d")
        ax.set_xlabel("x (m)")
        ax.set_aspect("equal")
    axes[0].set_ylabel("y (m)")
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.01)
    cb.set_label(LBL["so4"])
    fig.suptitle(f"{title_tag}SO$_4$ plan view -- layer {layer+1} of {mg.nlay} "
                 f"(head contours in m; $\\triangledown$ inject, $\\triangle$ recover)",
                 fontweight="bold")
    return savefig(fig, name, stage)


def plot_xsection(gwf, hds, ucn, name="spatial_xsection_so4", stage=STAGE, title_tag=""):
    apply_style()
    mg = gwf.modelgrid
    vmin, vmax = _so4_levels(ucn)
    hlev = _head_levels(hds)

    # section line along the well axis (cells sit at y>=~0.5; run just inside)
    yc = mg.ycellcenters
    ysec = float(np.min(yc))
    x0 = float(np.min(mg.xcellcenters)) - 2.0
    x1 = float(np.max(mg.xcellcenters)) + 2.0
    line = {"line": [(x0, ysec), (x1, ysec)]}

    # well cells (for screen z) + plot-x (distance along the horizontal line = x - x0)
    ix = GridIntersect(mg)
    wcell = {nm: int(ix.intersect([(x, y)], "point").cellids[0])
             for nm, (x, y) in WELLS.items()}
    botm = np.asarray(mg.botm)
    top = np.asarray(mg.top)

    def screen_segments(nm):
        cell = wcell[nm]
        px = WELLS[nm][0] - x0
        segs = []
        for L in SCREENS[nm]:
            ztop = float(top[cell]) if L == 0 else float(botm[L - 1, cell])
            zbot = float(botm[L, cell])
            segs.append((px, zbot, ztop))
        return px, segs

    fig, axes = plt.subplots(len(TIMES), 1, figsize=(11, 3.0 * len(TIMES)), sharex=True)
    im = None
    for ax, t in zip(axes, TIMES):
        pxs = flopy.plot.PlotCrossSection(model=gwf, ax=ax, line=line, geographic_coords=False)
        im = pxs.plot_array(_so4(ucn, t), cmap=CMAP, vmin=vmin, vmax=vmax, masked_values=MASK)
        cs = pxs.contour_array(_head(hds, t), levels=hlev, colors="k",
                               linewidths=0.7, alpha=0.85)
        ax.clabel(cs, fmt="%.0f", fontsize=7, inline=True)
        pxs.plot_grid(lw=0.15, color="0.7", alpha=0.35)
        for nm in WELLS:
            px, segs = screen_segments(nm)
            col = C["red"] if nm == "wellin" else C["blue"]
            zmin = min(s[1] for s in segs)
            zmax = max(s[2] for s in segs)
            ax.plot([px, px], [zmin, zmax], color=col, lw=1.0, alpha=0.6, zorder=6)  # casing
            for (pxx, zb, zt) in segs:
                ax.plot([pxx, pxx], [zb, zt], color=col, lw=6, solid_capstyle="butt",
                        zorder=7)  # screen
            ax.annotate(nm, (px, zmax), textcoords="offset points", xytext=(0, 5),
                        ha="center", fontsize=8, fontweight="bold", color=col, zorder=8)
        ax.set_ylabel("elevation (m)")
        ax.set_title(f"t = {t:.0f} d", loc="left", fontsize=11)
    axes[-1].set_xlabel(f"distance along well axis (m); x0 = {x0:.0f} m")
    cb = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.01)
    cb.set_label(LBL["so4"])
    handles = [Line2D([0], [0], color=C["red"], lw=6, label="wellin screens"),
               Line2D([0], [0], color=C["blue"], lw=6, label="wellout screens")]
    axes[0].legend(handles=handles, loc="upper right", fontsize=8)
    fig.suptitle(f"{title_tag}SO$_4$ cross-section along the well axis (head contours in m)",
                 fontweight="bold")
    return savefig(fig, name, stage)


def _pick_times(ucn):
    """4 snapshot times snapped to available output: end-of-conditioning + spread to the end."""
    avail = np.array(ucn.get_times())
    tmax = float(avail[-1])
    targets = [252.0, 0.55 * tmax, 0.8 * tmax, tmax]
    picked = sorted({float(avail[np.argmin(np.abs(avail - t))]) for t in targets})
    return picked


def truth_field_from_obs(master, template, real="5"):
    """Reconstruct the SO4 field (per snapshot) for a realization from the prior obs ensemble
    (so4_field group) -- no model re-run. Returns {snap_day: (nlay, ncpl) array}."""
    import pyemu
    master, template = Path(master), Path(template)
    pst = pyemu.Pst(str(template / "pest.pst"))
    pst.try_parse_name_metadata()
    jcb = master / "pest.0.obs.jcb"
    oe = (pyemu.ObservationEnsemble.from_binary(pst=pst, filename=str(jcb)) if jcb.exists()
          else pyemu.ObservationEnsemble.from_csv(pst=pst, filename=str(master / "pest.0.obs.csv")))
    vals = pd.Series(np.asarray(oe.values)[list(map(str, oe.index)).index(str(real))],
                     index=oe.columns)
    fld = pst.observation_data
    fld = fld[fld.obgnme == "so4_field"].copy()
    fld["so4"] = fld["obsnme"].map(vals).astype(float)
    fld["snap"] = fld["snap"].astype(float)
    fld["layer"] = fld["layer"].astype(int)
    fld["cell2d"] = fld["cell2d"].astype(int)
    nlay = fld["layer"].max() + 1
    ncpl = fld["cell2d"].max() + 1
    out = {}
    for snap, d in fld.groupby("snap"):
        arr = np.full((nlay, ncpl), np.nan)
        arr[d["layer"].values, d["cell2d"].values] = d["so4"].values
        out[float(snap)] = arr
    return out


def plot_field_snapshots(model_ws, fields, name, stage, title_tag="", layer=PLAN_LAYER):
    """Plan + xsection grid of a reconstructed SO4 field {snap: (nlay,ncpl)} (from obs ensemble)."""
    apply_style()
    gwf = flopy.mf6.MFSimulation.load(sim_ws=str(model_ws), verbosity_level=0).get_model("gwf")
    mg = gwf.modelgrid
    snaps = sorted(fields)
    vmax = np.ceil(max(np.nanmax(fields[s]) for s in snaps) / 5.0) * 5.0
    yc = mg.ycellcenters
    ysec = float(np.min(yc))
    x0 = float(np.min(mg.xcellcenters)) - 2.0
    x1 = float(np.max(mg.xcellcenters)) + 2.0
    line = {"line": [(x0, ysec), (x1, ysec)]}

    fig, axes = plt.subplots(len(snaps), 2, figsize=(13, 2.9 * len(snaps)),
                             gridspec_kw={"width_ratios": [2, 3]})
    im = None
    for row, s in enumerate(snaps):
        arr = np.ma.masked_invalid(fields[s])
        axp = axes[row, 0]
        pmv = flopy.plot.PlotMapView(model=gwf, ax=axp, layer=layer)
        im = pmv.plot_array(arr, cmap=CMAP, vmin=0, vmax=vmax)
        for nm, (x, y) in {"wellin": (1.0, 0.0), "wellout": (-99.0, 0.0)}.items():
            axp.scatter([x], [y], s=40, marker="v" if nm == "wellin" else "^",
                        facecolor="w", edgecolor="k", zorder=6)
        axp.set_aspect("equal")
        axp.set_ylabel(f"t={s:.0f} d\ny (m)", fontsize=9)
        if row == 0:
            axp.set_title(f"plan (layer {layer + 1})", fontsize=11)
        axx = axes[row, 1]
        pxs = flopy.plot.PlotCrossSection(model=gwf, ax=axx, line=line, geographic_coords=False)
        pxs.plot_array(arr, cmap=CMAP, vmin=0, vmax=vmax, masked_values=MASK)
        for nm, (x, y) in {"wellin": (1.0, 0.0), "wellout": (-99.0, 0.0)}.items():
            axx.axvline(x - x0, color="k", lw=0.7, ls=":", alpha=0.6)
        axx.set_ylabel("elev (m)", fontsize=9)
        if row == 0:
            axx.set_title("cross-section (well axis)", fontsize=11)
    cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.01)
    cb.set_label(LBL["so4"])
    fig.suptitle(f"{title_tag}SO$_4$ field snapshots (reconstructed from the obs ensemble)",
                 fontweight="bold")
    return savefig(fig, name, stage)


def main(ws=WS, prefix="spatial", stage=STAGE, title_tag=""):
    global TIMES
    gwf, hds, ucn = _load(ws)
    TIMES = _pick_times(ucn)
    print(f"[plot_spatial] {ws.name} snapshot times (d): {[round(t) for t in TIMES]}")
    p1 = plot_plan(gwf, hds, ucn, name=f"{prefix}_plan_so4", stage=stage, title_tag=title_tag)
    p2 = plot_xsection(gwf, hds, ucn, name=f"{prefix}_xsection_so4", stage=stage,
                       title_tag=title_tag)
    print(f"[plot_spatial] wrote:\n  {p1}\n  {p2}")


if __name__ == "__main__":
    if "--truth" in sys.argv:
        main(ws=Path(__file__).parent / "_truth_props", prefix="truth_spatial",
             stage="04_obs_weights", title_tag="TRUTH ")
    else:
        main()
