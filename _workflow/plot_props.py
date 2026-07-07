"""Truth-model aquifer-property fields (plan + cross-section) for the two-well ASR study.

The base build is homogeneous; heterogeneity enters via the PstFrom pilot-point parameters. This
module materializes the SYNTHETIC-TRUTH realization's property fields (K, porosity, dispersivity,
pyrite m0) by applying its parameter vector to the model, then plots each property in plan (one
layer) and in cross-section along the well axis.

    conda run -n rtm_gmdsi python _workflow/plot_props.py

(Sulfate/SO4 is a transported concentration, not a static property -- see plot_spatial.py.)
"""

import os
import sys
import shutil
import warnings
import subprocess
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tutorials"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import flopy
from flopy.utils.gridintersect import GridIntersect

from wf_style import apply_style, savefig, C

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = Path(__file__).parent / "_s3_template"
TRUTH_DIR = Path(__file__).parent / "_truth"
OUT_WS = Path(__file__).parent / "_truth_props"
STAGE = "04_obs_weights"

# (label, per-layer file tag, colormap)
PROPS = [
    ("K (m/d)", "gwf.npf_k_layer", "viridis", True),          # log scale
    ("porosity (-)", "H2O.mst_porosity_layer", "cividis", False),
    ("dispersivity $\\alpha_L$ (m)", "H2O.dsp_alh_layer", "magma", True),
    ("pyrite m0 (mol/L)", "kinetic_phases.Pyrite.m0.layer", "YlOrBr", False),
]
PLAN_LAYER = 3
NLAY = 12


def materialize_truth(out_ws=OUT_WS, template=TEMPLATE, truth_dir=TRUTH_DIR):
    """Apply the truth realization's parameters to a fresh copy of the PEST template so the
    perturbed model-input arrays (K, porosity, dispersivity, pyrite m0) land on disk."""
    import pyemu

    out_ws = Path(out_ws).absolute()
    if out_ws.exists():
        shutil.rmtree(out_ws)
    # shutil.copytree silently drops files on this large dir -> use cp -R
    subprocess.run(["cp", "-R", str(template), str(out_ws)], check=True)

    pst = pyemu.Pst(str(out_ws / "pest.pst"))
    truth = pd.read_csv(truth_dir / "truth_pars.csv", index_col=0)["parval"]
    common = pst.parameter_data.index.intersection(truth.index)
    pst.parameter_data.loc[common, "parval1"] = truth.loc[common].astype(float).values
    pst.write_input_files(pst_path=str(out_ws))          # absolute path for mp workers
    cwd = os.getcwd()
    os.chdir(out_ws)
    try:
        pyemu.helpers.apply_list_and_array_pars(arr_par_file="mult2model_info.csv", chunk_len=50)
    finally:
        os.chdir(cwd)
    print(f"  [materialize_truth] truth property arrays written -> {out_ws}")
    return out_ws


def _read_prop(ws, tag, nlay=NLAY):
    return np.array([np.loadtxt(Path(ws, f"{tag}{L + 1}.txt")) for L in range(nlay)])


def _section(mg):
    ysec = float(np.min(mg.ycellcenters))
    x0 = float(np.min(mg.xcellcenters)) - 2.0
    x1 = float(np.max(mg.xcellcenters)) + 2.0
    return {"line": [(x0, ysec), (x1, ysec)]}, x0


def _cell_plotx(pxs, cell):
    """Distance-along-section where PlotCrossSection actually draws a cell -- the midpoint of its
    projected polygon. flopy measures distance from the grid entry, not from the line's x0, so we
    read the position back from the section rather than use (x - x0)."""
    xs = [p[0] for p in pxs.projpts[cell]]
    return 0.5 * (min(xs) + max(xs))


def plot_properties(ws=OUT_WS, plan_layer=PLAN_LAYER):
    apply_style()
    ws = Path(ws)
    gwf = flopy.mf6.MFSimulation.load(sim_ws=str(ws), verbosity_level=0).get_model("gwf")
    mg = gwf.modelgrid
    line, x0 = _section(mg)
    ix = GridIntersect(mg)
    wells = {"wellin": (1.0, 0.0), "wellout": (-99.0, 0.0)}

    fig, axes = plt.subplots(len(PROPS), 2, figsize=(13, 3.1 * len(PROPS)),
                             gridspec_kw={"width_ratios": [2, 3]})
    for row, (label, tag, cmap, logc) in enumerate(PROPS):
        arr = _read_prop(ws, tag)
        from matplotlib.colors import LogNorm
        norm = LogNorm(vmin=max(arr.min(), 1e-6), vmax=arr.max()) if logc else None
        vmin = None if logc else arr.min()
        vmax = None if logc else arr.max()

        # plan
        axp = axes[row, 0]
        pmv = flopy.plot.PlotMapView(model=gwf, ax=axp, layer=plan_layer)
        im = pmv.plot_array(arr, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
        for nm, (x, y) in wells.items():
            axp.scatter([x], [y], s=45, marker="v" if nm == "wellin" else "^",
                        facecolor="w", edgecolor="k", zorder=6)
        axp.set_aspect("equal")
        axp.set_ylabel(f"{label}\ny (m)", fontsize=9)
        if row == 0:
            axp.set_title(f"plan (layer {plan_layer + 1})", fontsize=11)
        if row == len(PROPS) - 1:
            axp.set_xlabel("x (m)")
        fig.colorbar(im, ax=axp, shrink=0.8, pad=0.01)

        # xsection
        axx = axes[row, 1]
        pxs = flopy.plot.PlotCrossSection(model=gwf, ax=axx, line=line, geographic_coords=False)
        imx = pxs.plot_array(arr, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
        for nm, (x, y) in wells.items():
            wc = int(ix.intersect([(x, y)], "point").cellids[0])
            axx.axvline(_cell_plotx(pxs, wc), color="k", lw=0.7, ls=":", alpha=0.6)
        axx.set_ylabel("elev (m)", fontsize=9)
        if row == 0:
            axx.set_title("cross-section (well axis)", fontsize=11)
        if row == len(PROPS) - 1:
            axx.set_xlabel(f"distance (m); x0={x0:.0f}")
        fig.colorbar(imx, ax=axx, shrink=0.8, pad=0.01, label=label)

    fig.suptitle("Synthetic-truth aquifer properties (pilot-point heterogeneity from the prior)",
                 fontweight="bold")
    return savefig(fig, "truth_properties", STAGE)


def main():
    materialize_truth()
    p = plot_properties()
    print(f"[plot_props] wrote {p}")


if __name__ == "__main__":
    main()
