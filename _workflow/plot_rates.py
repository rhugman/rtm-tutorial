"""Well pumping-rate timeseries for the two-well ASR build (stage 02).

Rates are set at build time (transmissivity-weighted per screen), so this reads the actual WEL
package rates from the built model (``_s2_model``) rather than the reactive output -- it runs as
soon as the model is built, without waiting on the reactive job. Draws:

* wellin injection rate (total + per screen), positive;
* wellout recovery rate (total + per screen), negative;
* net doublet rate (wellin + wellout) -- reveals the injection ramp vs constant recovery.

Window shading (conditioning / gap / recovery-forecast) and the +2 yr extension boundary are
marked. Figure -> gitignored _figs/02_build/ via wf_style.savefig.

    conda run -n rtm_gmdsi python _workflow/plot_rates.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt

import workflow as w
from wf_style import apply_style, savefig, C, ROLE, LBL, SPECIES

STAGE = "02_build"
DATA_D = "data"
ORIG_END = 728.0            # original sim length before the +2 yr extension
WS = Path(__file__).parent / "_s2_model"


def _load_gwf():
    import flopy
    return flopy.mf6.MFSimulation.load(sim_ws=str(WS), verbosity_level=0).get_model("gwf")


def _edges():
    """Stress-period day edges [0, d1, ..., DAY_END] (len NPER+1) for step plots."""
    return np.concatenate([[0.0], np.cumsum([p[0] for p in w.PERIODDATA])])


def _pkg_rates(gwf, pkg, layers):
    """(total[nper], {layer: perscreen[nper]}) actual rates read from a WEL package.

    Rates are the transmissivity-weighted splits written at build time; read them from the model
    (not the CSV) so the plot reflects what is actually injected/recovered.
    """
    spd = gwf.get_package(pkg).stress_period_data.get_data()
    total = np.zeros(w.NPER)
    per = {L: np.zeros(w.NPER) for L in layers}
    for kper in range(w.NPER):
        rec = spd.get(kper, spd[max(k for k in spd if k <= kper)])  # WEL may store only changes
        for c, q in zip(rec["cellid"], rec["q"]):
            per[int(c[0])][kper] = float(q)
        total[kper] = float(np.sum(rec["q"]))
    return total, per


def _wellin_rates(gwf):
    return _pkg_rates(gwf, "welin", w.LAYERS_IN)


def _wellout_rates(gwf):
    return _pkg_rates(gwf, "welout", w.LAYERS_OUT)


def _shade(ax):
    ax.axvspan(0, w.DAY_COND_END, color=ROLE["history"], alpha=0.12)
    ax.axvspan(w.DAY_COND_END, w.DAY_FORECAST_START, color=C["grey"], alpha=0.15)
    ax.axvspan(w.DAY_FORECAST_START, w.DAY_END, color=ROLE["forecast"], alpha=0.10)
    ax.axvline(ORIG_END, color=C["black"], ls=":", lw=1.2, alpha=0.7)


def plot_rates():
    apply_style()
    edges = _edges()
    gwf = _load_gwf()
    win, win_screens = _wellin_rates(gwf)
    wout, wout_screens = _wellout_rates(gwf)
    net = win + wout

    fig, (a0, a1, a2) = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)

    # --- wellin injection ---
    for i, (lay, s) in enumerate(win_screens.items()):
        a0.stairs(s, edges, color=C[list(C)[(i % 7) + 1]], lw=1.2,
                  label=f"screen L{lay}", alpha=0.9)
    a0.stairs(win, edges, color=C["red"], lw=2.6, label="total", baseline=None)
    a0.set_ylabel("injection\n(m$^3$/d)")
    a0.set_title("wellin -- injection rate (per screen + total)", loc="left", fontsize=11)
    a0.legend(ncol=3, fontsize=8, loc="upper left")
    a0.set_ylim(bottom=0)

    # --- wellout recovery ---
    for i, (lay, s) in enumerate(wout_screens.items()):
        a1.stairs(s, edges, color=C[list(C)[(i % 7) + 1]], lw=1.2, label=f"screen L{lay}")
    a1.stairs(wout, edges, color=C["blue"], lw=2.6, label="total", baseline=None)
    a1.set_ylabel("recovery\n(m$^3$/d)")
    a1.set_title("wellout -- recovery rate (per screen + total)", loc="left", fontsize=11)
    a1.legend(ncol=4, fontsize=8, loc="lower left")
    a1.set_ylim(top=0)

    # --- net doublet imbalance ---
    a2.stairs(net, edges, color=C["purple"], lw=2.4, baseline=0, fill=True, alpha=0.25)
    a2.stairs(net, edges, color=C["purple"], lw=2.4, baseline=None)
    a2.axhline(0, color=C["black"], lw=0.8)
    a2.set_ylabel("net\n(m$^3$/d)")
    a2.set_title("net doublet rate (wellin + wellout) -- >0 = net injection to aquifer",
                 loc="left", fontsize=11)
    a2.set_xlabel(LBL["time"])
    a2.set_xlim(0, w.DAY_END)

    for ax in (a0, a1, a2):
        _shade(ax)
    a0.annotate("original end (728 d)\n+2 yr extension >",
                (ORIG_END, a0.get_ylim()[1]), textcoords="offset points", xytext=(4, -14),
                fontsize=8, color=C["black"])
    fig.suptitle("Two-well ASR pumping schedule "
                 "(shaded: conditioning / gap / recovery-forecast)", fontweight="bold")
    return savefig(fig, "well_rates_timeseries", STAGE)


def plot_injectate_chem():
    """Injectate drivers of the redox front vs time: temperature, O2, nitrate.

    The three model inputs that drive pyrite oxidation (notebook: temperature sets the reaction
    rate; O(0)+N(+5) are the oxidants consumed). All vary seasonally over the field record, then
    hold constant through the extended window (continuous-constant, held at period HOLD_KPER).
    Baseline f_treat=0 -- the lever scales O(0) and N(+5) down toward 0.
    """
    apply_style()
    edges = _edges()
    df = w._wellin_df(DATA_D, w.NPER)                # identical across the 5 screens
    tmp_c = df.groupby("kper")["Tmp"].first().to_numpy() * 1000.0   # stored x1e-3 -> degC
    o2 = df.groupby("kper")["O(0)"].first().to_numpy()              # mol/L
    no3 = df.groupby("kper")["N(+5)"].first().to_numpy()            # mol/L

    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.stairs(tmp_c, edges, color=SPECIES["Tmp"], lw=2.4, baseline=None,
              label="temperature")
    ax.set_ylabel("injectate temperature (°C)", color=SPECIES["Tmp"])
    ax.tick_params(axis="y", colors=SPECIES["Tmp"])

    ax2 = ax.twinx()
    ax2.grid(False)
    ax2.stairs(o2, edges, color=SPECIES["O2"], lw=2.0, baseline=None, label="O(0) (O$_2$)")
    ax2.stairs(no3, edges, color=SPECIES["NO3"], lw=2.0, baseline=None, label="N(+5) (NO$_3$)")
    ax2.set_ylabel("injected oxidant (mol/L)")
    ax2.set_ylim(bottom=0)

    _shade(ax)
    ax.set_xlim(0, w.DAY_END)
    ax.set_xlabel(LBL["time"])
    ax.text(ORIG_END + 12, 0.06 * ax.get_ylim()[1], "held constant\n(period %d) >" % w.HOLD_KPER,
            fontsize=8, color=C["black"], va="bottom")
    # merged legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", ncol=1, fontsize=9,
              framealpha=0.9, facecolor="white")
    ax.set_title("Injectate drivers of the redox front (baseline; f_treat scales O$_2$ & NO$_3$)",
                 fontweight="bold")
    return savefig(fig, "injectate_chem_timeseries", STAGE)


if __name__ == "__main__":
    p = plot_rates()
    p2 = plot_injectate_chem()
    print(f"[plot_rates] wrote:\n  {p}\n  {p2}")
