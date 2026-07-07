"""Shared plotting style + QA-gate helpers for the two-well ASR workflow prototype.

Every workflow stage imports this so (a) all figures read as one deck (fixed
rcParams, colorblind-safe palette, units always on the axes) and (b) every stage
gates on hard checks that ``raise`` on failure and drops its signature figure into
a gitignored ``_figs/<stage>/`` (ADR-0004).

Two things every stage calls:

    from wf_style import apply_style, qa_gate, savefig, C, SPECIES

    apply_style()                       # once, at module import / top of stage
    qa_gate("model build", ok, "mass balance %.2f%% > tol" % mb)   # fail-fast
    savefig(fig, "breakthrough", stage="02_build")                 # -> _figs/02_build/

No AI/Claude references anywhere; this is tutorial infrastructure.
"""

from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt


# --- palette -------------------------------------------------------------------
# Okabe-Ito colorblind-safe qualitative palette (8 colours, deuteranopia-safe).
C = {
    "black":  "#000000",
    "orange": "#E69F00",
    "sky":    "#56B4E9",
    "green":  "#009E73",
    "yellow": "#F0E442",
    "blue":   "#0072B2",
    "red":    "#D55E00",   # vermillion
    "purple": "#CC79A7",
    "grey":   "#999999",
}

# Fixed colour per conditioning species so a species reads the same in every stage.
# Conditioning species: SO4, O2, NO3, pH, Tmp (CONTEXT.md).
SPECIES = {
    "SO4": C["red"],
    "O2":  C["blue"],
    "NO3": C["green"],
    "pH":  C["purple"],
    "Tmp": C["orange"],
}

# Semantic roles reused across stages (prior/posterior/truth/forecast).
# Convention: PRIOR is grey, POSTERIOR is blue.
ROLE = {
    "prior":     C["grey"],   # prior ensembles (spaghetti/hist) -- grey
    "posterior": C["blue"],   # posterior ensembles -- blue
    "truth":     C["black"],  # synthetic truth / measured -- black (distinct from red noise)
    "noise":     "#E8000B",   # observation-noise realizations -- bright red (not vermillion)
    "forecast":  C["orange"],
    "emulated":  C["green"],
    "fom":       C["black"],   # full-output-model reference
    "history":   C["sky"],    # conditioning/history window shading (NOT the prior ensemble)
}

# Standard axis labels — units always present.
LBL = {
    "so4":  "SO$_4$ (mg/L)",
    "time": "time (days)",
    "conc": "concentration (mg/L)",
    "cost": "treatment cost (–)",
    "ftreat": "$f_{treat}$ (–)",
}

_PALETTE = [C["blue"], C["orange"], C["green"], C["red"],
            C["purple"], C["sky"], C["yellow"], C["grey"]]


def apply_style():
    """Set the house rcParams. Idempotent; call at the top of every stage."""
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.constrained_layout.use": True,
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": C["grey"],
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "axes.prop_cycle": mpl.cycler(color=_PALETTE),
    })


# --- QA gate -------------------------------------------------------------------

class QAError(AssertionError):
    """Raised when a stage QA/QC gate fails. Halts the workflow (fail-fast)."""


def qa_gate(check, ok, detail=""):
    """Fail-fast gate: ``raise QAError`` unless ``ok`` is truthy.

    Parameters
    ----------
    check : str
        Short name of what is being checked (for the message).
    ok : bool
        The check result. Falsy -> raise.
    detail : str, optional
        Actionable detail appended on failure.

    Returns
    -------
    bool
        ``True`` on pass (so callers can chain / count).
    """
    if ok:
        print(f"  [QA pass] {check}")
        return True
    msg = f"[QA FAIL] {check}"
    if detail:
        msg += f" -- {detail}"
    raise QAError(msg)


def qa_close(check, value, target, tol, kind="abs"):
    """Convenience gate for a numeric tolerance check.

    ``kind='abs'`` -> pass if ``|value-target| <= tol``.
    ``kind='rel'`` -> pass if ``|value-target| <= tol*|target|``.
    """
    err = abs(value - target)
    lim = tol if kind == "abs" else tol * abs(target)
    ok = err <= lim
    return qa_gate(
        check, ok,
        f"|{value:.6g} - {target:.6g}| = {err:.3g} > {lim:.3g} ({kind} tol)",
    )


# --- figure output -------------------------------------------------------------

# _figs/ lives next to this module (gitignored). Mothership: each stage gets a subdir.
_FIGS_ROOT = Path(__file__).resolve().parent / "_figs"


def figs_dir(stage):
    """Return (and create) the gitignored figure dir for ``stage``."""
    d = _FIGS_ROOT / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def savefig(fig, name, stage, close=True):
    """Save ``fig`` as ``_figs/<stage>/<name>.png`` (+ pdf for deck use).

    Returns the PNG path.
    """
    d = figs_dir(stage)
    png = d / f"{name}.png"
    fig.savefig(png)
    fig.savefig(d / f"{name}.pdf")
    if close:
        plt.close(fig)
    print(f"  [fig] {png.relative_to(_FIGS_ROOT.parent)}")
    return png
