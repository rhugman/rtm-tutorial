"""Two-well ASR prototype workflow (ADR-0003 / ADR-0004) -- Section 2: model build.

Dev scaffolding: prove the whole arc in one runnable ``workflow.py``, split into the
``part1_*`` notebook sequence later. This module owns **Section 2**: the two-well
continuous-constant reactive-transport model and the ``f_treat`` injectate-treatment
preprocessor. Sections 3+ (pstfrom, obs/weights/truth, prior MC, DSI, DSIVC) append below.

Design invariants enforced here (source of truth: ``docs/adr/0003-*``, ``docs/MOU_HANDOFF.md``):

  * Geometry: ``wellin`` injects on-axis for the whole sim; ``wellout`` (-99,0) is a
    *continuous* on-axis recovery well (balanced doublet); ``wellopt`` is deleted.
  * The ``f_treat`` lever scales the injectate redox inputs ``O(0)`` (dissolved O2) and
    ``N(+5)`` (nitrate) by ``(1 - f_treat)`` (pH untouched), re-equilibrates through
    PhreeqcRM, and rewrites **only** the ``wellin`` WEL aux + ``cost.dat``.
  * LANDMINE (i): never scale the transported lumped ``O``/``N`` component arrays -- act
    on the SOLUTION inputs ``O(0)``/``N(+5)`` and let PhreeqcRM re-lump.
  * LANDMINE (ii): the preprocessor never calls ``sim.write_simulation()`` (that clobbers
    the PstFrom K/porosity/pyrite multiplier arrays). It touches the ``wellin`` package
    only, via flopy Method A (``wel.write()``).
  * LANDMINE (iii): mothership -- the workspace lives inside ``_workflow/`` (``_s2_model``),
    never in ``data/``.

Regression invariant (ADR-0002): ``apply_treatment(ws, f_treat=0)`` reproduces the
baseline ``wellin`` aux (numeric, within %.8E precision).

Run ``python workflow.py`` to build the model offline (grid + packages + f_treat=0
regression + signature figure). Pass ``--run`` to also execute the ~6 min full model and
gate on convergence + mass balance (a maintainer / human step, not CI).
"""

import os

# macOS/conda duplicate-OpenMP guard: numpy/scipy and PhreeqcRM each ship a libomp, and PhreeqcRM's
# init (the in-python f_treat chemistry build) aborts with "OMP Error #15 ... already initialized"
# when two copies are linked. Set before any numeric/PhreeqcRM import so the whole arc + the in-worker
# treatment survive it. Safe here because the copies are the same conda-forge libomp; if you ever see
# suspect reactive results, dedupe libomp in the env instead of relying on this. Respect a user override.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import re
import sys
import glob
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --- path wiring ---------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
DATA_D = REPO / "data"
DEPS = REPO / "dependencies"
TUT = REPO / "tutorials"
sys.path.insert(0, str(TUT))            # herebedragons
sys.path.insert(0, str(Path(__file__).parent))  # wf_style

from wf_style import apply_style, qa_gate, qa_close, savefig, C, SPECIES, ROLE, LBL  # noqa: E402

# heavy geoscience deps -- imported lazily inside builders so a bare parse/import of
# this module stays cheap and does not require the full stack to be present.


# =============================================================================
# Section-2 constants
# =============================================================================

STAGE = "02_build"
WS = Path(__file__).parent / "_s2_model"          # mothership workspace (gitignored)
PRISTINE = Path(__file__).parent / "_s2_pristine"  # baseline welin aux snapshot (gitignored)

# grid / tdis -----------------------------------------------------------------
NLAY = 12                                   # depth-resolved redox zonation (fixed)
TOP_DATUM = -273.0
# Mesh refinement knobs (vorflow). `resolution` = target cell size (m) at a feature; smaller =
# finer. `dist_max` = distance (m) the refinement reaches before relaxing to background.
# BUILD-DECISION: refine more than the original (bg 150, corridor 20, wells 3). Lower the
# background and the plume-corridor (bottom-line) resolution, keep wide halos. wellopt dropped.
# BUILD-DECISION: coarsened for a practical reactive runtime (~10 min at 1458 d). These give
# ncpl ~= 166 (finer near the wells than the original 194 via the corridor line, but a coarser
# background). Dial bg/LINE_RES down for more resolution (bg50/line8 -> ~394) at ~linear cost.
BACKGROUND_LC = 95.0        # background voronoi cell size (m)
LINE_RES = 14.0            # bottom-line (plume corridor) target size (m)
LINE_DISTMAX = 140.0        # corridor refinement reach (m)
PT_RES = 5.0               # well-point target size (m)
PT_DISTMAX = 200.0          # well refinement reach (m)
# The original 35 stress periods, (perlen, nstp, tsmult); daily steps spanning 728 days.
_BASE_PERIODDATA = [(2, 2, 1), (4, 4, 1), (4, 4, 1), (4, 4, 1), (7, 7, 1),
                    (7, 7, 1), (7, 7, 1), (7, 7, 1), (14, 14, 1), (14, 14, 1),
                    (15, 15, 1), (13, 13, 1), (14, 14, 1), (14, 14, 1), (14, 14, 1),
                    (21, 21, 1), (35, 35, 1), (28, 28, 1), (28, 28, 1), (28, 28, 1),
                    (28, 28, 1), (28, 28, 1), (28, 28, 1), (28, 28, 1), (28, 28, 1),
                    (35, 35, 1), (35, 35, 1), (28, 28, 1), (28, 28, 1), (28, 28, 1),
                    (35, 35, 1), (35, 35, 1), (28, 28, 1), (28, 28, 1), (28, 28, 1)]
# BUILD-DECISION: extend the recovery/forecast window by ~2 yr so the recovered-SO4 peak
# lands *inside* the sim window (it was still rising at 728 d). Continuous doublet keeps
# pumping through the extension; 30-day SPs with daily steps preserve output resolution.
EXTEND_DAYS = 730
_EXT_SP = 30
_ext = [(_EXT_SP, _EXT_SP, 1)] * (EXTEND_DAYS // _EXT_SP)
if EXTEND_DAYS % _EXT_SP:
    _ext.append((EXTEND_DAYS % _EXT_SP, EXTEND_DAYS % _EXT_SP, 1))
PERIODDATA = _BASE_PERIODDATA + _ext
NPER = len(PERIODDATA)

# history / forecast split -- BUILD-DECISION (see docstring of stage2_build).
# Cumulative end-of-SP days: SP18 -> 252, SP20 -> 308, SP34 -> 728 (verified arithmetic).
# The continuous doublet pumps every SP; this split is a *time-window* read of the single
# wellout breakthrough curve, NOT a pumping-schedule change (ADR-0003). The base-window
# boundaries (252/308) are unchanged by the tail extension; the forecast window now runs to
# DAY_END (computed), i.e. it absorbs the extra ~2 yr of recovery.
DAY_COND_END = 252         # end of conditioning (monitored) window  == end of SP18
DAY_FORECAST_START = 308   # start of recovery/forecast window       == end of SP20
DAY_END = sum(p[0] for p in PERIODDATA)      # total sim length (days); ~1458 with the extension
COND_SP = range(0, 19)      # SP0..18 -> days 0-252   (conditioning data)
GAP_SP = range(19, 21)      # SP19..20 -> days 252-308 (neutral lead-time gap)
FORECAST_SP = range(21, NPER)  # SP21..end -> recovery/forecast (now spans the extended tail)

# well geometry ----------------------------------------------------------------
# BUILD-DECISION: base-case inject/recover rates split by TRANSMISSIVITY (T = K x layer
# thickness) across the screened layers -- a well distributes flow to each screen in proportion
# to that layer's ability to transmit it. Total in = total out = INJ_TOTAL (balanced doublet);
# only the per-screen split is T-weighted. The split is computed from the model K + geometry at
# build time here; when K becomes a PstFrom parameter (slice 3) this MUST move to a runtime
# pre-processor (analogous to apply_treatment) that recomputes rates from the perturbed K each
# forward run -- see _transmissivity_weights + docs note.
INJ_TOTAL = 360.0                  # total base injection / recovery rate (m3/d)
LAYERS_IN = [1, 2, 3, 5, 7]        # wellin screens (5 layers)
LAYERS_OUT = [1, 3, 5]             # wellout screens (3 layers)

# Regional flow: impose a left->right head gradient across the CHD boundaries (was flat h=0).
CHD_GRADIENT = 0.001               # m/m, head decreasing left->right (flow left->right)

# f_treat preprocessor ---------------------------------------------------------
START_SOL = 2                       # first injectate SOLUTION id (matches make_wel_in)
SO4_GMOL = 96.06                    # g/mol, for mol/L -> mg/L (x1000)
# BUILD-DECISION: c_unit scaled so cost ~ O(SO4 mg/L) for NSGA-II conditioning.
# V_inj ~= 360 m3/d * 728 d ~= 2.62e5 m3; at f_treat=0.5, -ln(0.5)=0.693 ->
# cost ~= C_UNIT * 1.82e5.  C_UNIT=2.75e-4 -> cost ~= 50 at f_treat=0.5.  RETUNE once the
# recovered-SO4 spine magnitude is known post-rebake (open item).
C_UNIT = 2.75e-4
F_TREAT_BOUNDS = (0.0, 0.999)       # decision-variable range; cost -> inf as f_treat -> 1

# QA tolerances ----------------------------------------------------------------
MB_TOL = 1.0        # |mass-balance percent discrepancy| tolerance (%), flow + transport
REG_ATOL = 1.0e-6   # f_treat=0 regression numeric tolerance on the welin aux columns


# =============================================================================
# (A) MODEL BUILD -- two-well continuous-constant reactive transport
# =============================================================================

def _build_grid(ws, data_d=DATA_D):
    """Build the refined voronoi (DISV) grid; write the shapefile into ``ws``.

    Ports the grid block of ``dizon_build_model.ipynb``. Returns ``(grid_shp, ncpl)``.
    ``nlay`` is fixed at :data:`NLAY`. The shapefile is a *generated* artifact -> it lands
    in the workspace, never in ``data/``.
    """
    import geopandas as gpd
    from shapely import box, Point, LineString
    from flopy.utils import cvfdutil
    from vorflow import ConceptualMesh, MeshGenerator, VoronoiTessellator

    ws = Path(ws)
    domain = gpd.read_file(Path(data_d, "domain.gpkg"))
    minx, miny, maxx, maxy = domain.geometry.total_bounds
    domain_ext = box(minx, miny, maxx, maxy)

    wells = pd.read_csv(Path(data_d, "wells.csv"))
    # ADR-0003 deleted wellopt -- drop its local mesh refinement (data/wells.csv is read-only,
    # so filter here rather than editing it). Only wellin/wellout drive refinement now.
    wells = wells[wells["name"] != "wellopt"].reset_index(drop=True)
    refinement = gpd.read_file(Path(data_d, "refinement.gpkg"))
    geom = refinement.geometry[0]
    rminx, rminy, rmaxx, rmaxy = geom.bounds
    bottom_line = LineString([(rminx, rminy), (rmaxx, rminy)])

    blueprint = ConceptualMesh()
    blueprint.add_polygon(domain_ext, zone_id=1)
    blueprint.add_line(bottom_line, line_id="Refinement-Line",
                       resolution=LINE_RES, dist_max=LINE_DISTMAX)
    for wid in wells.index:
        blueprint.add_point(Point(wells.loc[wid, "x"], wells.loc[wid, "y"]),
                            point_id=f"{wells.loc[wid, 'name']}",
                            resolution=PT_RES, dist_max=PT_DISTMAX)
    clean_polys, clean_lines, clean_pts = blueprint.generate()

    mesher = MeshGenerator(background_lc=BACKGROUND_LC, verbosity=0)
    mesher.generate(clean_polys, clean_lines, clean_pts)
    grid_gdf = VoronoiTessellator(mesher, blueprint, clip_to_boundary=True).generate()

    grid_shp = Path(ws, "mf6_grid.shp")
    grid_gdf.to_file(grid_shp)

    # ncpl for the chemistry init (nlay is fixed)
    verts, iverts = cvfdutil.shapefile_to_cvfd(str(grid_shp))
    gridprops = cvfdutil.get_disv_gridprops(verts, iverts, xcyc=None)
    return grid_shp, int(gridprops["ncpl"])


HOLD_KPER = len(_BASE_PERIODDATA) - 1   # =34: last period of the original 728-d schedule


def _wellin_df(data_d, nper_model):
    """``data/wellin.csv`` injectate chemistry per kper x screen-layer, with the SEASONAL signal
    continued (cyclically repeated) across the extended window.

    The field record (periods 0..HOLD_KPER, days 0..728 ~ two annual cycles) carries the
    seasonal injectate -- temperature, O(0), NO3 all swing with the seasons. The recovery window
    was extended by +2 yr; rather than freeze the injectate at the last field period, we keep it
    SEASONAL by wrapping the field record in time: each extended stress period takes the field
    period whose day-of-record matches its own mid-time modulo the field span. Injection
    chemistry therefore keeps cycling; the total rate stays 360 m3/d (every field period 0..34
    sums to 360, so any wrap is balanced) and the per-screen rate is overridden downstream to a
    transmissivity-weighted split (see ``_transmissivity_weights``). Returns a kper-column
    dataframe truncated to ``kper < nper``.
    """
    df = pd.read_csv(os.path.join(str(data_d), "wellin.csv"))
    df = df[df["kper"] <= HOLD_KPER].copy()          # field window (drops the unused ramp 35..44)
    if nper_model - 1 > HOLD_KPER:
        perlen = np.array([p[0] for p in PERIODDATA], dtype=float)
        cumend = np.cumsum(perlen)
        cumstart = cumend - perlen
        field_span = float(cumend[HOLD_KPER])        # end day of the field window (~728)
        fend = cumend[:HOLD_KPER + 1]
        rows = []
        for p in range(HOLD_KPER + 1, nper_model):
            mid = 0.5 * (cumstart[p] + cumend[p])
            wrapped = mid % field_span               # replay the seasonal record
            fp = min(int(np.searchsorted(fend, wrapped, side="right")), HOLD_KPER)
            rows.append(df[df["kper"] == fp].assign(kper=p))
        df = pd.concat([df] + rows, ignore_index=True)
    return df[df["kper"] < nper_model].copy()


def build_injectate_solutions(sim, data_d, ws, nlay, ncpl, f_treat=0.0, chem_wd=None,
                              decision_day=DAY_COND_END):
    """Build the mup3d chemistry model with the ``f_treat`` treated injectate.

    A near-verbatim port of the notebook's ``initialize_chemistry`` with the treatment
    lever added: the injectate rows ``O(0)`` (dissolved O2) and ``N(+5)`` (nitrate) are
    scaled by ``(1 - f_treat)`` *before* the per-(period,layer) groupby, so the treated
    values flow into the numbered PHREEQC SOLUTION blocks. ``pH`` and the background
    solution (column ``'value'``) are left untouched.

    FORECAST-ONLY treatment (ADR-0003 / DSIVC design 2026-07-06): the scaling applies only
    to stress periods that START at/after ``decision_day`` (=DAY_COND_END, day 252 = end of
    the monitored window). Injection during the monitored history stays untreated, so the
    conditioning obs are ``f_treat``-independent and the truth (baseline, f_treat=0) is
    conditioned cleanly. ``f_treat=0`` reproduces the baseline for every period (scale=1).

    LANDMINE (i): we scale the SOLUTION *inputs* here, never the transported lumped
    ``O``/``N`` arrays. PhreeqcRM re-equilibration (via ``set_chem_stress`` downstream)
    re-lumps the redox change correctly into ``O``/``N``/``Charge``/``H``.

    LANDMINE (ii): ``model.initialize()`` calls mup3d's own ``write_simulation()``, which
    (re)writes the ``kinetic_phases.Pyrite.m0.layer*.txt`` / ``equilibrium_phases.*`` /
    ``exchange_phases.*`` arrays from baseline CSVs -- the *exact* files PstFrom's pyrite
    multiplier lands on. In the forward run those arrays are already multiplier-applied, so
    writing this chem model into the live ``ws`` would clobber them. ``chem_wd`` isolates the
    write: pass a scratch dir when you only need the re-equilibrated component vectors
    (``apply_treatment``); leave it ``None`` for the initial build, where writing into ``ws``
    IS the intent. Only the in-memory component vectors are used downstream either way.

    Returns the initialized ``mup3d.Mup3d`` model (phinp now carries treated injectate).
    """
    from mf6rtm import utils, mup3d

    ws = Path(ws)
    perioddata = sim.tdis.perioddata.get_data()

    # background aquifer chemistry (SOLUTION 1) -- NOT scaled
    solutionsdf = pd.read_csv(Path(data_d, "ic_aq_chem.csv"), index_col=0)

    # injection chemistry: one column per (stress period, layer), forward-filled to nper_model
    # so the extended recovery window keeps a dense, gap-free SOLUTION set (see _wellin_df).
    nper_model = len(perioddata)
    injdf = _wellin_df(data_d, nper_model).set_index("kper")
    injdf = injdf[["layer"] + solutionsdf.index.tolist()].copy()

    # --- the f_treat lever: scale both oxidants (leave pH), FORECAST-ONLY ---
    # gate the scaling to periods starting at/after decision_day so the monitored history
    # stays untreated (see docstring). period_start[kper] = cumulative days before kper.
    f = float(f_treat)
    perlen = np.array([row[0] for row in perioddata], dtype=float)
    period_start = np.concatenate([[0.0], np.cumsum(perlen)])[:-1]       # start day per kper
    kper_vals = injdf.index.to_numpy()                                   # kper per (period,layer) row
    treat = (period_start[kper_vals] >= float(decision_day))             # bool mask, forecast-only
    scale = np.where(treat, 1.0 - f, 1.0)                                # =1 pre-decision or f_treat=0
    injdf["O(0)"] = injdf["O(0)"].to_numpy() * scale     # dissolved O2
    injdf["N(+5)"] = injdf["N(+5)"].to_numpy() * scale   # nitrate

    frames = []
    for per in injdf.index.unique():
        df = (injdf.loc[per].reset_index()
                    .drop(columns="kper")
                    .groupby("layer").mean()
                    .T)
        df.columns = [f"{per}_{layer}" for layer in df.columns]
        frames.append(df)
    injdf = pd.concat(frames, axis=1)
    solutionsdf = pd.concat([solutionsdf, injdf], axis=1)

    solutions = utils.solution_df_to_dict(solutionsdf)
    sol_ic = np.ones((nlay, ncpl), dtype=float)

    solution = mup3d.Solutions(solutions)
    solution.set_ic(sol_ic)

    # cation exchanger
    excdf = pd.read_csv(Path(data_d, "ic_exchanger.csv"), comment="#")
    ex_names = {"Ca_ex": "CaX2", "Fe_ex": "FeX2", "K_ex": "KX",
                "Mg_ex": "MgX2", "Na_ex": "NaX"}
    excdf["name"] = excdf["var"].map(ex_names)
    excdf["layer"] -= 1
    excdf = excdf.pivot(index="name", columns="layer", values="value")
    exchanger_dict = excdf.to_dict()
    for _k, subdict in exchanger_dict.items():
        for key in subdict:
            subdict[key] = {"m0": subdict[key]}
    exchanger = mup3d.ExchangePhases(exchanger_dict)
    exchanger.set_ic(sol_ic)
    exchanger.set_equilibrate_solutions([1] * nlay)

    # mineral surfaces
    mindf = pd.read_csv(Path(data_d, "ic_surfaces.csv"), comment="#")
    mindf["value"] = [utils.concentration_volbulk_to_volwater(i, 0.35)
                      for i in mindf["value"].values]
    mindf = mindf.pivot(index="var", columns="layer", values="value")

    eq_m0 = utils.solution_df_to_dict(mindf.loc[["Ferrihydrite", "Orgmatter"], :])
    si = 0
    eq_dic = {}
    for ly in range(nlay):
        for key in eq_m0.keys():
            eq_dic[ly] = {key: {}}
            eq_dic[ly][key]["si"] = si
            eq_dic[ly][key]["m0"] = eq_m0[key][ly]
    equilibriums = mup3d.EquilibriumPhases(eq_dic)
    equilibriums.set_ic(sol_ic)

    # pyrite (kinetic) + organic carbon (kinetic, minor)
    py_m0 = utils.solution_df_to_dict(mindf.loc[["Pyrite"], :])
    kin_py_params = [1.600000e+01, 6.700000e-01, 5.000000e-01, -1.100000e-01]
    kin_orgc_params = [1.570000e-09, 1.670000e-11, 1.000000e-13]
    orgc_form = "Orgc -1.0 CH2O 1.0"
    orgc_steps = "8.640000e+04 in 1 steps"
    kin_dic = {}
    for ly in range(nlay):
        for key in py_m0.keys():
            kin_dic[ly] = {key: {}}
            kin_dic[ly][key]["m0"] = py_m0[key][ly]
            kin_dic[ly][key]["parms"] = kin_py_params
    for key in kin_dic.keys():
        kin_dic[key]["Orgc"] = {}
        kin_dic[key]["Orgc"]["m0"] = 1.0
        kin_dic[key]["Orgc"]["parms"] = kin_orgc_params
        kin_dic[key]["Orgc"]["formula"] = orgc_form
        kin_dic[key]["Orgc"]["steps"] = orgc_steps
    kinetics = mup3d.KineticPhases(kin_dic)
    kinetics.set_ic(sol_ic)

    model = mup3d.Mup3d("model", solution, nlay=nlay, ncpl=ncpl)
    # LANDMINE (ii): isolate the initialize()->write_simulation() to a scratch dir when the
    # caller only wants the re-equilibrated component vectors -- writing into the live ws would
    # overwrite the PstFrom-parameterized m0 arrays (pyrite kinetics etc.).
    chem_wd = Path(chem_wd) if chem_wd is not None else Path(ws)
    chem_wd.mkdir(parents=True, exist_ok=True)
    model.set_wd(chem_wd)
    shutil.copy(Path(data_d, "datab.dat"), Path(model.wd, "datab.dat"))
    model.set_database(Path("datab.dat"))
    model.set_postfix(Path(data_d, "postfix.phqr"))
    model.set_exchange_phases(exchanger)
    model.set_phases(kinetics)
    model.set_phases(equilibriums)

    import herebedragons as hbd
    # reactive OUTPUT timing (sout.csv / ucn) over the full (extended) schedule; herebedragons
    # default cadence (every 5 days). Covers all NPER periods since `perioddata` is the live tdis.
    tsteps = hbd.create_reactive_tsteps(perioddata, output_interval=5)
    model.set_config(reactive={"timing": "user", "externalio": True, "tsteps": tsteps})
    model.set_componenth2o(True)              # CRITICAL: O/N are lumped (see LANDMINE i)
    model.initialize(add_charge_flag=False)
    return model


def _transmissivity_weights(gwf, cell, layers):
    """Normalised transmissivity weights (T = K x layer thickness) for ``layers`` at ``cell``.

    A screened well splits its total rate between screened layers in proportion to each layer's
    transmissivity -- more flow through the more transmissive intervals. With the homogeneous base
    K this reduces to thickness-weighting; once K is a PstFrom pilot-point parameter (slice 3) the
    SAME function must drive a RUNTIME rate pre-processor that recomputes the wellin/wellout rates
    from the perturbed K on every forward run (analogous to ``apply_treatment``). Total rate is
    conserved (weights sum to 1); only the split changes.
    """
    k = np.asarray(gwf.get_package("npf").k.array)      # (nlay, ncpl)
    top = np.asarray(gwf.modelgrid.top).ravel()
    botm = np.asarray(gwf.modelgrid.botm)               # (nlay, ncpl)
    T = []
    for L in layers:
        ztop = top[cell] if L == 0 else botm[L - 1, cell]
        thick = float(ztop) - float(botm[L, cell])
        T.append(float(k[L, cell]) * thick)
    T = np.array(T, dtype=float)
    return T / T.sum()


def _make_wel_in_continuous(sim, mup3d_m, data_d=DATA_D):
    """Build the ``wellin`` injection package -- continuous, constant, no rate ramp.

    Mirrors ``hbd.make_wel_in``'s reactive branch but DROPS the second-half ``rate *= 10``
    (ADR-0003 continuous-constant operation). Screens :data:`LAYERS_IN`; runs all 35 SP;
    injectate chemistry from the ``mup3d_m`` chem stresses (same SOLUTION ordering as
    ``make_wel_in``: ``sol = START_SOL + per*nlay + layer_index``).
    """
    import flopy
    from mf6rtm import mup3d
    from collections import defaultdict
    import herebedragons as hbd

    nper_model = int(sim.tdis.nper.get_data())
    gwf = sim.get_model("gwf")
    layers = LAYERS_IN
    cellid = hbd.get_wel_coords(gwf, name="wellin", data_d=str(data_d))
    coords_in = {lay: (lay, cellid) for lay in layers}

    # BUILD-DECISION: continuous-constant balanced doublet. Total injection = INJ_TOTAL (360 m3/d),
    # split across wellin screens by TRANSMISSIVITY (T = K x thickness), constant in time. The CSV
    # per-screen rate + seasonal ramp are ignored for rate (only the CSV *chemistry* is used,
    # seasonally, via _wellin_df). At PstFrom this split moves to a runtime preprocessor (K varies).
    w_in = _transmissivity_weights(gwf, cellid, layers)
    rate_in = {lay: INJ_TOTAL * float(w_in[e]) for e, lay in enumerate(layers)}

    # Forward-filled to nper_model so wellin injects through the WHOLE (extended) window and its
    # SOLUTION ids line up 1:1 with build_injectate_solutions (see _wellin_df).
    df_inj = _wellin_df(data_d, nper_model)
    nper = df_inj.kper.max() + 1
    wellin_sp_data = defaultdict(list)

    nlay = len(layers)
    wel_chem_dir = {}
    for per in range(nper):
        sol_spd = list(range(START_SOL + per * nlay, START_SOL + (per + 1) * nlay))
        wellchem = mup3d.ChemStress("per_" + str(per))
        wellchem.set_spd(sol_spd)
        mup3d_m.set_chem_stress(wellchem)
        wel_chem_dir[per] = wellchem.data

    for _, r in df_inj.iterrows():
        layer = int(r["layer"])
        # transmissivity-weighted split across wellin screens (not the CSV per-screen rate)
        wellin_sp_data[int(r["kper"])].append([coords_in[layer], rate_in[layer]])

    for per in range(nper):
        for e, _layer in enumerate(layers):
            wellin_sp_data[per][e].extend(wel_chem_dir[per][e])

    wel_in = flopy.mf6.ModflowGwfwel(gwf, stress_period_data=wellin_sp_data,
                                     auxiliary=mup3d_m.components,
                                     pname="welin", filename=f"{gwf.name}.welin")
    wel_in.set_all_data_external()
    return wel_in


def _apply_chd_gradient(gwf, grad=CHD_GRADIENT):
    """Impose a regional right->left head gradient on the CHD boundary cells.

    ``hbd.make_chd`` sets every CHD cell to head 0 (flat, no regional flow). Here each CHD cell's
    head is set from its x-coordinate: ``head = grad * (x_cell - x_center)`` -- LOWER on the left
    edge, higher on the right, so the regional flow runs right->left at slope ``grad`` (m/m). This
    puts ``wellout`` (x=-99, west) DOWNgradient of ``wellin`` -- regional flow aids recovery. Heads
    are centred on 0 (range +/- grad*width/2). Only the head is changed; the CHD chemistry aux is
    left intact. Uses a per-package write (never ``sim.write_simulation``).
    """
    chd = gwf.get_package("chd")
    xc = np.asarray(gwf.modelgrid.xcellcenters).ravel()
    x_center = 0.5 * (xc.min() + xc.max())
    spd = chd.stress_period_data.get_data()
    for kper, rec in spd.items():
        # hbd.make_chd seeds head with the integer literal 0, so the 'head' column is INT dtype
        # and fractional gradient heads (+/- ~0.13 m) would truncate to 0. Rebuild the recarray
        # with a float 'head' column before assigning.
        names = list(rec.dtype.names)
        new_dt = [(n, (np.float64 if n == "head" else rec.dtype[n])) for n in names]
        newrec = np.empty(len(rec), dtype=new_dt).view(np.recarray)
        for n in names:
            newrec[n] = rec[n]
        newrec["head"] = np.array([grad * (xc[cid[1]] - x_center) for cid in rec["cellid"]],
                                  dtype=float)
        chd.stress_period_data.set_data({kper: newrec})
    chd.set_all_data_external()
    return chd


def _build_gwf(ws, grid_shp, data_d, mup3d_m):
    """Build the GWF flow model (two-well recast: wellopt deleted, wellout continuous).

    Ports ``make_gwf`` with two edits: (1) ``wellout`` is a *continuous* recovery well over
    all SP (reusing ``hbd.make_extraction_well`` with ``active_sp=range(0, nper)``);
    (2) ``make_wel_opt`` (the deleted supply well) is not called.
    """
    import flopy
    from flopy.utils import cvfdutil
    import herebedragons as hbd

    ws = Path(ws)
    name = "gwf"
    sim = flopy.mf6.MFSimulation(sim_name=name, version="mf6", sim_ws=str(ws))
    hbd.get_bins(str(ws))

    flopy.mf6.ModflowTdis(sim, pname="tdis", time_units="DAYS",
                          nper=NPER, perioddata=PERIODDATA)
    ims = flopy.mf6.ModflowIms(sim, complexity="complex",
                               outer_dvclose=1e-3, inner_dvclose=1e-3,
                               filename=f"{name}.ims")
    sim.register_ims_package(ims, [name])
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name, model_nam_file=f"{name}.nam",
                               exe_name="mf6")

    verts, iverts = cvfdutil.shapefile_to_cvfd(str(grid_shp))
    gridprops = cvfdutil.get_disv_gridprops(verts, iverts, xcyc=None)
    disv = flopy.mf6.ModflowGwfdisv(
        gwf, nlay=NLAY, ncpl=gridprops["ncpl"], nvert=gridprops["nvert"],
        vertices=gridprops["vertices"], cell2d=gridprops["cell2d"],
        top=TOP_DATUM, botm=0.0, filename=f"{name}.disv")
    botms = hbd.get_botms(gwf, str(ws), data_d=str(data_d))
    disv.botm.set_data(botms)
    disv.set_all_data_external()

    nlay = disv.nlay.get_data()
    ncpl = disv.ncpl.get_data()

    ic = flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=0.0 * np.ones((nlay, ncpl)))
    ic.set_all_data_external()
    npf = flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=np.ones((nlay, ncpl)),
                                  k33=np.ones((nlay, ncpl)))
    npf.set_all_data_external()
    sto = flopy.mf6.ModflowGwfsto(gwf, ss=np.ones((nlay, ncpl)) * 1.e-4,
                                  iconvert=0, transient={0: True})
    sto.set_all_data_external()

    # boundaries + wells (recast)
    hbd.make_chd(gwf, conservative_tracer=None, mup3d_m=mup3d_m, data_d=str(data_d))
    _apply_chd_gradient(gwf, grad=CHD_GRADIENT)    # regional right->left head gradient
    # continuous recovery well: active over the WHOLE sim (balanced doublet). Recovery total =
    # INJ_TOTAL, split across screens by transmissivity (T = K x thickness) -- same rule as wellin.
    out_cell = hbd.get_wel_coords(gwf, name="wellout", data_d=str(data_d))
    rates_out = list(-INJ_TOTAL * _transmissivity_weights(gwf, out_cell, LAYERS_OUT))
    hbd.make_extraction_well(sim, wellname="wellout", rates=rates_out,
                             active_sp=range(0, NPER), tag="welout",
                             conservative_tracer=None, mup3d_m=mup3d_m,
                             data_d=str(data_d))
    _make_wel_in_continuous(sim, mup3d_m, data_d=data_d)
    # NOTE: make_wel_opt (supply well) intentionally NOT built (ADR-0003).

    flopy.mf6.ModflowGwfoc(
        gwf, saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        head_filerecord=[f"{name}.hds"], budget_filerecord=[f"{name}.cbb"],
        printrecord=[("HEAD", "LAST")])
    return gwf, sim


def _build_gwt(sim, data_d, mup3d_m):
    """Build one GWT model per transported component (recast: welopt source dropped)."""
    import flopy
    import herebedragons as hbd

    ne = 0.35
    long_disp = 0.1
    disp_tr_vert = long_disp * 0.01
    disp_tr_hor = long_disp * 0.1
    diffc = 0

    gwf = sim.get_model("gwf")
    components = mup3d_m.components

    for comp in components:
        gwt = flopy.mf6.MFModel(sim, model_type="gwt6", modelname=comp,
                                model_nam_file=f"{comp}.nam")
        ims = flopy.mf6.ModflowIms(sim, complexity="complex",
                                   outer_dvclose=1e-3, inner_dvclose=1e-3,
                                   filename=f"{comp}.ims")
        sim.register_ims_package(ims, [comp])

        dis = gwf.dis
        nlay = dis.nlay.get_data()
        ncpl = dis.ncpl.get_data()
        disv = flopy.mf6.ModflowGwfdisv(gwt, nlay=nlay, ncpl=ncpl,
                                        nvert=dis.nvert.get_data(),
                                        vertices=dis.vertices.get_data(),
                                        cell2d=dis.cell2d.get_data(),
                                        top=dis.top.get_data(),
                                        botm=dis.botm.get_data(),
                                        filename=f"{comp}.disv")
        disv.set_all_data_external()

        ic = flopy.mf6.ModflowGwtic(gwt, strt=mup3d_m.sconc[comp], filename=f"{comp}.ic")
        ic.set_all_data_external()
        adv = flopy.mf6.ModflowGwtadv(gwt, scheme="tvd")
        adv.set_all_data_external()
        dsp = flopy.mf6.ModflowGwtdsp(
            gwt, xt3d_off=True,
            alh=np.ones((nlay, ncpl)) * long_disp,
            ath1=np.ones((nlay, ncpl)) * disp_tr_hor,
            atv=np.ones((nlay, ncpl)) * disp_tr_vert,
            diffc=diffc, filename=f"{comp}.dsp")
        dsp.set_all_data_external()

        # recast: only wellin, wellout, chd sources (welopt deleted)
        sourcerecarray = [["welin", "aux", comp], ["welout", "aux", comp],
                          ["chd", "aux", comp]]
        ssm = flopy.mf6.ModflowGwtssm(gwt, sources=sourcerecarray,
                                      save_flows=True, print_flows=True,
                                      filename=f"{comp}.ssm")
        ssm.set_all_data_external()

        if comp == "Tmp":
            distcoef = np.ones((nlay, ncpl)) * 2.1141E-04
            sorption = "Linear"
            bulk_density = np.ones((nlay, ncpl)) * 1850
        else:
            distcoef = sorption = bulk_density = None
        mst = flopy.mf6.ModflowGwtmst(gwt, porosity=np.ones((nlay, ncpl)) * ne,
                                      first_order_decay=None, decay=None,
                                      decay_sorbed=None, sorption=sorption,
                                      bulk_density=bulk_density, distcoef=distcoef,
                                      sp2=None, filename=f"{comp}.mst")
        mst.set_all_data_external()

        flopy.mf6.ModflowGwtoc(
            gwt, budget_filerecord=f"{comp}.cbb", concentration_filerecord=f"{comp}.ucn",
            concentrationprintrecord=[("COLUMNS", 10, "WIDTH", 15, "DIGITS", 10, "GENERAL")],
            saverecord=[("CONCENTRATION", "ALL")], printrecord=[("CONCENTRATION", "LAST")])
        flopy.mf6.ModflowGwfgwt(sim, exgtype="GWF6-GWT6", exgmnamea="gwf",
                                exgmnameb=comp, filename=f"{comp}.gwfgwt")
        hbd.make_obs_pack(gwt, data_d=str(data_d))

    sim.write_simulation()
    return sim


def _model_is_built(ws):
    ws = Path(ws)
    return (ws / "mfsim.nam").exists() and (ws / "gwf.welin").exists()


def build_two_well_model(ws=WS, data_d=DATA_D, rebuild=False):
    """Build (or reuse) the two-well continuous reactive model in ``ws`` (mothership).

    Returns the loaded ``flopy`` simulation. Does NOT run mf6rtm.
    """
    import flopy

    ws = Path(ws)
    if _model_is_built(ws) and not rebuild:
        print(f"  [build] reusing existing model in {ws}")
        return flopy.mf6.MFSimulation.load(sim_ws=str(ws), verbosity_level=0)

    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)

    print("  [build] grid ...")
    grid_shp, ncpl = _build_grid(ws, data_d=data_d)

    print("  [build] chemistry (f_treat=0 baseline) ...")
    # a throwaway sim just to hand tdis/perioddata to the chemistry init
    tmp_sim = flopy.mf6.MFSimulation(sim_name="gwf", version="mf6", sim_ws=str(ws))
    flopy.mf6.ModflowTdis(tmp_sim, pname="tdis", time_units="DAYS",
                          nper=NPER, perioddata=PERIODDATA)
    mup3d_m = build_injectate_solutions(tmp_sim, data_d, ws, NLAY, ncpl, f_treat=0.0)

    print("  [build] gwf + gwt (recast: welopt deleted, wellout continuous) ...")
    gwf, sim = _build_gwf(ws, grid_shp, data_d, mup3d_m)
    sim = _build_gwt(sim, data_d, mup3d_m)
    return flopy.mf6.MFSimulation.load(sim_ws=str(ws), verbosity_level=0)


# =============================================================================
# (B) f_treat PREPROCESSOR -- apply_treatment
# =============================================================================

def _treated_aux_vectors(mup3d_m, sim, layers=LAYERS_IN):
    """Re-equilibrate treated injectate SOLUTIONs -> per-(kper, screen) component vectors.

    Reuses the tested ``set_chem_stress`` -> ``initialize_chem_stress`` path (PhreeqcRM,
    mol/m3, component-ordered). Keyed by ``(kper, e)`` where ``e`` is the screen slot
    0..4 over :data:`LAYERS_IN` (the solution slot, NOT the model layer value).
    """
    from mf6rtm import mup3d

    nper = int(sim.tdis.nper.get_data())
    nlay = len(layers)
    treated = {}
    for per in range(nper):
        sol_spd = list(range(START_SOL + per * nlay, START_SOL + (per + 1) * nlay))
        wc = mup3d.ChemStress("per_" + str(per))
        wc.set_spd(sol_spd)
        mup3d_m.set_chem_stress(wc)
        for e in range(nlay):
            treated[(per, e)] = wc.data[e]     # order == mup3d_m.components
    return treated


def _rewrite_welin_aux(ws, treated, layers=LAYERS_IN):
    """Overwrite ONLY the wellin WEL aux (flopy Method A: wel.write()).

    Never touches ``q`` (rate) and never calls ``sim.write_simulation()`` (LANDMINE ii).
    Maps each recarray row's 0-based layer (in {1,2,3,5,7}) to its solution slot
    ``e = layers.index(lay)`` before indexing ``treated`` (layers 5,7 would KeyError on the
    raw layer value).
    """
    import flopy

    ws = Path(ws)
    sim = flopy.mf6.MFSimulation.load(sim_ws=str(ws), verbosity_level=0)
    gwf = sim.get_model("gwf")
    wel = gwf.get_package("welin")
    spd = wel.stress_period_data.get_data()
    # Aux field names taken from the recarray dtype (authoritative + in file/component order).
    # NB: wel.auxiliary.array leads with the literal 'auxiliary' keyword, so don't use it here.
    sample = next(iter(spd.values()))
    aux_names = [n for n in sample.dtype.names if n not in ("cellid", "q")]
    for kper, rec in spd.items():
        for i in range(len(rec)):
            lay = rec["cellid"][i][0]
            e = layers.index(lay)
            vec = treated[(kper, e)]
            for j, cname in enumerate(aux_names):
                rec[cname][i] = vec[j]
        wel.stress_period_data.set_data({kper: rec})
    wel.set_all_data_external()
    wel.write()                                    # per-package write ONLY
    return aux_names


def _compute_v_inj(sim):
    """Injected volume V_inj = Sum_kper Sum_screen (q * perlen[kper]) from the wellin pkg."""
    perlen = np.array([p[0] for p in sim.tdis.perioddata.get_data()], dtype=float)
    gwf = sim.get_model("gwf")
    wel = gwf.get_package("welin")
    spd = wel.stress_period_data.get_data()
    v = 0.0
    for kper, rec in spd.items():
        v += float(np.sum(rec["q"])) * perlen[kper]
    return v


def _write_cost(ws, f_treat, v_inj, c_unit=C_UNIT):
    """cost = c_unit * V_inj * (-ln(1 - f_treat)); convex, -> inf as f_treat -> 1."""
    f = float(f_treat)
    if f >= 1.0:
        raise ValueError("f_treat must be < 1")
    cost = c_unit * v_inj * (-np.log(1.0 - f))
    Path(ws, "cost.dat").write_text(f"{cost:.10E}\n")
    return cost


def apply_treatment(ws, f_treat, data_d=DATA_D):
    """The f_treat preprocessor (forward-run pre-command).

    1. build the treated injectate SOLUTIONs (scales O(0)+N(+5) by (1-f_treat), pH untouched);
    2. re-equilibrate through PhreeqcRM to component vectors (mol/m3, component-ordered);
    3. rewrite ONLY the wellin WEL aux (flopy Method A);
    4. write cost.dat.

    Chemistry-only recompute: touches the wellin package + cost.dat and nothing else, so
    the PstFrom K/porosity/pyrite multipliers are left intact (LANDMINE ii). The PhreeqcRM
    re-equilibration is isolated to a throwaway ``_treat_scratch`` dir so mup3d's
    ``write_simulation()`` never lands on the live parameterized arrays.
    Returns ``cost``.
    """
    import flopy

    ws = Path(ws)
    sim = flopy.mf6.MFSimulation.load(sim_ws=str(ws), verbosity_level=0)
    gwf = sim.get_model("gwf")
    nlay = gwf.dis.nlay.get_data()
    ncpl = gwf.dis.ncpl.get_data()

    # scratch dir absorbs the initialize()->write_simulation() m0-array writes (LANDMINE ii)
    scratch = ws / "_treat_scratch"
    if scratch.exists():
        shutil.rmtree(scratch)
    mup3d_m = build_injectate_solutions(sim, data_d, ws, nlay, ncpl,
                                        f_treat=f_treat, chem_wd=scratch)
    treated = _treated_aux_vectors(mup3d_m, sim)
    _rewrite_welin_aux(ws, treated)
    shutil.rmtree(scratch, ignore_errors=True)

    v_inj = _compute_v_inj(sim)
    cost = _write_cost(ws, f_treat, v_inj)
    print(f"  [apply_treatment] f_treat={f_treat:.4f}  V_inj={v_inj:.3e}  cost={cost:.4g}")
    return cost


def apply_treatment_forward(ws="."):
    """FORWARD-RUN pre-command (SELF-CONTAINED stub embedded verbatim into forward_run.py).

    The treatment chain (PhreeqcRM re-equilibration of the injectate SOLUTIONs + surgical wellin-aux
    rewrite + module constants like DAY_COND_END) is far too large to inline the way apply_well_rates
    is, so instead we IMPORT the workflow module -- shipped into the template alongside wf_style.py and
    the four chem-data CSVs at build time (with_treatment) -- and delegate to the real implementation.
    Runs in the worker's cwd: treatment.dat + the data CSVs live there.

    Order: after apply_well_rates (final rate schedule), before copy_parameterized_transport_files +
    mf6rtm. Echoes f_treat to ftreat.csv (DSIVC's controllable-obs decvar) and cost to cost_obs.csv.
    """
    import os
    import sys
    from pathlib import Path
    os.environ.setdefault("MPLBACKEND", "Agg")               # headless worker: no GUI matplotlib backend
    ws = str(Path(ws).resolve())
    if ws not in sys.path:
        sys.path.insert(0, ws)
    import workflow as _wf
    _wf._apply_treatment_forward_impl(ws, data_d=ws)


def _apply_treatment_forward_impl(ws=".", data_d=None):
    """The real treatment forward-run logic (called via the imported workflow module in-worker, or
    directly at build time with data_d=DATA_D). Reads f_treat from treatment.dat, applies forecast-only
    treatment (rewrites the wellin aux + writes cost.dat), and echoes f_treat + cost as one-row obs CSVs.
    index col is 'item', NOT 'name' -- PstFrom treats 'name' as an obsnme alias (reload clash)."""
    ws = Path(ws)
    data_d = data_d if data_d is not None else DATA_D
    f_treat = float(np.loadtxt(ws / "treatment.dat"))
    cost = apply_treatment(str(ws), f_treat, data_d=data_d)   # rewrites wellin aux + writes cost.dat
    pd.DataFrame({"item": ["f_treat"], "value": [f_treat]}).to_csv(ws / "ftreat.csv", index=False)
    pd.DataFrame({"item": ["cost"], "value": [cost]}).to_csv(ws / "cost_obs.csv", index=False)
    print(f"  [apply_treatment_forward] f_treat={f_treat:.4f}  cost={cost:.4g}")


def _add_treatment_par(pst, template_ws):
    """Hand-template the single ``f_treat`` decision-variable parameter over ``treatment.dat``.

    partrans='none' (a fraction, not a log quantity); bounds [0, 0.999] (cost -> inf as f_treat->1).
    pargp='decvar' so the sweep/merge/DSIVC steps can select it by metadata. The forward run reads
    the filled ``treatment.dat`` via ``apply_treatment_forward``.
    """
    tpl = Path(template_ws, "treatment.dat.tpl")
    tpl.write_text("ptf ~\n~  f_treat  ~\n")                 # header line dropped by PEST; value line written
    pst.add_parameters(str(tpl), pst_path=".")
    par = pst.parameter_data
    par.loc["f_treat", ["parval1", "parlbnd", "parubnd", "pargp", "partrans"]] = \
        [0.0, 0.0, F_TREAT_BOUNDS[1], "decvar", "none"]


def omp_env_guard(ws="."):
    """FORWARD-RUN pre-command (self-contained, runs FIRST): set KMP_DUPLICATE_LIB_OK so the ``mf6rtm``
    subprocess spawned later in the forward run survives the duplicate-libomp clash (OMP Error #15 --
    numpy/PhreeqcRM each ship a libomp) whatever env launched the worker. os.environ set here is
    inherited by the mf6rtm subprocess. Embedded into forward_run.py, which does NOT import workflow,
    so the module-level guard in workflow.py cannot reach the worker -- this closes that gap."""
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def apply_well_rates(ws="."):
    """FORWARD-RUN pre-command: set transmissivity-weighted wellin/wellout rates from the
    (PstFrom-perturbed) K field.

    A screened well splits its total rate between screens by T = K x layer thickness. K is a
    PstFrom pilot-point parameter, so the split must be recomputed on EVERY forward run from the
    perturbed K (written to the npf arrays by ``apply_list_and_array_pars``, which runs first).
    Totals are conserved (wellin +360, wellout -360 m3/d); only the per-screen split moves.

    Self-contained (all imports + geometry constants inline) so PstFrom can inject it verbatim.
    Touches ONLY the wellin/wellout WEL ``q`` columns via per-package writes -- never
    ``sim.write_simulation()`` (would clobber the parameterized arrays), and never the aux
    (``apply_treatment`` owns that; run this BEFORE it).
    """
    import numpy as np
    import flopy
    from pathlib import Path

    LAYERS_IN = [1, 2, 3, 5, 7]
    LAYERS_OUT = [1, 3, 5]
    INJ_TOTAL = 360.0

    ws = Path(ws)
    sim = flopy.mf6.MFSimulation.load(sim_ws=str(ws), verbosity_level=0)
    gwf = sim.get_model("gwf")
    k = np.asarray(gwf.get_package("npf").k.array)      # perturbed K (nlay, ncpl)
    top = np.asarray(gwf.modelgrid.top).ravel()
    botm = np.asarray(gwf.modelgrid.botm)

    def _tw(cell, layers):
        T = []
        for L in layers:
            ztop = top[cell] if L == 0 else botm[L - 1, cell]
            T.append(float(k[L, cell]) * (float(ztop) - float(botm[L, cell])))
        T = np.array(T, dtype=float)
        return T / T.sum()

    for pkg, layers, total in [("welin", LAYERS_IN, INJ_TOTAL),
                               ("welout", LAYERS_OUT, -INJ_TOTAL)]:
        wel = gwf.get_package(pkg)
        spd = wel.stress_period_data.get_data()
        cell = int(next(iter(spd.values()))["cellid"][0][1])   # single well cell2d
        w = _tw(cell, layers)
        rate = {L: total * float(w[e]) for e, L in enumerate(layers)}
        for kper, rec in spd.items():
            for i in range(len(rec)):
                rec["q"][i] = rate[int(rec["cellid"][i][0])]
            wel.stress_period_data.set_data({kper: rec})
        wel.set_all_data_external()
        wel.write()
    print("  [apply_well_rates] transmissivity-weighted wellin/wellout rates written")


def process_spatial_snapshots(ws="."):
    """FORWARD-RUN post-command: full SO4 field (all cells x layers) at fixed snapshot times,
    written as observations so plan/xsection plots can be reconstructed for ANY realization from
    the prior obs ensemble -- no per-realization model re-run. Zero-weight (diagnostic).
    Self-contained for PstFrom injection.
    """
    import numpy as np
    import pandas as pd
    import flopy
    from pathlib import Path

    SNAP_TIMES = [252.0, 600.0, 900.0, 1300.0]
    SO4_GMOL = 96.06
    ws = Path(ws)
    ucn = flopy.utils.HeadFile(str(ws / "S.ucn"), text="CONCENTRATION")
    avail = np.array(ucn.get_times())
    rows = []
    for ts in SNAP_TIMES:
        tt = float(avail[np.argmin(np.abs(avail - ts))])
        arr = np.squeeze(ucn.get_data(totim=tt))          # (nlay, ncpl)
        for L in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                rows.append((int(ts), L, c, float(arr[L, c]) * SO4_GMOL))
    pd.DataFrame(rows, columns=["snap", "layer", "cell2d", "so4"]).to_csv(
        ws / "spatial_snapshots.csv", index=False)
    print(f"  [process_spatial_snapshots] {len(rows)} SO4 field obs at {SNAP_TIMES} d")
    return ws / "spatial_snapshots.csv"


def process_heads(ws="."):
    """FORWARD-RUN post-command: head timeseries at the monitoring locations (same obsids as the
    conditioning species). Conditioning data for K -- weighted in slice 4. Self-contained.
    Needs ``obs_loc.csv`` present in the run dir (shipped into the template).
    """
    import numpy as np
    import pandas as pd
    import flopy
    from pathlib import Path
    from flopy.utils.gridintersect import GridIntersect

    ws = Path(ws)
    sim = flopy.mf6.MFSimulation.load(sim_ws=str(ws), verbosity_level=0)
    gwf = sim.get_model("gwf")
    hds = gwf.output.head()
    times = np.array(hds.get_times())
    keep = np.arange(0, len(times), 5)                     # ~5-day cadence (heads saved daily)
    times = times[keep]
    all_h = np.squeeze(hds.get_alldata())[keep]            # (ntime, nlay, ncpl)
    ix = GridIntersect(gwf.modelgrid)
    loc = pd.read_csv(ws / "obs_loc.csv")
    rows = []
    for _, r in loc.iterrows():
        cell = int(ix.intersect([(r.x, r.y)], "point").cellids[0])
        L = int(r.layer)
        series = all_h[:, L, cell]
        for t, h in zip(times, series):
            rows.append((str(r.obsid), round(float(t), 3), float(h)))
    pd.DataFrame(rows, columns=["obsid", "time", "head"]).to_csv(ws / "heads.csv", index=False)
    print(f"  [process_heads] head timeseries at {len(loc)} monitoring points")
    return ws / "heads.csv"


def process_forecast(ws="."):
    """FORWARD-RUN post-command: recovered-SO4 forecast at ``wellout``.

    Reads ``sout.csv``, takes the flow-weighted (by actual recovery rate) mean SO4 across the
    wellout screens per reactive-output time over the recovery window, and writes
    ``forecast_so4.csv`` (time, so4_mean). Mirrors the verified ``_fig_breakthrough`` indexing.
    Self-contained for PstFrom injection. This is the recovered-quality forecast (ADR-0003).
    """
    import numpy as np
    import pandas as pd
    import flopy
    from pathlib import Path

    LAYERS_OUT = [1, 3, 5]
    SO4_GMOL = 96.06
    REC_START = 308.0

    ws = Path(ws)
    sim = flopy.mf6.MFSimulation.load(sim_ws=str(ws), verbosity_level=0)
    gwf = sim.get_model("gwf")
    ncpl = int(gwf.dis.ncpl.get_data())
    orec = gwf.get_package("welout").stress_period_data.get_data()[0]
    cell = int(orec["cellid"][0][1])
    qmap = {int(c[0]): abs(float(q)) for c, q in zip(orec["cellid"], orec["q"])}

    s = pd.read_csv(ws / "sout.csv")
    s["layer"] = s["cell"].astype(int) // ncpl
    s["cell2d"] = s["cell"].astype(int) % ncpl
    bt = s[(s.cell2d == cell) & (s.layer.isin(LAYERS_OUT))].copy()
    bt["so4"] = bt["SO4"] * SO4_GMOL * 1000.0
    piv = bt.pivot_table(index="time", columns="layer", values="so4")
    wv = np.array([qmap[l] for l in piv.columns], dtype=float)
    mean = (piv.values * wv).sum(axis=1) / wv.sum()
    out = pd.DataFrame({"time": piv.index, "so4_mean": mean})
    out = out[out["time"] >= REC_START].reset_index(drop=True)
    out.to_csv(ws / "forecast_so4.csv", index=False)
    print(f"  [process_forecast] recovered-SO4 forecast: {len(out)} times, "
          f"peak {out.so4_mean.max():.1f} mg/L")
    return ws / "forecast_so4.csv"


# =============================================================================
# (C) QA GATE
# =============================================================================

def _snapshot_welin(ws, dest=PRISTINE):
    """Save a pristine copy of the as-built wellin external files (for the regression)."""
    ws, dest = Path(ws), Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for f in glob.glob(str(ws / "gwf.welin_stress_period_data_*.txt")):
        shutil.copy(f, dest)
    if (ws / "gwf.welin").exists():
        shutil.copy(ws / "gwf.welin", dest)


def _welin_regression(ws, pristine=PRISTINE, atol=REG_ATOL):
    """f_treat=0 == baseline wellin-aux regression (numeric, all 19 columns incl. q).

    Snapshot the pristine as-built externals, run apply_treatment(f_treat=0), then assert
    every rewritten period file matches numerically (formatting may differ; values must not).
    Returns the max absolute column difference seen.
    """
    ws = Path(ws)
    _snapshot_welin(ws, pristine)
    apply_treatment(ws, 0.0)
    max_diff = 0.0
    files = glob.glob(str(ws / "gwf.welin_stress_period_data_*.txt"))
    qa_gate("regression: wellin period files present", len(files) > 0,
            f"found {len(files)} files under {ws}")
    for f in files:
        new = np.loadtxt(f, ndmin=2)
        old = np.loadtxt(Path(pristine, os.path.basename(f)), ndmin=2)
        qa_gate(f"regression: {os.path.basename(f)} shape", new.shape == old.shape,
                f"{new.shape} != {old.shape}")
        max_diff = max(max_diff, float(np.max(np.abs(new - old))))
    qa_gate("f_treat=0 reproduces baseline wellin aux",
            max_diff <= atol, f"max |diff| = {max_diff:.3g} > {atol:.1g}")
    return max_diff


_M0_GLOBS = ("kinetic_phases.*.m0.layer*.txt",
             "equilibrium_phases.*.m0.layer*.txt",
             "exchange_phases.*.m0.layer*.txt")


def _m0_guard(ws):
    """LANDMINE (ii) enforced in code: the PstFrom-parameterized m0 arrays must be
    byte-unchanged across a non-trivial ``apply_treatment``.

    Pyrite kinetics (``kinetic_phases.Pyrite.m0.layer*.txt``) is exactly PstFrom's pyrite
    multiplier target; the treatment preprocessor must never rewrite it. Snapshot the m0
    files, run ``apply_treatment(f_treat=0.9)``, assert every one is byte-identical. Leaves
    the wellin aux perturbed -- callers run the ``f_treat=0`` regression afterwards to
    restore the baseline aux.
    """
    ws = Path(ws)
    files = []
    for g in _M0_GLOBS:
        files += glob.glob(str(ws / g))
    qa_gate("m0 arrays present for guard", len(files) > 0,
            f"no m0.layer files under {ws} (build first)")
    before = {f: Path(f).read_bytes() for f in files}
    apply_treatment(ws, 0.9)          # non-trivial treatment
    changed = [os.path.basename(f) for f in files if Path(f).read_bytes() != before[f]]
    qa_gate("apply_treatment leaves K/porosity/pyrite m0 arrays byte-unchanged (LANDMINE ii)",
            not changed, f"clobbered: {changed[:6]}{' ...' if len(changed) > 6 else ''}")


def _percent_discrepancy(lst_path):
    """Last |PERCENT DISCREPANCY| from an mf6 listing file, or None if absent."""
    p = Path(lst_path)
    if not p.exists():
        return None
    vals = re.findall(r"PERCENT DISCREPANCY\s*=\s*([-\d.Ee+]+)", p.read_text())
    return abs(float(vals[-1])) if vals else None


def _run_mf6rtm(ws):
    import pyemu
    pyemu.os_utils.run("mf6rtm", cwd=str(ws))


def qa_gate_model(ws, run_model=False, mb_tol=MB_TOL):
    """Section-2 QA gate.

    Always: the f_treat=0 == baseline wellin-aux regression (offline, no model run).
    With ``run_model``: mf6 converges (Normal termination) + |mass-balance %| < tol for
    flow (gwf.lst) and every transport species (<comp>.lst).
    """
    ws = Path(ws)

    # (C1a) regression FIRST -- snapshots the TRUE as-built baseline aux (no treatment has run
    #       yet), applies f_treat=0, confirms it reproduces baseline; leaves aux at baseline.
    _welin_regression(ws)

    # (C1b) LANDMINE ii guard -- applies f_treat=0.9 and asserts the parameterized m0 arrays are
    #       byte-unchanged. This perturbs the wellin aux, so restore baseline afterwards.
    _m0_guard(ws)
    apply_treatment(ws, 0.0)   # restore baseline injectate for the downstream QA model run

    if not run_model:
        print("  [QA] model run skipped (offline build); convergence + mass balance "
              "gate deferred to the live ~6 min run (pass --run).")
        return

    # (C2) run + convergence
    _run_mf6rtm(ws)
    mfsim = (ws / "mfsim.lst")
    converged = mfsim.exists() and ("Normal termination of simulation" in mfsim.read_text())
    qa_gate("MF6 converges (normal termination)", converged, f"see {mfsim}")

    # (C3) mass balance -- flow + every transport species
    flow_pd = _percent_discrepancy(ws / "gwf.lst")
    qa_gate("flow mass balance readable", flow_pd is not None, "no PERCENT DISCREPANCY in gwf.lst")
    qa_close("flow |mass balance %|", flow_pd, 0.0, mb_tol, kind="abs")
    for lst in sorted(glob.glob(str(ws / "*.lst"))):
        base = os.path.basename(lst)
        if base in ("gwf.lst", "mfsim.lst"):
            continue
        pd_ = _percent_discrepancy(lst)
        if pd_ is None:
            continue
        qa_close(f"transport |mass balance %| ({base})", pd_, 0.0, mb_tol, kind="abs")


# =============================================================================
# (D) SIGNATURE FIGURE
# =============================================================================

def _wellout_cellid(gwf, data_d=DATA_D):
    from flopy.utils.gridintersect import GridIntersect
    wells = pd.read_csv(Path(data_d, "wells.csv"))
    x, y = wells.loc[wells.name == "wellout", ["x", "y"]].values[0]
    ix = GridIntersect(gwf.modelgrid)
    return int(ix.intersect([(x, y)], "point").cellids[0])


def _fig_breakthrough(ws, data_d=DATA_D):
    """Signature figure: recovered-SO4 breakthrough at wellout + a monitor + timeline.

    Reads ``sout.csv`` (per-cell, per-day species output). If it is absent (offline build),
    falls back to a timeline / window schematic so the stage still emits a figure.
    """
    import matplotlib.pyplot as plt
    import flopy

    ws = Path(ws)
    apply_style()
    sout_p = ws / "sout.csv"

    if not sout_p.exists():
        # fallback: timeline + window schematic (no model output yet)
        fig, ax = plt.subplots(figsize=(9, 2.6))
        ax.axvspan(0, DAY_COND_END, color=ROLE["history"], alpha=0.18,
                   label=f"conditioning (0-{DAY_COND_END} d)")
        ax.axvspan(DAY_COND_END, DAY_FORECAST_START, color=C["grey"], alpha=0.18,
                   label="lead-time gap")
        ax.axvspan(DAY_FORECAST_START, DAY_END, color=ROLE["forecast"], alpha=0.18,
                   label=f"recovery / forecast ({DAY_FORECAST_START}-{DAY_END:.0f} d)")
        ax.set_xlim(0, DAY_END)
        ax.set_yticks([])
        ax.set_xlabel(LBL["time"])
        ax.set_title("Two-well ASR timeline (model built; run for breakthrough)")
        ax.legend(loc="upper left", ncol=3)
        return savefig(fig, "timeline", stage=STAGE)

    gwf = flopy.mf6.MFSimulation.load(sim_ws=str(ws), verbosity_level=0).get_model("gwf")
    ncpl = int(gwf.dis.ncpl.get_data())
    cid = _wellout_cellid(gwf, data_d=data_d)

    sout = pd.read_csv(sout_p)
    sout["layer"] = sout["cell"].astype(int) // ncpl
    sout["cell2d"] = sout["cell"].astype(int) % ncpl
    bt = sout[(sout.cell2d == cid) & (sout.layer.isin(LAYERS_OUT))].copy()
    bt["so4_mgL"] = bt["SO4"] * SO4_GMOL * 1000.0

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    # recovered SO4 breakthrough at wellout, per screen + screen-weighted mean
    for lay in LAYERS_OUT:
        s = bt[bt.layer == lay].sort_values("time")
        ax0.plot(s.time, s.so4_mgL, alpha=0.6, label=f"wellout screen L{lay}")
    piv = bt.pivot_table(index="time", columns="layer", values="so4_mgL")
    # flow-weight by the ACTUAL wellout per-screen recovery rates (transmissivity-weighted)
    orec = gwf.get_package("welout").stress_period_data.get_data()[0]
    qmap = {int(c[0]): abs(float(q)) for c, q in zip(orec["cellid"], orec["q"])}
    w = np.array([qmap[l] for l in piv.columns], dtype=float)
    wmean = (piv.values * w).sum(axis=1) / w.sum()
    ax0.plot(piv.index, wmean, color=SPECIES["SO4"], lw=3,
             label="flow-weighted mean (objective)")
    ax0.set_ylabel(LBL["so4"])
    ax0.set_title("Recovered SO$_4$ at the recovery well (wellout)")
    ax0.legend(loc="upper left", fontsize=8)

    # timeline
    ax1.axvspan(0, DAY_COND_END, color=ROLE["history"], alpha=0.18)
    ax1.axvspan(DAY_COND_END, DAY_FORECAST_START, color=C["grey"], alpha=0.18)
    ax1.axvspan(DAY_FORECAST_START, DAY_END, color=ROLE["forecast"], alpha=0.18)
    ax1.set_yticks([])
    ax1.set_xlim(0, DAY_END)
    ax1.set_xlabel(LBL["time"])
    for x0, txt in [(120, "conditioning"), (280, "gap"), (520, "recovery/forecast")]:
        ax1.text(x0, 0.5, txt, ha="center", va="center", fontsize=9)
    return savefig(fig, "recovered_so4_breakthrough", stage=STAGE)


def _fig_layout(ws, data_d=DATA_D):
    """Signature figure (available pre-run): plan grid layout + boundary conditions.

    Domain mesh, CHD boundary cells (coloured by the imposed head gradient), wellin/wellout
    cells, well markers and the monitoring clusters (the conditioning-data locations).
    """
    import matplotlib.pyplot as plt
    import flopy
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    apply_style()
    ws = Path(ws)
    gwf = flopy.mf6.MFSimulation.load(sim_ws=str(ws), verbosity_level=0).get_model("gwf")

    fig, ax = plt.subplots(figsize=(11, 6.2))
    pmv = flopy.plot.PlotMapView(model=gwf, ax=ax, layer=0)
    pmv.plot_grid(lw=0.2, color="0.7", alpha=0.5)
    # CHD coloured by head (shows the left->right gradient)
    chd = gwf.get_package("chd")
    rec = chd.stress_period_data.get_data()[0]
    xc = np.asarray(gwf.modelgrid.xcellcenters).ravel()
    yc = np.asarray(gwf.modelgrid.ycellcenters).ravel()
    c2d = [cid[1] for cid in rec["cellid"]]
    sc = ax.scatter(xc[c2d], yc[c2d], c=rec["head"], cmap="coolwarm", s=18, zorder=4)
    cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.01)
    cb.set_label("CHD head (m)")
    pmv.plot_bc(package=gwf.get_package("welin"), color=C["red"])
    pmv.plot_bc(package=gwf.get_package("welout"), color=C["blue"])

    for nm, (x, y) in {"wellin": (1.0, 0.0), "wellout": (-99.0, 0.0)}.items():
        ax.scatter([x], [y], s=120, marker="v" if nm == "wellin" else "^",
                   facecolor=C["red"] if nm == "wellin" else C["blue"],
                   edgecolor="k", zorder=7)
        ax.annotate(nm, (x, y), textcoords="offset points", xytext=(6, 6),
                    fontsize=9, fontweight="bold", zorder=7)
    obs = pd.read_csv(Path(data_d, "obs_loc.csv"))
    obs["well"] = obs["obsid"].str.split("-").str[0]
    for wname, g in obs.groupby("well"):
        ax.scatter([g.x.iloc[0]], [g.y.iloc[0]], s=55, marker="s", facecolor="none",
                   edgecolor=C["green"], linewidths=1.6, zorder=8)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    handles = [Patch(fc=C["red"], label="wellin cells"),
               Patch(fc=C["blue"], label="wellout cells"),
               Line2D([0], [0], marker="s", mfc="none", mec=C["green"], ls="",
                      label="monitoring cluster")]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.set_title("Two-well ASR -- plan layout & BCs (CHD coloured by regional-gradient head)",
                 fontweight="bold")
    return savefig(fig, "setup_plan_layout", stage=STAGE)


# =============================================================================
# ORCHESTRATOR
# =============================================================================

def stage2_build(ws=WS, data_d=DATA_D, rebuild=False, run_model=False):
    """Section-2: build the two-well ASR model, gate it, emit the signature figure.

    Parameters
    ----------
    rebuild : bool
        Force a full rebuild even if ``ws`` already holds a model.
    run_model : bool
        Also execute the ~6 min full model and gate on convergence + mass balance.
        Default False (offline build + regression only); the live run is a human step.

    History/forecast split -- BUILD-DECISION (ADR-0003 open item resolved here):
      keep the 35-SP / 728-day tdis grid unchanged; conditioning window = SP0-18
      (days 0-252), lead-time gap = SP19-20 (252-308), recovery/forecast = SP21-34
      (308-728, == the retired supply slot). Pumping is continuous every SP; the split is
      only a time-window read of the wellout breakthrough curve.
    """
    apply_style()
    print(f"[stage2_build] ws={ws}  rebuild={rebuild}  run_model={run_model}")

    build_two_well_model(ws=ws, data_d=data_d, rebuild=rebuild)
    # PRE-RUN signature figure: layout + BCs need only the built structure, so emit it BEFORE
    # the (expensive) model run -- it lands immediately even if --run takes ~10 min or fails.
    _fig_layout(ws, data_d=data_d)
    qa_gate_model(ws, run_model=run_model)
    # POST-RUN signature figure: recovered-SO4 breakthrough needs the reactive output.
    fig_path = _fig_breakthrough(ws, data_d=data_d)

    print(f"[stage2_build] done. figure -> {fig_path}")
    return ws


# =============================================================================
# SECTION 3 -- PstFrom PEST interface (conditioning / prior-MC over the uncertain aquifer)
# =============================================================================
#
# Emulation-first (ADR-0002/0003): this FULL-MODEL interface is for history matching + the prior
# Monte Carlo that trains the DSI emulator. The decision lever f_treat and its treatment
# preprocessor live at the DSIVC stage (slice 8) ON THE EMULATOR, so they are NOT in this
# interface -- here f_treat == 0 (baseline injectate). The uncertain parameters are the aquifer
# properties (K, porosity, dispersivity, pyrite mass + rate); the transmissivity-weighted well
# rates are recomputed from the perturbed K each run by the apply_well_rates pre-command.

STAGE3 = "03_pstfrom"
WS3 = Path(__file__).parent / "_s3_template"       # PEST template (mothership, gitignored)
_S3_STAGE = Path(__file__).parent / "_s3_stage"    # staged copy of _s2_model PstFrom wraps
PP_SPACE = 10.0            # pilot-point spacing (m)
VARIO_RANGE = 100.0       # spherical variogram range (m), isotropic
N_REALS = 201             # PstFrom interface build placeholder for ies_num_reals (overridden at run time)
N_PRIOR_MC = 120          # PRODUCTION prior-MC ensemble size -- drives stage5, the paired sweep, and run_all
                          # (the DSI trains on it and the FOM/sweep reuse it, so all three must match)


def _add_array_prop(pf, ib, gs, tag, ws, bounds, base_fn, ult=None, second="constant",
                    tidy=True, ppu=False):
    """Add one array property as stacked multiplier pars (pilot points + lumped) + input obs.

    Mirrors the notebook recipe: a per-layer pilot-point field (pp_space 10 m) plus a lumped
    (constant/zone) multiplier on the same file, both par_style='m', same geostruct, plus an
    add_observations tracking the multiplied input array. `base_fn(f)` -> the per-layer par base.
    """
    import herebedragons as hbd
    files = hbd.get_input_filenames(tag, str(ws), extension=".txt")
    for f in files:
        if tidy:
            hbd.tidy_array(os.path.join(str(ws), f))
        base = base_fn(f)
        common = dict(zone_array=ib, geostruct=gs, par_name_base=base, pargp=base,
                      lower_bound=bounds[0], upper_bound=bounds[1])
        if ult is not None:
            common.update(ult_lbound=ult[0], ult_ubound=ult[1])
        ppopts = {"prep_hyperpars": False, "pp_space": PP_SPACE}
        if ppu:
            ppopts["try_use_ppu"] = True
        pf.add_parameters(f, par_type="pilotpoints", par_style="m", pp_options=ppopts, **common)
        pf.add_parameters(f, par_type=second, par_style="m", **common)
        pf.add_observations(f, prefix=base, obsgp=base)
    return files


def _inject_forward_run_header(fr_path):
    """Inject a header into the generated forward_run.py (before the first ``def`` -- after the imports,
    so os/pandas are available; before main() runs), doing two worker-hardening things:

    (1) Redirect fd 1 & 2 to ``forward_run.log`` so the ENTIRE forward run (every pre-command print, the
        model output, any uncaught traceback, subprocess output) is captured on the worker even when it
        dies in a pre-command before mf6rtm/mfsim.lst exist -- shipped back by panther_transfer_on_fail.
    (2) ``pandas.set_option('future.infer_string', False)`` BEFORE PstFrom's apply_list_and_array_pars
        runs: on a worker env with pyarrow present, pandas-3 infers Arrow-backed string columns and
        pyemu's ``.reshape()`` on them raises NotImplementedError. Forcing object strings (old behaviour)
        makes it env-independent -- works whether or not the slot's env has pyarrow. Idempotent."""
    import re
    fr_path = Path(fr_path)
    txt = fr_path.read_text()
    if "forward_run.log" in txt:
        return
    block = (
        "\n# --- worker hardening (injected by build_pest_interface) ---\n"
        "import os as _os\n"
        "_os.dup2(_os.open('forward_run.log', _os.O_WRONLY | _os.O_CREAT | _os.O_TRUNC, 0o644), 1)\n"
        "_os.dup2(1, 2)  # capture the whole forward run for panther_transfer_on_fail\n"
        "try:\n"
        "    import pandas as _pd; _pd.set_option('future.infer_string', False)  # no Arrow strings -> pyemu reshape ok\n"
        "except Exception:\n"
        "    pass\n\n"
    )
    m = re.search(r"^def ", txt, flags=re.M)
    if m:
        fr_path.write_text(txt[:m.start()] + block + txt[m.start():])


def build_pest_interface(model_ws=WS, template_ws=WS3, num_reals=N_REALS, with_treatment=False):
    """Section 3: build the PstFrom PEST interface over the two-well model.

    Returns the built ``pyemu.Pst``. Reuses the section-2 model in ``model_ws`` as the source.

    ``with_treatment=True`` (the DSIVC sweep interface, Section 7) adds the ``f_treat`` decision
    variable: a ``treatment.dat`` tpl parameter, the ``apply_treatment_forward`` PRE command, and
    ``f_treat``/``cost`` echoed obs. The default (prior MC / conditioning interface) leaves f_treat
    baked at 0 in the as-built wellin aux (no treatment step), so stage 5/6 stay clean.
    """
    import flopy
    import pyemu
    import herebedragons as hbd

    model_ws, template_ws = Path(model_ws), Path(template_ws)
    hbd_py = str(TUT / "herebedragons.py")
    wf_py = str(Path(__file__).resolve())

    # --- stage a fresh copy of the model for PstFrom to wrap ---------------------------------
    # NB: shutil.copytree can silently drop files on this large workspace -> use cp -R.
    import subprocess
    if _S3_STAGE.exists():
        shutil.rmtree(_S3_STAGE)
    subprocess.run(["cp", "-R", str(model_ws), str(_S3_STAGE)], check=True)
    hbd.get_bins(_S3_STAGE)
    sim = flopy.mf6.MFSimulation.load(sim_ws=str(_S3_STAGE), verbosity_level=0)
    gwf = sim.get_model("gwf")
    sr = gwf.modelgrid

    pf = pyemu.utils.PstFrom(original_d=str(_S3_STAGE), new_d=str(template_ws),
                             remove_existing=True, longnames=True,
                             spatial_reference=sr, zero_based=False, echo=False)
    ib = np.ones(sr.ncpl, dtype=int)
    gs = pyemu.geostats.GeoStruct(
        variograms=pyemu.geostats.SphVario(contribution=1.0, a=VARIO_RANGE,
                                           anisotropy=1, bearing=0),
        transform="log")

    # --- pre/post-processing functions injected into forward_run.py -------------------------
    pf.extra_py_imports.append("flopy")
    pf.extra_py_imports.append("shutil")
    for fn in ("tidy_array()", "get_input_filenames()", "extract_layer_number()",
               "node_to_layer_icell2d()", "time_interpolate()"):
        pf.add_py_function(hbd_py, fn, is_pre_cmd=None)            # helpers
    pf.add_py_function(wf_py, "omp_env_guard()", is_pre_cmd=True)      # PRE (first): KMP guard for mf6rtm
    pf.add_py_function(wf_py, "apply_well_rates()", is_pre_cmd=True)   # PRE: K -> T-weighted rates
    if with_treatment:                                                # PRE: f_treat -> treated wellin aux + cost
        pf.add_py_function(wf_py, "apply_treatment_forward()", is_pre_cmd=True)
    pf.add_py_function(hbd_py, "copy_parameterized_transport_files()", is_pre_cmd=True)  # PRE
    # capture the model stdout/stderr (os_utils.run uses os.system -> shell redirect works) so the
    # OMP/mf6rtm crash message survives on a worker and can be shipped back via panther_transfer_on_fail
    pf.mod_sys_cmds.append("mf6rtm > mf6rtm.stdout 2>&1")
    pf.add_py_function(hbd_py, "process_sim_conc()", is_pre_cmd=False)  # POST: conditioning obs
    pf.add_py_function(wf_py, "process_forecast()", is_pre_cmd=False)   # POST: recovered-SO4 forecast
    pf.add_py_function(wf_py, "process_heads()", is_pre_cmd=False)      # POST: monitoring head series
    pf.add_py_function(wf_py, "process_spatial_snapshots()", is_pre_cmd=False)  # POST: SO4 field snaps

    # --- observations: conditioning species + heads + forecast + spatial snapshots ----------
    shutil.copy(DATA_D / "obs_chem_cleaned.csv", template_ws / "obs_chem_cleaned.csv")
    shutil.copy(DATA_D / "obs_loc.csv", template_ws / "obs_loc.csv")   # process_heads needs it
    fname, _ = hbd.process_sim_conc(wd=str(template_ws))          # -> _obs.conc.simvsmeas.csv
    pf.add_observations(fname, index_cols=["time", "obsid", "variable"], use_cols="sim",
                        prefix="conc", obsgp="conc")
    hd = process_heads(ws=str(template_ws))                       # -> heads.csv (conditioning)
    pf.add_observations(os.path.basename(str(hd)), index_cols=["obsid", "time"], use_cols="head",
                        prefix="head", obsgp="head")
    fc = process_forecast(ws=str(template_ws))                    # -> forecast_so4.csv
    pf.add_observations(os.path.basename(str(fc)), index_cols=["time"], use_cols="so4_mean",
                        prefix="fore", obsgp="forecast")
    snp = process_spatial_snapshots(ws=str(template_ws))         # -> spatial_snapshots.csv (plotting)
    pf.add_observations(os.path.basename(str(snp)), index_cols=["snap", "layer", "cell2d"],
                        use_cols="so4", prefix="so4field", obsgp="so4_field")

    if with_treatment:                                           # DSIVC sweep: f_treat (decvar) + cost as obs
        (template_ws / "treatment.dat").write_text("0.0\n")      # baseline; tpl added post-build_pst
        # the worker runs the treatment by IMPORTING workflow -> ship the modules it needs (workflow +
        # wf_style + herebedragons) and every chem-data file build_injectate_solutions reads.
        wfdir = Path(__file__).resolve().parent
        shutil.copy(wfdir / "workflow.py", template_ws / "workflow.py")
        shutil.copy(wfdir / "wf_style.py", template_ws / "wf_style.py")
        shutil.copy(TUT / "herebedragons.py", template_ws / "herebedragons.py")
        for _f in ("ic_aq_chem.csv", "wellin.csv", "ic_exchanger.csv", "ic_surfaces.csv",
                   "postfix.phqr", "datab.dat"):
            shutil.copy(DATA_D / _f, template_ws / _f)
        _apply_treatment_forward_impl(str(template_ws), data_d=str(DATA_D))  # -> ftreat.csv + cost_obs.csv (f_treat=0)
        pf.add_observations("ftreat.csv", index_cols=["item"], use_cols="value",
                            prefix="ftreat", obsgp="ftreat")     # the controllable-obs decvar
        pf.add_observations("cost_obs.csv", index_cols=["item"], use_cols="value",
                            prefix="cost", obsgp="cost")         # exact cost (carried; excluded from DSI train)

    # --- parameters: uncertain aquifer properties -------------------------------------------
    _add_array_prop(pf, ib, gs, "npf_k_", template_ws, (0.001, 10.0),
                    base_fn=lambda f: f.split(".")[1].replace("_", ""), ppu=True)
    _add_array_prop(pf, ib, gs, "h2o.mst_porosity_", template_ws, (0.5, 1.5),
                    base_fn=lambda f: f.split(".")[1].replace("_", "."),
                    ult=(5e-2, 0.65))
    _add_array_prop(pf, ib, gs, "h2o.dsp_alh_", template_ws, (0.5, 10.0),
                    base_fn=lambda f: f.split(".")[1].replace("_", "."))
    _add_array_prop(pf, ib, gs, "kinetic_phases.pyrite.m0", template_ws, (0.05, 5.0),
                    base_fn=lambda f: f.split(".txt")[0].replace("_", ".").lower(),
                    ult=(1e-5, 10.0), second="zone", tidy=False)

    pst = pf.build_pst(str(template_ws / "pest.pst"), version=2)

    # --- worker hardening: full-run log capture + pandas Arrow-string workaround ------------
    # Workers can die in a PRE-command (before mf6rtm/mfsim.lst exist), so redirect the whole
    # forward_run.py to forward_run.log for panther_transfer_on_fail; and force pandas object strings
    # so PstFrom's apply_list_and_array_pars survives a slot env that has pyarrow (see the helper).
    _inject_forward_run_header(template_ws / "forward_run.py")

    # --- pyrite reaction rate: single global hand-templated par -----------------------------
    _add_pyrite_rate_par(pst, template_ws)

    # --- f_treat decision variable (DSIVC sweep interface only) ------------------------------
    if with_treatment:
        _add_treatment_par(pst, template_ws)

    # --- forecast group + pestpp options ----------------------------------------------------
    pst.try_parse_name_metadata()
    obs = pst.observation_data
    fore_names = obs.loc[obs.obgnme == "forecast", "obsnme"].tolist()
    pst.pestpp_options["forecasts"] = ",".join(fore_names)
    pst.pestpp_options["ies_num_reals"] = num_reals
    # PANTHER: on a FAILED run, ship these diagnostic files from the worker back to the master so a
    # remote/worker crash (e.g. the mf6rtm OMP abort) can be inspected without shell access to the slot.
    pst.pestpp_options["panther_transfer_on_fail"] = "forward_run.log,mf6rtm.stdout,mfsim.lst,gwf.lst"
    pst.control_data.noptmax = 0
    pst.write(str(template_ws / "pest.pst"), version=2)
    return pf, pst


def _add_pyrite_rate_par(pst, template_ws):
    """Hand-template a single global pyrite log-rate par over the 12 Pyrite -parms blocks in
    phinp.dat (parm(1) = log10 A/V). partrans='none' (already a log quantity)."""
    import re
    phinp = Path(template_ws, "phinp.dat")
    text = phinp.read_text()
    # each Pyrite kinetics block carries a `-parms <lograte> ...` line; replace the first value
    n = [0]

    def _sub(m):
        n[0] += 1
        return f"{m.group(1)}~  pyr-lograte  ~{m.group(3)}"

    tpl = re.sub(r"(Pyrite[\s\S]*?-parms\s+)([-\d.eE+]+)(\s)", _sub, text)
    qa_gate("pyrite rate: 12 blocks templated", n[0] == 12, f"templated {n[0]} (expected 12)")
    tpl_path = Path(template_ws, "phinp.dat.tpl")
    tpl_path.write_text("ptf ~\n" + tpl)
    pst.add_parameters(str(tpl_path), pst_path=".")
    par = pst.parameter_data
    par.loc["pyr-lograte", ["parval1", "parlbnd", "parubnd", "pargp", "partrans"]] = \
        [16.0, 15.0, 17.0, "pyr.rate", "none"]


def _fig_pstfrom(pst, template_ws=WS3, data_d=DATA_D):
    """Signature figure: pilot-point network + obs/forecast locations + parameter summary."""
    import matplotlib.pyplot as plt

    apply_style()
    par = pst.parameter_data
    pp = par[(par.x.notna()) & (par.pargp.str.contains("npfklayer1$", na=False))]  # one K layer
    obs_loc = pd.read_csv(Path(data_d, "obs_loc.csv"))
    obs_loc["well"] = obs_loc["obsid"].str.split("-").str[0]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5.2),
                                   gridspec_kw={"width_ratios": [3, 2]})
    # (a) pilot-point network + monitoring + wells
    ax0.scatter(pp.x.astype(float), pp.y.astype(float), s=8, color=C["sky"],
                label=f"pilot points ({len(pp)}/layer)")
    for wname, g in obs_loc.groupby("well"):
        ax0.scatter([g.x.iloc[0]], [g.y.iloc[0]], s=60, marker="s", facecolor="none",
                    edgecolor=C["green"], linewidths=1.6, zorder=6)
    for nm, (x, y) in {"wellin": (1.0, 0.0), "wellout": (-99.0, 0.0)}.items():
        ax0.scatter([x], [y], s=110, marker="v" if nm == "wellin" else "^",
                    facecolor=C["red"] if nm == "wellin" else C["blue"],
                    edgecolor="k", zorder=7)
        ax0.annotate(nm, (x, y), textcoords="offset points", xytext=(5, 5),
                     fontsize=8, fontweight="bold", zorder=7)
    ax0.scatter([], [], s=60, marker="s", facecolor="none", edgecolor=C["green"],
                label="monitoring (conditioning)")
    ax0.scatter([], [], marker="^", color=C["blue"], label="wellout (recovered-SO$_4$ forecast)")
    ax0.set_xlabel("x (m)")
    ax0.set_ylabel("y (m)")
    ax0.set_aspect("equal")
    ax0.legend(fontsize=8, loc="upper right")
    ax0.set_title("Pilot-point network + observation geometry", fontsize=12)

    # (b) parameter count by property tier
    def _tier(g):
        for key in ("npfk", "mst.porosity", "dsp.alh", "kinetic.phases.pyrite", "pyr.rate"):
            if key in g:
                return {"npfk": "K", "mst.porosity": "porosity", "dsp.alh": "dispersivity",
                        "kinetic.phases.pyrite": "pyrite m0", "pyr.rate": "pyrite rate"}[key]
        return "other"
    tiers = par.pargp.map(_tier).value_counts()
    ax1.barh(tiers.index[::-1], tiers.values[::-1], color=C["blue"])
    for i, v in enumerate(tiers.values[::-1]):
        ax1.text(v, i, f" {v}", va="center", fontsize=9)
    ax1.set_xlabel("adjustable parameters")
    ax1.set_title(f"Prior parameters by tier (npar={pst.npar})", fontsize=12)

    fig.suptitle("PstFrom interface -- parameterization & observations "
                 "(f_treat deferred to DSIVC/emulator)", fontweight="bold")
    return savefig(fig, "pstfrom_interface", STAGE3)


def qa_gate_phi(template_ws=WS3, tol=1.0):
    """EXPENSIVE gate (one base forward run): the interface must reproduce the model outputs at
    base parameters -> phi ~ 0. Run pestpp-ies noptmax=0 first; this reads the resulting phi."""
    import pyemu
    pst = pyemu.Pst(str(Path(template_ws) / "pest.pst"))
    qa_gate("interface reproduces model at base params (phi ~ 0)", pst.phi < tol,
            f"phi = {pst.phi:.3g} (>= {tol}); check phi_components")
    return pst.phi


def stage3_pstfrom(model_ws=WS, template_ws=WS3, num_reals=N_REALS):
    """Section-3 orchestrator: build the interface, gate it, emit signature figures."""
    apply_style()
    print(f"[stage3_pstfrom] model_ws={model_ws} -> template_ws={template_ws}")
    pf, pst = build_pest_interface(model_ws, template_ws, num_reals)
    qa_gate_interface(pst)
    _fig_pstfrom(pst, template_ws)
    print(f"[stage3_pstfrom] npar={pst.npar} nobs={pst.nobs} "
          f"forecasts={len(pst.pestpp_options.get('forecasts', '').split(','))}")
    print("  [stage3] next: run `pestpp-ies pest.pst` (noptmax=0) then qa_gate_phi() to verify phi~0")
    return pf, pst


def qa_gate_interface(pst):
    """Cheap interface-build gates (ADR-0004): finite phi structure, forecast present, f_treat
    absent (emulation-first), no NaN obs, sane parameter counts."""
    qa_gate("build_pst produced parameters", pst.npar > 0, f"npar={pst.npar}")
    qa_gate("build_pst produced observations", pst.nobs > 0, f"nobs={pst.nobs}")
    qa_gate("forecast group present", "forecast" in set(pst.observation_data.obgnme),
            "no obs in group 'forecast'")
    qa_gate("pyrite rate par present", "pyr-lograte" in pst.parameter_data.parnme.values,
            "pyr-lograte missing")
    qa_gate("no NaN in observation values",
            not pst.observation_data["obsval"].isna().any(), "NaN obsval present")
    qa_gate("no NaN in parameter values",
            not pst.parameter_data["parval1"].isna().any(), "NaN parval1 present")


# =============================================================================
# SECTION 5 -- prior Monte Carlo (evaluate the prior parameter ensemble)
# =============================================================================
#
# Draw a prior parameter ensemble from the PstFrom geostatistical prior, run each realization
# through the full model (PANTHER workers), and collect the recovered-SO4 forecast distribution.
# The synthetic truth (ADR-0001/0003) is then chosen from the UPPER QUARTILE of that distribution.

WS5_MASTER = Path(__file__).parent / "_s5_master"
_S5_WORKER_ROOT = Path(__file__).parent / "_s5_workers"


def draw_prior_ensemble(pf, num_reals, template_ws=WS3, seed=20260706):
    """Draw the prior parameter ensemble from the PstFrom prior + append the pyrite log-rate.

    Spatial pars (pilot points + lumped multipliers) are drawn from the geostatistical prior;
    ``pyr-lograte`` (hand-templated, not a PstFrom par) is drawn independently N(16, 0.5) clipped
    to [15, 17] and appended as a column. Enforced to bounds. Written to ``prior_pe.jcb``.
    """
    template_ws = Path(template_ws)
    if pf.pst.npar < 35000:
        pf.build_prior(fmt="coo", filename=str(template_ws / "prior_cov.jcb"))
    pe = pf.draw(num_reals=num_reals, use_specsim=False)
    pe.enforce()
    rng = np.random.default_rng(seed)
    pe.loc[:, "pyr-lograte"] = np.clip(rng.normal(16.0, 0.5, pe.shape[0]), 15.0, 17.0)
    pe.to_binary(str(template_ws / "prior_pe.jcb"))
    print(f"  [draw_prior_ensemble] {pe.shape[0]} reals x {pe.shape[1]} pars -> prior_pe.jcb")
    return pe


# =============================================================================
# FOM (full-model) pestpp deployment -- HTCondor when available, else local
# =============================================================================
# The expensive full-model pestpp ensembles (prior MC, DSIVC sweep) run their PANTHER worker agents
# either on an HTCondor pool (via the vendored ``condor_deploy``: env travels as a conda-pack tarball,
# the template as a zip -- no shared filesystem) or, when no pool is present, as LOCAL workers exactly
# as before. Auto-routing on ``condor_submit`` availability keeps local behaviour unchanged off-cluster.


# Default HTCondor slot request for the FOM ensembles, grounded in MEASURED single-run usage (2026-07-07):
# peak mf6rtm RSS ~1.8 GB -> 4 GB memory (~2.2x headroom); per-slot disk = unpacked conda-pack env (~1.6 GB)
# + template (~0.25 GB) + full reactive outputs (sout.csv ~1.6 GB + *.ucn ~0.37 GB + cbb/hds) ~2.1 GB +
# tarball/headroom -> 8 GB. Reactive transport is spiky + HTCondor evicts over-limit jobs, so err generous.
# 60 workers, 1 cpu each. Overridable per-run via condor_kwargs.
CONDOR_DEFAULTS = {"n_workers": 60, "memory_mb": 4096, "disk_mb": 8192, "cpus_per_worker": 1}


def _htcondor_available():
    """True iff an HTCondor pool is reachable (``condor_submit`` on PATH) AND condor_deploy imports."""
    import shutil as _sh
    if _sh.which("condor_submit") is None:
        return False
    try:
        import condor_deploy  # noqa: F401
    except ImportError:
        return False
    return True


def _configure_condor():
    """One-time condor_deploy project defaults: the four vendored deps travel as editable installs,
    the model/pestpp binaries get +x on each slot, bulky run outputs are excluded from the worker zip,
    and the pool is macOS -- the template ships the committed bin/mac binaries + the conda-pack env is
    built on the mac submit node, so slots must match (bin/linux is not populated)."""
    from condor_deploy import configure
    configure(
        worker_pip_editable=[str(DEPS / d) for d in ("flopy", "pyemu", "mf6rtm", "vorflow")],
        worker_chmod_exes=["mf6", "pestpp-ies", "pestpp-mou", "libmf6.dylib", "mf6rtm", "gridgen"],
        zip_exclude=["*.ucn", "*.hds", "*.cbb", "*.cbc", "sout.csv", "*.lst", "*.list"],
        platform_requirements='( (OpSys == "MacOS") )',
        worker_python_version="3.12",
    )


def _ensure_env_zip():
    """Resolve the worker-env conda-pack tarball, BUILDING it once if absent (automatic --build-env),
    reusing it if present. Path = ``$CONDOR_ENV_ZIP`` or ``REPO/worker_env.tar.gz``. The build
    (force_recreate=False) uses the packages + vendored editable deps set by _configure_condor, and
    needs conda/mamba + conda-pack on the submit node."""
    import os
    from condor_deploy import build_worker_env_zip
    env_zip = Path(os.environ.get("CONDOR_ENV_ZIP") or (REPO / "worker_env.tar.gz"))
    if env_zip.exists():
        print(f"  [deploy] reusing worker env zip: {env_zip}")
    else:
        print(f"  [deploy] worker env zip absent -> auto --build-env (once) -> {env_zip}")
        build_worker_env_zip(str(env_zip), force_recreate=False)
    return env_zip


def _slim_template(ws):
    """Delete regenerated model OUTPUTS from a template before a worker run -- workers recreate them,
    so PANTHER/Condor copy a small template instead of the multi-GB reactive outputs (disk headroom).
    Keeps every input (.txt/.tpl/.ins/.dat/.jcb/.csv-except-sout/.pst/.phqr + binaries)."""
    ws = Path(ws)
    for f in ("sout.csv", "gwf.cbb", "gwf.lst", "mfsim.lst"):
        (ws / f).unlink(missing_ok=True)
    for pat in ("*.ucn", "*.hds", "*.lst", "*.list"):
        for p in ws.glob(pat):
            p.unlink(missing_ok=True)


def _deploy_pestpp(template_ws, pst_name, master_dir, num_workers, worker_root,
                   condor_kwargs=None, pestpp_exe="pestpp-ies"):
    """Run any pestpp-* ensemble over workers: HTCondor pool when available, else local PANTHER.

    Used for the expensive full-model ensembles (pestpp-ies: prior MC, DSIVC sweep, FOM history match)
    AND the DSIVC outer optimization (pestpp-mou) -- pass the executable via ``pestpp_exe``.
    ``condor_kwargs`` forwards to condor_deploy (e.g. ``env_zip``, ``memory_mb``, ``cpus_per_worker``,
    ``n_workers``, ``force_local``). ``env_zip`` defaults to ``$CONDOR_ENV_ZIP`` or an auto-built
    conda-pack tarball. The pestpp master always runs locally; only the worker agents deploy.
    """
    import os
    template_ws, master_dir, worker_root = Path(template_ws), Path(master_dir), Path(worker_root)
    if _htcondor_available():
        from condor_deploy import submit_condor_workers_from_env
        _configure_condor()
        kw = dict(CONDOR_DEFAULTS)                          # n_workers=60, memory/disk/cpus from measured usage
        kw["pestpp_exe"] = pestpp_exe
        kw.update(condor_kwargs or {})                     # per-run overrides win
        if not kw.get("env_zip"):                          # automatic --build-env if the zip is absent
            kw["env_zip"] = str(_ensure_env_zip())
        print(f"  [deploy] HTCondor pool detected -> submitting {kw['n_workers']} {pestpp_exe} worker "
              f"agents (env_zip={kw['env_zip']})")
        submit_condor_workers_from_env(template_dir=str(template_ws), pst_name=pst_name,
                                       master_dir=str(master_dir), **kw)
    else:
        import pyemu
        if worker_root.exists():
            shutil.rmtree(worker_root)
        worker_root.mkdir(parents=True)
        print(f"  [deploy] no HTCondor pool -> {num_workers} local {pestpp_exe} workers")
        pyemu.os_utils.start_workers(str(template_ws), pestpp_exe, pst_name,
                                     num_workers=num_workers, master_dir=str(master_dir),
                                     worker_root=str(worker_root))
    return master_dir


def run_prior_mc(template_ws=WS3, num_workers=15, master_dir=WS5_MASTER, num_reals=None,
                 condor_kwargs=None):
    """Run the prior ensemble once (pestpp-ies noptmax=-1) over PANTHER workers.

    ``num_reals`` sizes the ensemble evaluated (default: use every realization in prior_pe.jcb).
    It is INDEPENDENT of ``num_workers`` (the PANTHER worker count) -- 15 workers can chew through
    120 reals. Passing num_workers here would silently truncate the ensemble (historic bug).
    """
    import pyemu

    template_ws, master_dir = Path(template_ws), Path(master_dir)
    pst = pyemu.Pst(str(template_ws / "pest.pst"))
    pst.pestpp_options["ies_par_en"] = "prior_pe.jcb"
    if num_reals is None:                              # default: run the full drawn ensemble
        pe = pyemu.ParameterEnsemble.from_binary(pst=pst, filename=str(template_ws / "prior_pe.jcb"))
        num_reals = pe.shape[0]
    pst.pestpp_options["ies_num_reals"] = num_reals
    pst.pestpp_options["ies_no_noise"] = True          # prior MC: no obs-noise realizations
    pst.pestpp_options["save_binary"] = True           # write obs/par ensembles as .jcb
    pst.control_data.noptmax = -1                       # evaluate prior ensemble only, no update
    pst.write(str(template_ws / "pest.pst"), version=2)
    _slim_template(template_ws)                          # drop regenerated outputs -> disk headroom
    return _deploy_pestpp(template_ws, "pest.pst", master_dir, num_workers, _S5_WORKER_ROOT,
                              condor_kwargs=condor_kwargs)


def prior_forecast(master_dir=WS5_MASTER, template_ws=WS3):
    """Peak recovered-SO4 per realization from the prior obs ensemble (forecast group).

    Returns a DataFrame (realization index) with the peak recovered-SO4 (the ADR-0003 conditioning
    forecast) plus the full forecast obs ensemble for spaghetti plotting.
    """
    import pyemu

    master_dir, template_ws = Path(master_dir), Path(template_ws)
    pst = pyemu.Pst(str(template_ws / "pest.pst"))
    jcb, csv = master_dir / "pest.0.obs.jcb", master_dir / "pest.0.obs.csv"
    oe = (pyemu.ObservationEnsemble.from_binary(pst=pst, filename=str(jcb)) if jcb.exists()
          else pyemu.ObservationEnsemble.from_csv(pst=pst, filename=str(csv)))
    obs = pst.observation_data
    ff = obs.loc[obs.obgnme == "forecast"].copy()
    ff["time"] = ff["time"].astype(float)                        # NUMERIC sort (was lexicographic)
    ff = ff.sort_values("time")
    fore = oe.loc[:, ff.obsnme.values]                           # (nreal, ntime), time-ordered
    times = ff["time"].values
    peak = fore.max(axis=1)
    return peak, fore, times


# --- Section 4: observations / weights / truth --------------------------------------------
# Per-species measurement sigma (in the observation units: mol/L for SO4/O0/NO3, pH units, Tmp
# stored x1e-3). Raw measurement sigma sets BOTH the weight (1/sigma) and the pestpp obs-noise
# (no sqrt(n) deflation) -- see the noise-weights-separate design note. PLACEHOLDER magnitudes,
# tune against real analytical precision at re-bake.
SPECIES_SIGMA = {"so4": 2.0e-5, "o0": 3.0e-6, "no3": 5.0e-6, "ph": 0.1, "tmp": 3.0e-4}
COND_SPECIES = list(SPECIES_SIGMA)              # conditioning species (cations held back)
HEAD_SIGMA = 0.05                               # m -- head measurement sigma (weight = 1/sigma)
MONITOR_PREFIX = ("wp", "pp", "ip")             # monitoring wells (exclude welopt / forecast)
# per-time measurement sigma model (noise-weights-separate memo): concentrations = 7% proportional
# + per-species floor; pH/Tmp/head absolute. Raw sigma sets BOTH weight (1/sigma) and noise.
SIGMA_FLOOR = {"so4": 2.0e-5, "o0": 3.0e-6, "no3": 5.0e-6}
SIGMA_ABS = {"ph": 0.1, "tmp": 3.0e-4, "head": HEAD_SIGMA}
PROP_SIGMA = 0.07
N_NOISE = 20              # noise realizations drawn for display (correlated-noise spaghetti)


def obs_sigma(vals, kind):
    """Per-time measurement sigma for a series. Concentrations: max(7%*|val|, floor); else absolute."""
    vals = np.abs(np.asarray(vals, dtype=float))
    if kind in SIGMA_FLOOR:
        return np.maximum(PROP_SIGMA * vals, SIGMA_FLOOR[kind])
    return np.full(vals.shape, SIGMA_ABS[kind])


def draw_obs_noise(truth_vals, kind, n_noise, rng):
    """`n_noise` TEMPORALLY-CORRELATED noise realizations of a site:species series.

    One correlated shock ``z ~ N(0,1)`` per series per realization (an instrument bias), applied
    across the whole series scaled per-time by sigma: ``noisy_t = truth_t + z * sigma_t``.
    Concentrations clipped >= 0. Returns (n_noise, ntime).
    """
    truth_vals = np.asarray(truth_vals, dtype=float)
    sig = obs_sigma(truth_vals, kind)
    z = rng.standard_normal(n_noise)[:, None]
    noisy = truth_vals[None, :] + z * sig[None, :]
    if kind in SIGMA_FLOOR:
        noisy = np.clip(noisy, 0.0, None)
    return noisy


def inject_truth_and_weights(template_ws=WS3, truth_dir=None, sigma=None):
    """Section 4: make the truth the data + weight the conditioning obs (vectorized).

    (1) Inject the synthetic-truth realization's simulated values as the observed values
        (``obsval``) for every obs, so the truth IS the reality being conditioned on.
    (2) Zero all weights, then weight only the CONDITIONING species (SO4/O2/NO3/pH/Tmp) at the
        monitoring wells, in the HISTORY window (t <= DAY_COND_END), at the real measurement
        times, with weight = 1/sigma_species. Cations are held back (dataworth, later); the
        recovered-SO4 forecast stays zero-weight.
    """
    import pyemu

    template_ws = Path(template_ws)
    truth_dir = Path(truth_dir) if truth_dir else (Path(__file__).parent / "_truth")
    sigma = sigma or SPECIES_SIGMA
    pst = pyemu.Pst(str(template_ws / "pest.pst"))
    pst.try_parse_name_metadata()
    obs = pst.observation_data

    # (1) truth -> obsval (vectorized map by obsnme)
    truth = pd.read_csv(truth_dir / "truth_obs.csv", index_col=0)["obsval"]
    obs["obsval"] = obs["obsnme"].map(truth).fillna(obs["obsval"])

    # (2) weights. Real-measurement (obsid, variable, time) triples from the sim-vs-meas table.
    svm = pd.read_csv(template_ws / "_obs.conc.simvsmeas.csv")
    meas_ok = svm[np.abs(svm["meas"].astype(float)) < 1e29].copy()
    meas_ok["key"] = (meas_ok["obsid"].astype(str) + "|" + meas_ok["variable"].str.lower()
                      + "|" + meas_ok["time"].round(3).astype(str))
    meas_keys = set(meas_ok["key"])

    obs["weight"] = 0.0
    key = (obs["obsid"].astype(str) + "|" + obs["variable"].astype(str).str.lower()
           + "|" + obs["time"].astype(float).round(3).astype(str))
    is_monitor = obs["obsid"].astype(str).str.startswith(MONITOR_PREFIX)
    in_history = obs["time"].astype(float) <= DAY_COND_END
    for sp, sig in sigma.items():
        m = (obs["variable"].astype(str).str.lower() == sp) & is_monitor & in_history \
            & key.isin(meas_keys)
        obs.loc[m, "weight"] = 1.0 / sig
    # heads: monitoring points, history window, ~monthly sampling (avoid over-conditioning)
    monthly = (obs["time"].astype(float) % 28.0) < 5.0
    mh = (obs["obgnme"] == "head") & is_monitor & in_history & monthly
    obs.loc[mh, "weight"] = 1.0 / HEAD_SIGMA
    pst.write(str(template_ws / "pest.pst"), version=2)
    return pst


def stage4_truth_weights(master_dir=WS5_MASTER, template_ws=WS3, quantile=0.75):
    """Section-4 orchestrator: lock the synthetic truth (the realization nearest ``quantile`` of the
    prior-MC recovered-SO4 forecast) as the measured data + inject proportional 1/sigma weights on the
    conditioning obs, then emit the obs/weights figure. Runs AFTER the prior MC (needs its forecast) and
    BEFORE stage 6 (the DSI keepobs + conditioning depend on these weights). Returns the truth name."""
    apply_style()
    peak, fore, times = prior_forecast(master_dir, template_ws)
    tname, tpeak, q = pick_truth(peak, quantile=quantile)
    lock_truth(tname, master_dir=master_dir, template_ws=template_ws)
    pst = inject_truth_and_weights(template_ws)
    try:
        _fig_weights(pst)
    except Exception as exc:                              # no-stops: a fig hiccup must not abort the arc
        print(f"  [stage4] weights fig skipped: {exc}")
    print(f"[stage4] truth=r{tname} peak={tpeak:.1f} (P{int(round(quantile * 100))}={q:.1f}) mg/L; "
          f"weights on {int((pst.observation_data.weight.astype(float) > 0).sum())} conditioning obs")
    return tname


def qa_gate_weights(pst):
    """Slice-4 gates (ADR-0004): weights land only on conditioning species in the history window;
    forecast stays zero-weight; some obs are actually weighted."""
    obs = pst.observation_data
    nz = obs[obs.weight > 0]
    qa_gate("some conditioning obs are weighted", len(nz) > 0, "no non-zero weights")
    weighted_kinds = set(nz[nz.obgnme != "head"].variable.str.lower())
    qa_gate("weighted obs are conditioning species or heads only",
            weighted_kinds <= set(COND_SPECIES),
            f"unexpected: {weighted_kinds - set(COND_SPECIES)}")
    qa_gate("weighted obs are within the history window",
            (nz.time.astype(float) <= DAY_COND_END).all(), "weight past history window")
    qa_gate("forecast stays zero-weight",
            (obs.loc[obs.obgnme == 'forecast', 'weight'] == 0).all(), "forecast weighted")
    return nz


def _fig_weights(pst, truth_dir=None):
    """Signature figure: weighted conditioning data (the truth) as timeseries at each monitoring
    LOCATION, with the measurement-noise band -- one panel per conditioning species + heads."""
    import numpy as np
    import matplotlib.pyplot as plt

    apply_style()
    obs = pst.observation_data
    nz = obs[obs.weight > 0].copy()
    nz["time"] = nz["time"].astype(float)
    nz["obsval"] = nz["obsval"].astype(float)
    nz["well"] = nz["obsid"].astype(str).str.split("-").str[0]      # cluster (wp1, pp1, ...)
    wells = sorted(nz["well"].unique())
    cmap = plt.get_cmap("tab10")
    wcol = {w: cmap(i % 10) for i, w in enumerate(wells)}

    panels = [(sp, sp.upper(), SPECIES_SIGMA[sp], "variable", sp) for sp in COND_SPECIES]
    panels.append(("head", "HEAD (m)", HEAD_SIGMA, "obgnme", "head"))

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), constrained_layout=True)
    for ax, (key, label, sig, col, val) in zip(axes.flat, panels):
        sub = nz[nz[col].astype(str).str.lower() == val] if col == "variable" \
            else nz[nz["obgnme"] == "head"]
        seen = set()
        rng = np.random.default_rng(20260706)
        for oid, d in sub.groupby("obsid"):        # per SCREEN line (no cross-screen zigzag)
            d = d.sort_values("time")
            wl = str(oid).split("-")[0]
            lab = wl if wl not in seen else None
            seen.add(wl)
            tv = d.obsval.astype(float).values
            for nr in draw_obs_noise(tv, key, N_NOISE, rng):     # correlated-noise spaghetti
                ax.plot(d.time, nr, color=ROLE["noise"], alpha=0.45, lw=0.6)
            ax.plot(d.time, tv, color=wcol[wl], lw=1.0, marker="o", ms=2.5, label=lab)
        ax.axvspan(0, DAY_COND_END, color=ROLE["history"], alpha=0.08)
        ax.set_title(f"{label}  (w={1/sig:g}, n={len(sub)})", fontsize=10)
        ax.set_xlabel(LBL["time"])
    axes.flat[0].legend(fontsize=7, ncol=2, title="location", title_fontsize=7, loc="best")
    fig.suptitle("Conditioning data = synthetic truth (P75) by monitoring location + measurement "
                 "noise (history window; species + heads; cations held back)", fontweight="bold")
    return savefig(fig, "obs_weights_truth", "04_obs_weights")


def _fig_prior_vs_data(master_dir=WS5_MASTER, template_ws=WS3, truth_real="5"):
    """Grid figure: rows = monitoring sites, cols = obs types (species + head). Each panel shows
    the PRIOR ensemble simulated timeseries (spaghetti), the MEASURED conditioning data (the
    truth) with its measurement-NOISE band, at that site. Diagnoses prior coverage of the data."""
    import numpy as np
    import pyemu
    import matplotlib.pyplot as plt

    apply_style()
    master_dir, template_ws = Path(master_dir), Path(template_ws)
    pst = pyemu.Pst(str(template_ws / "pest.pst"))
    pst.try_parse_name_metadata()
    jcb = master_dir / "pest.0.obs.jcb"
    oe = (pyemu.ObservationEnsemble.from_binary(pst=pst, filename=str(jcb)) if jcb.exists()
          else pyemu.ObservationEnsemble.from_csv(pst=pst, filename=str(master_dir / "pest.0.obs.csv")))
    oe_df = pd.DataFrame(oe.values, index=oe.index.astype(str), columns=oe.columns)
    obs = pst.observation_data.copy()
    obs["time"] = obs["time"].astype(float)

    # one representative (mid-depth) screen per cluster
    sites = {}
    mon = obs[obs.obsid.astype(str).str.startswith(MONITOR_PREFIX)]
    for wl, g in mon.groupby(mon.obsid.astype(str).str.split("-").str[0]):
        scr = sorted(g.obsid.unique())
        sites[wl] = scr[len(scr) // 2]
    site_ids = [sites[k] for k in sorted(sites)]
    types = [("so4", "SO$_4$", SPECIES_SIGMA["so4"]), ("o0", "O$_2$", SPECIES_SIGMA["o0"]),
             ("no3", "NO$_3$", SPECIES_SIGMA["no3"]), ("ph", "pH", SPECIES_SIGMA["ph"]),
             ("tmp", "Tmp", SPECIES_SIGMA["tmp"]), ("head", "head", HEAD_SIGMA)]

    fig, axes = plt.subplots(len(site_ids), len(types),
                             figsize=(2.4 * len(types), 1.9 * len(site_ids)),
                             sharex=True)
    for i, sid in enumerate(site_ids):
        for j, (tp, tlab, sig) in enumerate(types):
            ax = axes[i, j]
            if tp == "head":
                sel = obs[(obs.obgnme == "head") & (obs.obsid == sid)].sort_values("time")
            else:
                sel = obs[(obs.obgnme == "conc") & (obs.obsid == sid)
                          & (obs.variable.str.lower() == tp)].sort_values("time")
            if len(sel) == 0:
                ax.set_axis_off()
                continue
            t = sel["time"].values
            cols = sel["obsnme"].values
            prior = oe_df[cols].values                      # (nreal, ntime)
            for r in range(prior.shape[0]):
                ax.plot(t, prior[r], color=ROLE["prior"], alpha=0.55, lw=0.9)
            ax.plot(t, oe_df.loc[str(truth_real), cols].values, color=ROLE["truth"], lw=1.3)
            wsel = sel[sel.weight > 0].sort_values("time")
            if len(wsel):
                tw = wsel["time"].values
                yw = wsel["obsval"].astype(float).values
                rng = np.random.default_rng(20260706)
                for nr in draw_obs_noise(yw, tp, N_NOISE, rng):   # correlated-noise spaghetti
                    ax.plot(tw, nr, color=ROLE["noise"], alpha=0.5, lw=0.5)
                ax.plot(tw, yw, color=ROLE["truth"], marker="o", ms=2, ls="", zorder=5)
            ax.axvline(DAY_COND_END, color=C["black"], ls=":", lw=0.6, alpha=0.5)
            if i == 0:
                ax.set_title(tlab, fontsize=10)
            if j == 0:
                ax.set_ylabel(sid, fontsize=8)
            ax.tick_params(labelsize=6)
    for ax in axes[-1]:
        ax.set_xlabel("d", fontsize=7)
    handles = [plt.Line2D([0], [0], color=ROLE["prior"], alpha=0.5, label="prior sim reals"),
               plt.Line2D([0], [0], color=ROLE["truth"], label="truth"),
               plt.Line2D([0], [0], color=ROLE["truth"], marker="o", ls="", label="measured"),
               plt.Line2D([0], [0], color=ROLE["noise"], alpha=0.7, lw=1.0,
                          label="correlated-noise reals")]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Prior simulated vs measured (truth) at monitoring sites -- species + head "
                 "(dotted = end of history window)", fontweight="bold", y=1.015)
    fig.tight_layout()
    return savefig(fig, "prior_vs_data", "05_prior_mc")


def pick_truth(peak, quantile=0.75):
    """Synthetic truth = the realization whose peak recovered-SO4 is nearest the given quantile
    of the prior forecast distribution (ADR-0003: upper quartile). Returns (real_name, peak)."""
    import numpy as np
    peak = peak.astype(float)
    q = np.percentile(peak.values, quantile * 100)
    i = int(np.argmin(np.abs(peak.values - q)))
    return peak.index[i], float(peak.values[i]), float(q)


def lock_truth(truth_name, master_dir=WS5_MASTER, template_ws=WS3, out=None):
    """Persist the chosen synthetic-truth realization (ADR-0001/0003).

    Extracts the truth realization's parameter vector (prior_pe) and its full simulated
    observation vector (prior obs ensemble). The simulated conditioning species become the
    'measured' history-match data in slice 4 (obs/weights/truth); the recovered-SO4 obs is the
    truth forecast to recover. Writes truth_pars.csv + truth_obs.csv + truth_meta.txt.
    """
    import pyemu

    master_dir, template_ws = Path(master_dir), Path(template_ws)
    out = Path(out) if out else (Path(__file__).parent / "_truth")
    out.mkdir(exist_ok=True)
    pst = pyemu.Pst(str(template_ws / "pest.pst"))

    pe = pyemu.ParameterEnsemble.from_csv(pst=pst, filename=str(master_dir / "pest.0.par.csv")) \
        if (master_dir / "pest.0.par.csv").exists() else \
        pyemu.ParameterEnsemble.from_binary(pst=pst, filename=str(template_ws / "prior_pe.jcb"))
    oe = pyemu.ObservationEnsemble.from_csv(pst=pst, filename=str(master_dir / "pest.0.obs.csv")) \
        if (master_dir / "pest.0.obs.csv").exists() else \
        pyemu.ObservationEnsemble.from_binary(pst=pst, filename=str(master_dir / "pest.0.obs.jcb"))

    # plain pandas preserving the string realization index ('5' etc.) -- pd.DataFrame(oe) would
    # reset it to an integer RangeIndex.
    pe_df = pd.DataFrame(pe.values, index=pe.index.astype(str), columns=pe.columns)
    oe_df = pd.DataFrame(oe.values, index=oe.index.astype(str), columns=oe.columns)
    tname = str(truth_name)
    tpar = pe_df.loc[tname]
    tobs = oe_df.loc[tname]
    tpar.to_frame("parval").to_csv(out / "truth_pars.csv")
    tobs.to_frame("obsval").to_csv(out / "truth_obs.csv")

    obs = pst.observation_data
    fore = obs.loc[obs.obgnme == "forecast", "obsnme"].values
    peak = float(tobs.loc[fore].astype(float).max())
    (out / "truth_meta.txt").write_text(
        f"truth_realization={tname}\npeak_recovered_so4_mgL={peak:.4f}\n"
        f"n_pars={len(tpar)}\nn_obs={len(tobs)}\n")
    print(f"  [lock_truth] truth realization '{tname}' locked: peak recovered-SO4 {peak:.1f} mg/L "
          f"-> {out}")
    return tname, peak


def _fig_prior_mc(peak, fore, times, truth_name=None):
    """Signature figure: prior forecast peak distribution (hist+CDF) + recovered-SO4 spaghetti,
    with the synthetic-truth realization highlighted."""
    import numpy as np
    import matplotlib.pyplot as plt

    apply_style()
    peak = peak.astype(float)
    if truth_name is None:
        truth_name, _, _ = pick_truth(peak)
    p75 = np.percentile(peak.values, 75)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))
    # (a) peak distribution: hist + CDF + truth
    ax0.hist(peak.values, bins=8, color=ROLE["prior"], alpha=0.6, edgecolor="w")
    ax0.axvline(p75, color=C["black"], ls="--", lw=1.2, label=f"P75 = {p75:.1f}")
    ax0.axvline(peak[truth_name], color=ROLE["truth"], lw=2.5,
                label=f"truth ({peak[truth_name]:.1f})")
    ax0.set_xlabel("peak recovered SO$_4$ (mg/L)")
    ax0.set_ylabel("realizations")
    ax0.set_title(f"Prior forecast distribution (n={len(peak)})", fontsize=12)
    ax0.legend(fontsize=8)

    # (b) spaghetti recovered-SO4 breakthrough + truth
    for r in fore.index:
        ax1.plot(times, fore.loc[r].values, color=ROLE["prior"], alpha=0.55, lw=1.1)
    ax1.plot(times, fore.loc[truth_name].values, color=ROLE["truth"], lw=2.5, label="truth")
    ax1.plot([], [], color=ROLE["prior"], alpha=0.5, label="prior realizations")
    ax1.set_xlabel(LBL["time"])
    ax1.set_ylabel(LBL["so4"])
    ax1.set_title("Recovered-SO$_4$ breakthrough (prior spaghetti)", fontsize=12)
    ax1.legend(fontsize=8, loc="upper right")

    fig.suptitle("Prior Monte Carlo -- recovered-SO$_4$ forecast + synthetic truth (P75)",
                 fontweight="bold")
    return savefig(fig, "prior_mc_forecast", "05_prior_mc")


def stage5_prior_mc(num_reals=None, num_workers=None, condor_kwargs=None):
    """Section-5 orchestrator: draw + run the prior MC, summarize the forecast distribution."""
    import os
    apply_style()
    nw = num_workers or (os.cpu_count() - 1)
    nr = num_reals if num_reals is not None else N_PRIOR_MC   # production default (not the worker count)
    print(f"[stage5_prior_mc] num_reals={nr} num_workers={nw}")
    pf, pst = build_pest_interface()                    # rebuild to obtain pf for the draw
    draw_prior_ensemble(pf, nr)
    run_prior_mc(num_workers=nw, num_reals=nr, condor_kwargs=condor_kwargs)  # nr reals across nw workers
    peak, fore, times = prior_forecast()
    print(f"  [stage5] prior peak recovered-SO4: "
          f"P25={np.percentile(peak, 25):.1f} P50={np.percentile(peak, 50):.1f} "
          f"P75={np.percentile(peak, 75):.1f} mg/L (n={len(peak)})")
    return peak, fore, times


# =============================================================================
# SECTION 6 -- DSI emulator (train on the prior MC, cross-validate, then condition)
# =============================================================================
#
# Data Space Inversion: learn the joint prior distribution of the observables directly from the
# prior MC obs ensemble (a normal-score + SVD emulator), then condition it on the data with IES
# in SECONDS (no full model in the loop). This section covers the PLUMBING: build + fit + CROSS-
# VALIDATE the emulator. The conditioning run (ies_* options) is wired after the user signs off.
# (Full-model-IES comparison is intentionally SKIPPED per scope.)

WS6_DSI = Path(__file__).parent / "_s6_dsi_template"
DSI_ENERGY = 1.0                                  # SVD energy threshold (1.0 = full rank, no truncation)
DSI_TRANSFORMS = [{"type": "normal_score", "quadratic_extrapolation": True}]


def _dsi_keepobs(pst):
    """Obs columns the DSI emulator is trained on: the WEIGHTED conditioning obs (species + heads,
    at the real measurement times) + the recovered-SO4 forecast. Keeping only the weighted points
    (not every conc timestep) keeps the normal-score fit + SVD fast and is exactly what conditioning
    + forecast-prediction need. Excludes the array-tracking / spatial bookkeeping obs."""
    obs = pst.observation_data
    keep = obs.index[(obs.weight.astype(float) > 0) | (obs.obgnme == "forecast")]
    return list(keep)


def build_dsi_emulator(master_dir=WS5_MASTER, template_ws=WS3, energy_threshold=DSI_ENERGY,
                       truth_real="5", n_val=None, seed=0, data_cache=None, pst_cache=None):
    """Build + fit the DSI emulator on the prior MC obs ensemble, holding out the truth realization
    and a validation set. Returns (dsi, train, validation, truth, pst, keepobs).

    ``data_cache`` / ``pst_cache`` (the keep-subset obs DataFrame + parsed Pst from ``_dsi_load_all``)
    skip the per-call reload of the multi-hundred-MB obs.jcb + the 464k-obs control file -- essential
    for the LOO loops (dsi_loo_coverage / reinflate_test), which otherwise reload once PER real."""
    import pyemu
    from pyemu.emulators import DSI

    master_dir, template_ws = Path(master_dir), Path(template_ws)
    if pst_cache is not None:
        pst = pst_cache                                   # already try_parse_name_metadata()'d
    else:
        pst = pyemu.Pst(str(template_ws / "pest.pst"))
        pst.try_parse_name_metadata()
    if data_cache is not None:
        data = data_cache.copy()                          # keep-subset, lowercased, float (from _dsi_load_all)
        keep = list(data.columns)
    else:
        jcb = master_dir / "pest.0.obs.jcb"
        oe = (pyemu.ObservationEnsemble.from_binary(pst=pst, filename=str(jcb)) if jcb.exists()
              else pyemu.ObservationEnsemble.from_csv(pst=pst, filename=str(master_dir / "pest.0.obs.csv")))
        oe_df = pd.DataFrame(oe.values, index=oe.index.astype(str), columns=[c.lower() for c in oe.columns])
        keep = [c.lower() for c in _dsi_keepobs(pst)]
        data = oe_df.loc[:, keep].astype(float)
    truth = data.loc[[str(truth_real)]].copy()
    truth[truth.abs() < 1e-8] = 0.0                       # zero out numerical dust
    data = data.drop(index=str(truth_real))               # truth NEVER in training

    rng = np.random.default_rng(seed)
    # n_val is None -> default fidelity-check holdout; n_val=0 -> TRUE leave-one-out: drop ONLY the
    # single truth real, train the DSI on ALL the rest. (The `n_val or ...` idiom silently turned 0
    # into a 23-real holdout, so LOO was really leave-24-out -- fixed.)
    if n_val is None:
        n_val = max(2, data.shape[0] // 5)
    val = (rng.choice(data.index.values, size=min(n_val, data.shape[0] - 2), replace=False)
           if n_val > 0 else np.array([], dtype=object))
    validation = data.loc[val].copy()
    train = data.drop(index=val).copy()

    dsi = DSI(data=train, transforms=DSI_TRANSFORMS, energy_threshold=energy_threshold)
    dsi.fit()
    print(f"  [build_dsi] train={train.shape[0]} val={validation.shape[0]} obs_cols={train.shape[1]} "
          f"latent_dim={dsi.latent_dim} (energy={energy_threshold})")
    return dsi, train, validation, truth, pst, keep


def dsi_encode(dsi, df):
    """Project held-out obs rows to DSI latent space, with the two numerical guards that keep the
    normal-score quadratic extrapolation from blowing up: clip transformed values into the training
    range, and pseudo-invert the projection matrix (rcond=1e-8)."""
    xt = dsi.transformer_pipeline.transform(df.copy()).loc[:, dsi.ovals.index]
    lo = dsi.data_transformed[dsi.ovals.index].min(axis=0)
    hi = dsi.data_transformed[dsi.ovals.index].max(axis=0)
    xt = xt.clip(lower=lo, upper=hi, axis=1)
    dev = xt.values - dsi.ovals.values[np.newaxis, :]
    pinv = np.linalg.pinv(dsi.pmat, rcond=1e-8)
    pvals = pinv @ dev.T
    return pd.DataFrame(pvals.T, index=df.index,
                        columns=[f"p_{i}" for i in range(dsi.pmat.shape[1])])


def _cond_noise_ensemble(target, meta, cond, num_reals, rng):
    """Build the obs+noise ensemble for the DIRECT-PROJECTION posterior: `num_reals` correlated-noise
    realizations of the conditioning obs around the `target` (truth) values. One shock z~N(0,1) per
    site:species series per realization (matching the IES noise convention), scaled per-obs by sigma;
    concentrations clipped >= 0. Returns a (num_reals, len(cond)) DataFrame over the `cond` columns."""
    tv = target[cond].astype(float).values
    species = np.where(meta.loc[cond, "obgnme"].values == "head", "head",
                       meta.loc[cond, "variable"].astype(str).values)
    sig = np.array([float(obs_sigma(np.array([v]), s)[0]) for v, s in zip(tv, species)])
    grp = np.array([f"{meta.loc[o, 'obsid']}:{s}" for o, s in zip(cond, species)])
    ens = np.tile(tv, (num_reals, 1)).astype(float)
    for g in np.unique(grp):
        idx = np.where(grp == g)[0]
        z = rng.standard_normal(num_reals)[:, None]
        ens[:, idx] = tv[idx][None, :] + z * sig[idx][None, :]
        if g.split(":")[-1] not in ("ph", "tmp", "head"):
            ens[:, idx] = np.clip(ens[:, idx], 0.0, None)
    return pd.DataFrame(ens, columns=cond)


def dsi_project_posterior(dsi, target, meta, forecast_cols, num_reals=500, seed=20260706, mode="reg"):
    """DIRECT-PROJECTION DSI posterior (fast; no pestpp-ies in the loop). Build the obs+noise ensemble
    around `target`'s CONDITIONING obs, project each realization to the DSI latent space using ONLY the
    conditioning (weighted) obs, then predict the full obs vector. Returns (posterior peak recovered-SO4
    per realization, full predicted obs df). Reported ALONGSIDE the IES-conditioned posterior.

    ``mode``:
      'raw' -- minimum-norm least squares (pinv of the conditioning-row submatrix). Over-dispersed:
               obs-noise maps straight into the weakly-constrained latent directions (nonphysical tails).
      'reg' -- prior-informed MAP: latent = (Pc^T R^-1 Pc + I)^-1 Pc^T R^-1 dev, with the N(0,I) latent
               prior and R = empirical transformed-space obs-noise variance. The proper DSI linear-
               Gaussian update; tighter, physical.
    """
    dsi_cols = list(dsi.ovals.index)
    cond = [o for o in dsi_cols if o in meta.index and float(meta.loc[o, "weight"]) > 0]
    rng = np.random.default_rng(seed)
    cond_ens = _cond_noise_ensemble(target, meta, cond, num_reals, rng)

    xt_raw = dsi.transformer_pipeline.transform(cond_ens.copy()).loc[:, cond]
    lo = dsi.data_transformed[cond].min(axis=0)
    hi = dsi.data_transformed[cond].max(axis=0)
    xt = xt_raw.clip(lower=lo, upper=hi, axis=1)
    dev = xt.values - dsi.ovals[cond].values[np.newaxis, :]        # (num_reals, n_cond)
    rows = [dsi.ovals.index.get_loc(c) for c in cond]
    Pc = dsi.pmat[rows, :]                                          # (n_cond, latent)
    if mode == "raw":
        latent = (np.linalg.pinv(Pc, rcond=1e-8) @ dev.T).T
    elif mode == "reg":
        # R = transformed-space obs-noise variance (empirical across the noise ensemble); prior = I
        r = np.maximum(xt_raw.var(axis=0).values, 1e-8)
        rinv = 1.0 / r
        A = Pc.T @ (rinv[:, None] * Pc) + np.eye(Pc.shape[1])       # (latent, latent)
        B = Pc.T @ (rinv[:, None] * dev.T)                          # (latent, num_reals)
        latent = np.linalg.solve(A, B).T
    else:
        raise ValueError(f"mode must be 'raw' or 'reg', got {mode!r}")
    pred = dsi.predict(pd.DataFrame(latent, columns=[f"p_{i}" for i in range(latent.shape[1])]))
    pred = pred if isinstance(pred, pd.DataFrame) else pd.DataFrame(pred, columns=dsi_cols)
    fc = [c for c in forecast_cols if c in pred.columns]
    return pred.loc[:, fc].max(axis=1).values, pred


def xvalidate_dsi(dsi, validation, forecast_cols):
    """Cross-validate: encode held-out reals -> DSI.predict -> compare to actual. Returns a dict
    of fidelity metrics (negative-concentration fraction, per-obs RMSE, forecast emulated-vs-actual)."""
    pvals = dsi_encode(dsi, validation)
    pred = dsi.predict(pvals)
    pred = pred if isinstance(pred, pd.DataFrame) else pd.DataFrame(pred, index=validation.index,
                                                                    columns=validation.columns)
    pred = pred.reindex(columns=validation.columns)
    neg_frac = float((pred.values < -1e-9).mean())
    rmse = ((pred - validation) ** 2).mean(axis=0) ** 0.5
    fc_cols = [c for c in forecast_cols if c in validation.columns]
    fc_actual = validation.loc[:, fc_cols].max(axis=1)
    fc_pred = pred.loc[:, fc_cols].max(axis=1)
    return {"neg_frac": neg_frac, "rmse": rmse, "fc_actual": fc_actual, "fc_pred": fc_pred,
            "pred": pred}


def _fig_dsi_xval(dsi, validation, forecast_cols, stage="06_dsi"):
    """Signature figure: emulator cross-validation -- held-out emulated vs actual (forecast peak
    1:1 + negative-concentration fraction)."""
    import matplotlib.pyplot as plt
    apply_style()
    r = xvalidate_dsi(dsi, validation, forecast_cols)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.6))
    lo = min(r["fc_actual"].min(), r["fc_pred"].min()) * 0.8
    hi = max(r["fc_actual"].max(), r["fc_pred"].max()) * 1.1
    a0.plot([lo, hi], [lo, hi], color=C["grey"], ls="--", lw=1)
    a0.scatter(r["fc_actual"], r["fc_pred"], s=45, color=ROLE["emulated"], edgecolor="k", zorder=5)
    a0.set_xlabel("actual peak recovered SO$_4$ (mg/L)")
    a0.set_ylabel("DSI-emulated (mg/L)")
    a0.set_title(f"held-out forecast fidelity (n={len(r['fc_actual'])})", fontsize=11)
    a1.hist(r["rmse"].values, bins=30, color=ROLE["emulated"], alpha=0.7, edgecolor="w")
    a1.set_xlabel("per-obs RMSE (emulated vs actual)")
    a1.set_ylabel("obs count")
    a1.set_title(f"neg-conc fraction = {r['neg_frac']:.3f} (want ~0)", fontsize=11)
    fig.suptitle(f"DSI emulator cross-validation (latent_dim={dsi.latent_dim}, "
                 "setup ensemble)", fontweight="bold")
    return savefig(fig, "dsi_xval", stage)


def _fig_dsi_posterior(prior_fc, post_fc, truth_fc, raw_fc=None, reg_fc=None, stage="06_dsi"):
    """Signature figure: prior->posterior recovered-SO4 forecast. Overlays the IES-conditioned
    posterior (blue) with the two DIRECT-PROJECTION posteriors -- regularized/MAP (green) and raw
    pinv (orange) -- against grey prior + black truth. Hist (prior vs IES) + CDF (all methods)."""
    import numpy as np
    import matplotlib.pyplot as plt
    apply_style()
    prior_fc = np.asarray(prior_fc, float)
    post_fc = np.asarray(post_fc, float)
    methods = [(prior_fc, ROLE["prior"], "prior")]
    if reg_fc is not None:
        methods.append((np.asarray(reg_fc, float), ROLE["emulated"], "proj (regularized)"))
    if raw_fc is not None:
        methods.append((np.asarray(raw_fc, float), C["orange"], "proj (raw pinv)"))
    methods.append((post_fc, ROLE["posterior"], "IES posterior"))

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 4.8))
    a0.hist(prior_fc, bins=30, color=ROLE["prior"], alpha=0.5, density=True, label="prior")
    a0.hist(post_fc, bins=30, color=ROLE["posterior"], alpha=0.5, density=True, label="IES posterior")
    if reg_fc is not None:
        a0.hist(np.asarray(reg_fc, float), bins=30, color=ROLE["emulated"], alpha=0.4,
                density=True, label="proj (regularized)")
    a0.axvline(truth_fc, color=ROLE["truth"], lw=2.5, label=f"truth ({truth_fc:.1f})")
    a0.set_xlabel("peak recovered SO$_4$ (mg/L)")
    a0.set_ylabel("density")
    a0.legend(fontsize=8)
    # x-limit to a physical window so the raw-pinv tail can't crush the comparison
    xhi = float(np.percentile(np.concatenate([prior_fc, post_fc]), 99.5)) * 1.3
    for arr, col, lab in methods:
        s = np.sort(arr)
        a1.plot(s, np.linspace(0, 1, len(s)), color=col, lw=2, label=lab)
    a1.axvline(truth_fc, color=ROLE["truth"], lw=2.5)
    a1.set_xlim(0, xhi)
    a1.set_xlabel("peak recovered SO$_4$ (mg/L)  (raw-pinv tail extends off-axis)")
    a1.set_ylabel("cumulative probability")
    a1.legend(fontsize=8)

    def _cov(a):
        a = np.asarray(a, float)
        return "Y" if a.min() <= truth_fc <= a.max() else "N"
    bits = [f"IES {np.percentile(post_fc,50):.1f}({_cov(post_fc)})"]
    if reg_fc is not None:
        bits.append(f"reg {np.percentile(reg_fc,50):.1f}({_cov(reg_fc)})")
    if raw_fc is not None:
        bits.append(f"raw {np.percentile(raw_fc,50):.1f}({_cov(raw_fc)})")
    fig.suptitle(f"DSI posterior forecast (truth {truth_fc:.1f} mg/L; P50 & min/max-cover) -- "
                 + ", ".join(bits), fontweight="bold", fontsize=11)
    return savefig(fig, "dsi_forecast", stage)


def _dsi_load_all(master_dir=WS5_MASTER, template_ws=WS3):
    """Full prior obs ensemble (all reals) restricted to the DSI keepobs columns + forecast cols."""
    import pyemu
    pst = pyemu.Pst(str(Path(template_ws) / "pest.pst"))
    pst.try_parse_name_metadata()
    jcb = Path(master_dir) / "pest.0.obs.jcb"
    oe = (pyemu.ObservationEnsemble.from_binary(pst=pst, filename=str(jcb)) if jcb.exists()
          else pyemu.ObservationEnsemble.from_csv(pst=pst, filename=str(Path(master_dir) / "pest.0.obs.csv")))
    oe_df = pd.DataFrame(oe.values, index=oe.index.astype(str),
                         columns=[c.lower() for c in oe.columns])
    keep = [c.lower() for c in _dsi_keepobs(pst)]
    data = oe_df.loc[:, keep].astype(float)
    obs = pst.observation_data
    fc = [c.lower() for c in obs.index[obs.obgnme == "forecast"] if c.lower() in data.columns]
    return data, pst, keep, fc


def _locked_truth_name(truth_dir=None):
    """The synthetic-truth realization name locked by lock_truth (reads _truth/truth_meta.txt).
    Falls back to '5' if not yet locked. Keeps the figure drivers tracking the current truth."""
    truth_dir = Path(truth_dir) if truth_dir else (Path(__file__).parent / "_truth")
    meta = truth_dir / "truth_meta.txt"
    if meta.exists():
        for line in meta.read_text().splitlines():
            if line.startswith("truth_realization="):
                return line.split("=", 1)[1].strip()
    return "5"


def xvalidate_dsi_loo(master_dir=WS5_MASTER, template_ws=WS3, frac=1.0, energy=DSI_ENERGY, seed=0,
                      cache=None):
    """LEAVE-ONE-OUT projection cross-validation: for each left-out realization, refit DSI on the
    rest, encode (direct projection) the held-out obs, predict, and compare. Iterates the left-out
    over at least ``frac`` of the ensemble (frac=1.0 -> full LOO). Returns a per-real DataFrame.
    ``cache`` = the _dsi_load_all tuple (load-once)."""
    from pyemu.emulators import DSI
    data, pst, keep, fc = cache if cache is not None else _dsi_load_all(master_dir, template_ws)
    reals = list(data.index)
    rng = np.random.default_rng(seed)
    n_test = max(int(np.ceil(frac * len(reals))), 1)
    test = reals if n_test >= len(reals) else list(rng.choice(reals, n_test, replace=False))
    rows = []
    for r in test:
        train = data.drop(index=r)
        dsi = DSI(data=train, transforms=DSI_TRANSFORMS, energy_threshold=energy).fit()
        left = data.loc[[r]]
        pred = dsi.predict(dsi_encode(dsi, left)).reindex(columns=data.columns)
        actual = float(left[fc].max(axis=1).iloc[0])
        emul = float(pred[fc].max(axis=1).iloc[0])
        rows.append((r, actual, emul, float((pred.values < -1e-9).mean()), dsi.latent_dim))
    df = pd.DataFrame(rows, columns=["real", "actual", "emulated", "neg_frac", "latent_dim"])
    print(f"  [xvalidate_dsi_loo] {len(df)} left-out; forecast R (actual vs emulated)="
          f"{np.corrcoef(df.actual, df.emulated)[0,1]:.3f}; mean neg_frac={df.neg_frac.mean():.4f}")
    return df


def dsi_loo_coverage(master_dir=WS5_MASTER, template_ws=WS3, frac=0.30, num_reals=500,
                     seed=20260706, select_seed=0, cache=None):
    """LEAVE-ONE-OUT conditioning coverage: for at least ``frac`` of the ensemble, train DSI on the
    rest, CONDITION it on the left-out realization's obsvals (noise redrawn around them), and check
    whether the posterior 5-95 forecast interval covers the left-out real's actual forecast.

    ``cache`` = the ``_dsi_load_all`` tuple (data, pst, keep, fc); passed to avoid reloading the big
    obs.jcb + control file once per left-out real (load-once)."""
    data, pst, keep, fc = cache if cache is not None else _dsi_load_all(master_dir, template_ws)
    reals = list(data.index)
    n_test = max(int(np.ceil(frac * len(reals))), 1)
    test = list(np.random.default_rng(select_seed).choice(reals, n_test, replace=False))
    dtemp = WS6_DSI.parent / "_s6_loo_dsi"
    rows, posteriors = [], {}
    for r in test:
        # SAME build_dsi_conditioning + SAME noise seed as the real truth conditioning
        _dpst, fore, target = build_dsi_conditioning(
            master_dir, template_ws, dsi_template=dtemp, truth_real=r, num_reals=num_reals, seed=seed,
            data_cache=data, pst_cache=pst)
        run_dsi(dtemp)
        _pr, po, tr = dsi_posterior(dtemp, forecast_cols=fore, truth=target)
        po = np.asarray(po, float)
        posteriors[r] = po                              # full posterior forecast ensemble
        # coverage = actual within the FULL posterior range (min..max), not a 90% interval
        rows.append((r, tr, float(po.min()), float(np.median(po)), float(po.max()),
                     bool(po.min() <= tr <= po.max())))
    df = pd.DataFrame(rows, columns=["real", "actual_fc", "pmin", "p50", "pmax", "covered"])
    print(f"  [dsi_loo_coverage] {len(df)} left-out conditioned; coverage rate="
          f"{df.covered.mean():.2f} (posterior min..max range)")
    return df, posteriors


def _fig_dsi_loo(loo_xval, cov, posteriors, stage="06_dsi"):
    """Signature figure: (a) LOO projection forecast 1:1; (b) LOO conditioning posterior VIOLINS
    per left-out real, with the actual forecast marked (violin blue=covered / red=missed)."""
    import matplotlib.pyplot as plt
    apply_style()
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 5))
    lo = min(loo_xval.actual.min(), loo_xval.emulated.min()) * 0.8
    hi = max(loo_xval.actual.max(), loo_xval.emulated.max()) * 1.1
    a0.plot([lo, hi], [lo, hi], color=C["grey"], ls="--", lw=1)
    a0.scatter(loo_xval.actual, loo_xval.emulated, s=45, color=ROLE["emulated"],
               edgecolor="k", zorder=5)
    a0.set_xlabel("actual peak recovered SO$_4$ (mg/L)")
    a0.set_ylabel("DSI-emulated (projection)")
    a0.set_title(f"LOO projection fidelity (n={len(loo_xval)}, "
                 f"R={np.corrcoef(loo_xval.actual, loo_xval.emulated)[0,1]:.2f})", fontsize=11)

    cov = cov.sort_values("actual_fc").reset_index(drop=True)
    pos = np.arange(len(cov))
    parts = a1.violinplot([posteriors[r] for r in cov.real], positions=pos, vert=False,
                          showmedians=True, showextrema=False, widths=0.85)
    for body, covered in zip(parts["bodies"], cov.covered):
        body.set_facecolor(ROLE["posterior"] if covered else ROLE["noise"])
        body.set_edgecolor("none")
        body.set_alpha(0.55)
    parts["cmedians"].set_color(C["black"])
    parts["cmedians"].set_linewidth(1.0)
    for i, row in cov.iterrows():
        a1.plot(row.actual_fc, i, "|", color=ROLE["truth"], ms=16, mew=2.5, zorder=6)
    a1.set_yticks(pos)
    a1.set_yticklabels(cov.real)
    a1.set_xlabel("peak recovered SO$_4$ (mg/L)")
    a1.set_ylabel("left-out realization")
    a1.set_title(f"LOO conditioning posterior (min-max coverage={cov.covered.mean():.0%}; | = actual)",
                 fontsize=11)
    fig.suptitle("DSI leave-one-out validation -- projection fidelity + conditioning posteriors",
                 fontweight="bold")
    return savefig(fig, "dsi_loo", stage)


def _dsi_condition_figs(dsi_template, master_dir, template_ws, target_real, tag,
                        forecast_cols, stage="06_dsi"):
    """For one DSI conditioning run, emit the prior/posterior analogues of prior_vs_data and
    prior_mc_forecast: emulator PRIOR (grey) vs POSTERIOR (blue) vs measured/target (black) + noise
    (red) at monitoring sites, and the prior->posterior recovered-SO4 forecast."""
    import pyemu
    import matplotlib.pyplot as plt

    apply_style()
    dsi_template = Path(dsi_template)
    dpst = pyemu.Pst(str(dsi_template / "dsi.pst"))

    def _oe(jcb):
        oe = pyemu.ObservationEnsemble.from_binary(pst=dpst, filename=str(jcb))
        return pd.DataFrame(oe.values, index=oe.index.astype(str), columns=oe.columns)
    prior = _oe(dsi_template / "dsi.0.obs.jcb")
    last = max(i for i in range(4) if (dsi_template / f"dsi.{i}.obs.jcb").exists())
    post = _oe(dsi_template / f"dsi.{last}.obs.jcb")
    pidx = prior.index[::max(1, len(prior) // 120)]     # thin spaghetti to ~120 lines
    qidx = post.index[::max(1, len(post) // 120)]

    data, pst, keep, fc = _dsi_load_all(master_dir, template_ws)
    target = data.loc[str(target_real)]
    meta = _dsi_meta(pst)
    rng = np.random.default_rng(20260706)

    # representative screen per monitoring cluster
    cond = meta.loc[[o for o in prior.columns if o in meta.index and float(meta.loc[o, "weight"]) > 0]]
    cond = cond[cond.obsid.astype(str).str.startswith(MONITOR_PREFIX)]
    sites = {}
    for wl, g in cond.groupby(cond.obsid.astype(str).str.split("-").str[0]):
        scr = sorted(g.obsid.unique())
        sites[wl] = scr[len(scr) // 2]
    site_ids = [sites[k] for k in sorted(sites)]
    types = [("so4", "SO$_4$"), ("o0", "O$_2$"), ("no3", "NO$_3$"),
             ("ph", "pH"), ("tmp", "Tmp"), ("head", "head")]

    # --- Fig A: prior/posterior sim vs data at sites ---------------------------------------
    figA, axes = plt.subplots(len(site_ids), len(types),
                              figsize=(2.4 * len(types), 1.9 * len(site_ids)), sharex=True)
    for i, sid in enumerate(site_ids):
        for j, (tp, tlab) in enumerate(types):
            ax = axes[i, j]
            sel = meta[(meta.obsid.astype(str) == sid)
                       & ((meta.obgnme == "head") if tp == "head"
                          else (meta.variable == tp))]
            cols = [o for o in sel.sort_values("time").index if o in prior.columns]
            if not cols:
                ax.set_axis_off()
                continue
            t = sel.loc[cols, "time"].astype(float).values
            for r in pidx:
                ax.plot(t, prior.loc[r, cols].values, color=ROLE["prior"], alpha=0.35, lw=0.5)
            for r in qidx:
                ax.plot(t, post.loc[r, cols].values, color=ROLE["posterior"], alpha=0.35, lw=0.5)
            yv = target[cols].astype(float).values
            for nr in draw_obs_noise(yv, tp, N_NOISE, rng):
                ax.plot(t, nr, color=ROLE["noise"], alpha=0.35, lw=0.4)
            ax.plot(t, yv, "o", color=ROLE["truth"], ms=2.5, zorder=6)
            if i == 0:
                ax.set_title(tlab, fontsize=10)
            if j == 0:
                ax.set_ylabel(sid, fontsize=8)
            ax.tick_params(labelsize=6)
    handles = [plt.Line2D([0], [0], color=ROLE["prior"], label="prior sim"),
               plt.Line2D([0], [0], color=ROLE["posterior"], label="posterior sim"),
               plt.Line2D([0], [0], color=ROLE["truth"], marker="o", ls="", label="measured"),
               plt.Line2D([0], [0], color=ROLE["noise"], alpha=0.6, label="noise")]
    figA.legend(handles=handles, loc="upper center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    figA.suptitle(f"DSI prior vs posterior vs data at monitoring sites ({tag})",
                  fontweight="bold", y=1.012)
    figA.tight_layout()
    pA = savefig(figA, f"dsi_prior_vs_data_{tag}", stage)

    # --- Fig B: prior->posterior recovered-SO4 forecast ------------------------------------
    fcc = [c for c in forecast_cols if c in prior.columns]
    ftimes = meta.loc[fcc, "time"].astype(float)
    order = ftimes.sort_values().index
    ft = ftimes[order].values
    prior_pk = prior[fcc].max(axis=1).values
    post_pk = post[fcc].max(axis=1).values
    truth_pk = float(target[fcc].max())
    figB, (b0, b1) = plt.subplots(1, 2, figsize=(13, 4.8))
    b0.hist(prior_pk, bins=30, color=ROLE["prior"], alpha=0.55, density=True, label="prior")
    b0.hist(post_pk, bins=30, color=ROLE["posterior"], alpha=0.55, density=True, label="posterior")
    b0.axvline(truth_pk, color=ROLE["truth"], lw=2.5, label=f"truth ({truth_pk:.1f})")
    b0.set_xlabel("peak recovered SO$_4$ (mg/L)")
    b0.set_ylabel("density")
    b0.legend(fontsize=9)
    covered = post_pk.min() <= truth_pk <= post_pk.max()
    for r in pidx:
        b1.plot(ft, prior.loc[r, order].values, color=ROLE["prior"], alpha=0.3, lw=0.5)
    for r in qidx:
        b1.plot(ft, post.loc[r, order].values, color=ROLE["posterior"], alpha=0.3, lw=0.5)
    b1.plot(ft, target[order].values, color=ROLE["truth"], lw=2.2, label="truth")
    b1.set_xlabel(LBL["time"])
    b1.set_ylabel(LBL["so4"])
    b1.legend(fontsize=9)
    figB.suptitle(f"DSI recovered-SO4 forecast ({tag}): prior P50={np.percentile(prior_pk,50):.1f} -> "
                  f"posterior P50={np.percentile(post_pk,50):.1f} mg/L (truth in range: {covered})",
                  fontweight="bold")
    pB = savefig(figB, f"dsi_forecast_{tag}", stage)
    return pA, pB


def _dsi_meta(pst):
    """pst.observation_data with a lowercased index (to match DSI obs names) + lowercased variable."""
    meta = pst.observation_data.copy()
    meta.index = meta.index.str.lower()
    meta["variable"] = meta["variable"].astype(str).str.lower()
    meta["time"] = pd.to_numeric(meta["time"], errors="coerce")   # numeric -> correct time sort
    return meta


def build_dsi_conditioning(master_dir=WS5_MASTER, template_ws=WS3, dsi_template=WS6_DSI,
                           noptmax=1, num_reals=500, multimodal_alpha=None, subset_size=-100,
                           truth_real="5", seed=20260706,
                           n_iter_reinflate=None, reinflate_factor=None, reinflate_num_reals=None,
                           data_cache=None, pst_cache=None, autoadaloc=False):
    """Build the DSI-emulator conditioning PEST interface: train on the (non-truth) prior, write
    the emulator interface, set targets=truth + weights + per-site:species phi factors + correlated
    observation-noise ensemble, and the ies_* solver options the user chose. Returns the dsi Pst.

    ``n_iter_reinflate`` / ``reinflate_factor`` enable pestpp-ies REINFLATION (combats posterior
    collapse / LOO overconfidence): after every ``n_iter_reinflate`` iteration(s) the ensemble is
    reinflated by ``reinflate_factor`` (1.0 = 100%, back toward the prior spread). E.g. (1, 1.0) with
    noptmax=3 = "1 iteration -> reinflate 100%", three times."""
    import pyemu
    import herebedragons as hbd

    dsi_template = Path(dsi_template)
    # production emulator: train on ALL non-truth reals (no validation holdout)
    dsi, _train, _v, truth, pst, _keep = build_dsi_emulator(
        master_dir, template_ws, truth_real=truth_real, n_val=0,
        data_cache=data_cache, pst_cache=pst_cache)

    dpst = dsi.prepare_pestpp(str(dsi_template), use_runstor=True)   # dsi.pst + dsi.pickle + fwd run
    hbd.get_bins(str(dsi_template))
    meta = _dsi_meta(pst)
    dobs = dpst.observation_data

    # (a) targets = synthetic truth
    tvals = truth.iloc[0]
    common = [o for o in dobs.index if o in tvals.index]
    dobs.loc[common, "obsval"] = tvals[common].values
    dobs["weight"] = 0.0
    dobs["standard_deviation"] = np.nan

    # (b) weight the FOM-weighted conditioning obs; per-obs proportional sigma; obgnme = obsid:variable
    cond = [o for o in common if float(meta.loc[o, "weight"]) > 0]
    cm = meta.loc[cond]
    species = np.where(cm["obgnme"].values == "head", "head", cm["variable"].values)
    ovals = dobs.loc[cond, "obsval"].astype(float).values
    sig = np.array([float(obs_sigma(np.array([v]), s)[0]) for v, s in zip(ovals, species)])
    dobs.loc[cond, "standard_deviation"] = sig
    dobs.loc[cond, "weight"] = 1.0 / sig
    dobs.loc[cond, "obgnme"] = [f"{oid}:{s}" for oid, s in zip(cm["obsid"].values, species)]

    # (c) ies_phi_factors: equal phi share per site:species group
    groups = pd.unique(dobs.loc[cond, "obgnme"])
    pd.Series(1.0 / len(groups), index=groups).to_csv(
        dsi_template / "ies_phi_factors.csv", header=False)
    dpst.pestpp_options["ies_phi_factor_file"] = "ies_phi_factors.csv"

    # (d) correlated observation-noise ensemble (one shock per site:species series per real)
    noise = pyemu.ObservationEnsemble.from_gaussian_draw(dpst, num_reals=num_reals)
    rng = np.random.default_rng(seed)
    for grp, g in dobs.loc[cond].groupby("obgnme"):
        z = rng.standard_normal(num_reals)[:, None]
        vv = g["obsval"].astype(float).values[None, :] + z * g["standard_deviation"].astype(float).values[None, :]
        if grp.split(":")[-1] not in ("ph", "tmp", "head"):
            vv = np.clip(vv, 0.0, None)
        noise._df.loc[:, g.index] = vv
    noise.to_binary(str(dsi_template / "noise.jcb"))
    dpst.pestpp_options["ies_observation_ensemble"] = "noise.jcb"

    # (e) forecast obs + solver options (user's choices)
    fore = [o for o in dobs.index if meta.loc[o, "obgnme"] == "forecast"]
    dpst.pestpp_options["forecasts"] = ",".join(fore)
    dpst.pestpp_options["ies_num_reals"] = num_reals
    dpst.pestpp_options["save_binary"] = True
    dpst.pestpp_options["ies_subset_size"] = subset_size
    dpst.pestpp_options["ies_drop_conflicts"] = True       # drop prior-data-conflict obs
    dpst.pestpp_options.pop("ies_autoadaloc", None)
    if autoadaloc:                                         # adaptive automatic localization (expensive; off by default)
        dpst.pestpp_options["ies_autoadaloc"] = True
    dpst.pestpp_options.pop("ies_multimodal_alpha", None)   # off unless a value is given
    if multimodal_alpha is not None:
        dpst.pestpp_options["ies_multimodal_alpha"] = multimodal_alpha
    # reinflation (off unless requested): reinflate the ensemble every n_iter_reinflate iters
    for _opt in ("ies_n_iter_reinflate", "ies_reinflate_factor", "ies_reinflate_num_reals"):
        dpst.pestpp_options.pop(_opt, None)
    if n_iter_reinflate is not None:
        dpst.pestpp_options["ies_n_iter_reinflate"] = n_iter_reinflate
    if reinflate_factor is not None:
        dpst.pestpp_options["ies_reinflate_factor"] = reinflate_factor
    if reinflate_num_reals is not None:
        dpst.pestpp_options["ies_reinflate_num_reals"] = reinflate_num_reals
    dpst.control_data.noptmax = noptmax
    dpst.write(str(dsi_template / "dsi.pst"), version=2)
    _reinf = "off" if n_iter_reinflate is None else f"every {n_iter_reinflate} iter x{reinflate_factor}"
    print(f"  [build_dsi_conditioning] {len(cond)} weighted obs, {len(groups)} phi groups, "
          f"{len(fore)} forecast obs, {num_reals} reals, noptmax={noptmax}, "
          f"multimodal={'off' if multimodal_alpha is None else multimodal_alpha}, reinflate={_reinf}")
    return dpst, fore, truth


def run_dsi(dsi_template=WS6_DSI):
    """Condition the DSI emulator with pestpp-ies (runstor /e mode -- fast, no model in the loop)."""
    import pyemu
    pyemu.os_utils.run("pestpp-ies dsi.pst /e", cwd=str(Path(dsi_template)))


# --- FULL-MODEL (FOM) history match -- the gold-standard comparison for the DSI posterior -----------
WS_FOM = Path(__file__).parent / "_s6_fom_template"       # cloned from WS3 (keeps _s3_template clean)
WS_FOM_MASTER = Path(__file__).parent / "_s6_fom_master"
_S6_FOM_WORKER_ROOT = Path(__file__).parent / "_s6_fom_workers"


def build_fom_conditioning(src_template=WS3, fom_template=WS_FOM, prior_master=WS5_MASTER,
                           num_workers=60, seed=20260706, noptmax=1):
    """Build a FULL-OUTPUT-MODEL (FOM) pestpp-ies history match -- the gold-standard the DSI-emulator
    posterior is compared against. Mirrors ``build_dsi_conditioning`` EXACTLY on the real model
    interface: targets = synthetic truth (already in obsval from stage 4), per-obs PROPORTIONAL weights
    (1/obs_sigma -- not the fixed stage-4 weights), per-site:species phi factors, a correlated obs-noise
    ensemble (same seed/convention), drop_conflicts, no autoadaloc/multimodal, noptmax -- with the ONE
    exception ``ies_subset_size = num_workers`` (each lambda-test subset = one worker batch, since FOM
    runs are expensive). REUSES the prior-MC parameter ensemble (``prior_master/pest.0.par.jcb`` -- the
    exact ~120 realizations that produced the obs the DSI trained on), so the FOM conditions the SAME
    prior as the DSI (clean apples-to-apples). Deploy with run_fom_conditioning.

    Scale note: on the full obs set (~465k obs) the noise + per-iteration obs ensembles are large
    (120 reals -> ~0.45 GB each); master-side only.
    """
    import subprocess
    import pyemu
    import herebedragons as hbd

    src_template, fom_template = Path(src_template), Path(fom_template)
    if fom_template.exists():
        shutil.rmtree(fom_template)
    subprocess.run(["cp", "-R", str(src_template), str(fom_template)], check=True)  # copytree drops files
    hbd.get_bins(str(fom_template))

    pst = pyemu.Pst(str(fom_template / "pest.pst"))
    pst.try_parse_name_metadata()
    obs = pst.observation_data

    # (b) RECOMPUTE proportional weights (1/obs_sigma) + obgnme=obsid:species on the conditioning obs
    #     (stage-4 injected truth->obsval + fixed weights; we overwrite the weights to match the DSI).
    cond = list(obs.index[obs.weight.astype(float) > 0])
    cm = obs.loc[cond]
    species = np.where(cm["obgnme"].values == "head", "head",
                       cm["variable"].astype(str).str.lower().values)
    ovals = cm["obsval"].astype(float).values
    sig = np.array([float(obs_sigma(np.array([v]), s)[0]) for v, s in zip(ovals, species)])
    obs.loc[cond, "standard_deviation"] = sig
    obs.loc[cond, "weight"] = 1.0 / sig
    obs.loc[cond, "obgnme"] = [f"{oid}:{s}" for oid, s in zip(cm["obsid"].astype(str).values, species)]

    # (c) ies_phi_factors: equal phi share per site:species group
    groups = pd.unique(obs.loc[cond, "obgnme"])
    pd.Series(1.0 / len(groups), index=groups).to_csv(fom_template / "ies_phi_factors.csv", header=False)
    pst.pestpp_options["ies_phi_factor_file"] = "ies_phi_factors.csv"

    # (d) REUSE the prior-MC parameter ensemble (the exact reals that produced the prior MC obs the DSI
    #     trained on) -> the FOM conditions the SAME prior as the DSI. num_reals derived from it.
    src_par = Path(prior_master) / "pest.0.par.jcb"
    if not src_par.exists():
        src_par = src_template / "prior_pe.jcb"            # fallback: the drawn prior ensemble
    shutil.copy(src_par, fom_template / "fom_pe.jcb")
    pe = pyemu.ParameterEnsemble.from_binary(pst=pst, filename=str(fom_template / "fom_pe.jcb"))
    num_reals = pe.shape[0]

    # (e) correlated observation-noise ensemble -- identical construction/seed to build_dsi_conditioning
    noise = pyemu.ObservationEnsemble.from_gaussian_draw(pst, num_reals=num_reals)
    rng = np.random.default_rng(seed)
    for grp, g in obs.loc[cond].groupby("obgnme"):
        z = rng.standard_normal(num_reals)[:, None]
        vv = g["obsval"].astype(float).values[None, :] + z * g["standard_deviation"].astype(float).values[None, :]
        if grp.split(":")[-1] not in ("ph", "tmp", "head"):
            vv = np.clip(vv, 0.0, None)
        noise._df.loc[:, g.index] = vv
    noise.to_binary(str(fom_template / "noise.jcb"))

    # (f) ies_* solver options -- SAME as the DSI conditioning except ies_subset_size = num_workers
    fore = list(obs.index[obs.obgnme == "forecast"])
    pst.pestpp_options["ies_par_en"] = "fom_pe.jcb"
    pst.pestpp_options["ies_observation_ensemble"] = "noise.jcb"
    pst.pestpp_options["forecasts"] = ",".join(fore)
    pst.pestpp_options["ies_num_reals"] = num_reals
    pst.pestpp_options["save_binary"] = True
    pst.pestpp_options["ies_subset_size"] = num_workers    # THE exception (DSI used -100)
    pst.pestpp_options["ies_drop_conflicts"] = True
    pst.pestpp_options.pop("ies_autoadaloc", None)         # off, matching the DSI conditioning
    pst.pestpp_options.pop("ies_multimodal_alpha", None)   # off
    pst.pestpp_options["ies_no_noise"] = False             # use the provided correlated noise ensemble
    pst.control_data.noptmax = noptmax
    pst.write(str(fom_template / "pest.pst"), version=2)
    print(f"  [build_fom_conditioning] {len(cond)} weighted obs, {len(groups)} phi groups, "
          f"{len(fore)} forecast obs, {num_reals} reals, subset_size={num_workers}, noptmax={noptmax}")
    return pst, fore


def run_fom_conditioning(fom_template=WS_FOM, master_dir=WS_FOM_MASTER, num_workers=60,
                         condor_kwargs=None):
    """Deploy the FOM history match over workers (HTCondor when a pool is available, else local),
    reusing the same routing as the prior MC / DSIVC sweep. Expensive: each ensemble evaluation runs
    the full reactive model for every realization."""
    return _deploy_pestpp(fom_template, "pest.pst", master_dir, num_workers, _S6_FOM_WORKER_ROOT,
                              condor_kwargs=condor_kwargs)


def stage6_fom(num_workers=None, condor_kwargs=None):
    """Section-6 FOM history match orchestrator: build the full-model conditioning (mirrors the DSI IES;
    reuses the prior-MC parameter ensemble; ies_subset_size = worker count) then deploy it. On a pool the
    worker count defaults to CONDOR_DEFAULTS (60); locally to cpu_count-1. EXPENSIVE (one full reactive run
    per realization per ensemble evaluation) -- intended for HTCondor."""
    import os
    if num_workers is None:
        num_workers = CONDOR_DEFAULTS["n_workers"] if _htcondor_available() else max(1, (os.cpu_count() or 2) - 1)
    build_fom_conditioning(num_workers=num_workers)     # subset_size = num_workers; ensemble from prior MC
    run_fom_conditioning(num_workers=num_workers, condor_kwargs=condor_kwargs)
    return WS_FOM_MASTER


def dsi_posterior(dsi_template=WS6_DSI, forecast_cols=None, truth=None, noptmax=1):
    """Prior (iter 0) and posterior (last iter) recovered-SO4 forecast from the DSI obs ensembles."""
    import pyemu
    dsi_template = Path(dsi_template)
    dpst = pyemu.Pst(str(dsi_template / "dsi.pst"))
    if forecast_cols is None:
        meta = dpst.observation_data
        forecast_cols = list(meta.index[meta.obgnme == "forecast"]) if "forecast" in \
            set(meta.obgnme) else []

    def _fc(jcb):
        oe = pyemu.ObservationEnsemble.from_binary(pst=dpst, filename=str(jcb))
        df = pd.DataFrame(oe.values, index=oe.index.astype(str), columns=oe.columns)
        cols = [c for c in forecast_cols if c in df.columns]
        return df.loc[:, cols].max(axis=1)

    last = max(i for i in range(noptmax + 1) if (dsi_template / f"dsi.{i}.obs.jcb").exists())
    prior_fc = _fc(dsi_template / "dsi.0.obs.jcb")
    post_fc = _fc(dsi_template / f"dsi.{last}.obs.jcb")
    truth_fc = float(truth.loc[:, [c for c in forecast_cols if c in truth.columns]].max(axis=1).iloc[0]) \
        if truth is not None else None
    return prior_fc, post_fc, truth_fc


def stage6_figs(master_dir=WS5_MASTER, template_ws=WS3, frac=0.30, select_seed=0):
    """Emit the DSI prior/posterior-vs-data + forecast figures for the conditioned truth AND for
    each LOO realization that fails the posterior min/max coverage test."""
    apply_style()
    _dpst, fore, _truth = build_dsi_conditioning(dsi_template=WS6_DSI, truth_real="5")
    run_dsi(WS6_DSI)
    _dsi_condition_figs(WS6_DSI, master_dir, template_ws, "5", "truth_r5", fore)

    data, pst, keep, fc = _dsi_load_all(master_dir, template_ws)
    reals = list(data.index)
    n_test = max(int(np.ceil(frac * len(reals))), 1)
    test = list(np.random.default_rng(select_seed).choice(reals, n_test, replace=False))
    failed = []
    for r in test:
        dtmp = WS6_DSI.parent / f"_s6_fig_{r}"
        _dp, fore_r, target = build_dsi_conditioning(dsi_template=dtmp, truth_real=r)
        run_dsi(dtmp)
        _pr, po, tr = dsi_posterior(dtmp, forecast_cols=fore_r, truth=target)
        po = np.asarray(po, float)
        if not (po.min() <= tr <= po.max()):            # failed min/max coverage
            failed.append(r)
            _dsi_condition_figs(dtmp, master_dir, template_ws, r, f"loo_r{r}", fore_r)
    print(f"[stage6_figs] figs: truth_r5 + failed LOO reals {failed}")


def reinflate_test(master_dir=WS5_MASTER, template_ws=WS3, frac=0.30, num_reals=500,
                   n_iter_reinflate=1, reinflate_factor=1.0, noptmax=3,
                   seed=20260706, select_seed=0, demo_n=2):
    """Test pestpp-ies REINFLATION on the LOO xvals that FAIL baseline min/max coverage.

    Baseline (noptmax=1, no reinflation) identifies the failures; then re-condition EACH failed real
    with reinflation -- "after every ``n_iter_reinflate`` iteration reinflate ``reinflate_factor`` (1.0
    =100%)", run over ``noptmax`` iterations (so noptmax=3, n_iter=1 -> three reinflate cycles) -- and
    re-check whether the posterior min..max now covers the held-out actual. If the (larger) ensemble
    has no baseline failures, demo reinflation on the ``demo_n`` tightest-margin reals instead.
    Reports coverage recovered + how much the posterior spread widened (guarding against trivial
    'cover everything by exploding the range').
    """
    apply_style()
    cache = _dsi_load_all(master_dir, template_ws)         # load the big obs.jcb + control file ONCE
    data, pst, _keep, _fc = cache
    base, _bpost = dsi_loo_coverage(master_dir, template_ws, frac=frac, num_reals=num_reals,
                                    seed=seed, select_seed=select_seed, cache=cache)
    base["margin"] = np.minimum(base.actual_fc - base.pmin, base.pmax - base.actual_fc)  # <0 = missed
    failed = base[~base.covered]
    if failed.empty:
        pick = base.nsmallest(demo_n, "margin")
        print(f"[reinflate_test] baseline coverage {base.covered.mean():.0%} -- NO failures on this "
              f"{len(base)}-real LOO set; demoing reinflation on the {len(pick)} tightest-margin reals "
              f"{list(pick.real)} (margins {[round(m,1) for m in pick.margin]})")
    else:
        pick = failed
        print(f"[reinflate_test] baseline coverage {base.covered.mean():.0%}; {len(failed)} FAILED reals: "
              f"{list(failed.real)}")

    rows = []
    for _, br in pick.iterrows():
        r = br.real
        dtemp = WS6_DSI.parent / f"_s6_reinf_{r}"
        _dpst, fore, target = build_dsi_conditioning(
            master_dir, template_ws, dsi_template=dtemp, truth_real=r, num_reals=num_reals,
            seed=seed, noptmax=noptmax, n_iter_reinflate=n_iter_reinflate,
            reinflate_factor=reinflate_factor, data_cache=data, pst_cache=pst)
        run_dsi(dtemp)
        _pr, po, tr = dsi_posterior(dtemp, forecast_cols=fore, truth=target, noptmax=noptmax)
        po = np.asarray(po, float)
        rows.append(dict(real=r, actual=tr, base_cov=bool(br.covered),
                         base_min=br.pmin, base_max=br.pmax, base_range=br.pmax - br.pmin,
                         reinf_min=float(po.min()), reinf_max=float(po.max()),
                         reinf_range=float(po.max() - po.min()),
                         reinf_cov=bool(po.min() <= tr <= po.max())))
    res = pd.DataFrame(rows)
    print(f"\n[reinflate_test] reinflation = after {n_iter_reinflate} iter -> x{reinflate_factor}, "
          f"noptmax={noptmax} ({noptmax // max(n_iter_reinflate,1)} cycles). failed/tight LOO reals:")
    for _, x in res.iterrows():
        verdict = "COVER" if x.reinf_cov else "still miss"
        print(f"  real {x.real}: actual={x.actual:.1f} | base [{x.base_min:.1f},{x.base_max:.1f}] "
              f"{'cover' if x.base_cov else 'MISS'} -> reinf [{x.reinf_min:.1f},{x.reinf_max:.1f}] {verdict}"
              f"  (range {x.base_range:.1f}->{x.reinf_range:.1f} mg/L, "
              f"x{x.reinf_range / max(x.base_range, 1e-9):.1f})")
    newly = int(((~res.base_cov) & res.reinf_cov).sum())
    tot_fail = int((~res.base_cov).sum())
    print(f"[reinflate_test] reinflation recovered {newly}/{tot_fail} baseline-failed reals; "
          f"mean posterior range {res.base_range.mean():.1f} -> {res.reinf_range.mean():.1f} mg/L "
          f"(x{res.reinf_range.mean() / max(res.base_range.mean(), 1e-9):.1f})")
    return res


def dsi_loo_coverage_proj(master_dir=WS5_MASTER, template_ws=WS3, frac=0.30, num_reals=500,
                          seed=20260706, select_seed=0, cache=None):
    """FAST projection-based LOO coverage (raw pinv + regularized MAP), a companion to the IES-based
    dsi_loo_coverage. For each left-out real: refit DSI on the other 119, build the obs+noise ensemble
    around its conditioning obs, project (both modes) -> posterior peak forecast, min/max coverage.
    Uses the SAME left-out set as dsi_loo_coverage (same select_seed) so the methods are comparable."""
    from pyemu.emulators import DSI
    data, pst, keep, fc = cache if cache is not None else _dsi_load_all(master_dir, template_ws)
    meta = _dsi_meta(pst)
    reals = list(data.index)
    n_test = max(int(np.ceil(frac * len(reals))), 1)
    test = list(np.random.default_rng(select_seed).choice(reals, n_test, replace=False))
    rows, post_reg, post_raw = [], {}, {}
    for r in test:
        train = data.drop(index=r)
        dsi = DSI(data=train, transforms=DSI_TRANSFORMS, energy_threshold=DSI_ENERGY).fit()
        target = data.loc[r]
        actual = float(target[fc].max())
        raw, _ = dsi_project_posterior(dsi, target, meta, fc, num_reals=num_reals, seed=seed, mode="raw")
        reg, _ = dsi_project_posterior(dsi, target, meta, fc, num_reals=num_reals, seed=seed, mode="reg")
        post_reg[r], post_raw[r] = reg, raw
        rows.append((r, actual, bool(raw.min() <= actual <= raw.max()),
                     bool(reg.min() <= actual <= reg.max()),
                     float(np.median(reg)), float(reg.min()), float(reg.max())))
    df = pd.DataFrame(rows, columns=["real", "actual", "cov_raw", "cov_reg",
                                     "reg_p50", "reg_min", "reg_max"])
    print(f"  [dsi_loo_coverage_proj] projection min/max coverage: raw={df.cov_raw.mean():.2f} "
          f"reg={df.cov_reg.mean():.2f} (n={len(df)})")
    return df, post_reg, post_raw


def _fig_dsi_xval_loo(lx, stage="06_dsi"):
    """Signature figure: PURE leave-one-out projection fidelity -- each point is a realization projected
    through a DSI retrained without it. Emulated-vs-actual peak forecast 1:1 + per-real neg-conc frac."""
    import matplotlib.pyplot as plt
    apply_style()
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.6))
    lo = min(lx.actual.min(), lx.emulated.min()) * 0.8
    hi = max(lx.actual.max(), lx.emulated.max()) * 1.1
    a0.plot([lo, hi], [lo, hi], color=C["grey"], ls="--", lw=1)
    a0.scatter(lx.actual, lx.emulated, s=40, color=ROLE["emulated"], edgecolor="k", zorder=5)
    a0.set_xlabel("actual peak recovered SO$_4$ (mg/L)")
    a0.set_ylabel("DSI-emulated (LOO projection)")
    a0.set_title(f"pure LOO forecast fidelity (n={len(lx)}, "
                 f"R={np.corrcoef(lx.actual, lx.emulated)[0, 1]:.2f})", fontsize=11)
    a1.hist(lx.neg_frac, bins=20, color=ROLE["emulated"], alpha=0.7, edgecolor="w")
    a1.set_xlabel("negative-concentration fraction (per left-out real)")
    a1.set_ylabel("reals")
    a1.set_title(f"mean neg-frac = {lx.neg_frac.mean():.4f} (want ~0)", fontsize=11)
    fig.suptitle(f"DSI cross-validation -- pure leave-one-out (latent_dim={int(lx.latent_dim.iloc[0])})",
                 fontweight="bold")
    return savefig(fig, "dsi_xval", stage)


def _fig_dsi_proj_coverage(cov_ies, proj_df, post_reg, stage="06_dsi"):
    """Signature figure: LOO coverage COMPARISON across the three posteriors -- IES vs regularized
    projection vs raw projection. (a) coverage-rate bars; (b) regularized-projection P5-P95 band vs
    the held-out actual (1:1), points coloured by coverage."""
    import matplotlib.pyplot as plt
    apply_style()
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 4.8))
    rates = {"IES": cov_ies.covered.mean(), "proj\n(reg)": proj_df.cov_reg.mean(),
             "proj\n(raw)": proj_df.cov_raw.mean()}
    cols = [ROLE["posterior"], ROLE["emulated"], C["orange"]]
    a0.bar(list(rates), list(rates.values()), color=cols, edgecolor="k", alpha=0.85)
    for i, v in enumerate(rates.values()):
        a0.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=10, fontweight="bold")
    a0.set_ylim(0, 1.1)
    a0.set_ylabel("LOO min/max coverage")
    a0.set_title(f"posterior coverage (n={len(proj_df)} left-out)", fontsize=11)

    pj = proj_df.sort_values("actual")
    a1.plot([pj.actual.min(), pj.actual.max()], [pj.actual.min(), pj.actual.max()],
            color=C["grey"], ls="--", lw=1)
    for _, x in pj.iterrows():
        col = ROLE["emulated"] if x.cov_reg else ROLE["noise"]
        a1.plot([x.actual, x.actual], [x.reg_min, x.reg_max], color=col, lw=2, alpha=0.7, zorder=3)
        a1.plot(x.actual, x.reg_p50, "o", color=col, ms=5, zorder=4)
    a1.set_xlabel("actual peak recovered SO$_4$ (mg/L)")
    a1.set_ylabel("regularized-projection posterior (min-median-max)")
    a1.set_title("regularized projection: min-max vs actual (green=cover, red=miss)", fontsize=10)
    fig.suptitle("DSI LOO posterior coverage -- IES vs direct projection (regularized & raw)",
                 fontweight="bold", fontsize=11)
    return savefig(fig, "dsi_loo_proj_coverage", stage)


def regen_figs(master_dir=WS5_MASTER, template_ws=WS3, frac=0.30, select_seed=0):
    """Regenerate every stage-5/6 figure on the CORRECTED setup: 120-real prior MC, locked truth
    (read from _truth), TRUE leave-one-out (drop the single real, retrain, noptmax=1). Runs the slow
    conditioning LOO exactly ONCE and reuses it. Emits:
      05_prior_mc: prior forecast distribution + recovered-SO4 spaghetti (truth highlighted)
      06_dsi:      xval projection fidelity; LOO (projection 1:1 + conditioning coverage violins);
                   truth conditioning (prior_vs_data + forecast + posterior); prior_vs_data/forecast
                   for every LOO real that fails coverage.
    """
    from pyemu.emulators import DSI
    apply_style()
    truth_real = _locked_truth_name()
    cache = _dsi_load_all(master_dir, template_ws)
    data, pst, keep, fc = cache
    meta = _dsi_meta(pst)
    print(f"[regen_figs] truth=r{truth_real}; obs_cols={len(keep)}; reals={len(data)}", flush=True)

    # --- 05: prior MC forecast fig (truth highlighted) ---
    peak, fore, times = prior_forecast(master_dir, template_ws)
    _fig_prior_mc(peak, fore, times, truth_name=truth_real)

    # --- 06: xval fidelity -- PURE leave-one-out projection (drop-one, retrain each) ---
    lx = xvalidate_dsi_loo(master_dir, template_ws, frac=1.0, cache=cache)
    _fig_dsi_xval_loo(lx)

    # --- 06: LOO coverage -- IES conditioning (SLOW, once) + direct projection (raw+reg, fast) ---
    cov, post = dsi_loo_coverage(master_dir, template_ws, frac=frac, select_seed=select_seed, cache=cache)
    proj_df, post_reg, post_raw = dsi_loo_coverage_proj(
        master_dir, template_ws, frac=frac, select_seed=select_seed, cache=cache)
    _fig_dsi_loo(lx, cov, post)
    _fig_dsi_proj_coverage(cov, proj_df, post_reg)

    # --- 06: truth conditioning -- prior_vs_data + forecast; posterior overlays IES + proj(raw+reg) ---
    dpst, fore_c, target = build_dsi_conditioning(
        master_dir, template_ws, dsi_template=WS6_DSI, truth_real=truth_real,
        data_cache=data, pst_cache=pst)
    run_dsi(WS6_DSI)
    _dsi_condition_figs(WS6_DSI, master_dir, template_ws, truth_real, f"truth_r{truth_real}", fore_c)
    pr, po, tr = dsi_posterior(WS6_DSI, forecast_cols=fore_c, truth=target)
    dsi_t = DSI(data=data.drop(index=truth_real), transforms=DSI_TRANSFORMS,
                energy_threshold=DSI_ENERGY).fit()
    raw_t, _ = dsi_project_posterior(dsi_t, data.loc[truth_real], meta, fc, mode="raw")
    reg_t, _ = dsi_project_posterior(dsi_t, data.loc[truth_real], meta, fc, mode="reg")
    _fig_dsi_posterior(pr, po, tr, raw_fc=raw_t, reg_fc=reg_t)

    # --- 06: conditioning figs for each FAILED (IES) LOO real ---
    failed = list(cov.loc[~cov.covered, "real"])
    for r in failed:
        dt = WS6_DSI.parent / f"_s6_fig_{r}"
        _dp, fore_r, _tgt = build_dsi_conditioning(
            master_dir, template_ws, dsi_template=dt, truth_real=r, data_cache=data, pst_cache=pst)
        run_dsi(dt)
        _dsi_condition_figs(dt, master_dir, template_ws, r, f"loo_r{r}", fore_r)
    print(f"[regen_figs] proj LOO coverage: reg={proj_df.cov_reg.mean():.0%} raw={proj_df.cov_raw.mean():.0%}",
          flush=True)
    print(f"[regen_figs] DONE. truth=r{truth_real} (P75); LOO coverage={cov.covered.mean():.0%} "
          f"(n={len(cov)}); failed={failed}; figs -> _figs/05_prior_mc + _figs/06_dsi", flush=True)
    return cov


# =============================================================================
# SECTION 7 -- DSIVC optimization (cost vs recovered-SO4 over the DSI emulator)
# =============================================================================
#
# The decision lever f_treat enters here. DSIVC wraps an outer pestpp-mou around the DSI emulator:
# f_treat is a controllable OBS column; each outer candidate injects it as a high-weight target,
# runs a nested pestpp-ies /e conditioning, summarizes the "stack" into percentile stack-stats, and
# the objectives sit on those. Design (ADR-0003 + user 2026-07-06, see PROGRESS.md SECTION 7):
#   * f_treat forecast-only (treatment from day 252) -> clean conditioning (built into build_injectate_solutions)
#   * training ensemble = MERGE of the f_treat=0 prior MC (Section 5) + a f_treat~U[0,0.999] sweep that
#     REUSES the same 120 param draws (paired design)
#   * objectives: min cost (EXACT, compute_cost.py 2nd command) vs min P95(peak recovered-SO4) (emulated)
#
# Sweep runs AFTER (regardless of) the first prior MC. This section is DRAFTED, not yet run: it needs
# the 120-real prior MC (_s5_master) landed first, and the DSIVC/MOU tail (build_dsivc) is a scaffold
# whose stack-stats obs names must be confirmed against the first prepare_pestpp output.

WS7_SWEEP = Path(__file__).parent / "_s7_sweep_template"     # with_treatment PEST interface (f_treat varies)
WS7_SWEEP_MASTER = Path(__file__).parent / "_s7_sweep_master"
_S7_WORKER_ROOT = Path(__file__).parent / "_s7_workers"
WS7_DSIVC = Path(__file__).parent / "_s7_dsivc_template"      # runstore DSI + DSIVC (mou) template
WS7_DSIVC_MASTER = Path(__file__).parent / "_s7_dsivc_master"
_S7_DSIVC_WORKER_ROOT = Path(__file__).parent / "_s7_dsivc_workers"
N_SWEEP = N_PRIOR_MC                                          # paired with the prior MC -- MUST match


def build_sweep_interface(model_ws=WS, template_ws=WS7_SWEEP, num_reals=N_SWEEP):
    """The DSIVC sweep PEST interface: identical obs to Section 3 PLUS the f_treat decvar (param + echoed
    obs) and the apply_treatment_forward PRE command. Just build_pest_interface(with_treatment=True)."""
    return build_pest_interface(model_ws, template_ws, num_reals=num_reals, with_treatment=True)


def draw_sweep_ensemble(sweep_pst, prior_pe_path=WS3 / "prior_pe.jcb", template_ws=WS7_SWEEP,
                        s3_template=WS3, seed=20260707):
    """Sweep parameter ensemble = the SAME 120 prior param draws + an independent f_treat~U[0,0.999]
    column (the paired design). Reuses prior_pe.jcb so params are shared with the f_treat=0 prior MC;
    the orthogonal f_treat draw lets the emulator separate param vs treatment effects.
    """
    import pyemu
    pst_s3 = pyemu.Pst(str(Path(s3_template) / "pest.pst"))
    pe_prior = pyemu.ParameterEnsemble.from_binary(pst=pst_s3, filename=str(prior_pe_path))
    df = pe_prior._df.copy()
    rng = np.random.default_rng(seed)
    df["f_treat"] = rng.uniform(F_TREAT_BOUNDS[0], F_TREAT_BOUNDS[1], size=df.shape[0])
    df = df.loc[:, list(sweep_pst.par_names)]                 # align to the sweep pst par order (+f_treat)
    pe = pyemu.ParameterEnsemble(pst=sweep_pst, df=df)
    pe.enforce()
    out = Path(template_ws) / "sweep_pe.jcb"
    pe.to_binary(str(out))
    print(f"  [draw_sweep_ensemble] {pe.shape[0]} reals x {pe.shape[1]} pars (f_treat U{F_TREAT_BOUNDS}) -> {out.name}")
    return pe


def run_dsivc_sweep(template_ws=WS7_SWEEP, num_workers=15, master_dir=WS7_SWEEP_MASTER,
                    num_reals=N_SWEEP, condor_kwargs=None):
    """Evaluate the f_treat sweep once (pestpp-ies noptmax=-1) over PANTHER workers -- the Section-5
    run_prior_mc pattern, but on the with_treatment interface + sweep_pe.jcb. Deploys via HTCondor
    when a pool is available, else local (see _deploy_pestpp)."""
    import pyemu
    template_ws, master_dir = Path(template_ws), Path(master_dir)
    pst = pyemu.Pst(str(template_ws / "pest.pst"))
    pst.pestpp_options["ies_par_en"] = "sweep_pe.jcb"
    pst.pestpp_options["ies_num_reals"] = num_reals
    pst.pestpp_options["ies_no_noise"] = True
    pst.pestpp_options["save_binary"] = True
    pst.control_data.noptmax = -1
    pst.write(str(template_ws / "pest.pst"), version=2)
    _slim_template(template_ws)                          # drop regenerated outputs -> disk headroom
    return _deploy_pestpp(template_ws, "pest.pst", master_dir, num_workers, _S7_WORKER_ROOT,
                              condor_kwargs=condor_kwargs)


def merge_training_data(prior_master=WS5_MASTER, prior_template=WS3,
                        sweep_master=WS7_SWEEP_MASTER, sweep_template=WS7_SWEEP):
    """Merge the f_treat=0 prior MC obs + the f_treat-varying sweep obs into one 240-real training
    DataFrame for the DSIVC emulator (paired: same 120 params at two f_treat conditions).

    Columns = the DSI keep-obs (weighted species/heads + forecast) shared by both, PLUS ``f_treat``
    (=0 for the prior-MC half, varying for the sweep half) PLUS a derived scalar ``fore_peak_so4``
    (= max over the forecast timeseries per real) -- the P95 of which is the MOU objective. Cost is
    NOT included (deterministic in f_treat -> computed exactly by compute_cost.py, never emulated).
    Reals are re-indexed p0..p119 (prior) / s0..s119 (sweep) so the halves never collide.
    """
    import pyemu

    def _obs_df(master, template):
        pst = pyemu.Pst(str(Path(template) / "pest.pst"))
        pst.try_parse_name_metadata()
        oe = pyemu.ObservationEnsemble.from_binary(pst=pst, filename=str(Path(master) / "pest.0.obs.jcb"))
        df = pd.DataFrame(oe.values, index=oe.index.astype(str), columns=[c.lower() for c in oe.columns])
        return pst, df

    pst_p, dp = _obs_df(prior_master, prior_template)
    pst_s, ds = _obs_df(sweep_master, sweep_template)

    keep = [c.lower() for c in _dsi_keepobs(pst_p)]          # weighted + forecast, shared by both interfaces
    fore = [c for c in keep if pst_p.observation_data.loc[c, "obgnme"] == "forecast"]
    ft_obs = [o for o in pst_s.observation_data.index if pst_s.observation_data.loc[o, "obgnme"] == "ftreat"]
    assert len(ft_obs) == 1, f"expected one ftreat obs, got {ft_obs}"
    ftreat = ft_obs[0].lower()

    dp = dp.loc[:, keep].astype(float)
    dp["f_treat"] = 0.0                                      # prior-MC half: baseline (echoed obs = 0)
    ds = ds.loc[:, keep + [ftreat]].astype(float).rename(columns={ftreat: "f_treat"})

    dp.index = [f"p{i}" for i in range(dp.shape[0])]
    ds.index = [f"s{i}" for i in range(ds.shape[0])]
    merged = pd.concat([dp, ds], axis=0)
    merged["fore_peak_so4"] = merged.loc[:, fore].max(axis=1)   # derived scalar: peak recovered-SO4
    print(f"  [merge_training_data] {merged.shape[0]} reals ({dp.shape[0]} prior + {ds.shape[0]} sweep) "
          f"x {merged.shape[1]} obs; f_treat in [{merged.f_treat.min():.3f},{merged.f_treat.max():.3f}]")
    # PRE-MOU LEVERAGE GATE: forecast-only treatment only affects post-252 water -> confirm f_treat bites.
    lev = np.corrcoef(ds["f_treat"].values, ds.loc[:, fore].max(axis=1).values)[0, 1]
    print(f"  [merge_training_data] LEVERAGE corr(f_treat, peak-SO4) on sweep half = {lev:+.3f} "
          f"({'OK' if abs(lev) > 0.2 else 'WEAK -> front may be flat, see PROGRESS RISK'})")
    return merged, fore


def build_dsivc(merged=None, dsivc_template=WS7_DSIVC, truth_real="5", so4_percentile=0.95):
    """SCAFFOLD (draft): fit the DSI on the merged 240-real ensemble, condition on the baseline truth
    (inject f_treat=0), prepare the runstore DSI, wrap DSIVC over it with f_treat as the decvar, and
    declare the two objectives for pestpp-mou. Not yet run -- the stack-stats obs names emitted by
    DSIVC.prepare_pestpp must be confirmed against the first output before the objective wiring is final.

    Steps (each maps to PROGRESS.md SECTION 7 build steps d-h):
      d. fit ONE DSI on `merged` (f_treat is a varying obs column; fore_peak_so4 is the emulated target)
      e. condition on truth: inject the baseline truth's weighted obs AND f_treat=0, run -> posterior oe
      f. DSIVC(emulator, runstore_dsi_t_d, posterior_oe).prepare_pestpp(decvar_names=["f_treat"])
      g. objectives: min `cost` (compute_cost.py 2nd command) + min `fore_peak_so4_stat:95%`; pestpp-mou
      h. FOM-validate the optimum f_treat through the full model vs the emulated P5-P95 band
    """
    from pyemu.emulators import DSI, DSIVC
    import pyemu
    if merged is None:
        merged, _fore = merge_training_data()
    dsivc_template = Path(dsivc_template)

    # (d) fit the emulator on the merged ensemble (truth held out of training)
    train = merged.drop(index=[str(truth_real)], errors="ignore")
    dsi = DSI(data=train, transforms=DSI_TRANSFORMS, energy_threshold=DSI_ENERGY).fit()
    print(f"  [build_dsivc] DSI fit: train={train.shape[0]} obs_cols={train.shape[1]} latent_dim={dsi.latent_dim}")

    # (e) TODO: condition on the baseline truth (reuse build_dsi_conditioning weighting/noise/phi-factors,
    #     but on THIS emulator and with f_treat injected = 0 as a high-weight target) -> posterior oe.
    #     dpst = dsi.prepare_pestpp(runstore_dsi_t_d, use_runstor=True); hbd.get_bins(runstore_dsi_t_d)
    #     ... set targets/weights/phi/noise exactly as build_dsi_conditioning ... + f_treat target=0 ...
    #     run_dsi(runstore_dsi_t_d); posterior_oe = ObservationEnsemble.from_binary(dsi.<last>.obs.jcb)
    #
    # (f) DSIVC over the runstore DSI, f_treat as the controllable decvar
    #     dv = DSIVC(emulator=dsi, dsi_t_d=str(runstore_dsi_t_d), oe=posterior_oe)
    #     pst_mou = dv.prepare_pestpp(str(dsivc_template), decvar_names=["f_treat"],
    #                                 percentiles=(0.05, so4_percentile))   # P5/P95 stack-stats
    #
    # (g) objectives on the stack-stats + exact cost:
    #     - add compute_cost.py as a 2nd model command (reads the injected f_treat -> writes cost obs)
    #     - obj obs group 'less_than' (minimize), non-zero weight, listed in mou_objectives:
    #         cost                            (exact, from compute_cost.py)
    #         fore_peak_so4_stat:95%          (emulated risk band; NAME to confirm from prepare output)
    #     - pst_mou.pestpp_options['mou_objectives'] = 'cost,fore_peak_so4_stat:95%'
    #     - DEPLOY the outer pestpp-mou via run_dsivc_mou (HTCondor pool when available, else local) ->
    #       1-D convex cost-vs-SO4 risk-banded front
    #
    # (h) FOM validation: run the optimum f_treat through the full model, check the peak lands in P5-P95.
    print("  [build_dsivc] SCAFFOLD -- steps (e)-(h) pending the landed prior MC + sweep; see docstring.")
    return dsi, train


def run_dsivc_mou(dsivc_template=WS7_DSIVC, master_dir=WS7_DSIVC_MASTER, num_workers=None,
                  condor_kwargs=None):
    """Deploy the DSIVC OUTER optimization (pestpp-mou over the emulator) across workers -- HTCondor
    when a pool is available, else local. Same routing as the FOM ensembles, but pestpp_exe='pestpp-mou'.
    Each MOU candidate's forward run injects the decvars and runs a nested ``pestpp-ies dsi.pst /e``
    conditioning (emulator-only, so a slot is light -- override CONDOR_DEFAULTS memory/disk via
    condor_kwargs if you want to pack more per node)."""
    import os
    if num_workers is None:
        num_workers = CONDOR_DEFAULTS["n_workers"] if _htcondor_available() else max(1, (os.cpu_count() or 2) - 1)
    return _deploy_pestpp(dsivc_template, "dsivc.pst", master_dir, num_workers, _S7_DSIVC_WORKER_ROOT,
                          condor_kwargs=condor_kwargs, pestpp_exe="pestpp-mou")


def stage7_dsivc(num_workers=None, condor_kwargs=None):
    """Section-7 orchestrator (draft): build sweep interface -> reuse-120 sweep ensemble -> run sweep ->
    merge with the prior MC -> build DSIVC. Requires the Section-5 prior MC (_s5_master) already landed.
    Worker count defaults to CONDOR_DEFAULTS (60) on a pool, else cpu_count-1 (same rule as stage6_fom)."""
    import os
    apply_style()
    if num_workers is None:
        num_workers = CONDOR_DEFAULTS["n_workers"] if _htcondor_available() else max(1, (os.cpu_count() or 2) - 1)
    _pf, sweep_pst = build_sweep_interface()
    draw_sweep_ensemble(sweep_pst)
    run_dsivc_sweep(num_workers=num_workers, condor_kwargs=condor_kwargs)
    merged, _fore = merge_training_data()
    return build_dsivc(merged)


def run_all(num_reals=N_PRIOR_MC, num_workers=None, condor_kwargs=None, quantile=0.75):
    """ONE-SHOT DRIVER (``--all``): run the whole DIZON arc, stages 2 -> 7, in order, NO STOPS. Each
    stage writes the workspace the next consumes (mothership pattern). The heavy stages (prior MC,
    full-model IES history match, DSIVC sweep) auto-deploy to HTCondor when a pool is reachable, else
    to local PANTHER workers -- so this same command scales from a laptop to a cluster. End-to-end
    locally is many hours of full-model runs; on a pool it is the intended way to deploy the workflow.

    Stage 4 (lock truth + weights) is sequenced correctly here -- AFTER the prior MC (it needs the
    prior forecast to pick the P-quantile truth) and BEFORE stage 6.
    """
    def _hdr(s):
        print(f"\n{'=' * 72}\n[run_all] {s}\n{'=' * 72}", flush=True)

    _hdr("stage 2 -- model build + baseline run")
    stage2_build(run_model=True)
    _hdr("stage 3 -- PstFrom PEST interface")
    stage3_pstfrom()
    _hdr(f"stage 5 -- prior Monte Carlo ({num_reals} reals)")
    stage5_prior_mc(num_reals=num_reals, num_workers=num_workers, condor_kwargs=condor_kwargs)
    _hdr("stage 4 -- lock synthetic truth + inject weights")
    stage4_truth_weights(quantile=quantile)
    _hdr("stage 6 -- DSI condition + cross-validation + signature figures")
    regen_figs()
    _hdr("stage 6-FOM -- full-model IES history match (DSI gold-standard)")
    stage6_fom(num_workers=num_workers, condor_kwargs=condor_kwargs)
    _hdr("stage 7 -- DSIVC f_treat sweep + merge (240-real) + optimizer")
    stage7_dsivc(num_workers=num_workers, condor_kwargs=condor_kwargs)
    _hdr("DONE -- full DIZON arc complete")


if __name__ == "__main__":
    _run = "--run" in sys.argv
    _rebuild = "--rebuild" in sys.argv
    if "--all" in sys.argv:                                  # one-shot: whole arc, stages 2->7, no stops
        run_all()
    elif "--stage7sweep" in sys.argv:                        # build + run the f_treat sweep only
        _pf, _spst = build_sweep_interface()
        draw_sweep_ensemble(_spst)
        run_dsivc_sweep()
    elif "--stage7merge" in sys.argv:                        # merge + leverage gate (after sweep lands)
        merge_training_data()
    elif "--stage7" in sys.argv:
        stage7_dsivc()
    elif "--reinflate" in sys.argv:                          # test reinflation on failed LOO xvals
        reinflate_test()
    elif "--regen" in sys.argv:                              # regenerate all stage5/6 figs (corrected LOO)
        regen_figs()
    elif "--stage6figs" in sys.argv:
        stage6_figs()
    elif "--stage6loo" in sys.argv:
        _lx = xvalidate_dsi_loo(frac=1.0)
        _cv, _post = dsi_loo_coverage(frac=0.30)
        _fig_dsi_loo(_lx, _cv, _post)
        print(f"[stage6loo] LOO projection R={np.corrcoef(_lx.actual,_lx.emulated)[0,1]:.3f}; "
              f"LOO conditioning coverage={_cv.covered.mean():.0%}")
    elif "--stage6" in sys.argv:                             # DSI condition on the LOCKED truth + all figs
        regen_figs()                                         #   (same as run_all's stage 6, not truth_real=5)
    elif "--stage6fom" in sys.argv:                          # FULL-MODEL IES history match (DSI comparison)
        stage6_fom()
    elif "--stage5" in sys.argv:
        stage5_prior_mc()                                    # N_PRIOR_MC reals (same as run_all)
    elif "--stage4" in sys.argv:                             # lock truth + inject weights (after prior MC)
        stage4_truth_weights()
    elif "--stage3" in sys.argv:
        stage3_pstfrom()
    else:                                                    # stage 2: build + RUN the model (as run_all does)
        stage2_build(rebuild=_rebuild, run_model=True)
