"""
DSI variable control (DSIVC).

A workflow builder that wraps an outer PESTPP-MOU optimization around a fitted
DSI-family emulator.  Decision variables are *observations* in DSI-world (columns
of the training ensemble a manager can control).  Evaluating one outer candidate
means injecting its decision-variable values as high-weight, zero-noise observation
targets into the inner ``dsi.pst`` and running a full nested PESTPP-IES conditioning
in runstore mode (``pestpp-ies dsi.pst /e``).  The conditioned posterior ensemble
(the "stack") is summarized into per-output percentile observations (the "stack
stats"), which are the outer problem's observations.  The user defines objectives
and constraints on those stack-stats manually.
"""
from __future__ import print_function, division
import os
import shutil
import numpy as np
import pandas as pd

from pyemu.logger import Logger
from pyemu.pst.pst_handler import Pst
from pyemu.pst.pst_utils import csv_tpl_from_parnames
from pyemu.en import ObservationEnsemble, ParameterEnsemble
from pyemu.utils.helpers import series_to_insfile
from .base import Emulator

# On-disk filenames used by the DSIVC workflow (names reused exactly from the
# previous implementation for continuity).  NOTE: the forward-run functions at
# the bottom of this module hardcode these same names as string literals — they
# are embedded standalone into the generated dsivc_forward_run.py and CANNOT
# reference these constants.  If you rename a constant, update the matching
# literals in the embedded functions or prepare/run will silently desync.
DSI_PST = "dsi.pst"
DSI_PICKLE = "dsi.pickle"
DSI_FORWARD_RUN = "forward_run.py"
DSIVC_PST = "dsivc.pst"
DSIVC_PARS_CSV = "dsivc_pars.csv"
DSIVC_FORWARD_RUN = "dsivc_forward_run.py"
STACK_STATS_CSV = "dsi.stack_stats.csv"
STACK_CSV = "dsi.stack.csv"
INITIAL_DVPOP = "initial_dvpop.jcb"
NOISE_BASE_JCB = "dsi.noise_base.jcb"  # prepare-time noise source
NOISE_JCB = "dsi.noise.jcb"            # per-call noise written by the forward run


class DSIVC:
    """DSI variable control: an optimization workflow over a fitted DSI emulator.

    Parameters
    ----------
    emulator : DSI or DSIAE
        A fitted DSI-family emulator (``emulator.fitted`` must be True).
    dsi_t_d : str
        Existing template directory holding a runstore-prepared DSI interface
        (``dsi.pst``, ``dsi.pickle``, ``forward_run.py``).  Never modified;
        ``prepare_pestpp`` copies it into a fresh directory.
    oe : ObservationEnsemble
        A (typically posterior) DSI observation ensemble.  Its columns must
        match the DSI observation names exactly (the run-time stack covers all
        inner observations); decision variables are drawn from its columns and
        the stack-stats observations are summarized from it.
    verbose : bool, optional
        If True, enable verbose logging.  Default is False.
    """

    def __init__(self, emulator, dsi_t_d, oe, verbose=False):
        self.logger = Logger(verbose)
        self.log = self.logger.log

        if not getattr(emulator, "fitted", False):
            raise ValueError(
                "emulator must be a fitted DSI-family emulator "
                "(emulator.fitted is True); call fit() first")
        self.emulator = emulator

        if not os.path.isdir(dsi_t_d):
            raise FileNotFoundError(f"dsi_t_d directory not found: {dsi_t_d}")
        for fname in (DSI_PST, DSI_PICKLE, DSI_FORWARD_RUN):
            fpath = os.path.join(dsi_t_d, fname)
            if not os.path.exists(fpath):
                raise FileNotFoundError(
                    f"required file not found in dsi_t_d: {fpath}")
        self.dsi_t_d = dsi_t_d

        # The DSI template must be runstore-prepared: the inner run is runstore-only
        # (pestpp-ies dsi.pst /e).  Validate the artifact by checking forward_run.py
        # targets dsi_runstore_forward_run.  This is precondition validation, not
        # runtime dispatch.
        with open(os.path.join(dsi_t_d, DSI_FORWARD_RUN), "r") as f:
            frun_src = f.read()
        if "dsi_runstore_forward_run" not in frun_src:
            raise ValueError(
                f"{DSI_FORWARD_RUN} in {dsi_t_d} is not runstore-prepared "
                "(its __main__ does not call dsi_runstore_forward_run); "
                "re-run DSI.prepare_pestpp(..., use_runstor=True)")

        if not isinstance(oe, ObservationEnsemble):
            raise TypeError("oe must be a pyemu.ObservationEnsemble")
        self.pst = Pst(os.path.join(dsi_t_d, DSI_PST))
        # oe columns must EQUAL the DSI observation set (not a subset): the
        # run-time stack covers all inner observations, and the prepare-time
        # ins files must enumerate exactly the same names, in pst.obs_names
        # order at both ends (the ins read is positional).
        oe_cols = set(oe.columns)
        obs_names = set(self.pst.obs_names)
        missing = sorted(obs_names - oe_cols)
        extra = sorted(oe_cols - obs_names)
        if missing or extra:
            hint = ""
            if any(str(c) != str(c).lower() for c in extra):
                hint = " (note: DSI observation names are lowercase)"
            raise ValueError(
                "oe columns must match the DSI observation names exactly; "
                f"missing from oe: {missing}; not DSI observations: {extra}{hint}")
        self.oe = oe

    def prepare_pestpp(self, t_d, decvar_names, percentiles=(0.25, 0.5, 0.75),
                       track_stack=False, inner_noptmax=3, decvar_weight=1.0,
                       inner_num_reals=None, mou_population_size=None,
                       seed=358, ies_exe_path="pestpp-ies"):
        """Build the outer PESTPP-MOU interface for a DSIVC optimization.

        Parameters
        ----------
        t_d : str
            Fresh template directory to create.  Must differ from ``dsi_t_d``.
            If it exists it is removed.
        decvar_names : str or list of str
            Decision variables: DSI observations the manager controls.  Must be
            zero-weight in the inner ``dsi.pst`` (a weighted decvar would be both a
            history-matching target and a control).
        percentiles : iterable of float, optional
            Percentiles in [0, 1] used to summarize the stack into stack-stats.
            Default ``(0.25, 0.5, 0.75)``.
        track_stack : bool, optional
            If True, also emit every realization of the stack as observations.
            The stack obs are positions ``0..(inner n_reals - 1)`` per output
            (POSITIONAL naming): the run-time stack is the inner IES posterior
            whose row labels IES controls, so realization identities are dropped
            and the obs COUNT (the inner n_reals decided in the noise step) is the
            contract.  Default False.
        inner_noptmax : int, optional
            ``noptmax`` for the inner PESTPP-IES conditioning.  Must be >= 1.
            Default 3.
        decvar_weight : float, optional
            Observation weight assigned to the decvars in the inner ``dsi.pst``.
            Must be > 0.  Default 1.0.
        inner_num_reals : int, optional
            Number of inner realizations / noise realizations.  Defaults to the
            harvested noise source size, or the ``oe`` row count when drawing.
        mou_population_size : int, optional
            Outer MOU population size.  Defaults to ``2 * len(decvar_names)``.
        seed : int, optional
            Seed for deterministic noise draws and the initial decvar population.
            Default 358.
        ies_exe_path : str, optional
            Path to the PESTPP-IES executable used by the inner run.
            Default "pestpp-ies".

        Returns
        -------
        Pst
            The outer DSIVC control file (``dsivc.pst``).  ``noptmax`` is 0; the
            user must still define objectives and constraints.
        """
        # --- validation -----------------------------------------------------
        if isinstance(decvar_names, str):
            decvar_names = [decvar_names]
        if not isinstance(decvar_names, (list, tuple)):
            raise TypeError("decvar_names must be a str or a list of str")
        decvar_names = [str(d).lower() for d in decvar_names]
        if len(decvar_names) == 0:
            raise ValueError("decvar_names must be non-empty")
        if len(set(decvar_names)) != len(decvar_names):
            dupes = sorted({d for d in decvar_names if decvar_names.count(d) > 1})
            raise ValueError(f"duplicate decvar names: {dupes}")

        missing = [d for d in decvar_names if d not in self.oe.columns]
        if missing:
            raise ValueError(f"decvars not found in oe columns: {missing}")
        inner_obs = self.pst.observation_data
        missing = [d for d in decvar_names if d not in inner_obs.index]
        if missing:
            raise ValueError(
                f"decvars not found in inner dsi.pst observations: {missing}")
        weighted = [d for d in decvar_names if float(inner_obs.loc[d, "weight"]) != 0.0]
        if weighted:
            raise ValueError(
                "decvars must be zero-weight in the inner dsi.pst "
                f"(these are weighted history-matching targets): {weighted}")

        percentiles = np.asarray(list(percentiles), dtype=float)
        if percentiles.size == 0:
            raise ValueError("percentiles must be non-empty")
        if np.any((percentiles < 0.0) | (percentiles > 1.0)):
            raise ValueError("percentiles must all be in [0, 1]")
        percentiles = np.unique(percentiles)

        if not isinstance(inner_noptmax, (int, np.integer)) or inner_noptmax < 1:
            raise ValueError(
                "inner_noptmax must be an integer >= 1 "
                "(0 would silently disable the inner conditioning)")
        if decvar_weight <= 0:
            raise ValueError("decvar_weight must be > 0")

        if mou_population_size is None:
            mou_population_size = 2 * len(decvar_names)
        elif mou_population_size < 2 * len(decvar_names):
            self.logger.warn(
                f"mou_population_size ({mou_population_size}) is smaller than "
                f"2*len(decvar_names) ({2 * len(decvar_names)}); this may be too small")

        # --- 1. fresh template dir -----------------------------------------
        if os.path.abspath(t_d) == os.path.abspath(self.dsi_t_d):
            raise ValueError("t_d must differ from dsi_t_d")
        if os.path.exists(t_d):
            shutil.rmtree(t_d)
        shutil.copytree(self.dsi_t_d, t_d)
        self.logger.statement(f"copied DSI template {self.dsi_t_d} -> {t_d}")

        inner_pst = Pst(os.path.join(t_d, DSI_PST))

        # --- 2. noise source (hybrid: harvest or draw) ---------------------
        harvest_jcb = os.path.join(t_d, "dsi.obs+noise.jcb")
        harvest_csv = os.path.join(t_d, "dsi.obs+noise.csv")
        if os.path.exists(harvest_jcb):
            self.logger.statement(f"harvesting noise from {harvest_jcb}")
            noise = ObservationEnsemble.from_binary(inner_pst, harvest_jcb)
            source_reals = noise.shape[0]
            if inner_num_reals is not None:
                if inner_num_reals > source_reals:
                    raise ValueError(
                        f"inner_num_reals ({inner_num_reals}) exceeds harvested "
                        f"noise realizations ({source_reals})")
                noise = ObservationEnsemble(pst=inner_pst,
                                            df=noise._df.iloc[:inner_num_reals])
            n_reals = noise.shape[0]
        elif os.path.exists(harvest_csv):
            self.logger.statement(f"harvesting noise from {harvest_csv}")
            noise = ObservationEnsemble.from_csv(inner_pst, harvest_csv)
            source_reals = noise.shape[0]
            if inner_num_reals is not None:
                if inner_num_reals > source_reals:
                    raise ValueError(
                        f"inner_num_reals ({inner_num_reals}) exceeds harvested "
                        f"noise realizations ({source_reals})")
                noise = ObservationEnsemble(pst=inner_pst,
                                            df=noise._df.iloc[:inner_num_reals])
            n_reals = noise.shape[0]
        else:
            n_reals = inner_num_reals if inner_num_reals is not None else self.oe.shape[0]
            self.logger.statement(f"drawing {n_reals} noise realizations from inner pst weights")
            rng = np.random.RandomState(seed)
            noise = ObservationEnsemble.from_gaussian_draw(
                inner_pst, num_reals=n_reals, fill=True, rng=rng)
        noise.to_binary(os.path.join(t_d, NOISE_BASE_JCB))

        # --- 3. stack-stats observations -----------------------------------
        # _dsivc_stack_stats is the single shared naming/summary function (used at
        # both prepare time and run time), so the prepare-time obs names and the
        # run-time output names can never diverge.
        # The ins read is POSITIONAL, so both ends must also summarize columns in
        # the same order: pst.obs_names is the canonical order (the run-time
        # posterior is reindexed to it inside dsivc_forward_run).
        oe_df = self.oe._df.loc[:, inner_pst.obs_names]
        stack_stats = _dsivc_stack_stats(oe_df, percentiles)
        out_files = []
        ss_file = os.path.join(t_d, STACK_STATS_CSV)
        stack_stats.to_csv(ss_file, float_format="%.6e")
        series_to_insfile(ss_file, ins_file=None)
        out_files.append(ss_file)
        # explicit obsnme -> (org_obsnme, stat) mapping built directly from the
        # melt (no string prefix matching).  Its obsnme index must match the
        # shared function's output exactly.
        ss_map = self._stack_stats_mapping(oe_df, percentiles)
        if list(ss_map.index) != list(stack_stats.index):
            raise RuntimeError(
                "stack-stats obsnme mapping diverged from the shared naming "
                "function; this is a bug")

        # --- 4. optional per-realization stack observations ----------------
        # The stack obs are positions 0..(inner n_reals - 1) per column.  Naming
        # is POSITIONAL via the shared _dsivc_stack_long: the run-time stack is
        # the inner IES posterior whose row labels IES controls and need not match
        # self.oe's labels, so identities are dropped and the COUNT is the contract.
        track_map = None
        if track_stack:
            stack, track_map = self._stack_mapping(oe_df, n_reals)
            stk_file = os.path.join(t_d, STACK_CSV)
            stack.to_csv(stk_file, float_format="%.6e")
            series_to_insfile(stk_file, ins_file=None)
            out_files.append(stk_file)
            if list(track_map.index) != list(stack.index):
                raise RuntimeError(
                    "stack obsnme mapping diverged from the shared naming "
                    "function; this is a bug")

        # --- 5. decvar template + input file -------------------------------
        in_file = os.path.join(t_d, DSIVC_PARS_CSV)
        tpl_file = in_file + ".tpl"
        train = self.emulator.data
        csv_tpl_from_parnames(decvar_names, tpl_file)
        with open(in_file, "w") as fin:
            fin.write("parnme,parval1\n")
            for dv in decvar_names:
                init = float(train.loc[:, dv].median())
                fin.write(f"{dv},{init:.6e}\n")

        # --- 6. build the outer pst ----------------------------------------
        pst_dsivc = Pst.from_io_files(
            [tpl_file], [in_file],
            [f + ".ins" for f in out_files], out_files, pst_path=".")

        par = pst_dsivc.parameter_data
        par.loc[:, "partrans"] = "fixed"
        par.loc[decvar_names, "pargp"] = "decvars"
        par.loc[decvar_names, "partrans"] = "none"
        par.loc[decvar_names, "parlbnd"] = train.loc[:, decvar_names].min()
        par.loc[decvar_names, "parubnd"] = train.loc[:, decvar_names].max()
        par.loc[decvar_names, "parval1"] = train.loc[:, decvar_names].median()

        # --- 7. observation data: weights, metadata from explicit maps ------
        obs = pst_dsivc.observation_data
        obs.loc[:, "weight"] = 0.0
        obs.loc[ss_map.index, "org_obsnme"] = ss_map["org_obsnme"].values
        obs.loc[ss_map.index, "stat"] = ss_map["stat"].values
        obs.loc[ss_map.index, "obgnme"] = "stack_stats"
        if track_map is not None:
            obs.loc[track_map.index, "org_obsnme"] = track_map["org_obsnme"].values
            obs.loc[track_map.index, "real"] = track_map["real"].values
            obs.loc[track_map.index, "obgnme"] = "stack"

        # --- 8. initial decvar population for MOU --------------------------
        rng = np.random.RandomState(seed)
        dvpop = ParameterEnsemble.from_uniform_draw(
            pst_dsivc, num_reals=mou_population_size, rng=rng)
        dvpop.to_binary(os.path.join(t_d, INITIAL_DVPOP))
        pst_dsivc.pestpp_options["mou_dv_population_file"] = INITIAL_DVPOP
        pst_dsivc.pestpp_options["mou_population_size"] = mou_population_size
        pst_dsivc.pestpp_options["mou_save_population_every"] = 1

        # --- 9. generated runner script ------------------------------------
        # Embed the source of the forward-run functions (rather than importing
        # them) so the script is robust to PEST++ shelling out with a bare `python`
        # that may resolve a different pyemu install.  All run-time config is passed
        # as literal kwargs.
        pst_dsivc.model_command = f"python {DSIVC_FORWARD_RUN}"
        pct_literal = "[" + ", ".join(repr(float(p)) for p in percentiles) + "]"
        # inner_noptmax is intentionally NOT baked into call_args: it is written
        # into the t_d copy of dsi.pst (step 10), which is the single source of
        # truth.  dsivc_forward_run reads noptmax from the loaded dsi.pst, so a
        # later edit to the pst's noptmax stays consistent with the
        # dsi.{noptmax}.obs.jcb the inner run produces.
        call_args = (
            f"ies_exe_path={ies_exe_path!r}, percentiles={pct_literal}, "
            f"track_stack={bool(track_stack)!r}")
        Emulator._write_forward_run_script_body(
            os.path.join(t_d, DSIVC_FORWARD_RUN),
            [_dsivc_inject_decvars, _dsivc_stack_stats, _dsivc_stack_long,
             dsivc_forward_run],
            "dsivc_forward_run",
            call_args)

        # --- 10. update the inner dsi.pst copy in t_d ----------------------
        inner_obs = inner_pst.observation_data
        inner_obs.loc[decvar_names, "weight"] = decvar_weight
        inner_pst.control_data.noptmax = inner_noptmax
        inner_pst.pestpp_options["ies_observation_ensemble"] = NOISE_JCB
        inner_pst.pestpp_options["ies_num_reals"] = n_reals
        inner_pst.write(os.path.join(t_d, DSI_PST), version=2)

        # --- 11. finalize outer pst ----------------------------------------
        pst_dsivc.control_data.noptmax = 0
        pst_dsivc.write(os.path.join(t_d, DSIVC_PST), version=2)
        self.logger.statement(
            "DSIVC control files created; the user must still define "
            "objectives and constraints on the stack-stats observations")
        return pst_dsivc

    @staticmethod
    def _stack_stats_mapping(oe_df, percentiles):
        """Explicit obsnme -> (org_obsnme, stat) mapping for stack-stats.

        Built directly from the describe/melt (no string prefix matching).
        The 'count' row is dropped exactly as in _dsivc_stack_stats.
        """
        desc = oe_df.describe(percentiles=percentiles).drop(index="count")
        desc = desc.reset_index().melt(id_vars="index")
        desc = desc.rename(columns={"index": "stat", "variable": "org_obsnme"})
        desc["obsnme"] = desc["org_obsnme"] + "_stat:" + desc["stat"]
        return desc.set_index("obsnme")[["org_obsnme", "stat"]]

    @staticmethod
    def _stack_mapping(oe_df, n_reals):
        """Per-realization stack series and its obsnme -> (org_obsnme, real) map.

        The stack obs count is the contract: there must be exactly ``n_reals``
        positions per column (the inner-run size).  ``oe_df`` is shaped to
        ``n_reals`` rows positionally first -- truncated if longer, padded with
        per-column means if shorter (those initial values are cosmetic
        placeholders for zero-weight obs).  Naming is POSITIONAL via the shared
        ``_dsivc_stack_long`` (the 'real' metadata is the 0-based position as a
        string), so prepare-time and run-time obs names cannot diverge.
        """
        df = oe_df.copy().reset_index(drop=True)
        if df.shape[0] > n_reals:
            df = df.iloc[:n_reals]
        elif df.shape[0] < n_reals:
            means = df.mean(axis=0)
            for i in range(df.shape[0], n_reals):
                df.loc[i] = means
        stack = _dsivc_stack_long(df)
        # derive the positional obsnme -> (org_obsnme, real) map from the same
        # shape, mirroring _dsivc_stack_long's row-major iteration order.
        records = []
        for i in range(df.shape[0]):
            for col in df.columns:
                records.append((f"{col}_real:{i}", col, str(i)))
        mapping = pd.DataFrame(records, columns=["obsnme", "org_obsnme", "real"])
        mapping = mapping.set_index("obsnme")[["org_obsnme", "real"]]
        return stack, mapping


# ---------------------------------------------------------------------------
# Forward-run functions.
#
# These are embedded (via inspect.getsource) into the generated
# dsivc_forward_run.py, so each does its imports INSIDE the function body and
# references only each other plus stdlib / pandas / pyemu.  Filenames are
# hardcoded literals here on purpose: the embedded source cannot reference the
# module constants above — the literals MUST be kept byte-identical to them.
# ---------------------------------------------------------------------------

def _dsivc_inject_decvars(pst, decvars, ws='.'):
    """Inject decision-variable values into the inner dsi.pst and noise file.

    Parameters
    ----------
    pst : pyemu.Pst
        The inner dsi.pst (loaded from ws).
    decvars : pandas.Series
        Decision-variable values indexed by parnme (read from dsivc_pars.csv).
    ws : str, optional
        Working directory holding dsi.noise_base.jcb.  Default '.'.

    Returns
    -------
    pyemu.ObservationEnsemble
        The per-call noise ensemble (with decvar columns set to the scalar
        targets, zero noise), also written to dsi.noise.jcb.
    """
    import os
    import pyemu

    obs = pst.observation_data
    missing = [d for d in decvars.index if d not in obs.index]
    if missing:
        raise ValueError(f"decvars not found in observation data: {missing}")
    unweighted = [d for d in decvars.index if float(obs.loc[d, "weight"]) <= 0.0]
    if unweighted:
        raise ValueError(
            f"decvars must have weight > 0 in the inner pst: {unweighted}")

    # assign from a Series aligned on index (pandas>=2 safe; never an (n,1) array)
    obs.loc[decvars.index, "obsval"] = decvars

    noise = pyemu.ObservationEnsemble.from_binary(
        pst, os.path.join(ws, "dsi.noise_base.jcb"))
    missing = [d for d in decvars.index if d not in noise._df.columns]
    if missing:
        raise ValueError(f"decvars not found in noise ensemble columns: {missing}")
    for col in decvars.index:
        noise._df[col] = float(decvars.loc[col])  # whole column -> scalar, zero noise
    noise.to_binary(os.path.join(ws, "dsi.noise.jcb"))
    return noise


def _dsivc_stack_stats(oe_df, percentiles):
    """Summarize a stack DataFrame into a stack-stats Series.

    The single shared naming/summary function: prepare time and run time both
    call this, so prepare-time obs names and run-time output names cannot diverge.

    Parameters
    ----------
    oe_df : pandas.DataFrame
        The stack (rows = realizations, columns = outputs).
    percentiles : iterable of float
        Percentiles in [0, 1].

    Returns
    -------
    pandas.Series
        Stack-stats indexed by ``f"{col}_stat:{stat}"`` (pandas describe stat
        keys: mean/std/min/<pct>%/max; the realization 'count' row is dropped —
        it differs between prepare time and run time and is useless as an
        objective or constraint).
    """
    desc = oe_df.describe(percentiles=percentiles).drop(index="count")
    desc = desc.reset_index().melt(id_vars="index")
    desc = desc.rename(columns={"index": "stat", "variable": "org_obsnme",
                                "value": "obsval"})
    desc["obsnme"] = desc["org_obsnme"] + "_stat:" + desc["stat"]
    series = desc.set_index("obsnme")["obsval"]
    series.name = "obsval"
    return series


def _dsivc_stack_long(oe_df):
    """Flatten a stack DataFrame into a long Series with POSITIONAL real names.

    The single shared naming function for the per-realization stack: prepare
    time and run time both call this, so the obs names cannot diverge.

    Naming is POSITIONAL (``f"{col}_real:{i}"`` with i the 0-based row position)
    rather than label-based on purpose.  The run-time stack is the inner IES
    posterior, whose row labels IES controls (drawn indices, a possible 'base'
    real, etc.) and which need not match the prepare-time ensemble's labels.
    Realizations are exchangeable samples, so their identities carry no meaning
    for the outer problem; leaking IES-controlled labels into the obs names would
    break the ins read.  The contract is the row COUNT, not the labels.

    Parameters
    ----------
    oe_df : pandas.DataFrame
        The stack (rows = realizations, columns = outputs).

    Returns
    -------
    pandas.Series
        Stack values indexed by ``f"{col}_real:{i}"`` where i is the 0-based
        positional row number (row order as-is).
    """
    import pandas as pd

    rows = []
    for i, (_, row) in enumerate(oe_df.iterrows()):
        for col in oe_df.columns:
            rows.append((f"{col}_real:{i}", row[col]))
    series = pd.Series({name: val for name, val in rows})
    series.index.name = "obsnme"
    series.name = "obsval"
    return series


def dsivc_forward_run(ies_exe_path='pestpp-ies', percentiles=(0.25, 0.5, 0.75),
                      track_stack=False, ws='.'):
    """Outer-MOU model run: condition the DSI emulator on the candidate decvars.

    Injects the decision-variable values into dsi.pst as high-weight zero-noise
    targets, runs the inner PESTPP-IES conditioning in runstore mode, and
    summarizes the conditioned posterior into stack-stats.

    Parameters
    ----------
    ies_exe_path : str, optional
        PESTPP-IES executable.  Default 'pestpp-ies'.
    percentiles : iterable of float, optional
        Percentiles for the stack-stats summary.  Default ``(0.25, 0.5, 0.75)``.
    track_stack : bool, optional
        If True, also write the full stack (dsi.stack.csv).  Default False.
    ws : str, optional
        Working directory.  Default '.'.
    """
    import os
    import pandas as pd
    import pyemu

    pst = pyemu.Pst(os.path.join(ws, "dsi.pst"))

    # iteration obs files are dsi.<N>.obs.jcb|csv.  The inner IES may stop
    # BEFORE noptmax (phi-based termination criteria are on by default), so the
    # produced iterations are discovered from disk, never assumed from noptmax.
    def _iteration_obs_files(ws):
        found = {}
        for fname in os.listdir(ws):
            parts = fname.split(".")
            if (len(parts) == 4 and parts[0] == "dsi" and parts[1].isdigit()
                    and parts[2] == "obs" and parts[3] in ("jcb", "csv")):
                found.setdefault(int(parts[1]), []).append(fname)
        return found

    # remove stale outputs (silent if absent), including ALL iteration obs
    # files from any previous call
    stale = ["dsi.stack_stats.csv", "dsi.stack.csv", "dsi.noise.jcb"]
    stale += [f for fnames in _iteration_obs_files(ws).values() for f in fnames]
    for fname in stale:
        fpath = os.path.join(ws, fname)
        if os.path.exists(fpath):
            os.remove(fpath)

    # read decvars as a Series (index parnme, single value column)
    decvars = pd.read_csv(os.path.join(ws, "dsivc_pars.csv"), index_col=0).iloc[:, 0]
    if decvars.shape[0] == 0:
        raise ValueError("no decvars found in dsivc_pars.csv")

    noise = _dsivc_inject_decvars(pst, decvars, ws=ws)
    pst.write(os.path.join(ws, "dsi.pst"), version=2)

    # runstore-only inner run
    pyemu.os_utils.run(f"{ies_exe_path} dsi.pst /e", cwd=ws)

    # load the conditioned posterior (the stack) from the LAST iteration the
    # inner IES actually produced.  Iteration 0 is the unconditioned prior, so
    # a run that produced nothing past 0 failed to condition.
    produced = _iteration_obs_files(ws)
    last_iter = max(produced) if produced else None
    if last_iter is None or last_iter < 1:
        raise FileNotFoundError(
            "inner IES run failed: no post-iteration dsi.<N>.obs.[jcb|csv] "
            f"produced in {ws} (iterations found: {sorted(produced)})")
    jcb = os.path.join(ws, f"dsi.{last_iter}.obs.jcb")
    csv = os.path.join(ws, f"dsi.{last_iter}.obs.csv")
    if os.path.exists(jcb):
        oe = pyemu.ObservationEnsemble.from_binary(pst, jcb)
    else:
        oe = pyemu.ObservationEnsemble.from_csv(pst, csv)
    if oe.shape[0] != noise.shape[0]:
        raise ValueError(
            f"stack rows ({oe.shape[0]}) do not match noise rows "
            f"({noise.shape[0]}); failed inner runs?")

    # reindex to the canonical pst.obs_names column order: the prepare-time ins
    # files were built in this order and the ins read is positional
    oe_df = oe._df.loc[:, pst.obs_names]

    # write stack-stats (shared naming function with prepare time)
    stack_stats = _dsivc_stack_stats(oe_df, percentiles)
    stack_stats.to_csv(os.path.join(ws, "dsi.stack_stats.csv"), float_format="%.6e")

    if track_stack:
        # POSITIONAL naming: the inner-run posterior row labels are IES-controlled
        # and must not leak into obs names (see _dsivc_stack_long).
        stack = _dsivc_stack_long(oe_df)
        stack.to_csv(os.path.join(ws, "dsi.stack.csv"), float_format="%.6e")
