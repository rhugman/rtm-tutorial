"""Shared helpers for the DIZON tutorial series.

Functions are grouped into three sections:

* model build helpers — called interactively while constructing the flopy/mf6rtm
  model in the notebooks;
* runtime / forward-run helpers — the functions injected into the PEST++ forward
  run by ``PstFrom.add_py_function``; these MUST be self-contained (all imports
  inside the function body) because PstFrom serialises only the function source;
* misc — everything else.

There is no module-level state: heavy objects such as ``PestUtilsLib`` are
instantiated inside the function that needs them.
"""

import os
import numpy as np
import pandas as pd
import flopy
import platform
import shutil
import geopandas as gpd
from pathlib import Path
from collections.abc import Iterable
from collections import defaultdict
from mf6rtm import mup3d, utils
from flopy.utils.gridintersect import GridIntersect
from shapely.geometry import LineString


# --- model build helpers ---

def create_reactive_tsteps(perioddata, output_interval=5):
    """Build the (kper, kstp) pairs at which reactive output is written.

    Walks the stress-period definition day by day and emits a time-step
    reference every ``output_interval`` days.

    Parameters
    ----------
    perioddata : list of tuple
        MODFLOW 6 ``perioddata``; each entry is ``(perlen, nstp, tsmult)``.
    output_interval : int, optional
        Number of days between successive output time steps (default 5).

    Returns
    -------
    list of tuple
        ``(kper, kstp)`` pairs (both 1-based) at which to write output.
    """
    pairs = []
    cumulative_day = 0
    next_output_day = 0

    last_kper = None
    last_day = None

    for kper, (perlen, nstp, tsmult) in enumerate(perioddata):
        period_days = int(perlen)

        for day_in_period in range(period_days):
            last_kper = kper + 1
            last_day = day_in_period + 1

            if cumulative_day == next_output_day:
                pairs.append((last_kper, last_day))
                next_output_day += output_interval

            cumulative_day += 1

    return pairs


def append_values_to_inner_lists(d, values, *, in_place=False, boundnme='wel'):
    """Append values (and a boundname) to every inner list in a nested dict.

    Used to tack auxiliary concentrations and a per-layer boundname onto each
    stress-period record in a ``{kper: list[list]}`` well dictionary.

    Parameters
    ----------
    d : dict
        Nested ``{key: list[list]}`` dictionary.
    values : object or iterable
        If not an iterable (or a ``str``/``bytes``), appended once as a single
        item; otherwise each element is appended in order.
    in_place : bool, optional
        ``True`` to modify ``d`` directly and return it; ``False`` (default) to
        operate on a copy and return a new dictionary.
    boundnme : str, optional
        Prefix for the boundname appended to each inner list (default ``'wel'``).

    Returns
    -------
    dict
        The dictionary with updated inner lists.
    """
    # Decide whether to work on the original or a shallow copy
    target = d if in_place else {k: [lst[:] for lst in v] for k, v in d.items()}

    is_iterable = (
        isinstance(values, Iterable) and
        not isinstance(values, (str, bytes))  # treat strings/bytes as scalars
    )

    for outer in target.values():
        for inner in outer:
            if is_iterable:
                inner.extend(values)   # add every element in order
                inner.extend([f"{boundnme}-ly{inner[0][0]}"])
            else:
                inner.append(values)   # add the single value
                inner.append(f"{boundnme}-ly{inner[0][0]}")
    return target


def get_wel_coords(gwf, name="wellin", data_d="data"):
    """Return the model cellid of a named well from ``<data_d>/wells.csv``.

    Parameters
    ----------
    gwf : flopy.mf6.ModflowGwf
        The groundwater flow model (used for its model grid).
    name : str, optional
        Well name to look up in ``<data_d>/wells.csv`` (default ``'wellin'``).
    data_d : str, optional
        Directory holding the model input files (default ``'data'``).

    Returns
    -------
    tuple
        The cellid of the cell containing the well location.
    """
    from flopy.utils.gridintersect import GridIntersect
    mg = gwf.modelgrid
    ix = GridIntersect(mg)
    wells = pd.read_csv(os.path.join(data_d, "wells.csv"))
    wells = gpd.GeoDataFrame(wells, geometry=gpd.points_from_xy(wells.x, wells.y))

    assert name in wells.name.values, f"{name} not in well"
    wells = wells[wells.name == name]
    geom = wells.geometry[wells.name == name].values

    assert len(geom) == 1, f"more than one well with name {name} in wells.csv"
    cellid = ix.intersect(geom[0], 'point').cellids

    return cellid[0]


def make_stress_period_data(coords, rates):
    """Pair well cellids with rates into stress-period records.

    Parameters
    ----------
    coords : sequence
        Well cellids.
    rates : sequence
        Pumping rates (same length as ``coords``).

    Returns
    -------
    list of list
        One ``[cell, rate]`` record per well.
    """
    if len(coords) != len(rates):
        raise ValueError("Coordinate and rate lists must be the same length")
    return [[cell, q] for cell, q in zip(coords, rates)]


def make_obs_pack(gwf, data_d="data"):
    """Build a MODFLOW 6 continuous concentration observation package.

    Reads monitoring locations from ``<data_d>/obs_loc.csv`` and registers a
    concentration observation at each location that falls inside the grid.

    Parameters
    ----------
    gwf : flopy.mf6.ModflowGwf
        The groundwater flow (or transport) model to attach the package to.
    data_d : str, optional
        Directory holding the model input files (default ``'data'``).

    Returns
    -------
    flopy.mf6.ModflowUtlobs
        The observation package.
    """
    from flopy.utils.gridintersect import GridIntersect
    ix = GridIntersect(gwf.modelgrid)
    obsloc = pd.read_csv(os.path.join(data_d, "obs_loc.csv"))

    obs_list = []

    for obsid in obsloc.obsid.unique():
        x, y = obsloc.loc[obsloc.obsid == obsid, ['x', 'y']].values[0]
        cellid = ix.intersect([(x, y)], shapetype="point").cellids
        if len(cellid) == 0:
            print(f"{obsid} not in model domain")
            continue
        else:
            cellid = cellid[0]
            print(f"{obsid} is in model domain")
        obs_layer = int(obsloc.loc[obsloc.obsid == obsid, 'layer'].values[0])
        obs_list.append((obsid, 'concentration', (obs_layer, cellid)))

    obs_recarray = {f'obs_{gwf.name}.csv': obs_list}
    obs_package = flopy.mf6.ModflowUtlobs(gwf,
                                          digits=0,
                                          pname=f'obs_{gwf.name}',
                                          continuous=obs_recarray)
    return obs_package


def make_wel_in(sim, conservative_tracer=None, mup3d_m=None, data_d="data"):
    """Build the injection well (``wellin``) package.

    The injection well runs for the whole simulation. When a conservative
    tracer name is given the well carries that single auxiliary; otherwise it
    carries the full mf6rtm component set, with per-stress-period injectate
    chemistry pulled from the mup3d chemistry stresses.

    Parameters
    ----------
    sim : flopy.mf6.MFSimulation
        The simulation (used for ``tdis`` and the ``gwf`` model).
    conservative_tracer : str, optional
        Auxiliary species name for a conservative-tracer build; if ``None``,
        the full reactive component set is used.
    mup3d_m : mf6rtm.mup3d.Mup3d, optional
        The mup3d model, required for the reactive (non-tracer) build.
    data_d : str, optional
        Directory holding the model input files (default ``'data'``).

    Returns
    -------
    dict
        The stress-period data dictionary that was written to the package.
    """
    nper_model = sim.tdis.nper.get_data()
    gwf = sim.get_model("gwf")
    layers = [1, 2, 3, 5, 7]
    cellid = get_wel_coords(gwf, name="wellin", data_d=data_d)
    coords_in = {lay: (lay, cellid) for lay in layers}

    df_inj = pd.read_csv(os.path.join(data_d, "wellin.csv"))
    df_inj = df_inj[df_inj.kper < nper_model].copy()
    mask = df_inj.kper > int(nper_model / 2)
    df_inj.loc[mask, 'rate'] *= 10
    nper = df_inj.kper.max() + 1
    wellin_sp_data = defaultdict(list)

    if conservative_tracer is not None:
        assert conservative_tracer in df_inj.columns, print("compound not in wellin csv")
        # get all unique cells
        for _, r in df_inj.iterrows():
            layer = int(r["layer"])
            cell = coords_in[layer]  # zero-indexed
            wellin_sp_data[int(r["kper"])].append([cell, r["rate"], r[f"{conservative_tracer}"]])
        wel_in = flopy.mf6.ModflowGwfwel(gwf,
                                         stress_period_data=wellin_sp_data,
                                         auxiliary=conservative_tracer,
                                         pname='welin',
                                         filename=f'{gwf.name}.welin')
        wel_in.set_all_data_external()

    else:
        wel_chem_dir = {}
        start_sol = 2
        nlay = len(layers)
        indices = [
            list(range(start_sol + per * nlay,
                       start_sol + (per + 1) * nlay))
            for per in range(nper)
        ]
        for per in range(nper):
            sol_spd = indices[per]
            wellchem = mup3d.ChemStress('per_' + str(per))
            wellchem.set_spd(sol_spd)
            mup3d_m.set_chem_stress(wellchem)
            wel_chem_dir[per] = wellchem.data

        for _, r in df_inj.iterrows():
            layer = int(r["layer"])
            cell = coords_in[layer]  # zero-indexed
            wellin_sp_data[int(r["kper"])].append([cell, r["rate"]])

        for per in range(nper):
            for e, layer in enumerate(layers):
                chem_arr = wel_chem_dir[per][e]
                wellin_sp_data[per][e].extend(chem_arr)
        wel_in = flopy.mf6.ModflowGwfwel(gwf,
                                         stress_period_data=wellin_sp_data,
                                         auxiliary=mup3d_m.components,
                                         pname='welin',
                                         filename=f'{gwf.name}.welin')
        wel_in.set_all_data_external()
    return wellin_sp_data


def make_extraction_well(sim, wellname, rates, active_sp, tag,
                         conservative_tracer=None, mup3d_m=None, data_d="data"):
    """Build an extraction-well package (the flush well or the supply well).

    The flush well (``wellout``) and supply well (``wellopt``) are the same
    package construction with different pumping rates, active stress periods,
    and package tags. The well pumps at a constant rate over its active stress
    periods, carrying zero-concentration auxiliaries (a conservative tracer, or
    the full reactive component set).

    Parameters
    ----------
    sim : flopy.mf6.MFSimulation
        The simulation (used for ``tdis`` and the ``gwf`` model).
    wellname : str
        Well name to look up in ``<data_d>/wells.csv`` (e.g. ``'wellout'``).
    rates : sequence of float
        Per-layer extraction rates (negative), one per screened layer.
    active_sp : iterable of int
        Stress periods over which the well pumps.
    tag : str
        Package short name and file suffix (e.g. ``'welout'``).
    conservative_tracer : str, optional
        Auxiliary species name for a conservative-tracer build; if ``None``,
        the full reactive component set is used.
    mup3d_m : mf6rtm.mup3d.Mup3d, optional
        The mup3d model, required for the reactive (non-tracer) build.
    data_d : str, optional
        Directory holding the model input files (default ``'data'``).

    Returns
    -------
    flopy.mf6.ModflowGwfwel
        The extraction-well package.
    """
    gwf = sim.get_model("gwf")
    layers = [1, 3, 5]
    cellid = get_wel_coords(gwf, name=wellname, data_d=data_d)
    coords_out = [(lay, cellid) for lay in layers]

    active_sp = tuple(active_sp)
    wellout_sp_data = {sp: make_stress_period_data(coords_out, rates) for sp in active_sp}

    if conservative_tracer is not None:
        wellout_sp_data = append_values_to_inner_lists(wellout_sp_data, 0.0, boundnme=tag)
        aux = conservative_tracer
    else:
        wellout_sp_data = append_values_to_inner_lists(
            wellout_sp_data, [0.0] * len(mup3d_m.components), boundnme=tag)
        aux = mup3d_m.components

    wel_out = flopy.mf6.ModflowGwfwel(gwf,
                                      stress_period_data=wellout_sp_data,
                                      auxiliary=aux,
                                      pname=tag,
                                      boundnames=True,
                                      filename=f'{gwf.name}.{tag}')
    wel_out.set_all_data_external()
    return wel_out


def make_wel_out(sim, conservative_tracer=None, mup3d_m=None, wellname="wellout",
                 data_d="data"):
    """Build the flush well (``wellout``); thin wrapper over ``make_extraction_well``.

    See :func:`make_extraction_well` for the parameters. The flush well extracts
    during the history period at the rates ``[-300, -30, -30]``.
    """
    nper = sim.tdis.nper.get_data()
    return make_extraction_well(
        sim,
        wellname=wellname,
        rates=[-300, -30, -30],
        active_sp=range(0, 20),
        tag='welout',
        conservative_tracer=conservative_tracer,
        mup3d_m=mup3d_m,
        data_d=data_d,
    )


def make_wel_opt(sim, conservative_tracer=None, mup3d_m=None, wellname="wellopt",
                 data_d="data"):
    """Build the supply well (``wellopt``); thin wrapper over ``make_extraction_well``.

    See :func:`make_extraction_well` for the parameters. The supply well extracts
    during the supply period at the rates ``[-1300, -300, -300]``.
    """
    nper = sim.tdis.nper.get_data()
    return make_extraction_well(
        sim,
        wellname=wellname,
        rates=[-1300, -300, -300],
        active_sp=range(21, nper),
        tag='welopt',
        conservative_tracer=conservative_tracer,
        mup3d_m=mup3d_m,
        data_d=data_d,
    )


def make_chd(gwf, conservative_tracer=None, mup3d_m=None, data_d="data"):
    """Build the constant-head boundary on the left and right domain edges.

    Reads the domain polygon from ``<data_d>/domain.gpkg``, picks the cells
    along its left and right edges, and assigns a constant head with either a
    conservative-tracer auxiliary (from ``<data_d>/ic_aq_chem.csv``) or the full
    reactive component set (from the mup3d chemistry stress).

    Parameters
    ----------
    gwf : flopy.mf6.ModflowGwf
        The groundwater flow model.
    conservative_tracer : str, optional
        Auxiliary species name for a conservative-tracer build; if ``None``,
        the full reactive component set is used.
    mup3d_m : mf6rtm.mup3d.Mup3d, optional
        The mup3d model, required for the reactive (non-tracer) build.
    data_d : str, optional
        Directory holding the model input files (default ``'data'``).

    Returns
    -------
    flopy.mf6.ModflowGwfchd
        The constant-head package.
    """
    from pathlib import Path
    from flopy.utils.gridintersect import GridIntersect
    l_hd = 0
    domain = gpd.read_file(Path(data_d, 'domain.gpkg'))
    geom = domain.dissolve().geometry[0]
    minx, miny, maxx, maxy = geom.bounds
    left_boundary = LineString([(minx, miny), (minx + 0.1, maxy)])
    right_boundary = LineString([(maxx, miny), (maxx - 0.1, maxy)])

    ix = GridIntersect(gwf.modelgrid)
    left_cells = ix.intersect(left_boundary, 'line').cellids.tolist()
    right_cells = ix.intersect(right_boundary, 'line').cellids.tolist()
    left_cells.extend(right_cells)
    boundary_cells = left_cells

    nlay = gwf.dis.nlay.get_data()
    ncpl = gwf.dis.ncpl.get_data()

    if conservative_tracer is not None:
        df_inj = pd.read_csv(os.path.join(data_d, "ic_aq_chem.csv"), index_col=0)
        assert conservative_tracer in df_inj.index, f"compound {conservative_tracer} not in ic_aq_chem csv"
        c_list = [df_inj.loc[conservative_tracer, 'value']]
        aux = conservative_tracer
    else:
        chdchem = mup3d.ChemStress('chdchem')
        sol_spd = [1]
        chdchem.set_spd(sol_spd)
        mup3d_m.set_chem_stress(chdchem)
        c_list = mup3d_m.chdchem.data[0]
        aux = mup3d_m.components

    chdspd = []
    for i in range(nlay):          # layers
        for icpl in boundary_cells:      # rows
            chdspd.append([(i, icpl), l_hd])           # left boundary

    for i in range(len(chdspd)):
        chdspd[i].extend(c_list)

    chd = flopy.mf6.ModflowGwfchd(
        gwf,
        maxbound=len(chdspd),
        stress_period_data=chdspd,
        save_flows=True,
        auxiliary=aux,
        pname="CHD",
        filename=f"{gwf.name}.chd",
    )
    chd.set_all_data_external()
    return chd


def get_avg_distance(points, npoints=10):
    """Average distance to the nearest ``npoints`` neighbours for each point.

    Parameters
    ----------
    points : numpy.ndarray
        Array of point coordinates.
    npoints : int, optional
        Number of nearest neighbours to average over (default 10).

    Returns
    -------
    numpy.ndarray
        Mean distance to the nearest ``npoints`` neighbours, per input point.
    """
    from scipy.spatial import distance
    distances = distance.cdist(points, points, 'euclidean')
    np.fill_diagonal(distances, np.inf)
    nearest_n = np.partition(distances, npoints, axis=1)[:, :npoints]
    average_distances = np.mean(nearest_n, axis=1)
    return average_distances


def get_botms(gwf, ws, data_d="data"):
    """Krige layer-bottom elevations from borehole points onto the model grid.

    Reads borehole bottoms from ``<data_d>/botm.gpkg`` and uses ordinary kriging
    (via ``pypestutils``) to interpolate each layer's bottom onto the cell
    centres.

    Parameters
    ----------
    gwf : flopy.mf6.ModflowGwf
        The groundwater flow model (provides the grid).
    ws : str
        Workspace directory where the kriging factor file is written.
    data_d : str, optional
        Directory holding the model input files (default ``'data'``).

    Returns
    -------
    list of numpy.ndarray
        Bottom elevations for each layer.
    """
    from pypestutils.pestutilslib import PestUtilsLib
    lib = PestUtilsLib()

    bps = gpd.read_file(os.path.join(data_d, 'botm.gpkg'))
    bps['x'] = bps.geometry.centroid.x
    bps['y'] = bps.geometry.centroid.y

    ppeasting = bps.x.values
    ppnorthing = bps.y.values
    anis = 1
    bearing = 0.0
    aa = 1.5 * get_avg_distance(bps[['x', 'y']].values, 2).max()

    easting = gwf.modelgrid.xcellcenters.flatten()
    northing = gwf.modelgrid.ycellcenters.flatten()

    max_pts = 50  # pp are same as cell centers, so kind of irrelevant
    min_pts = 1
    search_dist = 1.e+10
    aa_pp = aa
    zone_pp = np.ones_like(ppeasting, dtype=int)
    fac_file = os.path.join(ws, "factors.bin")

    ib = np.ones_like(easting, dtype=int)
    lib.calc_kriging_factors_2d(ppeasting,
                                ppnorthing,
                                zone_pp,
                                easting,
                                northing,
                                ib.flatten(),
                                "exp", "ordinary",
                                aa_pp, anis, bearing, search_dist, max_pts, min_pts, fac_file, "binary")

    botms = []
    icpls = gwf.dis.ncpl.get_data()
    for layer in range(1, 13):
        # get COND multiplier
        ppval = bps[f"botm_{layer}"].values
        result = lib.krige_using_file(os.path.join(ws, "factors.bin"),
                                      "binary",
                                      icpls,
                                      "ordinary",
                                      "none",
                                      np.array(ppval),
                                      np.zeros_like(icpls),
                                      0)
        botms.append(np.round(result['targval'], 1))
    return botms


# --- runtime / forward-run helpers (PstFrom add_py_function targets) ---
# These are injected verbatim into the PEST++ forward run by
# PstFrom.add_py_function, which serialises only the function source. They must
# therefore be FULLY SELF-CONTAINED: every import lives inside the function body.

def time_interpolate(sim_times, sim_vals, obs_times):
    """Interpolate simulated values from simulation times onto observation times.

    Parameters
    ----------
    sim_times : array-like
        Simulation output times.
    sim_vals : array-like
        Simulated values at ``sim_times``.
    obs_times : array-like
        Times at which interpolated values are required.

    Returns
    -------
    numpy.ndarray
        Simulated values interpolated (and extrapolated at the ends) onto
        ``obs_times``.
    """
    import numpy as np
    from scipy import interpolate
    t0 = min(sim_times)
    sim_times = [(i - t0).astype(float) for i in sim_times]
    obs_times = [(i - t0).astype(float) for i in obs_times]

    # Create interpolation function
    f = interpolate.interp1d(sim_times, sim_vals, fill_value='extrapolate')
    # Interpolate at new times
    new_values = f(obs_times)
    return new_values


def extract_layer_number(filename):
    """Extract the integer layer number from a ``...layerN...`` filename.

    Parameters
    ----------
    filename : str
        Filename containing a ``layer<N>`` token.

    Returns
    -------
    int or float
        The layer number, or ``float('inf')`` if no ``layer<N>`` token is found
        (so such files sort last).
    """
    import re
    match = re.search(r'layer(\d+)', filename)
    return int(match.group(1)) if match else float('inf')


def get_input_filenames(tag, template_ws=os.path.join('pest', 'pst_template'),
                        extension='.txt', startswith=False):
    """List template-workspace input files matching a tag, sorted by layer.

    Parameters
    ----------
    tag : str
        Substring (or prefix, if ``startswith``) to match, case-insensitive.
    template_ws : str, optional
        Template workspace directory to search.
    extension : str, optional
        Required file extension (default ``'.txt'``).
    startswith : bool, optional
        If ``True``, match files whose name starts with ``tag``; otherwise match
        files that contain ``tag`` (default ``False``).

    Returns
    -------
    list of str
        Matching filenames, sorted by their layer number.
    """
    if startswith:
        files = [
            f for f in os.listdir(template_ws)
            if f.lower().startswith(tag) and f.endswith(extension)
        ]
    else:
        files = [
            f for f in os.listdir(template_ws)
            if tag in f.lower() and f.endswith(extension)
        ]
    files = sorted(files, key=extract_layer_number)
    return files


def copy_parameterized_transport_files(ws=".",
                                       parameterized_species="h2o",
                                       dsp_par=['alh'],
                                       mst_par=['porosity']):
    """Copy one species' parameterised transport inputs onto all other species.

    Dispersivity (``dsp_*``) and mobile-storage (``mst_*``) parameters are
    estimated for a single carrier species and then replicated, file by file,
    across every other transport species so they share the same fields.

    Parameters
    ----------
    ws : str, optional
        Workspace directory holding the simulation and input files.
    parameterized_species : str, optional
        The species whose parameterised input files are copied from.
    dsp_par : list of str, optional
        Dispersivity parameter tags to copy (default ``['alh']``).
    mst_par : list of str, optional
        Mobile-storage parameter tags to copy (default ``['porosity']``).

    Returns
    -------
    list of tuple
        The ``(source, destination)`` filename pairs copied for the last species.
    """
    import flopy
    import shutil
    from pathlib import Path

    def flatten(xss):
        return [x for xs in xss for x in xs]

    sim = flopy.mf6.MFSimulation.load(sim_ws=ws, verbosity_level=0)
    species = sim.model_names[1:]
    species.remove(parameterized_species)

    tag = []

    for e, par in enumerate(dsp_par):
        tag.append(f"dsp_{par}_")

    for e, par in enumerate(mst_par):
        tag.append(f"mst_{par}_")

    fnames_to_copy = [get_input_filenames(f"{parameterized_species}.{t}", template_ws=ws, startswith=True) for t in tag]
    fnames_to_copy = flatten(fnames_to_copy)

    print(
        f"Warning: copying files with tag: {', '.join(tag)} from species: {parameterized_species.upper()} to the following species: "
        f"{', '.join(sp.capitalize() for sp in species if sp != parameterized_species)}"
    )

    for sp in species:
        fnames_to_replace = [get_input_filenames(f"{sp}.{t}", template_ws=ws, startswith=True) for t in tag]
        fnames_to_replace = flatten(fnames_to_replace)
        assert sorted([x.split('.')[1] for x in fnames_to_copy]) == sorted([x.split('.')[1] for x in fnames_to_replace]), 'list of files to replace and to copy does not contain the same files names '
        # sort fnames_to_copy and fnames_to_replace according to the assert above
        fnames_to_copy = [x for _, x in sorted(zip([x.split('.')[1] for x in fnames_to_copy], fnames_to_copy))]
        fnames_to_replace = [x for _, x in sorted(zip([x.split('.')[1] for x in fnames_to_replace], fnames_to_replace))]
        fileszipped = list(zip(fnames_to_copy, fnames_to_replace))
        [shutil.copyfile(Path(ws, f[0]), Path(ws, f[1])) for f in fileszipped]
    return fileszipped


def node_to_layer_icell2d(nodes, ncpl):
    """Convert 1-based MF6 node numbers to (layer, icell2d) pairs.

    Parameters
    ----------
    nodes : array-like
        MF6 node numbers (1-based).
    ncpl : int
        Number of 2D cells per layer.

    Returns
    -------
    tuple of numpy.ndarray
        ``(layer, icell2d)`` arrays, both 0-based.
    """
    import numpy as np
    nodes = np.asarray(nodes, dtype=int)
    idx = nodes - 1  # to 0-based
    layer0 = idx // ncpl
    icell2d0 = idx % ncpl
    return layer0, icell2d0


def tidy_array(fpath):
    """Rewrite a whitespace-delimited array file as a clean one-per-line column.

    Reads all whitespace-separated values from ``fpath`` and writes them back as
    a flat column in scientific notation.

    Parameters
    ----------
    fpath : str
        Path to the array file to rewrite in place.

    Returns
    -------
    None
    """
    import numpy as np
    # read unordered txt file
    with open(fpath, 'r') as f:
        data = f.read().split()
    data = [float(x) for x in data]
    arr = np.array(data)
    arr = arr.flatten()
    np.savetxt(fpath, arr, fmt='%1.6e')
    return


def process_sim_conc(wd='.'):
    """Match simulated concentrations to observations and write the comparison.

    Loads the simulation in ``wd``, maps the ``sout.csv`` cell output onto the
    model grid, lines it up against the cleaned observation data
    (``obs_chem_cleaned.csv``), interpolates simulated values onto the
    observation times per ``(obsid, variable)``, and writes a tidy
    simulated-vs-measured table to ``_obs.conc.simvsmeas.csv``. This is the
    forward run's observation post-processor.

    Parameters
    ----------
    wd : str, optional
        Working directory containing the simulation and observation files
        (default ``'.'``).

    Returns
    -------
    tuple
        ``(fname, dfmerged)`` — the output filename and the merged
        simulated-vs-measured ``DataFrame``.
    """
    import os
    import numpy as np
    import pandas as pd
    import flopy
    from flopy.utils.gridintersect import GridIntersect
    from pathlib import Path
    sim = flopy.mf6.MFSimulation.load(sim_ws=wd,
                                      sim_name='gwf',
                                      version='mf6',
                                      exe_name='mf6',
                                      verbosity_level=0)
    gwf = sim.get_model("gwf")
    sout = pd.read_csv(os.path.join(wd, "sout.csv"))
    sout['cell'] += 1

    layers, icell2ds = node_to_layer_icell2d(sout['cell'], gwf.disv.ncpl.get_data())
    sout['layer'] = layers
    sout['cell2d'] = icell2ds

    ix = GridIntersect(gwf.modelgrid)
    obsdata = pd.read_csv(os.path.join(wd, "obs_chem_cleaned.csv"))
    obsdata.rename(columns={'var': 'variable'}, inplace=True)

    obs_list = {}

    for obsid in obsdata.obsid.unique():
        x, y = obsdata.loc[obsdata.obsid == obsid, ['x', 'y']].values[0]
        cellid = ix.intersect([(x, y)], shapetype="point").cellids
        if len(cellid) == 0:
            continue
        else:
            obs_layer = int(obsdata.loc[obsdata.obsid == obsid, 'layer'].values[0])
            obs_list[obsid] = cellid[0]
    obsdata['cell2d'] = obsdata['obsid'].map(obs_list)

    missvar = set(obsdata['variable'].unique()) - set(sout.columns)
    obs_ = sout[['time', 'cell2d', 'layer'] + list(set(obsdata['variable'].unique()) - missvar)].copy()
    obs_ = obs_.melt(id_vars=['time', 'layer', 'cell2d'])
    obs_ = obs_.merge(obsdata[['obsid', 'cell2d', 'layer']].drop_duplicates(), how='left')
    obs_.dropna(subset='obsid', inplace=True)

    dfmerged = pd.merge(obs_[['time', 'obsid', 'cell2d', 'layer', 'variable', 'value']],
                        obsdata[['time', 'obsid', 'cell2d', 'layer', 'variable', 'value']],
                        on=['time', 'obsid', 'cell2d', 'layer', 'variable'], how='outer')
    dfmerged.rename(columns={'value_x': 'sim',
                             'value_y': 'meas'}, inplace=True)

    dfmerged.sort_values(['obsid', 'time'], inplace=True)
    dfmerged.set_index('time', inplace=True)

    for oid in obs_.obsid.unique():
        print(f"Processing obs for: {oid:>5}")
        for var in obs_.variable.unique():
            mask = (dfmerged.obsid == oid) & (dfmerged.variable == var)

            tmp = dfmerged.loc[mask].copy()
            tmp.dropna(subset=['sim'], inplace=True)
            if tmp.shape[0] == 0:
                continue
            obs_times = dfmerged.loc[mask].index.values
            dfmerged.loc[mask, 'sim'] = time_interpolate(tmp.index.values,
                                                         tmp.sim.values,
                                                         obs_times)

    fname = '_obs.conc.simvsmeas.csv'
    dfmerged = dfmerged.reset_index()
    dfmerged.drop_duplicates(subset=['time', 'obsid', 'variable'], inplace=True)
    dfmerged = dfmerged.set_index('time')
    dfmerged[['layer', 'cell2d']] += 1  # back to 1-based
    dfmerged.dropna(subset=['variable'], inplace=True)  # housekeeping for "fake obs"
    dfmerged.replace(np.nan, 1e30).to_csv(Path(wd, fname), float_format="%.5e")
    print(f"Processed conc saved in {wd}/{fname}")

    return fname, dfmerged


# --- misc ---

def get_bins(local_dir):
    """Copy the platform's solver/PEST++ binaries into a local working dir.

    Resolves the repository ``bin/`` directory relative to this module's
    location (walking up to the repo root), picks the subdirectory for the
    current platform (``win``, ``mac`` or ``linux``), and copies every binary
    into ``local_dir``.

    Parameters
    ----------
    local_dir : str
        Destination directory; created if it does not exist.

    Returns
    -------
    str
        Path to the platform binary directory that was copied from.

    Raises
    ------
    FileNotFoundError
        If the expected ``bin/<platform>`` directory does not exist.
    """
    # figure out if mac, linux or windows
    if platform.system() == "Windows":
        plat_dir = "win"
    elif platform.system() == "Darwin":
        plat_dir = "mac"
    else:
        plat_dir = "linux"

    # resolve bin/ relative to the repo root (walk up from this module)
    repo_root = Path(__file__).resolve().parent
    while repo_root != repo_root.parent and not (repo_root / "bin").is_dir():
        repo_root = repo_root.parent
    bindir = repo_root / "bin" / plat_dir

    if not bindir.is_dir():
        raise FileNotFoundError(
            f"No binaries for platform '{plat_dir}': expected directory {bindir}. "
            f"Add a 'bin/{plat_dir}' directory containing the mf6, libmf6, "
            f"pestpp-ies and pestpp-mou binaries for this platform."
        )

    # copy all the exes to a local bin dir
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    for fname in os.listdir(bindir):
        src = os.path.join(bindir, fname)
        dst = os.path.join(local_dir, fname)
        if not os.path.isfile(src):
            continue
        if os.path.isfile(dst):
            os.remove(dst)
        shutil.copy(src, dst)
    return str(bindir)
