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
from pypestutils.pestutilslib import PestUtilsLib

lib = PestUtilsLib()

def time_interpolate(sim_times, sim_vals, obs_times):
    import numpy as np
    from scipy import interpolate
    t0 = min(sim_times)
    sim_times = [(i-t0).astype(float) for i in sim_times]
    obs_times = [(i-t0).astype(float) for i in obs_times]

    # Create interpolation function
    f = interpolate.interp1d(sim_times, sim_vals, fill_value='extrapolate')
    # Interpolate at new times
    new_values = f(obs_times)
    return new_values

def create_reactive_tsteps(perioddata, output_interval=5):
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

    # Ensure last kper, ktsp is always included
    # if pairs[-1] != (last_kper, last_day):
    #     pairs.append((last_kper, last_day))

    return pairs

def append_values_to_inner_lists(d, values, *, in_place=False, boundnme='wel'):
    """
    Append a single value or all values from an iterable to every inner list
    inside a {key: list[list]} dictionary.

    Parameters
    ----------
    d : dict
        Your nested list dictionary.
    values : any or Iterable
        * If `values` is not an Iterable (or is str/bytes), its treated as a
          single item and appended once.
        * If `values` is an Iterable (list/tuple/set/range), each element is
          appended in order.
    in_place : bool, default False
        True  → modify `d` directly and return it.  
        False → leave `d` unchanged and return a *new* dictionary.

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
                # print(inner)
                inner.extend(values)   # add every element in order
                inner.extend([f"{boundnme}-ly{inner[0][0]}"])
            else:
                # print(inner)
                inner.append(values)   # add the single value
                inner.append(f"{boundnme}-ly{inner[0][0]}")
                # print(inner)
    return target

def get_wel_coords(gwf, name  = "wellin"):
    from flopy.utils.gridintersect import GridIntersect
    mg = gwf.modelgrid
    ix = GridIntersect(mg)
    wells = pd.read_csv(os.path.join("data", "wells.csv"))
    wells = gpd.GeoDataFrame(wells, geometry=gpd.points_from_xy(wells.x, wells.y))

    assert name in wells.name.values, f"{name} not in well"
    wells = wells[wells.name == name]
    geom = wells.geometry[wells.name==name].values

    assert len(geom)==1, f"more than one well with name {name} in wells.csv"
    cellid = ix.intersect(geom[0], 'point').cellids

    return cellid[0]

def make_stress_period_data(coords, rates, add_conc=True):
        if len(coords) != len(rates):
            raise ValueError("Coordinate and rate lists must be the same length")
        return [
            ([cell, q] if add_conc else [cell, q])
            for cell, q in zip(coords, rates)
        ]

def make_obs_pack(gwf):
    from flopy.utils.gridintersect import GridIntersect
    ix = GridIntersect(gwf.modelgrid)
    obsloc = pd.read_csv(os.path.join("data", "obs_loc.csv"))

    obs_list=[]

    for obsid in obsloc.obsid.unique():
        x,y = obsloc.loc[obsloc.obsid==obsid,['x','y']].values[0]
        cellid = ix.intersect([(x,y)],shapetype="point").cellids
        if len(cellid)==0:
            print(f"{obsid} not in model domain")
            continue
        else:
            cellid = cellid[0]
            print(f"{obsid} is in model domain") 
        obs_layer = int(obsloc.loc[obsloc.obsid==obsid,'layer'].values[0])
        obs_list.append((obsid, 'concentration', (obs_layer, cellid)))

    # obs_recarray = {'obs.head.sim.csv':obs_list}
    obs_recarray = {f'obs_{gwf.name}.csv':obs_list}
    # print(obs_list)
    obs_package = flopy.mf6.ModflowUtlobs(gwf, 
                                        digits=0, #print_input=True,
                                        pname=f'obs_{gwf.name}',
                                        continuous=obs_recarray)
    return obs_package

def make_wel_in(sim, conservative_tracer = None,
                mup3d_m=None):
    nper_model = sim.tdis.nper.get_data()
    # nper=39
    gwf = sim.get_model("gwf")
    layers = [1,2,3,5,7]
    # coords_in = {}
    cellid = get_wel_coords(gwf, name  = "wellin")
    coords_in = {lay: (lay, cellid) for lay in layers}

    df_inj = pd.read_csv(os.path.join("data", "wellin.csv"))
    df_inj = df_inj[df_inj.kper<nper_model].copy()
    mask = df_inj.kper>int(nper_model/2)
    df_inj.loc[mask, 'rate'] *= 10
    nper = df_inj.kper.max()+1
    wellin_sp_data = defaultdict(list)

    if conservative_tracer is not None:
        assert conservative_tracer in df_inj.columns, print("compound not in wellin csv")
        #get all unique cells
        for _, r in df_inj.iterrows():
            layer = int(r["layer"])
            cell = coords_in[layer]  # zero‑indexed
            wellin_sp_data[int(r["kper"])].append([cell, r["rate"], r[f"{conservative_tracer}"]])
        wel_in  = flopy.mf6.ModflowGwfwel(gwf, 
                                       stress_period_data=wellin_sp_data,
                                       auxiliary=conservative_tracer,
                                       pname = 'welin',
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
        # indices = [list(range(i, i + 5)) for i in range(2, 197, 5)]
        for per in range(nper):
            sol_spd = indices[per]
            wellchem = mup3d.ChemStress('per_'+str(per))
            wellchem.set_spd(sol_spd)
            mup3d_m.set_chem_stress(wellchem)
            wel_chem_dir[per] = wellchem.data

        for _, r in df_inj.iterrows():
            layer = int(r["layer"])
            cell = coords_in[layer]  # zero‑indexed
            wellin_sp_data[int(r["kper"])].append([cell, r["rate"]])

        for per in range(nper):
            for e, layer in  enumerate(layers):
                chem_arr = wel_chem_dir[per][e]
                wellin_sp_data[per][e].extend(chem_arr)
        wel_in  = flopy.mf6.ModflowGwfwel(gwf, 
                                        stress_period_data=wellin_sp_data,
                                        auxiliary=mup3d_m.components,
                                        pname = 'welin',
                                        filename=f'{gwf.name}.welin')
        wel_in.set_all_data_external()
    return wellin_sp_data

def make_wel_out(sim, conservative_tracer = None, mup3d_m=None, wellname = "wellout"):

    nper = sim.tdis.nper.get_data()
    gwf = sim.get_model("gwf")
    layers = [1,3,5]
    cellid = get_wel_coords(gwf, name  = wellname)
    coords_out = [(lay, cellid) for lay in layers]
    # print(coords_out)

    init_rates_out  = [-300,  -30,  -30]                # 3 negatives
    fini_rates_out  = [-400,  -40,  -40]
    fini_rates_out  = [0,  0,  0]

    init_sp = range(0, 20)   # stress periods 0 – 35
    fini_sp = range(20, nper)  # stress periods 36 – 38
    all_sp  = (*init_sp, 
            #    *fini_sp
               )

    # Time‑invariant blocks for each phase
    wellout_init = {sp: make_stress_period_data(coords_out, init_rates_out) for sp in all_sp}
    wellout_fini = {sp: make_stress_period_data(coords_out, fini_rates_out, add_conc=True) 
                    for sp in all_sp}
    wellout_sp_data = {sp: (wellout_init[sp] if sp in init_sp else wellout_fini[sp])
                for sp in all_sp}

    if conservative_tracer is not None:
        wellout_sp_data = append_values_to_inner_lists(wellout_sp_data, 0.0, boundnme='welout')

        wel_out = flopy.mf6.ModflowGwfwel(gwf, 
                                            stress_period_data=wellout_sp_data, 
                                            auxiliary=conservative_tracer,
                                            pname = 'welout' ,
                                            boundnames=True,
                                            filename=f'{gwf.name}.welout')
        wel_out.set_all_data_external()
    else:
        wellout_sp_data = append_values_to_inner_lists(wellout_sp_data, [0.0]*len(mup3d_m.components), boundnme='welout')
        # wellout_sp_data = 
        wel_out = flopy.mf6.ModflowGwfwel(gwf, 
                                            stress_period_data=wellout_sp_data, 
                                            auxiliary=mup3d_m.components,
                                            pname = 'welout',
                                            boundnames=True,
                                            filename=f'{gwf.name}.welout')
        
        wel_out.set_all_data_external()
    return wel_out


def make_wel_opt(sim, conservative_tracer = None, mup3d_m=None, wellname = "wellopt"):

    nper = sim.tdis.nper.get_data()
    gwf = sim.get_model("gwf")
    layers = [1,3,5]
    cellid = get_wel_coords(gwf, name  = wellname)
    coords_out = [(lay, cellid) for lay in layers]
    # print(coords_out)

    init_rates_out  = [-1300,  -300,  -300]                # 3 negatives
    fini_rates_out  = [-1400,  -400,  -400]

    init_sp = range(21, nper)   # stress periods 0 – 35
    # fini_sp = range(39, nper)  # stress periods 36 – 38
    all_sp  = (*init_sp, 
            #    *fini_sp
               )

    # Time‑invariant blocks for each phase
    wellout_init = {sp: make_stress_period_data(coords_out, init_rates_out) for sp in all_sp}
    wellout_fini = {sp: make_stress_period_data(coords_out, fini_rates_out, add_conc=True) 
                    for sp in all_sp}
    wellout_sp_data = {sp: (wellout_init[sp] if sp in init_sp else wellout_fini[sp])
                for sp in all_sp}

    if conservative_tracer is not None:
        wellout_sp_data = append_values_to_inner_lists(wellout_sp_data, 0.0,  boundnme='welopt')
        wel_out = flopy.mf6.ModflowGwfwel(gwf, 
                                            stress_period_data=wellout_sp_data, 
                                            auxiliary=conservative_tracer,
                                            pname = 'welopt' ,
                                            boundnames=True,
                                            filename=f'{gwf.name}.welopt')
        wel_out.set_all_data_external()
    else:
        wellout_sp_data = append_values_to_inner_lists(wellout_sp_data, [0.0]*len(mup3d_m.components),  boundnme='welopt')
        wel_out = flopy.mf6.ModflowGwfwel(gwf, 
                                            stress_period_data=wellout_sp_data, 
                                            auxiliary=mup3d_m.components,
                                            pname = 'welopt',
                                            boundnames=True,
                                            filename=f'{gwf.name}.welopt')
        # welred_obslist = [(i, "wel-reduction", )]
        # obslist = [(i, "WEL", i) for i in well_loc.obs_id.unique()]
        # _obs = {(f'{gwf.name}.obs.welred.opt.csv'):welred_obslist,
        #         (f'{gwf.name}.obs.wel.opt.csv'):obslist,}
        # wel_out.obs.initialize(digits=10, print_input=False,continuous=_obs)
        
        wel_out.set_all_data_external()
    return wel_out

def make_chd(gwf, conservative_tracer = None, mup3d_m=None):
    from pathlib import Path
    from flopy.utils.gridintersect import GridIntersect
    l_hd= 0
    domain = gpd.read_file(Path('data', 'domain.gpkg'))
    geom = domain.dissolve().geometry[0]
    minx, miny, maxx, maxy = geom.bounds
    left_boundary = LineString([(minx, miny), (minx+0.1, maxy)])
    right_boundary = LineString([(maxx, miny), (maxx-0.1, maxy)])

    ix = GridIntersect(gwf.modelgrid)
    left_cells = ix.intersect(left_boundary, 'line').cellids.tolist()
    right_cells = ix.intersect(right_boundary, 'line').cellids.tolist()
    left_cells.extend(right_cells)
    boundary_cells = left_cells

    nlay = gwf.dis.nlay.get_data()
    ncpl = gwf.dis.ncpl.get_data()

    if conservative_tracer is not None:
        df_inj = pd.read_csv(os.path.join("data", "ic_aq_chem.csv"), index_col=0)
        assert conservative_tracer in df_inj.index, f"compound {conservative_tracer} not in ic_aq_chem csv"
        c_list = [df_inj.loc[conservative_tracer, 'value']]
        aux = conservative_tracer
        # print(c_list)
    else:
        chdchem = mup3d.ChemStress('chdchem')
        sol_spd = [1]
        chdchem.set_spd(sol_spd)
        mup3d_m.set_chem_stress(chdchem)
        c_list = mup3d_m.chdchem.data[0]
        aux=mup3d_m.components

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
    """
    Calculate the average distance to the nearest n points for each point in a set of points.

    Parameters
    ----------
    points : numpy array
        Array of points.
    npoints : int
        Number of nearest points to calculate the average distance to.
    Returns
    -------
    average_distances : numpy array
        Array of average distances to the nearest n points for each point in the input array.
    """

    from scipy.spatial import distance
    distances = distance.cdist(points, points, 'euclidean')
    np.fill_diagonal(distances, np.inf)
    nearest_n = np.partition(distances, npoints, axis=1)[:, :npoints]
    average_distances = np.mean(nearest_n, axis=1)
    return average_distances

def get_botms(gwf, ws):
    """
    Get botms using kriging from borehole points.
    
    Parameters
    ----------
    gwf : flopy.mf6.ModflowGwf
        The groundwater flow model object.
    ws : str
        The workspace directory where temporary files will be stored.
    Returns
    -------
    botms : list
        List of bottom elevations for each layer.
    """

    bps = gpd.read_file(os.path.join("data", 'botm.gpkg'))
    bps['x'] = bps.geometry.centroid.x
    bps['y'] = bps.geometry.centroid.y
    bps

    ppeasting = bps.x.values
    ppnorthing = bps.y.values
    anis = 1
    bearing= 0.0
    aa = 1.5 * get_avg_distance(bps[['x','y']].values, 2).max()

    ib = gwf.dis.idomain.get_data()
    # cellids = df.loc[df.layer==layer+1].icpl.values - 1 # zero-based
    easting = gwf.modelgrid.xcellcenters.flatten()
    northing = gwf.modelgrid.ycellcenters.flatten()

    max_pts = 50 # pp are same as cell centers, so kind of irrelevant
    min_pts = 1
    search_dist = 1.e+10
    aa_pp = aa #?
    zone_pp = np.ones_like(ppeasting,dtype=int)
    fac_file = os.path.join(ws,f"factors.bin")

    ib = np.ones_like(easting,dtype=int)
    ipts = lib.calc_kriging_factors_2d(ppeasting,
                                    ppnorthing,
                                    zone_pp,
                                    easting,
                                    northing,
                                    ib.flatten(),
                                    "exp","ordinary",
                                    aa_pp,anis,bearing,search_dist,max_pts,min_pts,fac_file,"binary")

    botms = []
    icpls = gwf.dis.ncpl.get_data()
    for layer in range(1, 13):
        # get COND multiplier
        ppval = bps[f"botm_{layer}"].values
        result = lib.krige_using_file(os.path.join(ws,f"factors.bin"),
                                        "binary",
                                        icpls,
                                        "ordinary",
                                        "none",
                                        np.array(ppval),
                                        np.zeros_like(icpls),
                                        0)
        botms.append(np.round(result['targval'], 1))
    return botms

def get_bins(local_dir):
    #figure out if mac,lilnux or windows
    if platform.system() == "Windows":
        bin_dir = "win"
    elif platform.system() == "Darwin":
        bin_dir = "mac"
    else:
        bin_dir = "linux"
    bindir = os.path.join("bin", bin_dir)

    # copy all the exes to a local bin dir
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    for fname in os.listdir(bindir):
        src = os.path.join(bindir, fname)
        dst = os.path.join(local_dir, fname)
        if os.path.isfile(dst):
            os.remove(dst)
        shutil.copy(src, dst)
    return bindir

def tidy_array(fpath):
    # read unordered txt file
    with open(fpath, 'r') as f:
        data = f.read().split()
    data = [float(x) for x in data]
    arr = np.array(data)
    arr = arr.flatten()
    #arr = arr.reshape(sr.ncpl)
    np.savetxt(fpath, arr, fmt='%1.6e')
    return

def get_input_filenames(tag, template_ws=os.path.join('pest','pst_template'), extension='.txt', startswith = False):
    """
    Get the input filenames from the template workspace

    Parameters:
        tag: str, tag to search for
        template_ws: str, template workspace
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

def extract_layer_number(filename):
    import re
    match = re.search(r'layer(\d+)', filename)
    return int(match.group(1)) if match else float('inf')

def copy_parameterized_transport_files(ws=".",
                                parameterized_species="h2o",
                                dsp_par =  ['alh'], 
                                mst_par = ['porosity']
                                 ):

    def flatten(xss):
        return [x for xs in xss for x in xs]
    from pathlib import Path
    sim = flopy.mf6.MFSimulation.load(sim_ws=ws, verbosity_level=0)
    species = sim.model_names[1:]
    species.remove(parameterized_species)

    tag = []

    for e, par in enumerate(dsp_par):
        tag.append(f"dsp_{par}_")

    for  e, par in enumerate(mst_par):
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
        assert sorted([x.split('.')[1] for x in fnames_to_copy]) == sorted([x.split('.')[1] for x in fnames_to_replace]), f'list of files to replace and to copy does not contain the same files names '
        # sort fnames_to_copy and fnames_to_replace according to the assert above
        fnames_to_copy = [x for _, x in sorted(zip([x.split('.')[1] for x in fnames_to_copy], fnames_to_copy))]
        fnames_to_replace = [x for _, x in sorted(zip([x.split('.')[1] for x in fnames_to_replace], fnames_to_replace))]
        fileszipped = list(zip(fnames_to_copy, fnames_to_replace))
        [shutil.copyfile(Path(ws, f[0]), Path(ws, f[1])) for f in fileszipped]
    return fileszipped

def node_to_layer_icell2d(nodes, ncpl):
    """
    nodes: array-like of MF6 node numbers (1-based)
    ncpl: number of 2D cells per layer
    Returns:
      layer (1-based), icell2d (1-based)  [change to 0-based if you prefer]
    """
    nodes = np.asarray(nodes, dtype=int)
    idx = nodes - 1  # to 0-based
    layer0 = idx // ncpl
    icell2d0 = idx % ncpl
    return layer0, icell2d0

def process_sim_conc(wd='.'):
    from flopy.utils.gridintersect import GridIntersect
    from pathlib import Path
    sim = flopy.mf6.MFSimulation.load(sim_ws = wd,
                                    sim_name = 'gwf', 
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
    obsdata.rename(columns={'var': 'variable'},  inplace=True)

    obs_list={}

    for obsid in obsdata.obsid.unique():
        x,y = obsdata.loc[obsdata.obsid==obsid,['x','y']].values[0]
        cellid = ix.intersect([(x,y)],shapetype="point").cellids
        if len(cellid)==0:
            continue
        else:
            obs_layer = int(obsdata.loc[obsdata.obsid==obsid,'layer'].values[0])
            obs_list[obsid] = cellid[0]
    obsdata['cell2d'] = obsdata['obsid'].map(obs_list)

    missvar = set(obsdata['variable'].unique()) - set(sout.columns)
    obs_ = sout[['time', 'cell2d', 'layer']+list(set(obsdata['variable'].unique())  - missvar)].copy()
    obs_ = obs_.melt(id_vars = ['time', 'layer', 'cell2d'])
    obs_ = obs_.merge(obsdata[['obsid', 'cell2d', 'layer']].drop_duplicates(), how = 'left')
    obs_.dropna(subset='obsid', inplace=True)

    dfmerged = pd.merge(obs_[['time', 'obsid','cell2d', 'layer', 'variable', 'value']],
                        obsdata[['time', 'obsid','cell2d', 'layer', 'variable', 'value']],
                        on=['time','obsid','cell2d', 'layer', 'variable'], how='outer')
    dfmerged.rename(columns={'value_x':'sim',
                            'value_y': 'meas'}, inplace=True)

    dfmerged.sort_values(['obsid','time'], inplace=True)
    dfmerged.set_index('time', inplace=True)

    for oid in obs_.obsid.unique():
        print(f"Processing obs for: {oid:>5}")
        for var in obs_.variable.unique():
            mask=(dfmerged.obsid==oid)&(dfmerged.variable==var)

            tmp = dfmerged.loc[mask].copy()
            tmp.dropna(subset=['sim'], inplace=True)
            if tmp.shape[0]==0:
                continue
            obs_times = dfmerged.loc[mask].index.values
            dfmerged.loc[mask,'sim'] = time_interpolate(tmp.index.values,
                                                        tmp.sim.values,
                                                        obs_times)

    fname = '_obs.conc.simvsmeas.csv'
    dfmerged = dfmerged.reset_index()
    dfmerged.drop_duplicates(subset=['time', 'obsid', 'variable'], inplace=True)
    dfmerged = dfmerged.set_index('time')
    dfmerged[['layer', 'cell2d']] += 1 #back to 1-based
    dfmerged.dropna(subset=['variable'], inplace=True) #housekeeping for "fake obs"
    dfmerged.replace(np.nan,1e30).to_csv(Path(wd, fname), float_format = "%.5e")
    print(f"Processed conc saved in {wd}/{fname}")

    return fname, dfmerged