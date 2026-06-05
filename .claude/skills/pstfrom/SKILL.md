---
name: pstfrom
description: Guide for using pyemu.PstFrom to set up a PEST++ interface around a MODFLOW 6 model - parameterisation, observations, weights, and prior ensembles.
user-invocable: true
disable-model-invocation: true
allowed-tools: Read Grep Glob Bash Agent Edit Write
---

# PstFrom: PEST++ Interface Setup for MODFLOW 6

When this skill is invoked, guide the user through setting up a PEST++ interface using `pyemu.utils.PstFrom`. Follow the workflow, conventions, and patterns described below. Adapt to the user's specific model and requirements.

Reference tutorials in this repo for working examples:
- `tutorials/part2_01_pstfrom_pest_setup/freyberg_pstfrom_pest_setup.ipynb` — full PstFrom workflow
- `tutorials/part2_02_obs_and_weights/freyberg_obs_and_weights.ipynb` — observation and weight setup

---

## Overall Workflow

1. **Initialise PstFrom** — point it at the model directory
2. **Define geostatistical structures** — for spatial and temporal correlation
3. **Add parameters** — for each model input to parameterise
4. **Add model run command(s)** — MODFLOW 6 (and optionally MODPATH, etc.)
5. **Add helper functions** — pre/post-processing Python functions
6. **Add observations** — from model output files
7. **Build the PST** — generates control file and `forward_run.py`
8. **Post-build modifications** — weights, bounds, transforms, observed values, standard deviations
9. **Build prior covariance / draw ensemble** — for uncertainty analysis
10. **Write and test** — write PST to disk, run with `noptmax=0` to verify

---

## 1. Initialisation

```python
import pyemu
import os

pf = pyemu.utils.PstFrom(
    original_d=org_model_ws,        # Directory with original model files
    new_d=template_ws,              # PEST template directory (created/overwritten)
    remove_existing=True,           # Clean start
    longnames=True,                 # Required for PEST++ (not limited by PEST name lengths)
    spatial_reference=sr,           # flopy.discretization.StructuredGrid or pyemu.helpers.SpatialReference
    zero_based=False,               # False for MODFLOW (1-based layer/row/col indexing)
    start_datetime="1-1-2008",      # Needed for temporal correlation
    echo=False
)
```

**Key points:**
- All model input files must use **external files** (array `.txt` or list `.txt`), not inline in the `.nam`/package files.
- PstFrom copies everything from `original_d` to `new_d`.
- `spatial_reference` is needed for pilot points and spatial geostatistics.

---

## 2. Geostatistical Structures

Define structures for different scales of spatial and temporal correlation:

```python
# Grid-scale (short correlation length)
v_grid = pyemu.geostats.ExpVario(contribution=1.0, a=500)  # a = correlation length in model units
grid_gs = pyemu.geostats.GeoStruct(variograms=v_grid, transform='log')

# Pilot-point scale (longer correlation length)
v_pp = pyemu.geostats.ExpVario(contribution=1.0, a=2000)
pp_gs = pyemu.geostats.GeoStruct(variograms=v_pp, transform='log')

# Temporal correlation
v_time = pyemu.geostats.ExpVario(contribution=1.0, a=365.25)  # a in days
temporal_gs = pyemu.geostats.GeoStruct(variograms=v_time, transform='none')
```

- Use `transform='log'` for multiplicative parameters (K, conductance, recharge).
- Use `transform='none'` for additive parameters (heads, offsets).

---

## 3. Adding Parameters

### Parameter Types

| `par_type` | Scale | Use case |
|-----------|-------|----------|
| `"constant"` | Coarse — single multiplier for entire array/file | Regional scaling |
| `"pilotpoints"` | Medium — interpolated from sparse points | Hydraulic conductivity, storage |
| `"grid"` | Fine — unique parameter per cell/entry | Maximum flexibility |

### Parameter Styles

| `par_style` | Description |
|------------|-------------|
| `"m"` (default) | Multiplicative — parameter multiplies original value |
| `"a"` | Additive — parameter is added to original value |

### Array-Type Files (e.g., hydraulic conductivity)

```python
# Grid-scale parameters
pf.add_parameters(
    'model.npf_k_layer1.txt',
    par_type="grid",
    zone_array=ib,                  # idomain array to mask inactive cells
    geostruct=grid_gs,
    par_name_base="npfkgr",
    pargp="npfkgr",
    lower_bound=0.2, upper_bound=5.0,      # Multiplier bounds
    ult_lbound=0.01, ult_ubound=100         # Final model value bounds
)

# Pilot point parameters
pf.add_parameters(
    'model.npf_k_layer1.txt',
    par_type="pilotpoints",
    zone_array=ib,
    geostruct=pp_gs,
    par_name_base="npfkpp",
    pargp="npfkpp",
    lower_bound=0.2, upper_bound=5.0,
    ult_lbound=0.01, ult_ubound=100,
    pp_space=5                              # Every 5th row/col; or path to CSV/shapefile
)

# Constant (single multiplier)
pf.add_parameters(
    'model.npf_k_layer1.txt',
    par_type="constant",
    zone_array=ib,
    geostruct=grid_gs,
    par_name_base="npfkcn",
    pargp="npfkcn",
    lower_bound=0.2, upper_bound=5.0,
    ult_lbound=0.01, ult_ubound=100
)
```

**Multiple scales on the same file are supported** — PstFrom chains multipliers together.

**Pilot point locations** via `pp_space`:
- Integer (e.g., `5`): regular grid every Nth row/col
- CSV path: file with `"name"`, `"x"`, `"y"` columns
- Shapefile path: point-type shapefile

### List-Type Files (e.g., boundary conditions, wells)

```python
# GHB conductance — multiplicative grid
pf.add_parameters(
    'model.ghb_stress_period_data_1.txt',
    par_type="grid",
    index_cols=[0, 1, 2],           # layer, row, col columns
    use_cols=[4],                   # conductance column
    par_name_base="ghbcondgr",
    pargp="ghbcondgr",
    geostruct=grid_gs,
    lower_bound=0.1, upper_bound=10.0
)

# GHB head — additive grid
pf.add_parameters(
    'model.ghb_stress_period_data_1.txt',
    par_type="grid",
    par_style="a",                  # Additive
    index_cols=[0, 1, 2],
    use_cols=[3],                   # head column
    par_name_base="ghbheadgr",
    pargp="ghbheadgr",
    geostruct=grid_gs,
    transform="none",               # No log-transform for additive
    lower_bound=-2.0, upper_bound=2.0,
    ult_lbound=32.5, ult_ubound=42
)
```

### Temporal Parameters (across stress periods)

Loop over stress-period files with `datetime` and `geostruct` for temporal correlation:

```python
dts = pd.to_datetime("1-1-2008") + pd.to_timedelta(np.cumsum(sp_data['perlen']), unit='d')

for f in wel_files:
    kper = int(f.split('.')[1].split('_')[-1]) - 1
    pf.add_parameters(
        filenames=f,
        index_cols=[0, 1, 2],
        use_cols=[3],
        par_type="constant",        # Single multiplier per stress period
        par_name_base="welcst",
        pargp="welcst",             # Same group enables temporal correlation
        upper_bound=4, lower_bound=0.25,
        datetime=dts[kper],
        geostruct=temporal_gs
    )
```

---

## 4. Model Run Command

```python
pf.mod_sys_cmds.append("mf6")
# Optionally:
pf.mod_sys_cmds.append("mp7 model_mp.mpsim")
```

---

## 5. Helper Functions (Pre/Post-Processing)

```python
pf.add_py_function(
    "helpers.py",                           # Source file (copied into template_ws)
    "extract_hds_arrays_and_list_dfs()",    # Function call string
    is_pre_cmd=False                        # False = post-processor
)
```

Common post-processing tasks:
- Extract head arrays from binary `.hds` files
- Extract budget components from `.lst` files
- Calculate secondary observations (temporal/spatial differences)

---

## 6. Adding Observations

### CSV-Based Observations

```python
hds_df = pf.add_observations(
    "heads.csv",
    insfile="heads.csv.ins",
    index_cols="time",
    use_cols=list(df.columns.values),
    prefix="hds"
)
```

### Secondary Observations (Temporal Differences)

Calculate differences from a baseline time step to reduce model-error influence:

```python
# In a helper function:
df = pd.read_csv("heads.csv", index_col='time')
df_tdiff = df - df.iloc[0, :]  # Difference from first time step
df_tdiff.to_csv("heads.tdiff.csv")

# Then register:
pf.add_observations("heads.tdiff.csv", index_cols="time",
                     use_cols=list(df.columns), prefix="hdstd")
```

### Array-Based Observations

```python
pf.add_observations("hdslay0_t1.txt", prefix="hdslay0_t1", obsgp="hdslay0_t1")
```

### Observation Naming Convention

Generated names follow: `{prefix}_{location}_{time_index}`
Example: `hds_trgw-2-1_time:365.0`

The `oname` column in `pst.observation_data` stores the prefix for batch selection.

---

## 7. Build the PST

```python
pst = pf.build_pst()
```

This generates the `.pst` control file and `forward_run.py`. All subsequent modifications are in-memory until `pst.write()`.

---

## 8. Post-Build Modifications

### Setting Observation Weights

```python
obs = pst.observation_data

# Step 1: Zero all weights
obs.loc[:, 'weight'] = 0

# Step 2: Activate only observations with measured data (by time window)
obs_in_calibration = obs.loc[(obs.time > 3652.5) & (obs.time <= 4018.5)].index
obs.loc[obs_in_calibration, 'weight'] = 1.0

# Step 3: Balance phi contributions across observation groups
balanced = {grp: pst.phi / len(pst.nnz_obs_groups) for grp in pst.nnz_obs_groups}
pst.adjust_weights(obsgrp_dict=balanced)
```

### Setting Standard Deviations for Uncertainty Analysis

**Critical**: weights control calibration visibility; `standard_deviation` controls noise for pestpp-ies/FOSM. They are NOT the same thing.

```python
obs.loc[:, "standard_deviation"] = np.nan
obs.loc[hds_obs, "standard_deviation"] = 0.3          # metres
obs.loc[hdstd_obs, "standard_deviation"] = 0.001       # metres
obs.loc[sfr_obs, "standard_deviation"] = obs.loc[sfr_obs, "obsval"] * 0.15  # 15% of value
```

### Updating Observed Values

Replace model-output values with actual measured data:

```python
# Load measured data, align to model time steps, update obs.obsval
obs.loc[measured_obs, 'obsval'] = measured_values
```

### Parameter Adjustments

```python
par = pst.parameter_data

# For parameters with initial value = 0, use offset
par.loc[par.parval1 == 0, 'offset'] = -10
# Adjust parval1/bounds accordingly

# Set increment type to absolute for near-zero parameters
pst.parameter_groups.loc[head_pargps, 'inctyp'] = 'absolute'
```

### Control Data

```python
pst.control_data.noptmax = 0   # 0 = test run (no optimisation)
```

---

## 9. Prior Covariance and Ensemble

```python
# Covariance matrix
cov = pf.build_prior(fmt='coo', filename=os.path.join(template_ws, "prior_cov.jcb"))

# Prior ensemble
pe = pf.draw(num_reals=1000, use_specsim=True)
pe.enforce()  # Enforce parameter bounds
pe.to_binary(os.path.join(template_ws, "prior_pe.jcb"))
```

---

## 10. Write and Test

```python
pst.write(os.path.join(template_ws, 'pest.pst'), version=2)  # version=2 for >10k parameters

# Test run
pyemu.os_utils.run("pestpp-glm pest.pst", cwd=template_ws)
```

---

## Parameter Naming Conventions

| Suffix | Meaning |
|--------|---------|
| `gr` | Grid-scale |
| `pp` | Pilot points |
| `cn` | Constant |
| `tcn` | Temporal constant |

Examples: `npfkgr` (NPF K grid), `welcst` (well constant), `ghbcondgr` (GHB conductance grid), `sfrgr` (SFR inflow grid).

## Key Reminders

- **Multiple parameterisation scales** can be applied to the same file (e.g., constant + pilot points + grid). PstFrom chains multipliers.
- **`zone_array`** masks inactive cells — typically use the `idomain` array.
- **Temporal correlation** requires `datetime` and a temporal `geostruct` on each `add_parameters` call; use the same `pargp` across stress periods.
- **`ult_lbound`/`ult_ubound`** constrain the final model value (after multipliers are applied), not the parameter itself.
- Always test with `noptmax=0` before running calibration.
- Write PST with `version=2` when parameter count exceeds 10,000.
