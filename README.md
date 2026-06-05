# Decision-Support Reactive Transport Modelling

A series of tutorial notebooks on doing decision-support reactive transport modelling under uncertainty. The worked case is **DIZON** — the "Deep well Injection in Zuid-Oost Nederland" aquifer storage and recovery experiment at Someren (NL), where pre-treated oxic surface water was injected ~300 m deep into an anoxic, pyrite-rich aquifer. Pyrite oxidation releases sulfate, and the decision question is whether water drawn back out later exceeds the drinking-water sulfate threshold. We use this case to teach how to carry a reactive transport model through history matching, uncertainty quantification, data worth and optimization — and how to think about doing so when each model run is expensive.

These tutorials provide examples of both "how to use" the software and "how to think" about using it. The workflows are implemented programmatically in Python — using `flopy`, `mf6rtm` (the MODFLOW 6 API coupled to PHREEQC) and `pyemu` to drive PEST++ — but the concepts are not limited to programmatic workflows. If you want to learn how to implement these workflows, you are encouraged to run the notebooks yourself. If you just want a high-level understanding, you can read through them without running anything.

## Who this is for

The intended audience is **reactive transport modellers**: people fluent in geochemistry and reactive transport who are new to PEST/PEST++ and to uncertainty quantification. No prior experience with parameter estimation, ensemble methods or decision-support modelling is assumed — we build those ideas up from the case. Some familiarity with Python, Jupyter and MODFLOW 6 is nice to have, but not required.

## Why emulation-first

A single DIZON model run takes about 6 minutes on a MacBook (longer on a workshop laptop). The ensemble workloads that decision-support modelling depends on — a 201-realisation prior Monte Carlo, full-model iterative ensemble smoother runs, an optimization training sweep — are simply not affordable to run live in a course setting. That cost *is* the thesis of this series.

So the central method here is **emulation**. We train a Data Space Inversion (DSI) emulator on the prior Monte Carlo ensemble, then do the history matching, data-worth analysis and optimization on the emulator — in seconds rather than days — always validating the emulator against held-out full-model runs before we trust it. The expensive full-model results are run once by the maintainer and shipped pre-baked (see [`prebaked/`](#prebaked-what-was-run-for-you) below), so the notebooks load them rather than regenerate them.

## Relationship to the GMDSI notebooks

This series is a standalone **sibling** of the [GMDSI_notebooks](https://github.com/gmdsi/GMDSI_notebooks) (the Freyberg curriculum). It clones their conventions, voice and notebook structure, but is a separate repository — it is not folded into them. Where this series needs the general theory of Bayes, ensemble methods, PEST++ or DSI in more depth than a reactive-transport reader needs to get going, it links out to the GMDSI notebooks rather than re-deriving it. Read the GMDSI notebooks alongside these if you want the fuller PEST/UQ treatment.

## Installation

**Get the repository.** Clone it (recommended, if you are comfortable with git) or download it as a zip from GitHub and unzip it.

**Create the Python environment.** Install [miniforge](https://github.com/conda-forge/miniforge) if you do not already have a conda distribution. From the repository folder:

```bash
conda env create -f environment.yml   # creates the env "rtm_gmdsi" (python 3.12)
conda activate rtm_gmdsi
```

This may take a while. `environment.yml` pip-installs four packages **editable from `dependencies/`** (`flopy`, `pyemu`, `mf6rtm`, `vorflow`). These are **vendored source snapshots committed to this repo**, not PyPI releases and not git submodules. The series relies on development-branch features of `pyemu` (the `pyemu.emulators` DSI/DSIVC machinery) and of `mf6rtm`, so please do not swap in released versions — see [`dependencies/PROVENANCE.md`](dependencies/PROVENANCE.md) for the exact provenance of each.

The platform executables (`mf6`, `gridgen`, `pestpp-ies`, `pestpp-mou`, `libmf6`) are committed under `bin/mac` and `bin/win`; the notebooks copy them into each workspace as needed, so you never need them on your `PATH`.

**Start jupyter.** With the environment activated, from the repository folder:

```bash
jupyter notebook
```

then navigate into `tutorials/` and open a notebook.

## How the notebooks are organised

The notebooks in `tutorials/` are organised in two parts. **part0** is standalone background — the notebooks can be run in any order and do not depend on each other. **part1** is the DIZON workflow proper and **must be run in strict order**: each notebook builds the workspace the next one consumes, opens with a prerequisite check that points you back upstream if you have skipped a step, and asserts it is using the vendored `pyemu`.

### part0 — background (any order)

| Notebook | What it covers |
|---|---|
| `part0_01_intro_to_dizon/intro_to_dizon.ipynb` | The DIZON field experiment, the pyrite-oxidation chemistry, and the decision question recast for this series. |
| `part0_02_intro_to_mf6rtm/intro_to_mf6rtm.ipynb` | The `mf6rtm` API on a tiny 1D pyritic column that runs in seconds — solutions, equilibrium phases, kinetics, exchange, and reading breakthrough. |
| `part0_03_uq_for_rtm/uq_for_rtm.ipynb` | A compressed uncertainty-quantification primer for reactive transport modellers: prior, realisation, ensemble, Bayes in pictures, and why "calibrate then forget" fails decision questions. |
| `part0_04_intro_to_pest_and_ies/intro_to_pest_and_ies.ipynb` | The model-as-black-box contract (template/instruction/control files), where `pyemu`/`PstFrom` fit, and how PESTPP-IES works. |
| `part0_05_intro_to_dsi/intro_to_dsi.ipynb` | Data Space Inversion: observation-space emulation, the latent space, and what an emulator can and cannot answer. |

We recommend the first two as background before starting part1; the rest stand alone as reference for topics that come up later.

### part1 — the DIZON workflow (strict order)

Run these in sequence:

1. `part1_01_build_model/dizon_build_model.ipynb` — build the base DIZON model (voronoi grid, MODFLOW 6 flow and transport, PHREEQC chemistry), run it once to feel the cost, and compare against measured field data for credibility.
2. `part1_02_pstfrom_setup/dizon_pstfrom_setup.ipynb` — set up the PEST interface with `PstFrom` around the layered prior (flow, transport, reaction tiers), name the forecast group, and draw the prior ensemble.
3. `part1_03_obs_weights_and_truth/dizon_obs_weights_and_truth.ipynb` — enact the timeline: choose the conditioning species and species-specific noise, zero the weights after the decision date, hold back the cations, and select the synthetic truth.
4. `part1_04_prior_mc/dizon_prior_mc.ipynb` — prior Monte Carlo: a token live run, then the pre-baked full prior ensemble, and the prior forecast distribution — do we even need to history match?
5. `part1_05_dsi_basics/dizon_dsi_basics.ipynb` — train the DSI emulator on the prior ensemble, check its fidelity against held-out realisations, then history-match on the emulator and compare the posterior forecast distribution to the prior.
6. `part1_06_full_model_check/dizon_full_model_check.ipynb` — load (don't run) the full-model ensemble smoother results and use them to validate the DSI posterior, honest about where they disagree.
7. `part1_07_dataworth/dizon_dataworth.ipynb` — the held-back cations cash in: retrain the emulator on different observation subsets and compare forecast uncertainty — data worth for nearly free.
8. `part1_08_optimization/dizon_optimization.ipynb` — DSIVC: the canonical decision problem, an optimization run over the emulator, and the Pareto front of volume supplied against the P95 of peak sulfate.
9. `part1_09_real_data_capstone/dizon_real_data_capstone.ipynb` — optional: swap the synthetic truth for measured data and confront model-structural error honestly.

### A note on parallel runs

When a notebook launches parallel model runs, PEST++ coordinates the workers over local network protocols. The first time this happens, your operating system may pop up a firewall warning (Windows Defender or similar). You can close it — clicking "cancel" — and the runs will proceed.

## `prebaked/`: what was run for you

The expensive full-model results live in `prebaked/`. These are **thinned ensembles** (cut down to the curated observation set plus the forecast group, so a few tens of MB rather than gigabytes), produced once by the maintainer and committed to the repository. Notebooks **load** them; they never regenerate them. Each artifact carries a provenance note recording the hardware, wall-time and date it was produced.

| Artifact | Consumed by |
|---|---|
| Prior Monte Carlo observation ensemble (201 realisations, thinned) | part1_04, part1_05 |
| Full-model iterative ensemble smoother results (thinned per iteration) | part1_06, part1_08 |
| DSIVC training sweep (posterior fields × resampled decision variables) | part1_08 |

This is a deliberate design choice (see [`docs/adr/0002-emulation-first-prebaked-artifacts.md`](docs/adr/0002-emulation-first-prebaked-artifacts.md)): re-running a tutorial can never dirty your git working tree, because the notebooks only read from `prebaked/`, never write to it.

## Runtime expectations

Most of part0 and the analysis cells throughout part1 run in seconds. The "feel the cost" beats — one live forward run per expensive notebook — take about 6 minutes each. Everything heavier than that has been pre-baked: you will load it, not run it. None of the notebooks asks you to reproduce the full 201-realisation prior Monte Carlo or the full-model ensemble smoother live.

## Reference and acknowledgments

The DIZON case is adapted from the field experiment described by Prommer & Stuyfzand (2005) and Stuyfzand et al. (2002). The tutorial model is a decision-support recast of that study — faithful in chemistry, not a reproduction. The notebook conventions, structure and voice follow the [GMDSI_notebooks](https://github.com/gmdsi/GMDSI_notebooks), developed with support from the U.S. Geological Survey and the Groundwater Modelling Decision Support Initiative (GMDSI), jointly funded by BHP and Rio Tinto.
