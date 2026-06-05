# Curriculum redesign plan

The executable spec for rebuilding this repo into a GMDSI-style tutorial series on decision-support reactive transport modelling. Domain language lives in [`CONTEXT.md`](../CONTEXT.md) (read it first — terms like *synthetic truth*, *history period*, *layered prior*, *held-back species* are used here without redefinition). Architectural decisions: [`docs/adr/`](adr/). Prose voice: `.claude/skills/gmdsi-voice/SKILL.md`.

## Identity

Standalone sibling of [GMDSI_notebooks](https://github.com/gmdsi/GMDSI_notebooks), cloning its conventions, **not** folded into it. Core audience: **reactive transport modellers** — fluent in reactions, new to PEST/UQ. The series introduces the mf6rtm (MODFLOW 6 API + PHREEQC) workflow, then teaches decision-support uncertainty quantification with **emulation (DSI) as the central method**, because RTM run costs (~6 min/run for DIZON on a MacBook) make full-model ensemble workflows impractical — that cost *is* the thesis; quote it often.

## Locked design decisions

| Topic | Decision |
|---|---|
| Truth | Synthetic truth spine (deliberately chosen prior realisation, upper prior quartile: peak SO₄ ≈ 87–95 mg/L); measured data = motivation in part1_01 + optional capstone (ADR-0001) |
| Forecast | Peak SO₄ at supply well over the supply period (max over all supply screens, mg/L = mol/L × 96.06×10³); named forecast group from PEST setup onward |
| Payoffs | **Minimize/design framing**: forecast distribution (median + P95) prior vs posterior — "truer news, not better news" (prior ~82 → posterior ~92 mg/L median, tighter); one-cell 90 mg/L risk lens (P 0.15→0.67); EU 250 mg/L = one sentence → dataworth (uncertainty reduction) → optimization (volume vs P95) |
| Compute | Emulation-first; full-model runs pre-baked in tracked `prebaked/`, thinned to curated obs set; one live forward run per expensive notebook max (ADR-0002) |
| Prior | Layered: K + porosity (+ dispersivity, to be verified) + pyrite m0 + pyrite rate (to be verified); OM excluded with citation; heat params fixed with stated reason |
| Timeline | History 0–252 d / decision date 252 d / supply 308–728 d; the lead-time gap is canonical |
| Obs | Conditioning species SO₄, O₂, NO₃, pH, Tmp; species-specific noise (proportional+floor for concs; absolute for pH ~0.1, Tmp ~0.5 °C); phi factors per site:species group; pe never conditioned; cations held back |
| DSI | Plain `pyemu.emulators.DSI` in core (DSIAE = later add-on); transform choice taught as a beat (untransformed failure first); `ies_multimodal_alpha=0.99` always, with an explanation cell; runstor workflow; fidelity check before conditioning, always |
| Optimization | dvars = supply-well rate multiplier + switch-on day; objectives = volume supplied vs P95(peak SO₄) — minimize-native, 90 mg/L trigger drawn as illustrative lens only; DSIVC training sweep off `master_hm` posterior fields |
| Dependencies | Vendored source trees kept; pyemu refreshed from `rhugman/pyemu@feat_dsivc` (e986b27); document SHA + refresh procedure per dependency |
| Testing | `autotest/nb_tests.py` (mothership pattern), maintainer/nightly; pyrite column + part0 = fast CI-able subset |

## Target layout

```
├── README.md            ← audience, env setup, sequencing DAG, prebaked/ explanation
├── CONTEXT.md           ← glossary (exists)
├── docs/{REDESIGN.md, adr/}
├── environment.yml      ← deduped
├── bin/{mac,win,linux}  ← linux to be added
├── data/                ← field data + GIS inputs ONLY; nothing generated is ever written here
├── dependencies/        ← vendored flopy, pyemu, mf6rtm, vorflow (+ PROVENANCE.md with SHAs)
├── prebaked/            ← tracked thinned artifacts (maintainer-written; see inventory)
├── etc/                 ← maintainer scripts (write prebaked, refresh deps)
├── autotest/nb_tests.py
└── tutorials/
    ├── herebedragons.py ← shared helpers (cleaned; see below)
    ├── part0_01_intro_to_dizon/
    ├── part0_02_intro_to_mf6rtm/
    ├── part0_03_uq_for_rtm/
    ├── part0_04_intro_to_pest_and_ies/
    ├── part0_05_intro_to_dsi/
    ├── part1_01_build_model/
    ├── part1_02_pstfrom_setup/
    ├── part1_03_obs_weights_and_truth/
    ├── part1_04_prior_mc/
    ├── part1_05_dsi_basics/
    ├── part1_06_full_model_check/
    ├── part1_07_dataworth/
    ├── part1_08_optimization/
    └── part1_09_real_data_capstone/
```

Every notebook's generated workspaces live **inside its own directory** (mothership pattern). part0 = standalone, any order. part1 = strict order; each notebook opens with a prerequisite check (`raise Exception("you need to run ...")` pointing at the upstream notebook) and `assert "dependencies" in pyemu.__file__`-style dependency asserts.

## Notebook specs

Format: **goal · salvage source · key beats · prebaked deps**. "Skeleton" = full markdown narrative + headed TODO code cells (prebaked artifacts don't exist yet); "draft" = complete code + prose, untested until the maintainer pass.

### part0 (standalone background)

- **part0_01_intro_to_dizon** *(draft)* — The field experiment, the chemistry, the decision question. Salvage: the strong prose of current `00_model.ipynb` (Background/Summary/Key-points/Decision-support-objective cells). Beats: site & experiment (Someren, 300 m, 854 d, Prommer & Stuyfzand 2005); the redox story (pyrite oxidation by O₂+NO₃, temperature-dependent rates — the paper's headline); the decision recast (supply well, treatment threshold); timeline figure (history/decision date/supply); one honest sentence on model provenance (728 vs 854 d abstraction, adapted not reproduced). Mostly markdown; figures may load from `data/`.
- **part0_02_intro_to_mf6rtm** *(draft)* — The mup3d API on the **pyrite column** (new ~50-line model: 1D pyritic column, oxic injectate, runs in seconds — crib chemistry blocks from the DIZON build). Beats: solutions → equilibrium phases → kinetics → exchanger → write → run via API → read/plot breakthrough; redox-front intuition (warm vs cold injectate as a teaser for temperature dependence). This model is also the regression-test canary.
- **part0_03_uq_for_rtm** *(draft)* — Compressed UQ primer for RTM modellers: prior, realisation/ensemble, Bayes in pictures, why "calibrate then forget" fails decision questions. Link out to GMDSI_notebooks part0 for depth; do not re-derive.
- **part0_04_intro_to_pest_and_ies** *(draft)* — The model-as-black-box contract (template/instruction/control files), where pyemu/PstFrom sits, and how PESTPP-IES works (iterative ensemble smoother, in pictures). Source material: mothership `part2_06_ies` long-form prose, compressed. Demonstrate on the pyrite column if a cheap demo helps, else prose+figures.
- **part0_05_intro_to_dsi** *(draft)* — DSI theory: obs-space emulation, SVD/latent space, what it can and cannot answer. Source: mothership `part0_intro_to_eva_and_dsi`, adapted (see `.refs/`).

### part1 (the DIZON workflow, strict order)

- **part1_01_build_model** *(draft)* — current `00_model.ipynb` cleaned: drop the prose that moved to part0_01 (keep a recap + link); kill empty/debug cells (idx 22, 36–41, 44); explain or name every magic number (`top=-273`, ICs, `kin_py_params`, `distcoef`, `pbulk`); remove commented-out experiments; grid outputs written to the notebook's workspace, **never** `data/`; canonical obsid spelling; ends by running the model once (the "feel the cost" beat) and comparing against measured data (credibility, not conditioning).
- **part1_02_pstfrom_setup** *(draft)* — current `01_pstfrom.ipynb` rebuilt around the **layered prior** narrative (flow/transport/reaction tiers as section headings). Fix the "Organic Matter" heading lie (it's pyrite kinetics; state why OM is excluded, cite the paper); resolve dead blocks (dispersivity un-commented *pending verification*; pyrite rate added *pending verification*); fix `SphVario`-vs-"exponential" comment, `np.NaN`→`np.nan`, inconsistent `ult_ubound`s; one canonical `pest.pst` write; reconcile realisation counts (draw and keep **201**); forecast group named here.
- **part1_03_obs_weights_and_truth** *(new draft; salvage weighting/noise code from `xx_dsi` cells 13–30)* — enacts the timeline: conditioning species + species-specific noise table (each σ justified in one honest sentence); weights zero after day 252; held-back cations stated as a promise ("we'll come back to these"); synthetic-truth selection (scan prior forecast distribution, pick a realisation from the upper quartile — peak SO₄ ≈ 87–95 mg/L — show it, name it); obsid scheme explained once; timeline figure reused.
- **part1_04_prior_mc** *(draft)* — current `02_priomc.ipynb` + the prior risk beat. Fix number drift (201 everywhere); `psutil` worker count; token live run (handful of reals) then load pre-baked full prior MC; payoff: the prior forecast distribution (median + P95 of peak SO₄) plus the one-cell 90 mg/L risk lens — "do we even need to history match?"; fix `raise Exception()` empty message, REPL-poke cells, "dsi-ae notebook" ghost reference.
- **part1_05_dsi_basics** *(draft; split from `xx_dsi` cells 1–37)* — train DSI on prior MC (minus truth); **fidelity check first** (held-out realisations; the transform beat: untransformed → negative concentrations → fix); then condition with PESTPP-IES on the emulator (runstor, `ies_multimodal_alpha=0.99` + explanation cell, noise ensemble); payoff: posterior vs prior forecast distribution (median + P95; "truer news, not better news") plus the 90 mg/L lens cell, seconds-vs-days compute contrast.
- **part1_06_full_model_check** *(skeleton until `prebaked/` exists; salvage `xx_dsi` cells 38–45)* — load-don't-run: the 201-real full-model IES (`master_hm` equivalent) as validation of the DSI posterior; honest about where they disagree; states the compute bill that was paid for you.
- **part1_07_dataworth** *(skeleton)* — the held-back cations cash in: retrain DSI on obs subsets (± cations, ± sites, ± species), compare forecast uncertainty; "dataworth for nearly free" as the emulation dividend. No new full-model runs.
- **part1_08_optimization** *(skeleton)* — DSIVC: the canonical decision problem (see CONTEXT.md), training sweep loaded from `prebaked/`, MOU over the emulator, Pareto front (volume vs P95 peak SO₄) with the 90 mg/L trigger drawn as the illustrative reliable-vs-robust lens. Requires pyemu `feat_dsivc`.
- **part1_09_real_data_capstone** *(skeleton)* — optional: swap synthetic truth for measured data, confront structural error honestly; what changes in the workflow, what breaks, what that teaches.

## `prebaked/` inventory (maintainer-produced)

| Artifact | Produces | Consumed by |
|---|---|---|
| Prior MC obs ensemble (thinned to curated obs + forecast group, 201 reals) | `etc/` script | 1_04, 1_05 |
| Full-model IES results (par+obs ensembles per iter, thinned) | `etc/` script | 1_06, 1_08 |
| DSIVC training sweep (posterior fields × resampled dvars, ~200 runs) | `etc/` script | 1_08 |
| Provenance note per artifact (hardware, wall-time, date, SHA) | maintainer | README + notebooks |

## `herebedragons.py` cleanup

Move to `tutorials/`; remove module-level `PestUtilsLib()` instantiation; delete dead code (`fini_rates_out` overwrite, vestigial `wellout_fini`/`fini_sp` machinery, no-op `add_conc` ternary, unused `exchanger_ic`); merge the near-duplicate `make_wel_out`/`make_wel_opt`; docstrings on every public function (`process_sim_conc` first); `get_bins` resolves `bin/` relative to repo root, supports linux, fails with a helpful message; runtime-injected functions (`add_py_function` targets) clearly sectioned and self-contained.

## Open verification items (test before the curriculum commits)

1. Pyrite **rate** parameterisation — is it exposed in a PstFrom-templatable mf6rtm input at all?
2. Dispersivity block — unknown why it was commented out; needs a test run.
3. Runstor end-to-end — currently mis-wired (no `.rns`; file-based tpl/ins deployed); root `forward_run.py` drifted from template copy (`__main__` differs); `dsi_forward_run` has a latent `NameError` (`pyemu` unimported).
4. `ies_multimodal_alpha` explanation cell must state the author's actual reasoning (draft: per-realisation solves are robust to outlier realisations/prior-data conflict without imposing strong localisation — **confirm with author**).
5. `bin/linux` binaries to source (mf6, pestpp-ies, pestpp-mou, libmf6).
6. Ghost obsids in salvaged plotting code (`wp3-*`, `ip2-*`, `wp1-f4/f5`) — verify against the obs data before reuse.

## Hygiene (one-time)

Untrack `data/mf6_grid.*` (generated); `.gitignore` gains generated-workspace patterns under `tutorials/`, root `*.pdf/*.png/*.log`, `.DS_Store`, `__pycache__/`; move root debris (`obs_plots*.pdf`, `results_tseries.png`, `PstFrom.log`, `pst.obs_data.nans.csv`) to a gitignored `attic/` (delete later at leisure); dedupe `geopandas` in `environment.yml`; retire `rtm-tutorial.code-workspace` path assumptions if any.
