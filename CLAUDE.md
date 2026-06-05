# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A GMDSI-style tutorial series on decision-support reactive transport modelling under uncertainty, built on the DIZON deep-well-injection ASR field experiment (Prommer & Stuyfzand, 2005): oxic water injected into a pyrite-rich anoxic aquifer, with the decision question being how much sulfate the drinking-water supply well will carry — a minimize/design framing (treatment capacity sized to the P95 of peak SO₄), with threshold exceedance appearing only as an illustrative risk lens. A single full-model run costs ~6 min on a MacBook, so the series is **emulation-first**: a DSI emulator does the history matching, data worth and optimization, and expensive full-model results ship pre-baked. The deliverable is the sequence of Jupyter notebooks in `tutorials/`, not a Python package.

## Read these first (in order)

1. **`CONTEXT.md`** — the canonical glossary. Use its terms and numbers exactly (201 realisations; history 0–252 d / decision date 252 / supply 308–728 d; forecast = peak SO₄, max over supply screens, mg/L; minimize/design framing with a 90 mg/L illustrative risk lens; conditioning species SO₄/O₂/NO₃/pH/Tmp; cations held back; well names injection/flush/supply with code ids `wellin`/`wellout`/`wellopt`). Read it before `REDESIGN.md`.
2. **`docs/REDESIGN.md`** — the executable spec for the curriculum: locked design decisions, target layout, per-notebook specs, the `prebaked/` inventory, and open verification items.
3. **`docs/adr/`** — architectural decision records (0001 synthetic-truth spine; 0002 emulation-first pre-baked artifacts). Consult before reopening a settled decision.

## Prose voice

All tutorial prose (notebook markdown cells, READMEs, docs) must follow the GMDSI tutorial voice. The skill `.claude/skills/gmdsi-voice/SKILL.md` defines it: instructor-led first-person plural, conversational but rigorous, terminology defined in quotes on first use, short markdown cells that lead into the next code cell. No AI/Claude references anywhere in content.

## Repository layout

```
tutorials/            ← the notebooks; part0 (background, any order) + part1 (DIZON workflow, strict order)
  herebedragons.py    ← shared helper module imported by every notebook
  part0_01..05_*/     ← standalone background notebooks
  part1_01..09_*/     ← the DIZON workflow; each notebook builds the workspace the next consumes
prebaked/             ← tracked, thinned full-model artifacts (maintainer-written; notebooks only read these)
etc/                  ← maintainer scripts (write prebaked, refresh deps); not run by tutorials
dependencies/         ← vendored flopy, pyemu, mf6rtm, vorflow source trees (+ PROVENANCE.md)
bin/{mac,win,linux}   ← committed platform binaries (linux to be added)
data/                 ← field data + GIS inputs ONLY; nothing generated is ever written here
docs/                 ← REDESIGN.md (spec) + adr/ (decision records)
autotest/             ← nb_tests.py (mothership pattern; maintainer/nightly)
```

Every notebook's generated workspace lives **inside its own directory** (mothership pattern). part1 notebooks open with a prerequisite check (`raise Exception("you need to run ...")` pointing upstream) and an `assert "dependencies" in pyemu.__file__`-style dependency assert.

## The `prebaked/` convention

Full-model runs are too expensive to run live (~6 min each; the prior MC is 201 realisations). They are run once by a maintainer script in `etc/`, **thinned** to the curated obs set plus forecast group, and committed to `prebaked/`. Tutorials **load** these artifacts; they **never write tracked artifacts**. This is load-bearing: re-running any tutorial must not dirty git (ADR-0002). When adding analysis to an expensive notebook, read from `prebaked/`; if you need a new artifact, add a generator script to `etc/`, do not write it from the notebook.

## Environment setup

```bash
conda env create -f environment.yml   # creates env "rtm_gmdsi" (python 3.12)
conda activate rtm_gmdsi
```

`environment.yml` pip-installs four packages **editable from `dependencies/`** (`flopy`, `pyemu`, `mf6rtm`, `vorflow`). These are vendored source snapshots committed to this repo, not submodules and not PyPI releases. The series relies on dev-branch features of `pyemu` (the `pyemu.emulators` DSI/DSIVC/DSIAE machinery) and `mf6rtm`, so do not swap in released versions. Provenance and the refresh procedure for each vendored dependency are recorded in `dependencies/PROVENANCE.md`.

The vendored `pyemu` is **`rhugman/pyemu` branch `feat_dsivc`, commit `e986b27e5ea0a83a0b8603cc167f481aaa120d08`** (refreshed 2026-06-05). The optimization notebook (`part1_08`) requires the `feat_dsivc` DSIVC additions specifically.

## Binaries and `get_bins`

Platform executables (`mf6`, `gridgen`, `pestpp-ies`, `pestpp-mou`, `libmf6`) are committed under `bin/mac` and `bin/win`. `tutorials/herebedragons.get_bins(local_dir)` resolves `bin/` relative to the repo root (walking up from the module), picks the subdirectory for the current platform (`win`/`mac`/`linux`), and copies the binaries into the workspace — so the notebooks never assume anything is on `PATH`. It raises a helpful `FileNotFoundError` if `bin/<platform>` is missing; `bin/linux` is not yet populated (see `docs/REDESIGN.md` open items).

## Testing

`autotest/nb_tests.py` runs notebooks end-to-end (mothership pattern), intended for the maintainer / nightly. The pyrite-column model (part0_02) and the part0 notebooks are the fast, CI-able subset; part1 is executable end-to-end by design (token live runs + pre-baked loads) but heavier.

## Conventions

- Models and PEST runs execute via `pyemu.os_utils.run(...)` / `start_workers(...)` with executables copied into each workspace by `hbd.get_bins()` — never assume `mf6`/`pestpp-ies` are on `PATH`.
- Paths in notebooks use `pathlib.Path`; generated workspaces live inside each notebook's own directory, never in `data/`.
- Notebooks are nbformat 4; build them with a script and verify they parse.

## Legacy directories (do not delete or modify)

The repository root still holds pre-redesign run artifacts: `model/`, `master_priormc/`, `master_hm/`, `pst_template/`, `pst_template2/`, `priormc/`, `dsi_template/`, `tmp/`. These hold expensive run results preserved for **prebaked artifact generation** — the `etc/` scripts thin them into `prebaked/`. Do not run, delete or modify them (each run cost ~6 min). They are gitignored and will be retired once the prebaked pipeline is complete. The salvaged original notebooks (`00_model`, `01_pstfrom`, `02_priomc`, `xx_dsi`) and `forward_run.py` live under `.refs/` as reference material; the live notebooks under `tutorials/` supersede them.
