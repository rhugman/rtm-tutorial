# Emulation-first curriculum; full-model runs ship as pre-baked artifacts

The flagship DIZON model costs ~6 min/run on a MacBook (worse on workshop laptops), so the ensemble workloads the curriculum depends on (201-realisation prior MC; noptmax=3 full-model IES; the DSIVC training sweep) can never run live in a workshop. We decided the curriculum is **emulation-first**: DSI is the canonical conditioning route, the full-model IES exists only as a pre-baked load-don't-run validation notebook, and all expensive artifacts ship as **thinned ensembles committed to a tracked `prebaked/` directory** (thinned to the curated obs set + forecast group, ~tens of MB), written by a maintainer script — never by the tutorial notebooks themselves, so re-running tutorials cannot dirty git. Each expensive notebook runs at most one live forward run (the "feel the cost" beat) and loads pre-baked results for analysis.

## Considered Options

- Git LFS — rejected: tooling prerequisite and quota pain; a known failure mode with 30 workshop attendees cloning at once.
- Release/Zenodo downloads via helper — rejected: adds a download step that fails exactly when workshop wifi does.
- A coarsened "mini-DIZON" that runs live — rejected for v1: a second model variant doubles maintenance and dilutes the "this is what RTM really costs" message the emulation thesis depends on.

## Consequences

- The repo carries binary result artifacts in git; refreshing them is a documented maintainer workflow, not a notebook side effect.
- Every part1 notebook must be executable end-to-end by design (token live runs + pre-baked loads); the nightly autotest depends on this.
