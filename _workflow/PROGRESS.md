# workflow.py prototype — progress tracker

Two-well ASR refactor (ADR-0003 / ADR-0004). Dev scaffolding: prove the whole arc in
one runnable `workflow.py`, split into `part1_*` notebooks later. This file tracks where we are.

**Source of truth:** `docs/adr/0003-*`, `docs/adr/0004-*`, `docs/MOU_HANDOFF.md`.
**Do NOT** re-litigate design or rewrite CONTEXT.md/REDESIGN.md numbers (pre-rebake).

## Arc (target)

`site → build (2-well continuous + f_treat preprocessor) → pstfrom (+f_treat.tpl) →
obs/weights/truth (recovered-SO4 spine) → prior MC → DSI condition + FOM-IES compare →
dataworth → DSIVC(f_treat) cost-vs-SO4 + FOM loop`

Every stage (ADR-0004): fail-fast QA gate (`raise`) + ≥1 signature figure → gitignored `_figs/`.

## Slices

- [x] **Slice 1 — shared style + QA-gate helper** (`_workflow/wf_style.py`) ✓ smoke-tested
  - `apply_style()` fixed rcParams; Okabe-Ito palette `C`; `SPECIES`/`ROLE`/`LBL` maps (units on axes)
  - `qa_gate(check, ok, detail)` + `qa_close(check, val, target, tol, kind)` fail-fast raise (`QAError`)
  - `savefig(fig, name, stage)` → gitignored `_workflow/_figs/<stage>/` (png+pdf)
  - `.gitignore` updated: `_workflow/_figs/`, `_workflow/__pycache__/`
- [x] **Slice 2 — section-2 model build** (`_workflow/workflow.py`) — DONE, all gates green live (2026-07-06)
  - build_two_well_model (wellin on-axis, wellout (−99,0) continuous recovery, wellopt dropped, ×10 ramp removed)
  - apply_treatment(ws, f_treat): scale injdf O(0)+N(+5)×(1−f), re-equilibrate, rewrite wellin aux (Method A), write cost.dat
  - QA: _m0_guard (landmine ii enforced) + _welin_regression (f_treat=0 ≡ baseline) + MF6 converge/mass-bal (--run)
  - sig fig: recovered-SO4 breakthrough @ wellout + timeline (falls back to schematic offline)
  - **Verify caught 1 REAL landmine violation** (interface): chem re-equilibration `model.initialize()→write_simulation()`
    clobbered PstFrom pyrite m0 arrays in live ws. **FIXED**: `chem_wd` scratch-dir isolation + `_m0_guard` byte-check.
  - geochem + regression verdicts CLEAN.
  - OFFLINE QA PASSES LIVE (2026-07-06): regression f_treat=0≡baseline ✓; m0-guard byte-unchanged ✓ (landmine ii
    fix confirmed — scratch isolation writes to _s2_model/_treat_scratch, live m0 arrays untouched); timeline fig ✓.
  - Fixed 2 runtime bugs during integration: (a) `_rewrite_welin_aux` aux names now from recarray dtype (was
    `wel.auxiliary.array` → KeyError 'auxiliary'); (b) QA order — regression BEFORE m0-guard (guard's
    apply_treatment(0.9) was corrupting the regression's pristine snapshot); restore apply_treatment(0.0) after.
  - LIVE RUN PASSED (2026-07-06): MF6 normal termination ✓; flow + all 16 species |mass-bal%|<1% ✓; regression ✓;
    m0-guard ✓; real recovered-SO4 breakthrough fig written (_figs/02_build/recovered_so4_breakthrough.png).
  - PHYSICS FLAGS (not gate failures; surfaced to user): (1) recovered-SO4 magnitudes LOW — flow-wt mean peaks
    ~5.8 mg/L, best screen L3 ~21 mg/L @728d (old supply framing was ~86-90); small lever dynamic range.
    (2) recovered-SO4 peak is BEYOND the 728-d window (still rising at edge) → "peak SO4" conditioning forecast
    (ADR-0003) is censored at endpoint. FORK for slice 4/8: (a) extend tdis until peak; (b) redefine forecast as
    end-of-recovery SO4 [my lean]; (c) accept endpoint-as-peak. DECISION DEFERRED to user; pre-rebake anyway.
  - OPEN Qs: (1) strictly byte-flat injection vs kept CSV ramp (dirties data/?); (2) Method A vs B git-cleanliness of
    aux externals; (3) C_UNIT=2.75e-4 placeholder → retune post-rebake.
  - BUILD-DECISION defaults (in-file): window split cond SP0–18 (0–252d) / gap SP19–20 / forecast SP21–34 (308–728d).
  - geometry: wellin on-axis, wellout (−99,0) recovery; wellopt deleted
  - tdis: monitored window (conditioning) → recovery window (forecast); boundaries TBD
  - f_treat preprocessor: PhreeqcRM re-equilibrate O(0)+N(+5)×(1−f_treat), rewrite wellin aux (Method A flopy)
  - QA gate: MF6 converges + |mass-bal%|<tol + f_treat=0 reproduces baseline wellin aux (regression)
  - sig fig: plume + breakthrough at wellout/monitors + timeline
- [x] **Slice 3 — pstfrom** (Section 3 in workflow.py) — INTERFACE BUILT + phi≈0 VERIFIED (2026-07-06)
  - phi = 1.7e-16 ≈ 0 (pestpp-ies noptmax=0, bg bpdv1t4b8): full injected forward-run chain reproduces the model
    at base params. apply_well_rates + process_sim_conc + process_forecast all work end-to-end. qa_gate_phi added.
  - Signature fig pstfrom_interface (pilot-pt net 312/layer + obs/forecast geometry + par tiers K/poro/disp/pyr=3756 each).
  - DEFERRED to proper slices: weighting (4), prior ensemble draw (5), f_treat/treatment/cost (8, emulator).
  - SCOPE (emulation-first): full-model interface = conditioning + prior-MC over UNCERTAIN AQUIFER params only.
    f_treat/apply_treatment/cost deferred to slice 8 (DSIVC on emulator) — during conditioning f_treat=0 (baseline).
  - Params (pilot pts pp_space=10 + constant/zone, SphVario a=100 log): K(0.001-10,ppu), porosity(0.5-1.5,ult
    0.05-0.65), dsp_alh(0.5-10), pyrite m0(0.05-5,ult 1e-5-10,zone), + pyr-lograte global hand-tpl [15,17]. npar=15025.
  - apply_well_rates (NEW forward-run pre-cmd, self-contained, VALIDATED standalone): reads perturbed K -> T-weighted
    wellin/wellout rates. Ordered after apply_list_and_array_pars, before mf6rtm.
  - Obs: conditioning via hbd.process_sim_conc (conc group) + NEW process_forecast (recovered-SO4 @ wellout, forecast
    group, 1152 times, peak 22 mg/L). nobs=450495 (most weight-1 array/conc bookkeeping -> slice4 zeros non-measured).
  - Cheap gates PASS: build_pst finite, forecast present, pyr-lograte present, no NaN, counts sane.
  - NOW: phi≈0 run (pestpp-ies noptmax=0, bg bpdv1t4b8) — one base forward run through the full injected chain.
  - TODO after phi: signature figs (pilot-pt map, forecast/obs locations, prior par summary); prior ensemble draw
    (slice 5); slice-4 weighting.
- [x] **Slice 4 — obs/weights/truth** (Section 4) — DONE (2026-07-06). truth locked realization 5 (P75, 31.5 mg/L).
  - lock_truth: truth realization's sim -> _truth/{truth_pars,truth_obs,truth_meta}.csv (pars+obs+peak).
  - inject_truth_and_weights (VECTORIZED, no row-wise apply): obsval=truth sim; weight=1/sigma_species on
    CONDITIONING species (SO4/O0/NO3/pH/Tmp) at monitoring wells (wp/pp/ip), history window t<=252, REAL
    measurement times only. 995 weighted obs. Cations held back; forecast zero-weight. Gates pass.
  - SPECIES_SIGMA (obs units, PLACEHOLDER→tune): so4 2e-5, o0 3e-6, no3 5e-6, ph 0.1, tmp 3e-4 mol/L etc.
    Raw sigma sets weight (=1/sigma); pestpp obs-noise from same sigma (no sqrt(n) deflation) [noise-weights memo].
  - Signature fig obs_weights_truth (5 species w/ sigma error bars + weighted counts).
  - TODO: verify noise stays inside prior spread (memo check); tune SPECIES_SIGMA at re-bake.
- 2026-07-06: added `plot_props.py` — TRUTH-model aquifer property fields (plan + xsection) for K, porosity,
  dispersivity, pyrite m0. materialize_truth() applies truth par vector (pst.write_input_files +
  apply_list_and_array_pars) -> perturbed arrays; heterogeneous (K 0.01-100 m/d, poro 0.2-0.55, pyrite 0.1-1.3).
  Fig truth_properties (4 props x plan/xsection, pilot-point structure visible; hi-pyrite band top-between-wells).
  ROBUSTNESS FIX: shutil.copytree silently dropped 8/97 files on the large template dir -> switched
  build_pest_interface staging + plot_props to `cp -R` (subprocess). copytree flaky on big dirs; avoid it.
- 2026-07-06: ARCHITECTURE CHANGE (user): make plotting quantities OBSERVATIONS so no per-realization re-run.
  Prior MC discards worker dirs + pestpp stores only obs values (not S.ucn), so truth spatial wasn't recoverable.
  FIX: added forward-run POST-processors process_spatial_snapshots (SO4 field all cells/layers @ [252,600,900,1300]d
  -> so4_field obs, ZERO weight, for plan/xsection reconstruction) + process_heads (head timeseries @ monitoring
  locations, ~5-day cadence -> head obs, WEIGHTED conditioning data for K). Wired into build_pest_interface
  (add_py_function POST + add_observations + ship obs_loc.csv). nobs 450k->490k (head 32k->thinned, so4_field 7968).
  inject_truth_and_weights now also weights heads (HEAD_SIGMA=0.05m, monthly sampling, history window).
  Rewrote obs_weights_truth fig: timeseries BY LOCATION (colored by monitoring cluster) + noise, species + heads panel.
  Rebuild+rerun stage5 bg bd38e6nka (~15min). AFTER: re-lock truth (now incl head/spatial obs) -> inject weights ->
  plot truth spatial FROM obs ensemble (no re-run). plot_spatial parametrized (ws/prefix/stage/title_tag).
  NOTE: earlier truth re-run was partial (~366d, pkilled) -> that stopgap plot spans early times only; superseded.
- 2026-07-06: stage5 rebuild+rerun bd38e6nka DONE (15/15). Bugfix: prior_forecast sorted forecast times
  LEXICOGRAPHICALLY ("1000"<"308") -> spaghetti/truth line "circled back to t=0"; now numeric sort. Post-run:
  re-locked truth (real 5, peak 31.5, now incl head+spatial obs), inject_truth_and_weights -> 1193 weighted
  (995 species + 198 heads). truth_field_from_obs + plot_field_snapshots: TRUTH SO4 plan/xsection reconstructed
  FROM OBS ENSEMBLE (so4_field group), no re-run — plume 252->1300d, 5->97 mg/L. obs_weights_truth reworked:
  per-SCREEN timeseries colored by location + noise, species + HEAD panel (fixed cluster-zigzag). ALL FIGS
  REGENERATED (02_build/03_pstfrom/04_obs_weights/05_prior_mc). Deleted stale partial truth_spatial_{plan,xsection}.
- 2026-07-06: added `_fig_prior_vs_data` (prior_vs_data.png): grid rows=monitoring sites, cols=obs types
  (SO4/O2/NO3/pH/Tmp/head), prior-sim spaghetti + measured(truth)+noise per site. WEIGHTING FINDINGS (user Q on O0):
  * O0 weight problematic: 172/187 (92%) weighted O0 obs are near-ZERO (consumed O2); w=333k gives O0 sum(w^2)=2e13
    = 44x SO4's -> O2 DOMINATES history-match phi, drowning SO4 (the forecast species). pH negligible (w=10).
    Fig shows O2~0 in history at most sites (noise band > signal at pp1). REC: keep O0 but REBALANCE weights
    (balance per-group phi contributions, or bump O0 sigma to detection-limit ~3e-5) — SPECIES_SIGMA still placeholder.
  * head prior spread ~0 across realizations (balanced doublet + fixed CHD dominate head field regardless of K)
    -> heads likely LOW data-worth for K here; flag for dataworth (slice 7).
  DECISION (user): DO NOT balance weights (that would corrupt noise, since raw sigma sets both). Keep weight=1/sigma.
  Balance phi PER TYPE AND SITE via pestpp-ies `ies_phi_factors` (already the documented design in the
  noise-weights-separate memory: "equal phi share per site:species group via ies_phi_factor_file, not weight
  deflation"). PREREQUISITE (enabler): regroup conditioning obs obgnme = obsid + ":" + variable (per site:species)
  + heads as obsid:head, so phi factors can act per group. Currently all conditioning obs are group "conc"/"head" ->
  TODO in inject_truth_and_weights (memory shows vectorized pattern obs.loc[nz.index,"obgnme"]=obsid+":"+variable).
  Do at slice 4 refine / slice 6 IES config. NOT changing weights.
- 2026-07-06: NOISE DISPLAY (user): show noise as temporally-CORRELATED spaghetti, not +/-sigma bars.
  Added obs_sigma (per-time: conc 7% proportional + per-species floor SIGMA_FLOOR; pH/Tmp/head absolute SIGMA_ABS)
  + draw_obs_noise (one correlated shock z~N(0,1) per series per real, noisy_t = truth_t + z*sigma_t; conc clipped>=0;
  rank-1 => perfectly temporally correlated, verified z-std~4e-16). N_NOISE=20. Updated _fig_weights + _fig_prior_vs_data
  to plot noise spaghetti (faint, colored) instead of errorbars. Matches noise-weights-separate memo (raw sigma sets
  both; noise stays inside prior — visibly tight vs prior spread). This is the DISPLAY noise model; the pestpp
  ies_observation_ensemble draw (slice 6) reuses the same obs_sigma + correlated-shock generator.
- 2026-07-06: STYLE convention (user): PRIOR=grey, POSTERIOR=blue. wf_style ROLE["prior"]->grey; added
  ROLE["history"]=sky for conditioning/history window shading (was routed through prior). Rerouted all axvspan
  window-shading (workflow _fig_breakthrough/_fig_weights, plot_rates _shade) to ROLE["history"]. Bolder spaghetti
  (prior alpha .35->.55 lw .8->1.1; noise alpha .12->.3 lw ->.5-.6). Legends updated. All ensemble/spaghetti figs regen.
- 2026-07-06: STYLE (user): NOISE = bright red, less faint. ROLE["noise"]="#E8000B" (bright red, not vermillion);
  noise spaghetti alpha ->0.45-0.5. To keep truth distinct from red noise, ROLE["truth"]->black (grey prior /
  black truth+measured / bright-red noise scheme). _fig_weights + _fig_prior_vs_data + prior_mc regen. If truth
  should stay a color instead of black, revert ROLE["truth"].
- [x] **Slice 5 — prior MC** (Section 5) — DONE (setup-sized n=15; full 201 later). bg bnwvbuoxe, 15/15 reals OK.
  - draw_prior_ensemble (build_prior coo + draw + enforce + pyr-lograte N(16,0.5)[15,17]) -> prior_pe.jcb.
    run_prior_mc: pestpp-ies noptmax=-1, ies_no_noise, save_binary, PANTHER x15. Fixed post-proc to read .csv
    when save_binary off (that run was csv; future runs binary).
  - PRIOR FORECAST (peak recovered-SO4): min 5.4 / P25 20.5 / P50 22.6 (≈base 22) / P75 30.8 / max 33.7 mg/L.
  - TRUTH CANDIDATE (P75, ADR-0003): realization idx 5, peak 31.5 mg/L. PROVISIONAL (n=15 coarse) — user to confirm.
  - Signature fig prior_mc_forecast (peak hist+P75+truth; recovered-SO4 spaghetti + truth). pick_truth() helper.
  - NEXT: user confirms truth -> slice 4 (weights + truth injection) / slice 6 (DSI condition).
- [x] **Slice 6 — DSI emulator** (Section 6 in workflow.py) — PLUMBING DONE (2026-07-06). FOM-IES compare SKIPPED (user scope).
  - build_dsi_emulator: prior MC obs ensemble -> keepobs (weighted conditioning + forecast, ~2345 cols) -> drop truth
    -> DSI(data=train, transforms=[normal_score,quadratic_extrapolation], energy_threshold=0.99).fit(). latent_dim=11.
  - dsi_encode (clip to training range + pinv rcond=1e-8 guards) + xvalidate_dsi (held-out emulated vs actual:
    neg-conc frac ~1.2%, forecast 1:1). _fig_dsi_xval.
  - build_dsi_conditioning: dsi.prepare_pestpp(use_runstor) -> emulator interface; targets=truth; per-obs
    PROPORTIONAL sigma (obs_sigma) -> weight=1/sigma + standard_deviation; obgnme=obsid:variable -> 132 phi groups
    (ies_phi_factor_file, equal share); correlated-noise ensemble (draw pattern, clip conc>=0) -> ies_observation_ensemble.
    USER ies_ CHOICES: noptmax=1, ies_num_reals=500, ies_subset_size=-100, save_binary. MULTIMODAL: user first said
    alpha=0.99 then KILLED it (too slow) -> multimodal OFF (multimodal_alpha=None default; pop unless given).
  - run_dsi: pestpp-ies dsi.pst /e (runstor, no workers, seconds). dsi_posterior: prior(iter0)/posterior(iterN) forecast.
  - RESULT (setup n=15): prior P50 23.9 -> posterior P50 28.5 mg/L, truth 31.5, covered=False (marginal, truth ~P96).
    Textbook DSI narrowing. 9-obs prior-data-conflict warning. Both artifacts = rank-11 emulator from 12 training reals;
    resolves with full 201-real prior MC. _fig_dsi_posterior (grey prior/blue posterior/black truth, hist+CDF).
  - NEXT (full run gate): rerun prior MC at 201 -> rebuild DSI -> expect calibrated coverage.
- 2026-07-06: LEAVE-ONE-OUT validation (user) — two modes:
  * xvalidate_dsi_loo (full LOO, frac=1.0): each real left out, refit DSI on rest, encode(direct projection)->predict,
    compare forecast. R(actual,emulated)=0.915, mean neg_frac=2.1%. Projection fidelity GOOD even at n=15.
  * dsi_loo_coverage (frac=0.30, 5 reals): leave real out, CONDITION dsi on its obsvals (noise REDRAWN around them
    via build_dsi_conditioning(truth_real=r)), check posterior P5-P95 covers the real's actual forecast.
    COVERAGE=20% (want ~90%) -> DSI conditioning OVERCONFIDENT + biased toward ensemble mean at n=15. Extreme reals
    (actual ~5 or ~33) get posteriors stuck ~18-21 (rank-11 emulator can't represent forecasts far from training center).
  * _fig_dsi_loo (projection 1:1 + conditioning-posterior VIOLINS per left-out real, blue=covered/red=missed,
    | = actual, median line). dsi_loo_coverage now returns (df, posteriors dict) for the violins. Violins make the
    overconfidence/mean-reversion visually obvious (bodies bunched ~18-25 while extreme actuals sit far outside).
    HARD GATE: don't trust DSI posterior until full 201-real prior MC. LOO coverage is the calibration diagnostic.
- 2026-07-06: DSI config updates (user): confirmed ies_drop_conflicts was NOT set (pestpp default off, only warned).
  Now: ies_drop_conflicts=True (drops 9 conflicted obs, 1193->1184), DSI_ENERGY 0.99->1.0 (full rank, latent_dim 11->12),
  ies_autoadaloc=True (adaptive localization). Verified live. Coverage still False at n=15 (ensemble size is the
  binding constraint, not solver opts). Config now correct for the full 201-real run.
- 2026-07-06: confirmed LOO conditioning == truth conditioning settings (both call build_dsi_conditioning; identical
  noptmax/num_reals/multimodal/subset/drop_conflicts/autoadaloc/energy/phi/noise). FOUND+FIXED discrepancy: noise seed
  (LOO was 0, main 20260706) -> dsi_loo_coverage now seed=20260706 (noise, matches main) + separate select_seed=0
  (test-real choice only). Re-ran stage6loo under current settings: LOO coverage 20% -> 40% (drop_conflicts+autoadaloc+
  full-rank widened posteriors; reals 4,6 now covered). Extreme reals (actual ~5, ~34) still missed = rank-12 emulator
  span limit at n=15. Projection R unchanged 0.915. Full 201-real prior MC remains the gate for ~90% coverage.
- 2026-07-06: LOO coverage criterion changed (user): posterior MIN/MAX range, not P5-P95. dsi_loo_coverage
  covered=(po.min()<=actual<=po.max()); df cols pmin/p50/pmax; fig title "min-max coverage". Result: still 40% (SAME
  as P5-P95) -> misses are a SPAN problem not interval-width: extreme actuals (~5,~25,~34) outside the emulator's
  ENTIRE posterior range (rank-12 span limit at n=15). min/max = honest test of whether emulator spans forecast range.
- 2026-07-06: DSI diagnostic figs (user): _dsi_condition_figs makes prior_vs_data + prior_mc_forecast ANALOGUES for a
  DSI conditioning (prior=grey dsi.0 / posterior=blue dsi.N / measured=black target / noise=red), for the conditioned
  truth AND each LOO real failing min/max coverage. stage6_figs driver: truth_r5 + loo_r{failed}. BUGFIX: _dsi_meta
  time was string -> sel.sort_values("time") lexicographic -> sawtooth timeseries. Fixed: meta["time"]=to_numeric.
  Also subsample spaghetti to ~120 reals (500 x 36 panels too slow -> 11s). Verified truth figs smooth + correct.
  stage6figs DONE: 8 figs (truth_r5 + failed LOO reals 3,7,9, each prior_vs_data + forecast).
- 2026-07-06: STARTED 120-real prior MC (user) bg buu021e6n: stage5_prior_mc(num_reals=120, num_workers=15).
  16 cores -> 15 workers, 8 reals/worker, ~80min + build/draw. OVERWRITES _s5_master (15->120) + rebuilds _s3_template.
  AFTER completes: re-inject truth (P75 of 120-real forecast dist) + weights -> rebuild+recondition DSI -> expect
  higher-rank emulator + better LOO coverage. Current 15-real DSI figs/truth superseded once 120-real lands.
- 2026-07-06: BUG FOUND + FIXED in run_prior_mc: it set ies_num_reals = num_workers (15), so pestpp-ies
  TRUNCATED the 120-real prior_pe.jcb to 15 (first buu021e6n run only evaluated 15 reals). Workers != reals.
  Fix: run_prior_mc(num_reals=None) defaults to prior_pe.jcb real count; stage5 passes num_reals=nr.
  RE-LAUNCHED bg b7wtlc986: run_prior_mc(num_workers=15, num_reals=120) directly on existing 120-real
  prior_pe.jcb (skipped rebuild+draw, seed-fixed). ies_num_reals=120 confirmed, 15 workers running. ~80min.

## Prior MC (120-real) -- two failures + fixes (2026-07-06)
- FAIL 1: ies_num_reals=num_workers(15) truncated the 120-real ensemble to 15. Fixed run_prior_mc (above).
- FAIL 2: run died at 110/120 with `RunStorage::update_run() stream not good` = DISK FULL. Root cause: a
  stale 1.6G sout.csv + 16 .ucn OUTPUT files sat in _s3_template, so each of 15 workers copied 2.3G AND
  regenerated ~2G of reactive output -> ~60G peak vs 36GiB free.
  FIX: (a) freed reclaimable dirs (_s3_stage, _truth_props, superseded _s6_* 15-real dirs, failed _s5_master);
  (b) SLIMMED _s3_template 2.3G->288M by deleting regenerated outputs (sout.csv, *.ucn, *.cbb, *.hds, *.lst
  -- all recreated per forward run; inputs .txt/.tpl/.ins/.dat/.jcb/.pst kept); (c) relaunched with 12 workers
  (not 15) for headroom. bg b3vtfmxkb. Verified workers regenerate sout.csv from the slim template.
  NOTE: if build_pest_interface is re-run it regenerates the full template; re-slim before any big worker run.

## REINFLATION test on failed LOO xvals (user 2026-07-06, harness READY, awaiting 120-real ensemble)
Goal: does pestpp-ies reinflation rescue the LOO cross-val reals that fail min/max coverage (posterior
collapses past the held-out truth)? Spec: after 1 iteration reinflate 100%, repeat 3 times.
- build_dsi_conditioning gained kwargs n_iter_reinflate / reinflate_factor / reinflate_num_reals ->
  sets ies_n_iter_reinflate / ies_reinflate_factor / ies_reinflate_num_reals (names confirmed from binary).
- reinflate_test() (--reinflate): baseline LOO coverage (noptmax=1) finds failures, then re-conditions each
  with ies_n_iter_reinflate=1, ies_reinflate_factor=1.0, noptmax=3 (=3 cycles); reports coverage recovered +
  posterior-range widening (guards vs trivial 'cover-all-by-exploding-range'). If the 120-real LOO has no
  failures (more data -> better calibration), demos on the 2 tightest-margin reals. RUN once b3vtfmxkb lands.

## Stage 4 refresh on 120-real (done 2026-07-07) + reinflation-test debugging
- Ran the stage-4 chain on the landed 120-real ensemble: prior_forecast -> pick_truth(P75) -> lock_truth
  -> inject_truth_and_weights. TRUTH now = real 13, peak 32.0 mg/L (P75=32.1). Weights set (1193 conditioning
  obs weighted, forecast 1152 zero-weight). This was a REQUIRED pending step.
- BUG that blocked the reinflation test: stage5_prior_mc's build_pest_interface reset ALL 464887 obs to
  PstFrom default weight 1.0, and inject_truth_and_weights had NOT been re-run -> _dsi_keepobs returned
  464887 cols -> DSI.fit() on 464k cols = catastrophically slow (>>min/real). Fixed by running stage 4
  (keepobs collapses to 2345 = 1193 weighted + 1152 forecast). LESSON: after any stage5 rebuild, RE-RUN
  stage 4 before any DSI work.
- Two tooling gotchas hit while debugging (both cost real time):
  (1) `conda run` CAPTURES all subprocess stdout until exit -> bg logs look empty / python -u useless.
      Use `export PATH=<env>/bin:$PATH; python -u ...` (or `conda run --no-capture-output`) for live output.
  (2) Calling the env python BINARY directly (not activating) leaves PATH unset -> pestpp-ies /e's nested
      `python forward_run.py` resolves to base python (no vendored pyemu) -> run fails. PATH-prepend fixes both.
- Added load-once cache (data_cache/pst_cache) threaded build_dsi_emulator <- build_dsi_conditioning <-
  dsi_loo_coverage <- reinflate_test, so the big obs.jcb + 464k-obs control file load ONCE, not per LOO real.

## REINFLATION test RESULT (2026-07-07, done) -- reinflation does NOT rescue failed LOO xvals
Baseline LOO coverage on the 120-real ensemble = 89% (36 left-out; up from ~40% at 15 reals -- the bigger
ensemble alone fixed most overconfidence). 4 failures: reals 99, 1, 57, 63 -- all EXTREME-value truths
(real 99 = ensemble MAX peak 59.3; reals 1/57/63 near the MIN, actual 1.0/1.5/6.5).

Reinflation (ies_n_iter_reinflate=1, ies_reinflate_factor=1.0, noptmax=3) does NOT rescue them:
- pestpp-ies reinflation "continues iterations": noptmax=3 yields dsi.0..dsi.6, ALTERNATING even=reinflated
  (bare, wide: range 70-180 mg/L, up to 181 -- nonphysical, >3x the 59.3 ensemble max) / odd=CONDITIONED
  (narrow: range ~28-32). The reportable posterior is the CONDITIONED (odd) iteration.
- Every conditioning cycle RE-COLLAPSES to the same narrow band that misses the extreme truth:
  real 99 conditioned [4.1,31.6]->[2.8,35.4]->[2.7,35.0], all miss 59.3; real 63 [13,40.6]->[19.8,52.1]->
  [22.5,51.3], all miss 6.5. So 0/4 conditioned posteriors recover.
- The transient "coverage" at even (reinflated, un-conditioned) iterations is trivially wide, not a posterior.
ROOT CAUSE: these are SPAN/extreme-value limits -- the emulator trained on 96 reals (truth + val held out)
cannot produce a forecast at/beyond the held-out global extreme, and reinflation (perturbs within the latent
span) cannot manufacture a forecast outside the training span. Reinflation cures COLLAPSE, not SPAN.
HARNESS CAVEAT (to fix): dsi_posterior reads range(noptmax+1) -> dsi.3, but reinflation writes past noptmax
(dsi.6); and the meaningful read is the LAST CONDITIONED (odd) iter, not the last file. Conclusion is robust
across all conditioned iters (all miss). reinflate_test's printed reinf coverage happened to read a
conditioned iter (dsi.3) so its 0/4 verdict was correct.

## LOO xval METHODOLOGY FIX (user 2026-07-07) -- re-running corrected baseline
User corrections to the DSI LOO cross-validation: (1) DSI IES conditioning runs = noptmax 1 (already the
default); (2) LOO drops ONLY the single left-out real, NOT a full validation holdout; (3) retrain the DSI
each time (already per-real). Bug fixed in build_dsi_emulator: `n_val = n_val or max(2, N//5)` turned an
explicit n_val=0 into a 23-real holdout, so "LOO" trained on only 96 reals (leave-24-out). Now n_val=0 ->
val empty -> train on all-but-truth = 119 (latent_dim 119). This also improves PRODUCTION conditioning
(build_dsi_conditioning passes n_val=0) -> trains on 119 not 96. Re-running dsi_loo_coverage(frac=0.30);
the prior 89% baseline + the reinflation finding were on the flawed 96-real train and are SUPERSEDED.
Expect coverage >= 89% (more training data -> better emulator); extreme-MAX real (99) may still miss on
span, near-min reals (1/57/63) may now be covered. bg bmble2ttc.

## DIRECT-PROJECTION posterior + pure-LOO figs (user 2026-07-07, regen bj61qyzoy)
User asks: (1) xval fidelity fig = PURE leave-one-out (not the 23-holdout); (2) ADD a direct-projection
posterior (project the obs+noise ensemble through the emulator) AS WELL AS IES; overlay BOTH raw + reg.
Implemented:
- dsi_project_posterior(dsi, target, meta, fc, mode): build obs+noise ensemble around the conditioning
  obs (correlated per site:species shock, same convention as IES noise), project to latent using ONLY
  the conditioning rows of the projection matrix, predict full obs -> posterior peak forecast dist.
  mode='raw' = min-norm pinv (OVER-dispersed: obs-noise floods the weakly-constrained latent dirs ->
  nonphysical tails, ~10-23% >60 mg/L, P95~140); mode='reg' = prior-informed MAP
  (Pc^T R^-1 Pc + I)^-1 Pc^T R^-1 dev, R = empirical transformed-space noise var -> TIGHT + physical
  (0% >60). For the P75 truth (r13, actual 32) reg P5-P95=[31.5,38.6]; reg is overconfident on extremes.
- _fig_dsi_xval_loo: pure-LOO forecast fidelity (R=0.993) + per-real neg-frac. Replaces the holdout xval.
- dsi_loo_coverage_proj: fast projection LOO coverage (raw+reg), same left-out set as the IES coverage.
  Smoke (n=24): raw min/max coverage 96% (over-cover by explosion), reg 54% (overconfident) vs IES.
- _fig_dsi_posterior now overlays prior / IES / proj-reg / proj-raw + truth (hist + CDF, x clipped so the
  raw tail doesn't crush the plot). New _fig_dsi_proj_coverage: IES vs reg vs raw coverage bars + reg
  P5-P95-vs-actual scatter.
- regen_figs wires all of it; runs the slow IES LOO once + the fast projections. Figs: 05_prior_mc +
  06_dsi/{dsi_xval, dsi_loo, dsi_loo_proj_coverage, dsi_forecast, dsi_prior_vs_data_truth_r13, ... loo_r*}.
Smoke test passed (all fig paths render). Full run bj61qyzoy pending (~1hr for the 36-real IES LOO).

## IES LOO speed: autoadaloc was the bottleneck (user dropped it 2026-07-07)
Profiled one conditioning: run_dsi (pestpp-ies /e IES) = the ENTIRE per-real cost; build_dsi_emulator +
prepare_pestpp + get_bins + from_gaussian_draw = ~2s total. run_dsi with ies_autoadaloc=True = 171.6s/real;
with it OFF = 11.0s/real -> 15.6x faster. Adaptive automatic localization over the latent-par x 1193-obs
space dominated. build_dsi_conditioning now takes `autoadaloc=False` (default OFF); pops ies_autoadaloc
unless requested. Full 36-real IES LOO now ~8min (was ~1.75hr). regen relaunched b7sbzf1br.

## FINAL corrected LOO coverage (regen b7sbzf1br DONE 2026-07-07; truth r13, true drop-one, noptmax=1, autoadaloc off)
- IES min/max coverage = 89% (n=36); failed reals 99, 1, 57, 63 (the DISTRIBUTION EXTREMES: r99=ensemble
  max peak 59.3; r1/r57/r63 near the min). SAME 89% as the flawed 96-train run -> more training data does
  NOT lift IES coverage; the misses are the held-out extremes (span limit), and dropping autoadaloc didn't
  hurt coverage (confirms it bought little).
- Direct-projection min/max coverage: RAW = 94% (over-covers by explosion, nonphysical tails), REG = 47%
  (regularized MAP is TIGHT -> more overconfident than IES). Clean 3-way story: raw over-disperses, reg
  over-confident, IES in between.
- Figs regenerated (stale r5/r3/r7/r9 removed): 05_prior_mc/prior_mc_forecast; 06_dsi/{dsi_xval (pure LOO
  R~0.99), dsi_loo, dsi_loo_proj_coverage, dsi_forecast (3-way posterior overlay), dsi_{prior_vs_data,
  forecast}_truth_r13, and _loo_r{99,1,57,63}}.
(NB: user originally ADDED autoadaloc for calibration quality; dropped here for speed -- the DSI has only
~119 latent pars so localization buys little. If conditioning quality regresses, reconsider.)

## SECTION 7 BUILD -- DSIVC sweep interface + run (2026-07-07)
- build_sweep_interface = build_pest_interface(with_treatment=True): f_treat tpl param [0,0.999],
  ftreat+cost obs, apply_treatment_forward PRE cmd. Two build bugs fixed:
  (1) obs index_cols=["name"] clashed with PstFrom's obsnme alias 'name' on RELOAD -> renamed to 'item'.
  (2) FORWARD-RUN EMBEDDING: add_py_function embeds only ONE function's source (self-contained, like
      apply_well_rates). apply_treatment_forward calls apply_treatment -> the whole PhreeqcRM chem chain
      + module constants -> can't embed. FIX: apply_treatment_forward is now a self-contained STUB that
      imports the workflow module (shipped into the template) and delegates to _apply_treatment_forward_impl.
      with_treatment build ships into the template: workflow.py, wf_style.py, herebedragons.py, and the 6
      chem-data files build_injectate_solutions reads (ic_aq_chem.csv, wellin.csv, ic_exchanger.csv,
      ic_surfaces.csv, postfix.phqr, datab.dat). Stub sets MPLBACKEND=Agg + sys.path.insert(cwd) and calls
      _wf._apply_treatment_forward_impl(ws, data_d=ws) so the copied CSVs are used (module DATA_D would be
      wrong in-worker). Verified worker-style: f_treat=0.5 -> cost=100.1, ftreat/cost obs written.
- draw_sweep_ensemble: reuse prior_pe.jcb 120 param reals + f_treat~U[0,0.999] (paired). SLIM the sweep
  template (remove sout.csv/*.ucn) before the worker run (disk).
- SWEEP RUNNING bn7lwqpgx: run_dsivc_sweep(12 workers, 120 reals). F=0, all workers ran treatment
  (ftreat.csv written) + into mf6rtm. ~60-80min. First attempt (bx3qayr8o) failed 120/120 fast on the
  embedding bug -> fixed. NEXT after it lands: merge_training_data (240-real) + build_dsivc.

## SECTION 7 -- DSIVC optimization (DESIGN LOCKED 2026-07-06, not yet built)

Framing: ADR-0003 f_treat lever (SUPERSEDES the old three-well dv-rate design in memory dsivc-part1-08).
DSIVC = outer pestpp-mou over the DSI emulator; decvar is an OBS COLUMN the manager controls, injected
as a high-weight zero-noise target -> nested pestpp-ies /e conditioning -> stack -> percentile stack-stats
-> outer objectives. Requires runstore-prepared dsi_t_d (stage6 already calls prepare_pestpp(use_runstor=True)).

STRUCTURAL GAP: stage5/6 prior MC is all f_treat=0 -> f_treat has zero correlation w/ obs -> injecting it
does nothing. DSIVC needs a NEW training sweep where f_treat VARIES. Separate artifact from stage5/6
(those MUST stay f_treat=0 for clean conditioning on the monitored truth).

LOCKED DECISIONS (user, 2026-07-06):
  1. f_treat TIMING = forecast-only: treatment scales injectate O(0)+N(+5) ONLY for periods starting at/after
     the DECISION DATE = day 252 (DAY_COND_END). Monitored history stays f_treat=0 for all sweep reals ->
     species conditioning is f_treat-independent/clean. (vs continuous-from-t=0, which entangles conditioning.)
  2. SWEEP DESIGN = reuse the 120 stage-5 param reals, pair each with an independent f_treat~U[0,0.999];
     ~120 full-model runs. Orthogonal f_treat draw -> emulator separates param vs f_treat effects. Cheapest.
  3. MOU SO4 OBJECTIVE = P95 of emulated peak recovered-SO4 (risk-averse / chance-constrained "risk band").
  COST = exact (deterministic in f_treat), NOT emulated -- compute_cost.py as a 2nd model command ->
     `cost` obs objective (the ±30-80% DSI-corruption lesson). cost=c_unit*V_inj*(-ln(1-f_treat)).
  4. MERGE (user, 2026-07-06): DSIVC emulator trains on 240 reals = the SAME 120 param draws at TWO
     f_treat conditions -- f_treat=0 (from the prior MC) + f_treat~U[0,0.999] (from the sweep). A PAIRED
     design that isolates the f_treat response. The sweep runs AFTER (regardless of) the first prior MC;
     the prior MC's runs are reused as the f_treat=0 half, not wasted.
     Consequences: (i) under forecast-only treatment each paired real has IDENTICAL monitored-window obs
     (params same, treatment only post-252) -> the merge adds the f_treat->forecast response, NOT new
     monitored covariance; conditioning power stays set by the 120 param draws. (ii) f_treat must be BOTH
     a per-real PARAMETER (treatment.dat via tpl) AND an echoed OBS column (DSIVC controls the obs). Prior-MC
     reals get f_treat-obs = 0 at merge time.

BUILD STEPS (sweep runs after the first prior MC; stage6 conditioning/xval stays on the f_treat=0 half):
  a. DONE 2026-07-06: build_injectate_solutions gates O(0)+N(+5) scaling on period-start-day >= decision_day
     (=DAY_COND_END=252); added decision_day arg. Verified f_treat=0 -> scale=1 everywhere (regression intact),
     f_treat=0.5 -> pre-252 untreated / post-252 x0.5, first treated period starts day 252. Compiles.
     SAFE vs running prior MC (frozen forward_run.py + f_treat=0 no-op).
  b. DRAFTED: build_pest_interface(with_treatment=True) adds treatment.dat tpl -> f_treat param (pargp=decvar,
     bnds [0,0.999], partrans none) + apply_treatment_forward() PRE cmd (reads treatment.dat -> treated welin
     aux + cost.dat, echoes ftreat.csv/cost_obs.csv) + ftreat & cost obs. Not yet RUN (rebuilds a template).
  c. DRAFTED: build_sweep_interface / draw_sweep_ensemble (reuse prior_pe.jcb 120 reals + f_treat~U[0,0.999],
     paired) / run_dsivc_sweep (noptmax=-1, 15 workers -> _s7_sweep_master). CLI: --stage7sweep.
  d. DRAFTED: merge_training_data -> 240-real DataFrame (p0..p119 f_treat=0 + s0..s119 sweep), keep-obs +
     f_treat + derived fore_peak_so4; prints LEVERAGE corr(f_treat,peak-SO4) gate. CLI: --stage7merge.
  e-h. SCAFFOLD in build_dsivc (docstring TODOs): fit DSI on merged -> condition on baseline truth (inject
     f_treat=0) -> DSIVC(...).prepare_pestpp(decvar_names=["f_treat"]) -> objectives {min cost via
     compute_cost.py 2nd cmd, min fore_peak_so4_stat:95%} -> pestpp-mou -> FOM-validate optimum vs P5-P95.
     compute_cost.py written (_workflow/compute_cost.py): exact cost from injected decvar + v_inj.dat.
  PRE-MOU GATE: merge_training_data prints corr(f_treat, peak-SO4). Weak (<0.2) -> flat front (RISK below).
  All drafts compile; SAFE vs running prior MC (source-only edits, frozen forward_run.py, f_treat=0 no-op).
  OPEN: (e) stack-stats obs name `fore_peak_so4_stat:95%` + decvar-file format in compute_cost confirm on
  first prepare_pestpp output; v_inj.dat must be written at DSIVC prepare time.

RISK to watch (MOU_HANDOFF "verify plume breakthrough first"): forecast-only treatment only affects water
injected after day 252; if the forecast peak is dominated by pre-252 (untreated) water, f_treat leverage is
weak -> flat front. Sweep validates leverage (corr(f_treat, peak-SO4)) before committing to the MOU.
- [ ] Slice 7 — dataworth
- [ ] Slice 8 — DSIVC(f_treat) cost-vs-SO4 + iterative FOM outer loop

## Landmines

- No scaling transported `O`/`N` arrays (geochem wrong; act on solution inputs O(0)+N(+5), re-equilibrate)
- No full-rebuild in forward_run (clobbers PstFrom K/porosity/pyrite mults; touch wellin aux + cost.dat only)
- No CONTEXT.md/REDESIGN.md number rewrite yet (pre-rebake)
- Emulation-first non-negotiable (ADR-0002): opt runs on DSI emulator, not full model
- Nothing generated tracked; never write to data/; mothership (workspace inside stage dir)
- pyemu stays rhugman/pyemu@feat_dsivc (e986b27); assert "dependencies" in pyemu.__file__

## Log

- 2026-07-06: created scaffolding dir + this tracker; starting slice 1.
- 2026-07-06: slice 1 done — `wf_style.py` (style + qa_gate + savefig), gitignore, smoke-tested. Next: slice 2 model build.
- 2026-07-06: slice 2 drafted via workflow (8 agents, Map→Build→Verify). Verify caught real landmine-ii violation
  (mup3d write_simulation clobbering pyrite m0); fixed w/ chem_wd scratch isolation + _m0_guard. Parse OK.
- 2026-07-06: slice 2 live run PASSED (728 d, 194 cells). Added `plot_spatial.py` (SO4 plan + xsection, head
  contours, wells+screens) — physics reads correctly; in-aquifer SO4 peaks ~80 mg/L (lever headroom > breakthrough).
- 2026-07-06: user asked to extend run +2 yr and rebuild mesh (drop wellopt refinement + more refinement).
  tdis 35→60 SP, DAY_END 728→1458 (~4 yr total, +2 yr recovery); FORECAST_SP/DAY_END now computed.
  Mesh: exposed vorflow knobs (BACKGROUND_LC/LINE_RES/LINE_DISTMAX/PT_RES/PT_DISTMAX); wellopt filtered from
  refinement; picked bg50/line8 → ncpl≈394 (~2× the 194). NOTE: first naive bg150→100 coupling made it COARSER
  (131) — decoupled dist_max, verified by sweep. plot_spatial times now adaptive.
- 2026-07-06: first extended run FAILED — PhreeqcRM "SOLUTION 227 not found". Root cause: data/wellin.csv defines
  only 45 periods × 5 screens (injectate sols 2..226); NPER=60 needed 227+. Also wel side silently capped at 45 →
  extension would've had ZERO injection. FIX: shared `_wellin_df(data_d, nper_model)` forward-fills last field
  period (rate 156.7 + its treated chem) across the extension → dense gap-free SOLUTION set, injection continues
  (continuous-constant per ADR-0003). Wired into build_injectate_solutions + _make_wel_in_continuous; unit-checked
  (60 dense periods, 300 rows). Relaunched rebuild+run bg b7vhtpy1c (~24min).
- 2026-07-06: added `plot_rates.py` (wellin/wellout rate timeseries + net doublet, window shading). It CAUGHT a
  design bug my forward-fill introduced: filling wellin.csv's LAST period (44, rate 480) pulled in a field-data
  rate ramp the original 728-d model never used → injection jumped 360→480 at day 728, net doublet +120 (violates
  ADR-0003 continuous-constant + balanced). FIX: `HOLD_KPER = len(_BASE_PERIODDATA)-1 = 34`; `_wellin_df` now holds
  period 34 (rate 360, balanced) across the extension, ignoring the unused CSV ramp (35..44). Verified: wellin flat
  360, net=0 everywhere. Killed b7vhtpy1c, relaunched bg bgrguzgzx (~24min).
- 2026-07-06: added `plot_setup.py` — static pre-run diagnostics (no model output needed):
  setup_plan_layout (domain/mesh/CHD/wells/monitoring clusters), setup_xsection_layout (12-layer grid + screens +
  monitoring depths + CHD), setup_properties (base pyrite m0 xsection + scalar summary). Confirmed base build is
  HOMOGENEOUS (K=1, phi=0.15, pyrite m0=0.1797 all uniform) — heterogeneity enters at PstFrom/slice3; property
  plotter built as a ready template. Mesh refinement confirmed on-axis (coarse at deleted wellopt). Minor: a few
  monitoring labels collide near x=0 in plan (cosmetic).
- User proposed increasing pump rate to bring SO4 peak earlier (vs +2yr extension). Flagged physics: advection
  rescales w/ Q but pyrite kinetics DON'T; faster flow = earlier BUT WEAKER peak. Waited for the run instead.
- 2026-07-06: EXTENDED RUN bgrguzgzx COMPLETE (394 cells, 1458 d, balanced doublet) — ALL 20 gates green.
  KEY RESULTS (from sout.csv SO4 = true sulfate; NOT S.ucn which is total S incl. sulfide — that confusion cost a
  detour): recovered-SO4 objective (flow-wt mean) now has an INTERIOR PEAK ~4.8 mg/L @ ~870 d, gentle decline after
  → the +2yr extension SOLVED the peak-in-window problem. Rate-up NOT needed for peak timing.
  BIG DESIGN FINDING: the objective is diluted by the recovery-rate split. RATES_OUT=[300,30,30] puts 83% of
  pumping on top screen L1, which carries ~0-2 mg/L SO4; the real sulfate (~29 mg/L peak @ ~1050-1230 d) is at the
  DEEP low-weight screens L3/L5. So recovered-SO4 objective (~5) massively understates mobilized sulfate (~29).
  f_treat has strong leverage on the deep 29 mg/L plume; whether the OBJECTIVE sees it depends on the [300,30,30]
  split. NEXT DECISION for user: is [300,30,30] intended, or should recovery be more balanced across screens
  (would raise recovered-SO4 signal + lever visibility)? All figures regenerated vs final run.
- 2026-07-06: added injectate_chem_timeseries (Tmp degC + O(0)/N(+5) mol/L, seasonal then held@period34) per user
  request. Tmp stored x1e-3 -> degC (swings 2-23C). Legend/annotation collision fixed.
- 2026-07-06: FOUR user fixes applied (rebuild+run bk6bfwqhr, ~10min):
  (1) RUNTIME: coarsened grid bg50/line8 (394) -> bg95/line14 (~166 cells, ~10min). Layout+BC figure now emitted
      BY the workflow (_fig_layout in stage2_build) — CHD coloured by gradient head.
  (2) SEASONAL: _wellin_df no longer freezes at period34 — wraps the field record cyclically (mid-time mod
      field_span) so injectate stays seasonal in the extension (O(0)/Tmp now 22 unique vals, Tmp 1.9-22.8C).
      Rate stays balanced (all field periods 0..34 total 360; wrap preserves it).
  (3) REGIONAL GRADIENT: _apply_chd_gradient sets CHD head = grad*(x_center - x) => 0.001 m/m left->right
      (was flat h=0). head +/-~0.125 m across the ~250m domain. Only head changed, chem aux intact, per-pkg write.
  (4) EQUAL RATES: wellin RATE_IN_PER=72 x5=360; wellout RATES_OUT=[-120,-120,-120], SCREEN_W_OUT=[120,120,120].
      Directly fixes the objective-dilution finding (was [300,30,30] hiding deep SO4). Recovery now samples the
      sulfate-rich deep screens equally -> expect much higher recovered-SO4 objective.
- 2026-07-06: 2 follow-up fixes: (a) PRE-RUN figure ordering — _fig_layout now runs BEFORE qa_gate_model (was
  after the ~10min run, defeating the point). (b) CHD GRADIENT BUG: hbd.make_chd seeds head with int literal 0 ->
  'head' column is INT dtype -> fractional gradient heads (+/-0.126) truncated to 0 (verified 0.0 on disk).
  Fixed _apply_chd_gradient to rebuild the recarray with a float 'head' column; verified +/-0.126 m persists.
  Killed buggy run bk6bfwqhr, relaunched b0id33gz9 (~10min).
- 2026-07-06: RUN b0id33gz9 COMPLETE (166 cells, ~10min) — all 20 gates green. VERIFIED:
  * CHD gradient +/-0.126 m on disk (red left / blue right in layout fig); sim heads -22.6..+19.4 m.
  * EQUAL RATES lifted the objective: recovered-SO4 flow-wt mean now peaks ~15 mg/L @ 870 d (was ~5 w/ [300,30,30]),
    a CLEAN interior peak declining to ~0 by ~1300 d — well-defined "peak SO4" forecast, comfortably in-window.
    Screens flipped: L1 now carries sulfate (~29 mg/L @ 877d) under equal pumping + regional flow.
  * Seasonal injectate confirmed in extension; pre-run layout fig emitted before the run.
  GEOMETRY NOTE for user: left->right gradient makes wellout (x=-99, west) UPGRADIENT of wellin (x=+1) — regional
  flow opposes doublet recovery. Literal impl of "slope left to right"; flip sign if recovery should be downgradient.
  Fixed filename collision: plot_setup plan -> "setup_plan_layout_annotated"; workflow _fig_layout owns the canonical
  gradient-coloured "setup_plan_layout". Section-2 build now stable with all 4 fixes.
- 2026-07-06: further user fixes (rebuild+run brmvp7row):
  (a) FLIPPED gradient: head=grad*(x - x_center) => left NEG / right POS, flow right->left, wellout DOWNgradient
      of wellin (regional flow now AIDS recovery). Verified left -0.126 / right +0.126.
  (b) TRANSMISSIVITY-WEIGHTED rates: new _transmissivity_weights(gwf,cell,layers) T=K*thickness; wellin/wellout
      totals still 360 (balanced) but split by T. Base (uniform K) => thickness-weighted: wellin [87,104,78,39,52],
      wellout [-225,-90,-45] (L1=62.5%). Removed equal RATE_IN_PER/RATES_OUT/SCREEN_W_OUT constants.
      _fig_breakthrough now flow-weights by ACTUAL welout package q. plot_rates reads rates from the model packages.
      **KEY**: at PstFrom (slice3) this MUST become a RUNTIME preprocessor (recompute rates from perturbed K each
      forward run, like apply_treatment) — noted in constants + _transmissivity_weights docstring.
  (c) create_reactive_tsteps output_interval 7 -> 5 (herebedragons default; finer reactive output; covers all 60 SP).
  All parse-checked + T-weights/gradient validated pre-run.
- 2026-07-06: RUN brmvp7row COMPLETE — all 20 gates green. Best breakthrough yet: T-weighted objective peaks
  22.0 mg/L @ 867 d (was 5 orig -> 15 equal -> 22 now), clean interior peak. All 3 screens contribute (L1~23@830,
  L3~19@900, L5~28@1067; L5 low weight so doesn't dominate). Flipped gradient (wellout downgradient) + T-weighting
  both lifted it. Section-2 build STABLE. All figs refreshed. Rate plot confirms T-split (wellin 40-104, wellout
  -225/-90/-45), net=0. NEXT: forecast-def confirmation + slice 3 (PstFrom) — where the T-weight rate preprocessor
  + f_treat preprocessor both get wired as add_py_function pre-commands.
