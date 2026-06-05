# rtm-tutorial

Working glossary for the reactive-transport decision-support tutorial curriculum. Standalone sibling of the GMDSI_notebooks (freyberg) curriculum; terms here are the RTM-curriculum language. PEST++/ensemble terminology (realisation, ensemble, prior/posterior) follows the GMDSI_notebooks usage and is not redefined here.

## Language

**DIZON**:
"Deep well Injection in Zuid-Oost Nederland" — the field ASR experiment at Someren (NL) this curriculum is built on: pre-treated oxic surface water injected ~300 m deep into an anoxic pyritic aquifer, monitored over 854 days (Prommer & Stuyfzand 2005; Stuyfzand et al. 2002). The tutorial model is a modified, decision-support recast of that study — faithful in chemistry, not a reproduction.

**Synthetic truth**:
The canonical "reality" of the curriculum: a single realisation held out from the prior ensemble, *deliberately chosen* from the upper quartile of the prior forecast distribution (peak SO₄ ≈ 87–95 mg/L) — so the prior median under-predicts it and conditioning visibly corrects toward truth (not randomly drawn). All history matching, posterior scoring ("did we cover the truth?") and emulator-fidelity checks in the core sequence are against the synthetic truth.
_Avoid_: calling the measured field data "truth" — it is **measured data**, never the truth.

**Measured data**:
The real DIZON field observations (multi-site, multi-depth, multi-species breakthrough). Appears in exactly two places: as motivation/credibility evidence when the base model is built, and in the optional real-data capstone. It does not drive the core history-matching sequence.

**Forecast (canonical)**:
The quantity the whole curriculum is judged on: the SO₄ concentration time series at the supply well during the supply period, summarised as **peak SO₄** — the maximum over *all* supply-well screens and supply-period times, reported in mg/L (model output is mol/L; × 96.06 g/mol × 10³). Carried as a named forecast group from the PEST-setup notebook onward; every method's payoff figure is what it does to this forecast distribution.

**Decision question (canonical)**:
"How much sulfate will the supplied water carry, and how sure are we?" — a *minimization/design* framing: treatment cost scales with concentration, and the operator designs treatment capacity to the **P95 of peak SO₄**. The payoff tracked through every notebook is the forecast distribution (median and P95), prior vs posterior. The lesson of the case: conditioning shifts the forecast *up* (prior median ≈ 82 → posterior ≈ 92 mg/L) while tightening it — the value of data is truer news, not better news.
_Avoid_: framing the decision as threshold exceedance. Thresholds appear only as a one-cell illustrative risk lens (the 90 mg/L worked example: prior P ≈ 0.15 → posterior P ≈ 0.67); the EU 250 mg/L standard is comfortably met and gets exactly one sentence (cost, not compliance, drives this decision).

**Layered prior**:
The canonical parameterisation, narrated in three tiers: *flow* (hydraulic conductivity), *transport* (porosity, dispersivity), and *reaction* (pyrite abundance and pyrite oxidation rate — the case's signature uncertainty). Organic matter is deliberately excluded (the source study found it a minor redox contributor); heat-exchange parameters are deliberately fixed. The tiers are the teaching device: RTM uncertainty is not just K-fields.

**History period**:
Day 0–252 of the simulation (injection + `wellout` operation): the window in which monitoring data exists and may be used for conditioning. Nothing after day 252 informs any history match in the core sequence.

**Decision date**:
Day 252 — when the treatment/operation decision must be committed, ~8 weeks *before* the supply well switches on. The gap is canonical: decisions are made with lead time, not the morning the pump starts.

**Supply period**:
Day 308–728: the supply well (`wellopt`) extracts drinking water. The canonical forecast (peak SO₄) lives entirely in this window. The tutorial timeline (728 days) is an abstraction of the 854-day field experiment, acknowledged but not reconciled.

**Conditioning species**:
The species whose history-period data informs the history match: SO₄ (forecast species), O₂ and NO₃ (oxidant-consumption signal), pH (buffering), Tmp (heat "tracer" constraining velocities). Noise is species-specific (proportional with a floor for concentrations; absolute for pH and Tmp), narrated as a judgment call. pe is never conditioned on.

**Held-back species**:
The major cations (Ca, Mg, Na, K, Fe) — deliberately excluded from conditioning in the core sequence and reserved as the dataworth question: "would measuring these have helped the forecast?"

**Fidelity check (validate-first)**:
The mandatory beat before any emulator is used for conditioning or decisions: predictions for held-out realisations the emulator never saw, compared against the full model's actual outputs. Series principle: never trust an emulator you haven't tested. Nearly free, since the prior-MC runs already exist.

**Decision problem (canonical, optimization)**:
Two decision variables — supply-well extraction rate (multiplier on the base rate) and switch-on day within the supply period. Bi-objective: maximize volume supplied vs minimize P95(peak SO₄) — minimize-native, no threshold required; a contractual trigger (the 90 mg/L worked example) may be drawn across the front as the illustrative chance-constrained ("reliable") lens. Requires its own training sweep (decision-variable values re-sampled across bounds against posterior parameter fields — coverage of decision space is designed, not inherited; see GMDSI_notebooks CONTEXT.md).

**Pyrite column**:
The miniature teaching model of part0: a 1D column of pyritic sand receiving oxic injectate — DIZON's chemistry in miniature, running in seconds. Used to teach the mf6rtm API and build redox-front intuition before the student meets the same reaction network at full scale; doubles as the fast regression-test model.

**Injection well / flush well / supply well**:
The prose names for the three wells (code IDs `wellin`, `wellout`, `wellopt`): injection runs the whole simulation; the flush well extracts during the history period; the supply well extracts drinking water during the supply period and is where the forecast lives. Prose always uses the descriptive names; code IDs appear once per notebook in parentheses.
_Avoid_: "optimization well" — `wellopt`'s "opt" is not explained by the model; the well is the *supply* well.

**Monitoring site obsids (`wp1-f3` scheme)**:
Observation IDs encode monitoring site (`wp1`…) and screened interval / filter depth (`-f3`); supply-well obs use layer suffixes. Canonical spellings: the well *package* id is `wellopt`; the supply-well *obsid* prefix is `welopt` (e.g. `welopt-ly3`) — this matches the field data and every existing run artifact, and is not to be "fixed" (renaming would force regenerating the expensive ensembles). The scheme is explained once, in the obs notebook, and used consistently everywhere.

**Real-data capstone**:
The optional closing exercise that re-runs the workflow against measured data instead of the synthetic truth, confronting model-structural error honestly. Opt-in, at the end of the sequence — structural error is a lesson of its own, not noise in the middle of the others.
