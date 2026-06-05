---
name: gmdsi-voice
description: Apply the GMDSI tutorial writing voice when writing or editing notebook markdown cells, README text, or any tutorial prose in this repo. Use whenever creating new notebooks or revising explanatory text.
user-invocable: true
disable-model-invocation: false
---

# GMDSI Tutorial Writing Voice

When writing or editing markdown/prose in this repository (notebook markdown cells, READMEs, docs), match the established tutorial voice. It is instructor-led, conversational, technically rigorous, and pedagogically scaffolded.

## Core voice

- **First-person plural, instructor-led.** "We will start by...", "Let's load the control file...", "Here we are going to...". The author and reader walk through the material together.
- **Direct address for actions.** Imperative instructions to the reader: "Open the new folder named `freyberg_k`...", "Run the cells, then inspect the new folder...". Remind novices how to execute when relevant: "(Just press `shift+enter` to run the cells)".
- **Conversational, lightly playful — but never at the cost of rigor.** Asides like "_Um, what?_ Let's use pictures to make this easier to understand.", "But there is more!", "choose your own adventure!", parenthetical callbacks ("history matching for 'nothing'..."). Use sparingly; most cells are plain and direct.
- **Honest about practice.** Acknowledge judgment calls and real-world limits: "Normally, this number can only be guessed.", "these mostly have pretty decent default values; however, depending on your setup you may wish to change them."

## Structure conventions

- `#` title at the top of the notebook; an optional epigraph or playful subtitle is acceptable (e.g. _"History Matching For 'Nothing', Uncertainty Analysis for Free"_).
- An **"Admin"** subsection (`### Admin`) near the top: what is pre-prepared, what convenience functions do, which folder gets created, and links to prerequisite notebooks.
- A **"The Current Tutorial"** or recap section situating the notebook in the sequence, with relative markdown links to other notebooks: `["observation and weights"](../part2_02_obs_and_weights/freyberg_obs_and_weights.ipynb)`.
- Short markdown cells that **end with a colon leading into the next code cell**: "Load the PEST control file as a `Pst` object:", "Prepare the template directory:".
- Reader exercises as numbered question lists or bold prompts: "**Do it yourself for the other TPL files:**".
- Define terminology in quotes on first use: 'Each parameter field is referred to as a "realisation". The suite of realisations is referred to as an "ensemble".'

## Technical register

- Inline code formatting for tools, files, classes: `pyemu`, `PstFrom`, `freyberg.pst`, `PESTPP-IES`. Software names in caps as conventionally written (PEST++, PESTPP-IES, TEMPCHEK, MODFLOW 6).
- LaTeX math for theory ($P\left(\boldsymbol{\theta}|\textbf{d}\right)$ style), with plain-language unpacking immediately after.
- Link to authoritative references rather than restating them: the [PEST++ user manual](https://github.com/usgs/pestpp/blob/master/documentation/pestpp_users_manual.md), GMDSI publications, literature.
- Spelling is mixed UK/US across the repo ("realisation" in PEST++ terminology, "realizations" elsewhere). **Match the surrounding text of the notebook you are editing**; do not bulk-normalize spelling.

## What to avoid

- Corporate/AI-flavored filler ("In this comprehensive guide...", "It's important to note that...", "Let's dive in").
- Over-hedging or stacked qualifiers; one honest hedge is the house style, two is mush.
- Long monolithic markdown cells — break prose at the point where the reader should run code.
- Emoji, exclamation-mark inflation, or humor in every cell. The playfulness is seasoning, not the dish.
- Comments or docstrings referencing AI/Claude in any generated content.

## Reference notebooks (canonical voice)

- `tutorials/part0_01_intro_to_bayes/intro_to_bayes.ipynb` — theory cell style, math + plain-language unpacking
- `tutorials/part1_03_calibrate_k/freyberg_k.ipynb` — Admin sections, reader exercises, imperative instructions
- `tutorials/part2_06_ies/freyberg_ies_1_basics.ipynb` — long-form conceptual prose, terminology definitions, cross-notebook links
