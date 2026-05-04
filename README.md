# Distributed Situation Models: Probabilistic Representations of Truth Conditions

This repository accompanies the paper "Distributed Situation Models: Probabilistic Representations of Truth Conditions." (Brasoveanu, CogSci 2026)

Paper PDF: [Distributed_Situation_Models_Probabilistic_Truth_Conditions_CogSci26.pdf](./Distributed_Situation_Models_Probabilistic_Truth_Conditions_CogSci26.pdf)

> In formal semantics, the interpretation function maps linguistic expressions “the pencil is in the cup” to set-theoretic objects `in(pencil, cup)` that mirror the compositional structure of the sentence, but do not encode soft inferences like “the pencil is vertical” that human comprehenders automatically make. Distributed situation models (Frank et al., 2009) were proposed as Gestalt / distributed representations of sentence meanings that enable direct probabilistic inference from their representational geometry. We show that previous implementations of distributed situation models systematically underestimate probabilistic inferences because of their sequential sampling of situations (possible worlds). We propose generating the truth-conditional training data using Markov Chain Monte Carlo sampling over a precomputed pool of valid situations, so that soft constraints are correctly evaluated on complete situations rather than partial ones. Training a competitive neural model on 250,000 MCMC samples yields distributed situation models that accurately encode these inferences, providing reliable, rich meaning representations for cognitive models of language comprehension.

The repository contains the 250,000 Pool MCMC and 250,000 sequential situation samples and the Python reimplementation of the Frank et al. (2009) competitive neural model for learning distributed situation models, as well as scripts for reproducing the main results in the paper.

It supports:

- comparing Pool MCMC samples with sequentially generated samples
- training the competitive neural model on Pool MCMC samples
- checking probability estimates computed from the learned vectors
- regenerating the paper's tables and figure from the shared files

The repository focuses on the main results reported in the paper.
It does not include the Z3-based microworld generation code, the Pool
MCMC implementation, or the sequential sampling implementation.
Instead, we directly provide the resulting sample files.

## Directory Structure

```text
metadata.json
data/
  pool_mcmc_samples.csv.gz
  sequential_samples.csv.gz
trained_models/
  competitive_150_units_seed19.npz
outputs/
  paper_figures/
    samples_to_vectors_focused.png
  paper_tables/
    hard_inference_table.tex
    sampling_comparison_table.tex
    soft_inference_table.tex
    vector_correlations_table.tex
src/
  competitive.py
  probabilities.py
scripts/
  compare_sampling_methods.py
  generate_paper_figures.py
  generate_paper_tables.py
  train_competitive_model.py
```

## Data and Model Files

The shared sample files contain binary situation vectors:

- `data/pool_mcmc_samples.csv.gz`: 250,000 Pool MCMC samples, 44 proposition
  columns.
- `data/sequential_samples.csv.gz`: 250,000 sequential-sampling comparison
  samples, with the same 44 proposition columns in the same order.

The trained competitive neural model is:

- `trained_models/competitive_150_units_seed19.npz`: 150 competitive units over
  the same 44 proposition dimensions. The file is included so the reported
  situation-model-based probability values can be checked immediately; can be
  regenerated with `scripts/train_competitive_model.py`.

The study metadata records the microworld attributes used by the paper / scripts:
17 entities, 44 atomic propositions, 167 hard constraints, and 28,676 valid
hard-constraint situations.

## Install

Use Python 3.10 or newer.

```bash
cd <repository-root>
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib
```
## Reproduce the Sampling Comparison

Run this command from the repository root:

```bash
python scripts/compare_sampling_methods.py
```

This command validates the shared sample files and reproduces the Pool MCMC
vs sequential-sampling comparison.

Expected headline output:

- sampling comparison: 1,892 pairwise conditionals, `r = 0.7546`, mean
  `|delta| = 0.1262`, max `|delta| = 0.5870`, mean delta `+0.0221`

## Train the Competitive Model

The trained seed-19 competitive weights are included in
`trained_models/competitive_150_units_seed19.npz`. To regenerate them from the
Pool MCMC samples, run:

```bash
python scripts/train_competitive_model.py
```

The script trains the competitive model on
`data/pool_mcmc_samples.csv.gz` and rewrites
`trained_models/competitive_150_units_seed19.npz`. The default training recipe
uses NumPy's `default_rng` with random seed 19.

## Regenerate Tables and Figure

Run these commands from the repository root:

```bash
python scripts/generate_paper_tables.py
python scripts/generate_paper_figures.py
```

The table script writes the four TeX table snippets in `outputs/paper_tables/`.
The figure script generates the paper figure in
`outputs/paper_figures/samples_to_vectors_focused.png` based the shared Pool
MCMC samples and trained competitive weights.
