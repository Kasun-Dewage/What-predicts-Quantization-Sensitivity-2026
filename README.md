<div align="center">

# What Predicts Quantization Sensitivity?

### Component Type, Not Reconstruction Error, Predicts Attention Quantization Sensitivity

**Kasun Dewage · Marianna Pensky · Suranadi De Silva**
University of Central Florida, Orlando, FL, USA

[![Paper](https://img.shields.io/badge/paper-PDF-b31b1b.svg)](paper/component-type-not-reconstruction-error.pdf)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776ab.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Measurements](https://img.shields.io/badge/measurements-3%2C808-blue.svg)](#key-results)

</div>

---

## Overview

Post-training quantization (PTQ) methods routinely minimize a layer-wise reconstruction
objective as a proxy for downstream model quality:

$$\min_{W_q} \; \lVert WX - W_qX \rVert_F^2$$

The implicit assumption is that a lower reconstruction error implies a smaller functional
impact. **This repository contains the code and results for a study that tests that
assumption directly at the level of individual attention projections.**

We sweep **nine open-weight language models** (1.3B–8B parameters; OPT, GPT-J,
LLaMA-1/2/3, Mistral, Qwen 2.5), quantize **one attention projection at a time** (Q, K, V,
or O at every layer) while every other weight stays at full precision, and record three
signals for each trial:

| Signal | Definition |
| :-- | :-- |
| Reconstruction error | $\lVert W - W_q\rVert_F / \lVert W \rVert_F$ |
| Activation-weighted error | $\hat{S} = \frac{1}{m}\sum_{i,j}(W_{ij}-W_{q,ij})^2\,\mathbb{E}_b[x_j^2]$ |
| Functional impact | $\Delta\mathrm{PPL} = \mathrm{PPL}_q - \mathrm{PPL}_{\text{base}}$ on WikiText-2 |

The full sweep yields **3,808 distinct measurements** — 2,144 under RTN across nine
models, and 1,664 under GPTQ across seven.

---

## Key results

**1. Reconstruction error is a weak within-component predictor of $\Delta$PPL.**
Of 36 model × component cells at 3-bit RTN, 27 (75%) have $R^2 < 0.10$; the overall
median is $R^2 = 0.044$.

| Model | Q | K | V | O |
| :-- | --: | --: | --: | --: |
| GPT-J-6B | 0.003 | 0.007 | 0.002 | 0.397 |
| LLaMA-1-7B | 0.027 | 0.092 | **0.560** | 0.013 |
| LLaMA-2-7B | 0.051 | 0.011 | 0.363 | 0.096 |
| LLaMA-3-8B | 0.114 | 0.006 | 0.064 | 0.027 |
| Mistral-7B | 0.000 | 0.038 | 0.056 | 0.000 |
| OPT-1.3B | 0.114 | 0.428 | 0.047 | 0.039 |
| OPT-6.7B | 0.040 | 0.009 | 0.128 | 0.137 |
| Qwen2.5-1.5B | 0.052 | 0.041 | 0.036 | 0.085 |
| Qwen2.5-7B | 0.001 | 0.155 | 0.024 | 0.047 |
| **Median** | **0.040** | **0.038** | **0.056** | **0.047** |

**2. Component type and layer identity both carry more information.**
Both categorical signals beat reconstruction error in all nine models. Layer identity is
the strongest individual predictor in 7 of 9 models; component type is strongest in the
remaining 2 (GPT-J-6B and Qwen2.5-7B).

**3. V projections dominate.** V is the most sensitive component in seven of nine models,
accounting for 38–51% of total positive $\Delta$PPL. Q is dominant in none.

| Model | Q% | K% | V% | O% | Dominant |
| :-- | --: | --: | --: | --: | :-: |
| GPT-J-6B | 14 | 19 | **48** | 19 | V |
| LLaMA-1-7B | 12 | 12 | **51** | 25 | V |
| LLaMA-2-7B | 13 | 12 | **50** | 25 | V |
| LLaMA-3-8B | 14 | 8 | **42** | 36 | V |
| Mistral-7B | 13 | 16 | **41** | 29 | V |
| OPT-1.3B | 12 | **39** | 34 | 14 | K |
| OPT-6.7B | 28 | 16 | **44** | 12 | V |
| Qwen2.5-1.5B | 12 | 15 | 32 | **42** | O |
| Qwen2.5-7B | 12 | 13 | **38** | 37 | V |

**4. The dominance picture largely survives a switch to GPTQ** (5 of 7 models match; both
mismatches are shifts within {K, V, O}, never to Q).

**5. Activation-weighted error is a substantially better V predictor.**
Median $R^2$ across V cells is **0.20** for $\log\hat{S}$ versus **0.06** for
reconstruction error, with four V cells exceeding $R^2 = 0.40$.

> **Takeaway.** Relative weight reconstruction error alone is insufficient for
> sensitivity-aware bit allocation. V projections merit dedicated consideration in
> mixed-precision schemes, and activation-weighted scoring should be preferred over
> reconstruction-error-based scoring within components.

---

## Repository layout

```
.
├── src/
│   ├── quant_sensitivity_experiment.py   # main sweep: one projection at a time → ΔPPL
│   ├── quant_methods.py                  # RTN, GPTQ, AWQ, QuIP quantizers
│   ├── analyze_quant_results.py          # aggregation, statistics, paper figures + LaTeX tables
│   └── downstream_eval.py                # optional downstream-task harness (see note below)
├── scripts/
│   ├── run_rtn_sweep.sh                  # full 9-model RTN sweep at 4 and 3 bits
│   ├── run_gptq_sweep.sh                 # 7-model GPTQ sweep at 4 and 3 bits
│   └── make_paper_figures.sh             # regenerate every figure and table in the paper
├── results/                              # all experimental outputs live here
│                       
├── paper/
│   └── component-type-not-reconstruction-error.pdf
├── docs/
│   └── EXPERIMENTS.md                    # detailed reproduction guide
├── requirements.txt
├── Makefile
├── CITATION.cff
└── LICENSE
```

> **Note on `downstream_eval.py`.** `quant_sensitivity_experiment.py` imports
> `run_downstream` and `summarize_downstream` from `src/downstream_eval.py` for the
> optional `--downstream_tasks` path. Drop your copy of that module into `src/` before
> running. The perplexity-based results in the paper do not depend on it.

---

## Installation

```bash
git clone https://github.com/Kasun-Dewage/What-predicts-Quantization-Sensitivity-2026.git
cd What-predicts-Quantization-Sensitivity-2026

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

Install a CUDA build of PyTorch matching your driver — see
[pytorch.org/get-started](https://pytorch.org/get-started/locally/). Gated checkpoints
(LLaMA-2, LLaMA-3, Mistral) require a Hugging Face token:

```bash
huggingface-cli login
```

**Hardware.** Experiments in the paper ran on a single NVIDIA H100 PCIe (80 GB), CUDA 12.1.
A 24 GB GPU is sufficient for the 1.3B–1.5B models; 7B–8B sweeps benefit from the default
CPU-offload path.

---

## Quick start

List the available model keys:

```bash
python src/quant_sensitivity_experiment.py --list
```

Run a single small model end to end (a good smoke test, ~15 minutes on one GPU):

```bash
python src/quant_sensitivity_experiment.py \
    --models qwen2.5-1.5b \
    --method rtn \
    --bits 4 3 \
    --group_size 128 \
    --n_tokens 16384 \
    --block_size 1024 \
    --output_dir results/raw
```

Then build the figures and tables:

```bash
python src/analyze_quant_results.py \
    --input "results/raw/*_quant_sensitivity_rtn_g128.json" \
    --bits 4 3 \
    --output_dir results/figures
```

Or use the Makefile shortcuts:

```bash
make smoke     # single-model sanity run
make rtn       # full RTN sweep
make gptq      # full GPTQ sweep
make figures   # regenerate all paper figures and tables
```

---

## Reproducing the paper

```bash
bash scripts/run_rtn_sweep.sh        # 2,144 measurements, 9 models
bash scripts/run_gptq_sweep.sh       # 1,664 measurements, 7 models
bash scripts/make_paper_figures.sh   # figures → results/figures, tables → results/tables
```

Experimental configuration used throughout the paper:

| Setting | Value |
| :-- | :-- |
| Evaluation corpus | WikiText-2, 16,384 tokens, non-overlapping 1,024-token blocks |
| Bit-widths | 4 and 3 |
| Group size | 128 |
| GPTQ calibration | 2,048 tokens from C4 |
| RTN activation moments | WikiText-2 evaluation activations |
| Determinism | RTN and GPTQ are deterministic given fixed weights and token sequences |

See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for the full flag reference, the output
file format, checkpoint/resume behavior, and runtime estimates.

---

## Quantization methods

`src/quant_methods.py` provides four weight-only quantizers behind a common dispatch:

| Method | Flag | Description |
| :-- | :-- | :-- |
| RTN | `--method rtn` | Symmetric per-row / per-group round-to-nearest |
| GPTQ | `--method gptq` | Row-by-row optimal-brain quantization with Hessian damping and Cholesky inverse |
| AWQ | `--method awq` | Activation-aware per-channel scale search over a grid of $\alpha$ |
| QuIP | `--method quip` | Incoherence processing via random-sign Hadamard rotation |

RTN is the primary method in the paper: its reconstruction error depends only on the
weight distribution and not on calibration data, which gives the cleanest possible test of
the reconstruction-error/sensitivity relationship.

---

## Results directory

`results/` is where every experimental artifact lands. It ships with the directory
structure in place and `.gitkeep` markers so the tree survives a fresh clone. Drop your
result files into the matching subdirectory:

- `results/raw/` — per-model JSON emitted by the sweep, e.g. `llama2-7b_quant_sensitivity_rtn_g128.json`
- `results/figures/` — PDF figures produced by the analysis script
- `results/tables/` — LaTeX tables (`within_component_r2_3bit.tex`, `dominance_3bit.tex`, …)

See [`results/README.md`](results/README.md) for the naming convention and the JSON schema.

---

## Citation

```bibtex
@inproceedings{dewage2026component,
  title     = {Component Type, Not Reconstruction Error, Predicts Attention
               Quantization Sensitivity},
  author    = {Dewage, Kasun and Pensky, Marianna and De Silva, Suranadi},
  year      = {2026},
  address   = {Orlando, Florida, USA},
  note      = {University of Central Florida}
}
```

---

## License

Released under the [MIT License](LICENSE). All evaluated checkpoints are publicly
available on Hugging Face and remain subject to their own licenses.
