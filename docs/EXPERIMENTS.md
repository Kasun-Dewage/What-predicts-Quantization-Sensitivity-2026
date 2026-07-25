# Experiments Guide

Detailed reference for running the sweep, understanding the outputs, and reproducing
every number in the paper.

---

## 1. The measurement loop

For each `(model, method, bit-width)` combination the pipeline does the following:

1. **Baseline.** Compute WikiText-2 perplexity over 16,384 tokens in non-overlapping
   1,024-token blocks with all weights at full precision.
2. **Calibration statistics.** Capture per-channel activation second moments
   $\mathbb{E}_b[x_j^2]$ via forward hooks. GPTQ and AWQ additionally accumulate the
   Hessian $H = XX^\top$ or the mean absolute activation.
3. **Single-projection quantization.** For one attention projection at a time (Q, K, V, or
   O at a given layer), replace $W$ with $W_q$ while every other weight in the model stays
   at full precision.
4. **Re-evaluate.** Recompute perplexity and record
   $\Delta\mathrm{PPL} = \mathrm{PPL}_q - \mathrm{PPL}_{\text{base}}$.
5. **Record three signals.** Relative reconstruction error
   $\lVert W - W_q\rVert_F / \lVert W\rVert_F$, the activation-weighted quantization error
   $\hat{S}$, and $\Delta\mathrm{PPL}$.
6. **Restore** the original weight and advance to the next projection.

A full sweep is at most 256 measurements per `(model, method)` pair — 4 components ×
number of layers × 2 bit-widths.

---

## 2. Model keys

```bash
python src/quant_sensitivity_experiment.py --list
```

| Key | Hugging Face checkpoint | Attention | In paper |
| :-- | :-- | :-: | :-: |
| `llama1-7b` | `huggyllama/llama-7b` | MHA | ✅ |
| `llama2-7b` | `meta-llama/Llama-2-7b-hf` | MHA | ✅ |
| `llama3-8b` | `meta-llama/Meta-Llama-3-8B` | GQA | ✅ |
| `mistral-7b` | `mistralai/Mistral-7B-v0.1` | GQA | ✅ |
| `qwen2.5-1.5b` | `Qwen/Qwen2.5-1.5B` | GQA | ✅ |
| `qwen2.5-7b` | `Qwen/Qwen2.5-7B` | GQA | ✅ |
| `opt-1.3b` | `facebook/opt-1.3b` | MHA | ✅ |
| `opt-6.7b` | `facebook/opt-6.7b` | MHA | ✅ |
| `gpt-j-6b` | `EleutherAI/gpt-j-6B` | MHA | ✅ |
| `llama3.2-1b` | `meta-llama/Llama-3.2-1B` | GQA | — |
| `opt-13b` | `facebook/opt-13b` | MHA | — |
| `falcon-7b` | `tiiuae/falcon-7b` | MHA | — |
| `pythia-6.9b` | `EleutherAI/pythia-6.9b` | MHA | — |

`llama1-7b`, `llama2-7b`, `llama3-8b`, and `mistral-7b` are gated on the Hub — run
`huggingface-cli login` first.

---

## 3. Sweep flags

`src/quant_sensitivity_experiment.py`

| Flag | Default | Description |
| :-- | :-- | :-- |
| `--models` | all | Space-separated model keys |
| `--method` | `rtn` | One of `rtn`, `gptq`, `awq`, `quip` |
| `--bits` | `4 3` | Bit-widths to sweep |
| `--group_size` | `0` | Quantization group size; `0` means per-row. The paper uses `128` |
| `--n_tokens` | `16384` | Evaluation tokens; `-1` for the full dataset |
| `--block_size` | `1024` | Perplexity block length |
| `--ppl_cap_mult` | `50.0` | Clamp catastrophic perplexity blow-ups at this multiple of baseline |
| `--eval_datasets` | `wikitext2` | Any of `wikitext2`, `c4`, `ptb` |
| `--output_dir` | `./output/quant_sensitivity` | Use `results/raw` in this repo |
| `--n_calib_tokens` | `2048` | Calibration tokens for GPTQ/AWQ |
| `--calib_source` | `c4` | `c4` or `wikitext2` |
| `--compute_hessian_sens` | off | Also record a Hessian-trace sensitivity score |
| `--mp` | off | Run the mixed-precision follow-up experiment |
| `--mp_dominant` | auto | Force the protected component (`Q`, `K`, `V`, or `O`) |
| `--no_cpu_offload` | off | Keep the whole model resident on GPU (needs more VRAM) |
| `--no_checkpoint` | off | Disable resumable checkpointing |
| `--downstream_tasks` | none | e.g. `hellaswag arc_easy arc_challenge piqa winogrande mmlu` |
| `--list` | — | Print the model table and exit |

### Checkpoint and resume

Unless `--no_checkpoint` is passed, partial progress is written to
`results/raw/<model>_quant_sensitivity_<suffix>_checkpoint.json` after each layer. Re-running
the same command picks up where it left off and deletes the checkpoint once the model
finishes. This matters: a full 7B sweep is hundreds of perplexity evaluations.

---

## 4. Output files

Filenames carry a suffix built from the method and group size — `_rtn_g128`, `_gptq_g128`,
and so on.

```
results/raw/
├── llama2-7b_quant_sensitivity_rtn_g128.json      # one file per model
├── ...
└── all_quant_sensitivity_rtn_g128.json            # combined across all models in the run
```

Each per-model JSON contains the baseline perplexity, run configuration, and a `layers`
map keyed by projection, with the reconstruction error, activation-weighted score, and
$\Delta$PPL per bit-width. The sweep also emits per-model scatter plots and heatmaps
alongside the JSON.

---

## 5. Analysis

`src/analyze_quant_results.py` accepts glob patterns and regenerates every figure and
table in the paper.

```bash
python src/analyze_quant_results.py \
    --input "results/raw/*_quant_sensitivity_rtn_g128.json" \
    --bits 4 3 \
    --n_boot 2000 \
    --output_dir results/figures
```

| Flag | Default | Description |
| :-- | :-- | :-- |
| `--input` | required | One or more paths or glob patterns |
| `--bits` | `4 3` | Bit-widths to analyze |
| `--output_dir` | `./output/.../paper_figs` | Use `results/figures` |
| `--n_boot` | `2000` | Bootstrap resamples for the $R^2$ confidence intervals |
| `--skip_per_model_figs` | off | Only produce the cross-model summary figures |
| `--split_proj_type` | on | Additionally split by projection shape (QO-square, KV-square, KV-rect) |

### Figures produced

| File | Content |
| :-- | :-- |
| `combined_scatter_<b>bit.pdf` | Reconstruction error vs. $\Delta$PPL, all models |
| `variance_decomposition_<b>bit.pdf` | $\eta^2_{\text{comp}}$ vs. $\eta^2_{\text{layer}}$ vs. $R^2_{\text{recon}}$ |
| `component_dominance_ci_<b>bit.pdf` | Share of positive $\Delta$PPL per component, bootstrapped CIs |
| `v_proj_natural_experiment_<b>bit.pdf` | Flat-reconstruction-error cells with varying $\Delta$PPL |
| `<model>_scatter_<b>bit.pdf` | Per-model scatter |
| `<model>_depth_profile_<b>bit.pdf` | $\Delta$PPL against layer depth |
| `<model>_boxplot_<b>bit.pdf` | $\Delta$PPL distribution by component |
| `<model>_proj_split_scatter_<b>bit.pdf` | Split by projection shape |

### Tables produced

| File | Paper table |
| :-- | :-- |
| `within_component_r2_<b>bit.tex` | Table II |
| `recon_cv_<b>bit.tex` | Table III |
| `variance_decomp_<b>bit.tex` | Table IV |
| `dominance_<b>bit.tex` | Tables V, VI |
| `correlation_table.tex` | Full correlation report |
| `correlation_table_proj_split.tex` | Correlations split by projection shape |
| `mp_summary.tex` | Mixed-precision follow-up |

---

## 6. Runtime notes

- Cost is dominated by the perplexity re-evaluations: one forward pass over 16,384 tokens
  per projection per bit-width.
- CPU offload is on by default. Pass `--no_cpu_offload` if you have the VRAM headroom —
  it is noticeably faster.
- GPTQ adds a calibration pass and a per-projection Cholesky solve, so GPTQ sweeps run
  meaningfully longer than RTN ones.
- Start with `--models qwen2.5-1.5b --bits 3` to validate the environment before
  committing to a full sweep.

---

## 7. Reproducibility

- RTN and GPTQ are deterministic given fixed weights and token sequences.
- QuIP uses a seeded random-sign Hadamard rotation (`seed=0` by default).
- All checkpoints are the public Hugging Face revisions listed above.
- Reference environment: NVIDIA H100 PCIe (80 GB), CUDA 12.1.
