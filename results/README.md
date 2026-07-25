# Results

All experimental artifacts live here. The directory structure is committed (via
`.gitkeep` files) so it survives a fresh clone even when empty.


## Naming convention

Filenames carry a suffix derived from the quantization method and group size:

```
<model_key>_quant_sensitivity_<method>_g<group_size>.json
```

Examples:

| File | Contents |
| :-- | :-- |
| `llama2-7b_quant_sensitivity_rtn_g128.json` | LLaMA-2-7B, RTN, group size 128 |
| `mistral-7b_quant_sensitivity_gptq_g128.json` | Mistral-7B, GPTQ, group size 128 |
| `all_quant_sensitivity_rtn_g128.json` | Combined across every model in one run |

Files matching `*_checkpoint.json` are resumable-sweep scratch files. They are
regenerated automatically and are excluded by `.gitignore`.

## JSON schema

Each per-model file follows this shape:

```jsonc
{
  "model": "llama2-7b",
  "hf_name": "meta-llama/Llama-2-7b-hf",
  "arch": "llama",
  "method": "rtn",
  "bits": [4, 3],
  "group_size": 128,
  "baseline_ppl": 5.47,
  "eval_datasets": ["wikitext2"],
  "primary_dataset": "wikitext2",
  "layers": {
    "<layer>.<component>": {
      // per bit-width: relative reconstruction error,
      // activation-weighted quantization error, quantized PPL, and delta PPL
    }
  }
}
```

## Adding results

Drop the JSON files into `results/raw/`, then run:

```bash
bash scripts/make_paper_figures.sh
```

Figures land in `results/figures/` and LaTeX tables in `results/tables/`.

## Size note

Raw JSON files are small (tens to hundreds of KB) and are safe to commit directly.
Model weights, activation caches, and Hessians are **not** stored here and are excluded
by `.gitignore`.
