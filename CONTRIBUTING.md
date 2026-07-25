# Contributing

Thanks for your interest in this project. Issues and pull requests are welcome.

## Reporting issues

When opening an issue, please include:

- The exact command you ran
- Model key, quantization method, bit-width, and group size
- Python, PyTorch, `transformers`, and CUDA versions
- GPU model and available VRAM
- The full traceback, if there is one

## Pull requests

1. Fork the repository and create a branch off `main`.
2. Keep changes focused — one concern per pull request.
3. Match the existing style: 4-space indentation, descriptive names, no unrelated
   reformatting of the numerical code paths.
4. If you change a quantizer or the measurement loop, run the smoke test and report the
   before/after numbers:
   ```bash
   make smoke
   ```
5. New quantization methods should be added to `src/quant_methods.py` and registered in
   `quantize_dispatch`, then exposed through the `--method` choices in
   `src/quant_sensitivity_experiment.py`.

## Reproducibility expectations

Results in this repository are deterministic for RTN and GPTQ given fixed weights and
token sequences. If a change introduces nondeterminism, please say so explicitly in the
pull request description and expose a seed argument.
