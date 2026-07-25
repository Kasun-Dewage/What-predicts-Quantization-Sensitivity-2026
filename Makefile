# Makefile for: What Predicts Quantization Sensitivity? (2026)

PY        ?= python
RAW       := results/raw
FIGS      := results/figures
TABLES    := results/tables
BITS      := 4 3
GROUP     := 128
NTOKENS   := 16384

.PHONY: help setup smoke rtn gptq figures tables clean-checkpoints list

help:
	@echo "Targets:"
	@echo "  setup             Install Python dependencies"
	@echo "  list              Print available model keys"
	@echo "  smoke             Single small model, quick sanity run"
	@echo "  rtn               Full 9-model RTN sweep at 4 and 3 bits"
	@echo "  gptq              7-model GPTQ sweep at 4 and 3 bits"
	@echo "  figures           Regenerate all paper figures and tables"
	@echo "  clean-checkpoints Remove resumable sweep checkpoints"

setup:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

list:
	$(PY) src/quant_sensitivity_experiment.py --list

smoke:
	$(PY) src/quant_sensitivity_experiment.py \
		--models qwen2.5-1.5b --method rtn --bits 3 \
		--group_size $(GROUP) --n_tokens 4096 --output_dir $(RAW)

rtn:
	bash scripts/run_rtn_sweep.sh

gptq:
	bash scripts/run_gptq_sweep.sh

figures:
	bash scripts/make_paper_figures.sh

clean-checkpoints:
	find $(RAW) -name '*_checkpoint.json' -delete
	@echo "Removed sweep checkpoints."
