# scripts

Slurm pipeline, run on a HPC system:

- `00-run-pipeline.sh` - build -> train -> evaluate
- `01-submit-build-graphs.sh` - build graph caches (r4/r6/r8)
- `02-submit-training.sh` - train each backend
- `submit-selection.sh` - helper to submit a chosen set of backends in one command
- `03-submit-baselines.sh` - optional, standalone non-GNN baselines (per-element-mean, linear, xgboost at each cutoff, plus zero-shot CHGNet), to compare the GNNs against
- `04-submit-chgnet-finetune.sh` - optional, fine-tune CHGNet on magmom21 and score on the test split (merged by eval as baseline-chgnet-ft)
- `05-submit-evaluate.sh` - submit test-set eval as a GPU job (runs `workers/evaluate.py`): per-model metrics, parity/error plots, and a ranked summary merging any baselines
- `06-generate-sweep-configs.py` - generate an HP-sweep grid of configs for a backend
- `07-submit-sweep.sh` - train an HP sweep for a backend (after 06-generate-sweep-configs.py)
- `slim-checkpoint.py` - strip optimizer state from a checkpoint for faster warm-start loading

`workers/` holds the code each job runs.
