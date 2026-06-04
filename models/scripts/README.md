# scripts

Slurm pipeline, run on a HPC system:

- `00-run-pipeline.sh` - build -> train -> evaluate
- `01-submit-build-graphs.sh` - build graph caches (r4/r6/r8)
- `02-submit-training.sh` - train each backend
- `submit-selection.sh` - helper to submit a chosen set of backends in one command
- `03-submit-baselines.sh` - optional, standalone non-GNN baselines (per-element-mean, linear, xgboost) at each cutoff, to compare the GNNs against
- `04-evaluate.py` - test-set metrics (overall and per-element for node-level targets), parity and error plots, and merges baseline results when present
- `05-generate-sweep-configs.py` - generate an HP-sweep grid of configs for a backend
- `06-submit-sweep.sh` - train an HP sweep for a backend (after 05-generate-sweep-configs.py)
- `slim-checkpoint.py` - strip optimizer state from a checkpoint for faster warm-start loading

`workers/` holds the code each job runs.
