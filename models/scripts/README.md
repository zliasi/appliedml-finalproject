# scripts

Slurm pipeline, run on a HPC system:

- `00-run-pipeline.sh` - build -> train -> evaluate
- `01-submit-build-graphs.sh` - build graph caches (r4/r6/r8)
- `02-submit-training.sh` - train each backend
- `submit-selection.sh` - helper to submit a chosen set of backends in one command
- `03-evaluate.py` - test-set metrics (overall and per-element for node-level targets) plus parity and error plots
- `04-submit-baselines.sh` - optional, standalone non-GNN baselines (per-element-mean + trees) to compare the GNNs against
- `05-generate-sweep-configs.py` - generate an HP-sweep grid of configs for a backend
- `06-submit-sweep.sh` - train an HP sweep for a backend (after 05-generate-sweep-configs.py)

`workers/` holds the code each job runs.
