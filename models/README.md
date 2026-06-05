# models

GNN training pipeline. Produces a GNN that predicts per-atom magnetic moments for metal slabs, that can be used as initial values for GPAW calculations.

- `src/` - shared library: graph building, the LMDB graph dataset, the training loop, metrics, the train/val/test split, and the target interface
- `targets/magmom/` - the magmom target: node-level GNN backends, the data iterator that reads `magmoms.h5`, and the per-backend training configs
- `scripts/` - Slurm pipeline that builds graphs, trains, and evaluates, plus the worker script each job runs
- `bench/` - benchmarks each backend on checkpoint load time, single-slab inference latency, and accuracy
- `runs/` - per-run outputs (graph caches, checkpoints, logs), created at runtime and mostly gitignored

## Running

Set your cluster environment in `scripts/env.sh` (`CONDA_SH`, `CONDA_ENV`), then from `models/`:

- `./scripts/00-run-pipeline.sh --target magmom --dataset magmom21-v0p1` - chained Slurm jobs: build graphs, train every config, evaluate.
- Or run the stages directly with the same `--target/--dataset`: `01-submit-build-graphs.sh`, `02-submit-training.sh`, `04-submit-evaluate.sh`.
- `python bench/bench-inference.py --checkpoints runs/magmom-magmom21-v0p1/checkpoints` - per-checkpoint load time and single-slab inference latency.
