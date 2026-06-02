# models

GNN training pipeline. Produces a GNN that predicts per-atom magnetic moments for metal slabs, that can be used as initial values for GPAW calculations.

- `src/` - shared library: graph building, the LMDB graph dataset, the training loop, metrics, the train/val/test split, and the target interface
- `targets/magmom/` - the magmom target: node-level GNN backends, the data iterator that reads `magmoms.h5`, and the per-backend training configs
- `scripts/` - Slurm pipeline that builds graphs, trains, and evaluates, plus the worker script each job runs
- `bench/` - benchmarks each backend on checkpoint load time, single-slab inference latency, and accuracy
- `runs/` - per-run outputs (graph caches, checkpoints, logs), created at runtime and mostly gitignored
