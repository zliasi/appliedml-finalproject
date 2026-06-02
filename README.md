# appliedml-finalproject

Repo for the final project for the Applied Machine Learning 2026 course. 

Submission: 2026-05-13. Presentations: 2026-06-10 / 2026-06-11.

## Project structure

- `models/` - the GNN training pipeline (Slurm: build graphs, train each backend, evaluate)
- `datasets/magmom21/` - per-atom magnetic moments of metal alloy slabs, the training data
- `applications/` - example use: predict initial moments for a slab to speed up a GPAW calculation
- `tutorial/` - standalone intro to GNNs (a work-function regression notebook)
- `requirements.txt` - pipeline dependencies
