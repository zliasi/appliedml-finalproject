# scripts

Slurm pipeline, run on a HPC system:

- `00-run-pipeline.sh` - build -> train -> evaluate
- `01-submit-build-graphs.sh` - build graph caches (r4/r6/r8)
- `02-submit-training.sh` - train each backend
- `03-evaluate.py` - per-atom and per-element metrics

`workers/` holds the code each job runs.
