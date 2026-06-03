#!/usr/bin/env bash
# Cluster environment for the pipeline jobs, sourced by each submitted job.
# Edit CONDA_SH / CONDA_ENV for your system, or set them in your shell before
# submitting.
module purge 2>/dev/null || true
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-magmom}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
