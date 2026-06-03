#!/usr/bin/env bash
# Cluster environment for the pipeline jobs, sourced by each submitted job.
# Defaults to my cluster setup, override CONDA_SH/CONDA_ENV for another system.
module purge
. "${CONDA_SH:-/groups/kemi/liasi/software/build/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-magmom21}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
