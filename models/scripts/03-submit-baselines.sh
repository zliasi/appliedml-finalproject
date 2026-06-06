#!/usr/bin/env bash
set -euo pipefail

# Optional, standalone non-GNN baselines for one (target, dataset). Trains a
# per-element-mean predictor, a linear regression, and XGBoost on per-atom
# features and scores the same test split the GNNs use, at each cutoff, so the
# eval table has a per-cutoff reference to beat. Independent of 00-run-pipeline.sh,
# just needs the graph caches already built (run 01-submit-build-graphs.sh first).
# 05-submit-evaluate.sh picks up the per-cutoff results if they are present.
#
# Usage:
#   ./scripts/03-submit-baselines.sh --target magmom --dataset magmom21-v0p1
#   ./scripts/03-submit-baselines.sh --target magmom --dataset magmom21-v0p1 --cutoffs "6"
#   ./scripts/03-submit-baselines.sh --target magmom --dataset magmom21-v0p1 --wandb

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

TARGET="magmom"
DATASET=""
CUTOFFS="4 6 8"
WANDB_FLAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)  TARGET="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --cutoffs) CUTOFFS="$2"; shift 2 ;;
        --wandb)   WANDB_FLAG="--wandb"; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done
[[ -n "${DATASET}" ]] || {
    echo "Usage: $0 --target T --dataset D-vMpN [--cutoffs \"4 6 8\"]" >&2
    exit 1
}
readonly TARGET DATASET CUTOFFS WANDB_FLAG

readonly RUN_DIR="runs/${TARGET}-${DATASET}"
readonly LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

batch_file="run-baselines-$$.tmp"
cat > "${batch_file}" << !EOSBATCH
#!/usr/bin/env bash
#SBATCH --job-name=${TARGET}-baselines-${DATASET}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --partition=katla_l40s
#SBATCH --output=${LOG_DIR}/baselines-%j.log

source "\${SLURM_SUBMIT_DIR}/scripts/env.sh"

cd "\${SLURM_SUBMIT_DIR}"

for R in ${CUTOFFS}; do
    printf "\nbaselines at r%s\n" "\${R}"
    python scripts/workers/magmom-baselines.py \\
        --target ${TARGET} \\
        --dataset ${DATASET} \\
        --cutoff \${R} \\
        --device cuda \\
        ${WANDB_FLAG}
done

printf "\nchgnet baseline\n"
python scripts/workers/chgnet-baseline.py \\
    --target ${TARGET} \\
    --dataset ${DATASET} \\
    --device cuda \\
    ${WANDB_FLAG} || echo "chgnet baseline skipped (pip install chgnet to enable)"
!EOSBATCH
job_id=$(sbatch --parsable "${batch_file}")
rm -f "${batch_file}"
printf "Submitted baselines: %s (cutoffs %s)\n" "${job_id}" "${CUTOFFS}"