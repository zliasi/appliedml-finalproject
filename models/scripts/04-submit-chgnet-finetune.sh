#!/usr/bin/env bash
set -euo pipefail

# Fine-tune CHGNet on magmom21 (train+val) and score on the held-out test split,
# as a GPU job. Writes baselines-chgnet-ft.json (+ plots) into the run's eval/,
# which 05-submit-evaluate.sh merges as baseline-chgnet-ft. Needs chgnet
# installed. Independent of the rest of the pipeline.
#
# Usage:
#   ./scripts/04-submit-chgnet-finetune.sh --target magmom --dataset magmom21-v0p1
#   ./scripts/04-submit-chgnet-finetune.sh --target magmom --dataset magmom21-v0p1 --epochs 200 --wandb

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

TARGET="magmom"
DATASET=""
EPOCHS=80
WANDB_FLAG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)  TARGET="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --epochs)  EPOCHS="$2"; shift 2 ;;
        --wandb)   WANDB_FLAG="--wandb"; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done
[[ -n "${DATASET}" ]] || {
    echo "Usage: $0 --target T --dataset D-vMpN [--epochs N] [--wandb]" >&2
    exit 1
}
readonly TARGET DATASET EPOCHS WANDB_FLAG

readonly RUN_DIR="runs/${TARGET}-${DATASET}"
readonly LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

batch_file="run-chgnet-ft-$$.tmp"
cat > "${batch_file}" << !EOSBATCH
#!/usr/bin/env bash
#SBATCH --job-name=${TARGET}-chgnet-ft-${DATASET}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --partition=katla_l40s
#SBATCH --output=${LOG_DIR}/chgnet-ft-%j.log

source "\${SLURM_SUBMIT_DIR}/scripts/env.sh"

cd "\${SLURM_SUBMIT_DIR}"

python scripts/workers/chgnet-finetune.py \\
    --target ${TARGET} \\
    --dataset ${DATASET} \\
    --epochs ${EPOCHS} \\
    --device cuda \\
    ${WANDB_FLAG}
!EOSBATCH
job_id=$(sbatch --parsable "${batch_file}")
rm -f "${batch_file}"
printf "Submitted chgnet fine-tune: %s (%s epochs)\n" "${job_id}" "${EPOCHS}"