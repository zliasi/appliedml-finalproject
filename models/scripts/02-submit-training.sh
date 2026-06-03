#!/usr/bin/env bash
set -euo pipefail

# Submit GNN training (--all) and/or baselines (--baselines) for one
# (target, dataset). Same SLURM shape as before: array job with %4
# throttle cap for GNN configs on katla_l40s, single CPU job for
# baselines on katla_long.
#
# Usage:
#   ./scripts/02-submit-training.sh --target hads --dataset fcc12-v1p1 --all
#   ./scripts/02-submit-training.sh --target hads --dataset fcc12-v1p1 --baselines
#   ./scripts/02-submit-training.sh --target hads --dataset fcc12-v1p1 \\
#       targets/hads/configs/schnet/c128l3h1-r8.yaml

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MODELS_DIR="${SCRIPT_DIR}/.."
cd "${MODELS_DIR}"

TARGET=""
DATASET=""
MODE=""
SINGLE_CONFIG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target) TARGET="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --all) MODE="all"; shift ;;
        --baselines) MODE="baselines"; shift ;;
        *)
            if [[ -z "${MODE}" ]]; then
                MODE="single"
                SINGLE_CONFIG="$1"
            else
                echo "Unknown arg: $1" >&2; exit 1
            fi
            shift
            ;;
    esac
done
[[ -n "${TARGET}" && -n "${DATASET}" && -n "${MODE}" ]] || {
    echo "Usage: $0 --target T --dataset D --all|--baselines|<cfg>" >&2
    exit 1
}
readonly TARGET DATASET MODE SINGLE_CONFIG

readonly RUN_DIR="runs/${TARGET}-${DATASET}"
readonly CONFIG_ROOT="targets/${TARGET}/configs"
readonly LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

submit_all() {
    mapfile -t configs < <(
        find "${CONFIG_ROOT}" -name "*.yaml" | sort
    )
    [[ ${#configs[@]} -gt 0 ]] || {
        echo "No yaml configs under ${CONFIG_ROOT}" >&2; return 1
    }
    local n=${#configs[@]}
    local list="${LOG_DIR}/.config-list.txt"
    printf "%s\n" "${configs[@]}" > "${list}"
    printf "GNN configs: %d (target=%s dataset=%s)\n" \
        "${n}" "${TARGET}" "${DATASET}"

    local batch_file="run-gnn-array-$$.tmp"
    cat > "${batch_file}" << !EOSBATCH
#!/usr/bin/env bash
#SBATCH --job-name=${TARGET}-train-${DATASET}
#SBATCH --array=1-${n}%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=katla_l40s
#SBATCH --output=${LOG_DIR}/04-train-%A_%a.log

source "\${SLURM_SUBMIT_DIR}/scripts/env.sh"

cd "\${SLURM_SUBMIT_DIR}"

CONFIG=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" "${list}")

printf "Job %s Task %s\n" "\${SLURM_JOB_ID}" "\${SLURM_ARRAY_TASK_ID}"
printf "Config: %s\n" "\${CONFIG}"
printf "Target ${TARGET} Dataset ${DATASET}\n\n"

python scripts/workers/train.py \\
    --target ${TARGET} \\
    --dataset ${DATASET} \\
    --config "\${CONFIG}" \\
    --device cuda \\
    --wandb

sleep 2
/usr/bin/sacct -n -j \${SLURM_JOB_ID} \\
    --format=JobID,JobName,MaxRSS,Elapsed,CPUTime --units=MB
!EOSBATCH
    sbatch "${batch_file}"
    rm -f "${batch_file}"
}

submit_single() {
    local cfg="$1"
    [[ -f "${cfg}" ]] || { echo "Config not found: ${cfg}" >&2; exit 1; }
    local stem
    stem="$(basename "${cfg}" .yaml)"
    local batch_file="run-${stem}-$$.tmp"

    cat > "${batch_file}" << !EOSBATCH
#!/usr/bin/env bash
#SBATCH --job-name=${TARGET}-${stem}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=katla_l40s
#SBATCH --output=${LOG_DIR}/04-train-${stem}-%j.log

source "\${SLURM_SUBMIT_DIR}/scripts/env.sh"

cd "\${SLURM_SUBMIT_DIR}"

python scripts/workers/train.py \\
    --target ${TARGET} \\
    --dataset ${DATASET} \\
    --config "${cfg}" \\
    --device cuda \\
    --wandb

sleep 2
/usr/bin/sacct -n -j \${SLURM_JOB_ID} \\
    --format=JobID,JobName,MaxRSS,Elapsed,CPUTime --units=MB
!EOSBATCH
    sbatch "${batch_file}"
    rm -f "${batch_file}"
}

submit_baselines() {
    local batch_file="run-baselines-$$.tmp"
    cat > "${batch_file}" << !EOSBATCH
#!/usr/bin/env bash
#SBATCH --job-name=${TARGET}-baselines-${DATASET}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH --partition=katla_long
#SBATCH --output=${LOG_DIR}/05-baselines-%j.log

source "\${SLURM_SUBMIT_DIR}/scripts/env.sh"

cd "\${SLURM_SUBMIT_DIR}"

python scripts/workers/train-baselines.py \\
    --target ${TARGET} \\
    --dataset ${DATASET}

sleep 2
/usr/bin/sacct -n -j \${SLURM_JOB_ID} \\
    --format=JobID,JobName,MaxRSS,Elapsed,CPUTime --units=MB
!EOSBATCH
    sbatch "${batch_file}"
    rm -f "${batch_file}"
}

case "${MODE}" in
    all)       submit_all ;;
    single)    submit_single "${SINGLE_CONFIG}" ;;
    baselines) submit_baselines ;;
esac
