#!/usr/bin/env bash
set -euo pipefail

# Submit a SLURM array of graph-build tasks for one (target, dataset).
#
# Builds one graph cache per cutoff in --cutoffs (default "4 6 8", Angstrom).
#
# Usage:
#   ./scripts/01-submit-build-graphs.sh --target magmom --dataset magmom21-v0p1
#   ./scripts/01-submit-build-graphs.sh --target magmom --dataset magmom21-v0p1 --signed
#   ./scripts/01-submit-build-graphs.sh --target magmom --dataset magmom21-v0p1 --cutoffs "3"

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MODELS_DIR="${SCRIPT_DIR}/.."
cd "${MODELS_DIR}"

TARGET=""
DATASET=""
SIGNED_FLAG=""
CUTOFFS="4 6 8"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target) TARGET="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --signed) SIGNED_FLAG="--signed"; shift ;;
        --cutoffs) CUTOFFS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done
[[ -n "${TARGET}" && -n "${DATASET}" ]] || {
    echo "Usage: $0 --target magmom --dataset <name-vMpN>" >&2
    exit 1
}
readonly TARGET DATASET SIGNED_FLAG CUTOFFS
SUFFIX=""
if [[ -n "${SIGNED_FLAG}" ]]; then SUFFIX="-signed"; fi
readonly SUFFIX

case "${TARGET}" in
    magmom) ;;
    *) echo "Unknown target: ${TARGET}" >&2; exit 1 ;;
esac
VARIANTS=()
for c in ${CUTOFFS}; do
    VARIANTS+=("r${c}:--cutoff ${c}")
done
readonly VARIANTS
readonly N=${#VARIANTS[@]}

readonly RUN_DIR="runs/${TARGET}-${DATASET}${SUFFIX}"
readonly LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"
readonly VARIANT_LIST="${LOG_DIR}/.variants-build-graphs.txt"
printf "%s\n" "${VARIANTS[@]}" > "${VARIANT_LIST}"

printf "target=%s dataset=%s variants=%d\n" \
    "${TARGET}" "${DATASET}" "${N}"

batch_file="run-build-graphs-$$.tmp"

cat > "${batch_file}" << !EOSBATCH
#!/usr/bin/env bash
#SBATCH --job-name=${TARGET}-build-${DATASET}
#SBATCH --array=1-${N}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=katla_medium
#SBATCH --output=${LOG_DIR}/build-graphs-%A_%a.log

source "\${SLURM_SUBMIT_DIR}/scripts/env.sh"

cd "\${SLURM_SUBMIT_DIR}"

LINE=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" "${VARIANT_LIST}")
TAG="\${LINE%%:*}"
ARGS="\${LINE#*:}"

printf "Job %s Task %s Variant %s\n" \\
    "\${SLURM_JOB_ID}" "\${SLURM_ARRAY_TASK_ID}" "\${TAG}"
printf "Target ${TARGET} Dataset ${DATASET}\n"
printf "Args: %s\n" "\${ARGS}"
printf "Cores: %s\n\n" "\${SLURM_CPUS_PER_TASK}"

python scripts/workers/build-graphs.py \\
    --target ${TARGET} \\
    --dataset ${DATASET} \\
    ${SIGNED_FLAG} \\
    \${ARGS} \\
    --n-workers \${SLURM_CPUS_PER_TASK}

sleep 2
/usr/bin/sacct -n -j \${SLURM_JOB_ID} \\
    --format=JobID,JobName,MaxRSS,Elapsed,CPUTime --units=MB
!EOSBATCH

JOB_ID=$(sbatch --parsable "${batch_file}")
rm -f "${batch_file}"
printf "Submitted array job: %s\n" "${JOB_ID}"
