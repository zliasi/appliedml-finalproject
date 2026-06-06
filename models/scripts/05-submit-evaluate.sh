#!/usr/bin/env bash
set -euo pipefail

# Submit test-set evaluation as a GPU job: runs workers/evaluate.py --all over
# every checkpoint in the run dir, writing per-model metrics, parity/error plots,
# and the ranked all-models-summary.json (merging any baselines present).
# Much faster than running the evaluation on the login node.
#
# Usage:
#   ./scripts/05-submit-evaluate.sh --target magmom --dataset magmom21-v0p1
#   ./scripts/05-submit-evaluate.sh --target magmom --dataset magmom21-v0p1 --signed

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

TARGET="magmom"
DATASET=""
SIGNED_FLAG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)  TARGET="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --signed)  SIGNED_FLAG="--signed"; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done
[[ -n "${DATASET}" ]] || {
    echo "Usage: $0 --target T --dataset D-vMpN [--signed]" >&2
    exit 1
}
readonly TARGET DATASET SIGNED_FLAG
SUFFIX=""
if [[ -n "${SIGNED_FLAG}" ]]; then SUFFIX="-signed"; fi
readonly SUFFIX

readonly RUN_DIR="runs/${TARGET}-${DATASET}${SUFFIX}"
readonly LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

batch_file="run-eval-$$.tmp"
cat > "${batch_file}" << !EOSBATCH
#!/usr/bin/env bash
#SBATCH --job-name=${TARGET}-eval-${DATASET}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=katla_l40s
#SBATCH --output=${LOG_DIR}/eval-%j.log

source "\${SLURM_SUBMIT_DIR}/scripts/env.sh"

cd "\${SLURM_SUBMIT_DIR}"

python scripts/workers/evaluate.py \\
    --target ${TARGET} \\
    --dataset ${DATASET} \\
    ${SIGNED_FLAG} \\
    --all
!EOSBATCH
job_id=$(sbatch --parsable "${batch_file}")
rm -f "${batch_file}"
printf "Submitted eval: %s\n" "${job_id}"