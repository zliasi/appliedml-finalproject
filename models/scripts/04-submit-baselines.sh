#!/usr/bin/env bash
set -euo pipefail

# Optional, standalone non-GNN baselines for one (target, dataset). Trains a
# per-element-mean predictor and tree ensembles on per-atom features and scores
# the same test split the GNNs use, so the eval table has a reference to beat.
# Independent of 00-run-pipeline.sh; just needs the graph caches already built
# (run 01-submit-build-graphs.sh first).
#
# Usage:
#   ./scripts/04-submit-baselines.sh --target magmom --dataset magmom21-v0p1
#   ./scripts/04-submit-baselines.sh --target magmom --dataset magmom21-v0p1 --cutoff 8

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

TARGET="magmom"
DATASET=""
CUTOFF=6

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)  TARGET="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --cutoff)  CUTOFF="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done
[[ -n "${DATASET}" ]] || { echo "Usage: $0 --target T --dataset D-vMpN [--cutoff N]" >&2; exit 1; }
readonly TARGET DATASET CUTOFF

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
#SBATCH --time=08:00:00
#SBATCH --partition=katla_long
#SBATCH --output=${LOG_DIR}/baselines-%j.log

source "\${SLURM_SUBMIT_DIR}/scripts/env.sh"

cd "\${SLURM_SUBMIT_DIR}"

python scripts/workers/magmom-baselines.py \\
    --target ${TARGET} \\
    --dataset ${DATASET} \\
    --cutoff ${CUTOFF}
!EOSBATCH
job_id=$(sbatch --parsable "${batch_file}")
rm -f "${batch_file}"
printf "Submitted baselines: %s (cutoff r%s)\n" "${job_id}" "${CUTOFF}"