#!/usr/bin/env bash
set -euo pipefail

# End-to-end pipeline orchestrator for one (target, dataset).
#
# Chains: build-graphs (array) -> train (--all) -> eval.
# SLURM dependencies link each stage to its predecessor (eval afterany).
#
# Usage:
#   ./scripts/00-run-pipeline.sh --target hads --dataset fcc12-v1p1
#   ./scripts/00-run-pipeline.sh --target hads --dataset fcc4-v1p1 \\
#       --skip-build           # if graphs already in place
#   ./scripts/00-run-pipeline.sh --target hads --dataset fcc4-v1p1 \\
#       --eval-only            # just submit eval (e.g. after partial)

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

TARGET=""
DATASET=""
SKIP_BUILD=0
EVAL_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)     TARGET="$2"; shift 2 ;;
        --dataset)    DATASET="$2"; shift 2 ;;
        --skip-build) SKIP_BUILD=1; shift ;;
        --eval-only)  EVAL_ONLY=1; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done
[[ -n "${TARGET}" && -n "${DATASET}" ]] || {
    echo "Usage: $0 --target T --dataset D-vMpN" >&2
    exit 1
}
readonly TARGET DATASET SKIP_BUILD EVAL_ONLY

readonly RUN_DIR="runs/${TARGET}-${DATASET}"
readonly LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

emit_eval_job() {
    local dep="${1:-}"
    local dep_arg=""
    [[ -n "${dep}" ]] && dep_arg="--dependency=afterany:${dep}"

    local batch_file="run-eval-$$.tmp"
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
#SBATCH --output=${LOG_DIR}/06-eval-%j.log

source "\${SLURM_SUBMIT_DIR}/scripts/env.sh"

cd "\${SLURM_SUBMIT_DIR}"

python scripts/03-evaluate.py \\
    --target ${TARGET} \\
    --dataset ${DATASET} \\
    --all
!EOSBATCH
    local eval_id
    eval_id=$(sbatch --parsable ${dep_arg} "${batch_file}")
    rm -f "${batch_file}"
    echo "${eval_id}"
}

if [[ ${EVAL_ONLY} -eq 1 ]]; then
    eval_id=$(emit_eval_job "")
    printf "Submitted eval-only: %s\n" "${eval_id}"
    exit 0
fi

# Stage 1: build-graphs (unless skipped).
build_dep=""
if [[ ${SKIP_BUILD} -eq 0 ]]; then
    build_output=$(
        ./scripts/01-submit-build-graphs.sh \
            --target "${TARGET}" --dataset "${DATASET}"
    )
    build_id=$(printf "%s\n" "${build_output}" |
        awk '/Submitted array job:/{print $NF}')
    [[ -n "${build_id}" ]] || {
        echo "Could not parse build job id" >&2; exit 1
    }
    build_dep="--dependency=afterok:${build_id}"
    printf "Build array: %s\n" "${build_id}"
fi

# Stage 2: training (--all) + baselines, both depending on build.
#
# 02-submit-training.sh wraps a single sbatch per call, so we submit
# twice (GNNs + baselines) sharing the build_dep.
gnn_batch="run-gnn-array-$$.tmp"
configs_list="${LOG_DIR}/.config-list.txt"
find "targets/${TARGET}/configs" -name "*.yaml" | sort > "${configs_list}"
n_configs=$(wc -l < "${configs_list}")

cat > "${gnn_batch}" << !EOSBATCH
#!/usr/bin/env bash
#SBATCH --job-name=${TARGET}-train-${DATASET}
#SBATCH --array=1-${n_configs}%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=katla_l40s
#SBATCH --output=${LOG_DIR}/04-train-%A_%a.log

source "\${SLURM_SUBMIT_DIR}/scripts/env.sh"

cd "\${SLURM_SUBMIT_DIR}"

CONFIG=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" "${configs_list}")
python scripts/workers/train.py \\
    --target ${TARGET} \\
    --dataset ${DATASET} \\
    --config "\${CONFIG}" \\
    --device cuda \\
    --wandb
!EOSBATCH
gnn_id=$(sbatch --parsable ${build_dep} "${gnn_batch}")
rm -f "${gnn_batch}"
printf "GNN array: %s (%d configs)\n" "${gnn_id}" "${n_configs}"

# Baselines skipped: the ported baselines worker is graph-level and does
# not fit the node-level magmom target (a per-element-mean baseline is a
# later batch). Re-add a baselines stage here once it exists.

# Stage 3: eval, depending on the GNN array (afterany, so one failing
# backend does not cancel the eval).
eval_id=$(emit_eval_job "${gnn_id}")
printf "Eval: %s (depends on %s)\n" "${eval_id}" "${gnn_id}"
printf "\nPipeline submitted.\n"
