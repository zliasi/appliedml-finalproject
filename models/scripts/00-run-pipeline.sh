#!/usr/bin/env bash
set -euo pipefail

# End-to-end pipeline orchestrator for one (target, dataset).
#
# Chains: build-graphs (array) -> train (--all) + baselines -> eval.
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
WANDB_FLAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)     TARGET="$2"; shift 2 ;;
        --dataset)    DATASET="$2"; shift 2 ;;
        --skip-build) SKIP_BUILD=1; shift ;;
        --eval-only)  EVAL_ONLY=1; shift ;;
        --wandb)      WANDB_FLAG="--wandb"; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done
[[ -n "${TARGET}" && -n "${DATASET}" ]] || {
    echo "Usage: $0 --target T --dataset D-vMpN" >&2
    exit 1
}
readonly TARGET DATASET SKIP_BUILD EVAL_ONLY WANDB_FLAG

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

python scripts/workers/evaluate.py \\
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

# Cutoffs to build: the distinct radius_cutoff values the configs actually use
# (so e.g. an r3 config is not left without its graph cache). Falls back to the
# build script default if none are found.
CUTOFFS=$(
    find "targets/${TARGET}/configs" -name '*.yaml' -exec \
        grep -h '^radius_cutoff:' {} + 2>/dev/null |
        awk '{print int($2)}' | sort -un | tr '\n' ' '
)
CUTOFFS="${CUTOFFS:-4 6 8}"
printf "Config cutoffs: %s\n" "${CUTOFFS}"

# Stage 1: build-graphs (unless skipped).
build_dep=""
if [[ ${SKIP_BUILD} -eq 0 ]]; then
    build_output=$(
        ./scripts/01-submit-build-graphs.sh \
            --target "${TARGET}" --dataset "${DATASET}" \
            --cutoffs "${CUTOFFS}"
    )
    build_id=$(printf "%s\n" "${build_output}" |
        awk '/Submitted array job:/{print $NF}')
    [[ -n "${build_id}" ]] || {
        echo "Could not parse build job id" >&2; exit 1
    }
    build_dep="--dependency=afterok:${build_id}"
    printf "Build array: %s\n" "${build_id}"
fi

# Stage 2: training (--all) and baselines, both depending on build.
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
    ${WANDB_FLAG}
!EOSBATCH
gnn_id=$(sbatch --parsable ${build_dep} "${gnn_batch}")
rm -f "${gnn_batch}"
printf "GNN array: %s (%d configs)\n" "${gnn_id}" "${n_configs}"

# Stage 2b: baselines (per-element-mean, linear, xgboost) over r4/r6/r8,
# depending on build. The eval stage merges the results into the ranking.
baselines_batch="run-baselines-$$.tmp"
cat > "${baselines_batch}" << !EOSBATCH
#!/usr/bin/env bash
#SBATCH --job-name=${TARGET}-baselines-${DATASET}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --partition=katla_l40s
#SBATCH --output=${LOG_DIR}/03-baselines-%j.log

source "\${SLURM_SUBMIT_DIR}/scripts/env.sh"

cd "\${SLURM_SUBMIT_DIR}"

for R in 4 6 8; do
    printf "\nbaselines at r%s\n" "\${R}"
    python scripts/workers/magmom-baselines.py \\
        --target ${TARGET} \\
        --dataset ${DATASET} \\
        --cutoff \${R} \\
        --device cuda \\
        ${WANDB_FLAG}
done

printf "\nchgnet zero-shot baseline\n"
python scripts/workers/chgnet-baseline.py \\
    --target ${TARGET} \\
    --dataset ${DATASET} \\
    --device cuda \\
    ${WANDB_FLAG} || echo "chgnet baseline skipped (pip install chgnet to enable)"
!EOSBATCH
baselines_id=$(sbatch --parsable ${build_dep} "${baselines_batch}")
rm -f "${baselines_batch}"
printf "Baselines: %s\n" "${baselines_id}"

# Stage 2c: CHGNet fine-tune (its own GPU job, longer wall time). Reads the
# dataset and the seeded split directly, so it only needs the build to have run.
ft_batch="run-chgnet-ft-$$.tmp"
cat > "${ft_batch}" << !EOSBATCH
#!/usr/bin/env bash
#SBATCH --job-name=${TARGET}-chgnet-ft-${DATASET}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=katla_l40s
#SBATCH --output=${LOG_DIR}/04-chgnet-ft-%j.log

source "\${SLURM_SUBMIT_DIR}/scripts/env.sh"

cd "\${SLURM_SUBMIT_DIR}"

python scripts/workers/chgnet-finetune.py \\
    --target ${TARGET} \\
    --dataset ${DATASET} \\
    --epochs 80 \\
    --device cuda \\
    ${WANDB_FLAG} || echo "chgnet fine-tune skipped (pip install chgnet to enable)"
!EOSBATCH
ft_id=$(sbatch --parsable ${build_dep} "${ft_batch}")
rm -f "${ft_batch}"
printf "CHGNet fine-tune: %s\n" "${ft_id}"

# Stage 3: eval, depending on the GNN array, baselines, and the CHGNet fine-tune
# (afterany, so a failing backend does not cancel the eval; all are merged).
eval_id=$(emit_eval_job "${gnn_id}:${baselines_id}:${ft_id}")
printf "Eval: %s (depends on %s + %s + %s)\n" \
    "${eval_id}" "${gnn_id}" "${baselines_id}" "${ft_id}"
printf "\nPipeline submitted.\n"
