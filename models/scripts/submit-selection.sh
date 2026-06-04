#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper to submit training for a chosen set of backends in one
# command, instead of one 02-submit-training.sh call per config. Resolves each
# backend name to its config(s) and hands them to 02-submit-training.sh, so each
# selected model runs as its own job exactly as if submitted by hand. Does not
# change 00 or 02.
#
# Defaults are set for the shared run, so only backend names need to be passed:
#   ./scripts/submit-selection.sh cgconv schnetconv gatv2conv
# Override any default if needed:
#   ./scripts/submit-selection.sh --cutoffs "6" cgconv schnetconv

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

TARGET="magmom"
DATASET="magmom21-v0p1"
CUTOFFS="4 6 8"
WANDB_PASS=""
SIGNED_PASS=""
BACKENDS=()

print_help() {
    local config_root="targets/${TARGET}/configs"
    cat <<USAGE
Submit magmom GNN training for the backends you pick (one job per config).

Usage:
  ./scripts/submit-selection.sh [options] backend [backend ...]

Defaults:
  --target ${TARGET}   --dataset ${DATASET}   --cutoffs "${CUTOFFS}"

Options:
  --cutoffs "4 6 8"  graph cutoff radii to train (default all three)
  --target T         prediction target (default ${TARGET})
  --dataset D-vMpN   dataset token (default ${DATASET})
  --wandb            log to Weights & Biases (default off, no account needed without it)
  --signed           use the signed-moment graphs (default: absolute |m|)
  -h, --help         show this help and the available backends

Available backends (see targets/${TARGET}/configs/README.md for descriptions):
USAGE
    if [[ -d "${config_root}" ]]; then
        for d in "${config_root}"/*/; do
            printf "  %s\n" "$(basename "${d}")"
        done
    fi
    echo
    echo "dimenet and visnet are heavy reference models, the rest are the candidates."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) print_help; exit 0 ;;
        --target)  TARGET="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --cutoffs) CUTOFFS="$2"; shift 2 ;;
        --cutoff)  CUTOFFS="$2"; shift 2 ;;
        --wandb)   WANDB_PASS="--wandb"; shift ;;
        --signed)  SIGNED_PASS="--signed"; shift ;;
        -*) echo "Unknown flag: $1 (try -h)" >&2; exit 1 ;;
        *) BACKENDS+=("$1"); shift ;;
    esac
done

[[ ${#BACKENDS[@]} -gt 0 ]] || {
    echo "Select at least one backend, e.g. cgconv schnetconv gatv2conv (see -h)" >&2
    exit 1
}

readonly TARGET DATASET CUTOFFS
readonly CONFIG_ROOT="targets/${TARGET}/configs"

# Resolve and validate every (backend, cutoff) before submitting anything.
configs=()
for backend in "${BACKENDS[@]}"; do
    for r in ${CUTOFFS//,/ }; do
        cfg="${CONFIG_ROOT}/${backend}/c128l3h1-r${r}.yaml"
        [[ -f "${cfg}" ]] || { echo "No config for ${backend} at r${r}: ${cfg} (try -h)" >&2; exit 1; }
        configs+=("${cfg}")
    done
done

printf "Submitting %d config(s) for %s on %s:\n" "${#configs[@]}" "${TARGET}" "${DATASET}"
for cfg in "${configs[@]}"; do
    printf "  %s\n" "${cfg}"
    ./scripts/02-submit-training.sh --target "${TARGET}" --dataset "${DATASET}" ${SIGNED_PASS} ${WANDB_PASS} "${cfg}"
done
