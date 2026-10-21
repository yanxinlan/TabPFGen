#!/bin/bash
#SBATCH --job-name=tabpfgen_diabetes_trace
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/prjs1237/project/tabpfgen/logs/diabetes_operation_trace/%j.out
#SBATCH --error=/projects/prjs1237/project/tabpfgen/logs/diabetes_operation_trace/%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="/projects/prjs1237/project/tabpfgen/TabPFGen"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/openml_cc18_tabpfgen}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/diabetes_operation_trace}"
CONDA_ENV="${CONDA_ENV:-tabpfgen}"
TABPFN_PAPER_ROOT="${TABPFN_PAPER_ROOT:-${REPO_ROOT}/third_party/tabpfn_paper}"
TABPFN_PAPER_CHECKPOINT="${TABPFN_PAPER_CHECKPOINT:-${TABPFN_PAPER_ROOT}/tabpfn/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt}"
TRACE_SEED="${TRACE_SEED:-0}"
TRACE_STEPS="${TRACE_STEPS:-1}"
TRACE_SAVE_MODE="${TRACE_SAVE_MODE:-full}"
OVERWRITE_FLAG="${OVERWRITE_FLAG:-}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
mkdir -p "${REPO_ROOT}/logs/diabetes_operation_trace" "${OUTPUT_ROOT}"
cd "${REPO_ROOT}"

python scripts/debug_diabetes_tabpfgen_operations.py \
  --data-root "${DATA_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --seed "${TRACE_SEED}" \
  --steps "${TRACE_STEPS}" \
  --device cuda \
  --n-ensembles 3 \
  --tabpfn-source-root "${TABPFN_PAPER_ROOT}" \
  --tabpfn-model-path "${TABPFN_PAPER_CHECKPOINT}" \
  --save-mode "${TRACE_SAVE_MODE}" \
  ${OVERWRITE_FLAG}
