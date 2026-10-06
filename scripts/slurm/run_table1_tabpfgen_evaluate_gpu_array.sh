#!/bin/bash
#SBATCH --job-name=tabpfgen_only_eval
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --array=0-431%120
#SBATCH --output=/projects/prjs1237/project/tabpfgen/logs/table1_tabpfgen_evaluate/%A_%a.out
#SBATCH --error=/projects/prjs1237/project/tabpfgen/logs/table1_tabpfgen_evaluate/%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/openml_cc18_tabpfgen}"
GENERATOR_ROOT="${GENERATOR_ROOT:-${REPO_ROOT}/outputs/table1_generators}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/table1_downstream}"
CONDA_BASE="${CONDA_BASE:-${HOME}/anaconda3}"
CONDA_ENV="${CONDA_ENV:-tabpfgen}"
DEVICE="${DEVICE:-cuda}"
OVERWRITE_FLAG="${OVERWRITE_FLAG:-}"

DATASET_IDS=(11 14 16 18 22 37 54 458 1049 1050 1063 1068 1462 1464 1494 1510 40982 40994)
SEEDS=(0 1 2)
MODES=(augmentation replacement)
DOWNSTREAM_MODELS=(xgb rf lr tabpfn)

N_DATASETS=${#DATASET_IDS[@]}
N_SEEDS=${#SEEDS[@]}
N_MODES=${#MODES[@]}
N_DOWNSTREAM=${#DOWNSTREAM_MODELS[@]}
TOTAL=$((N_DATASETS * N_SEEDS * N_MODES * N_DOWNSTREAM))

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if (( TASK_ID < 0 || TASK_ID >= TOTAL )); then
  echo "Task id ${TASK_ID} outside [0, $((TOTAL - 1))]"
  exit 1
fi

MODEL_IDX=$((TASK_ID % N_DOWNSTREAM))
TMP=$((TASK_ID / N_DOWNSTREAM))
MODE_IDX=$((TMP % N_MODES))
TMP=$((TMP / N_MODES))
SEED_IDX=$((TMP % N_SEEDS))
DATASET_IDX=$((TMP / N_SEEDS))

DATASET_ID="${DATASET_IDS[$DATASET_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
MODE="${MODES[$MODE_IDX]}"
DOWNSTREAM_MODEL="${DOWNSTREAM_MODELS[$MODEL_IDX]}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
PYTHON_BIN="${CONDA_PREFIX}/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found for Conda environment ${CONDA_ENV}: ${PYTHON_BIN}" >&2
  exit 1
fi

echo "REPO_ROOT=${REPO_ROOT}"
echo "DATASET_ID=${DATASET_ID} SEED=${SEED} GENERATOR=tabpfgen MODE=${MODE} DOWNSTREAM_MODEL=${DOWNSTREAM_MODEL}"
echo "CONDA_ENV=${CONDA_ENV}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

cd "${REPO_ROOT}"

"${PYTHON_BIN}" scripts/evaluate_table1_downstream.py \
  --data-root "${DATA_ROOT}" \
  --generator-root "${GENERATOR_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --dataset-id "${DATASET_ID}" \
  --generator tabpfgen \
  --mode "${MODE}" \
  --downstream-model "${DOWNSTREAM_MODEL}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  ${OVERWRITE_FLAG}
