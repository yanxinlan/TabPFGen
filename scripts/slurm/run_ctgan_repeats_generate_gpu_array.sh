#!/bin/bash
#SBATCH --job-name=ctgan_repeats_gen
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-539%60
#SBATCH --output=/projects/prjs1237/project/tabpfgen/logs/ctgan_repeats_generate/%A_%a.out
#SBATCH --error=/projects/prjs1237/project/tabpfgen/logs/ctgan_repeats_generate/%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/openml_cc18_tabpfgen}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/ctgan_repeats}"
CONDA_BASE="${CONDA_BASE:-${HOME}/anaconda3}"
CONDA_ENV="${CONDA_ENV:-synthcity}"
DEVICE="${DEVICE:-cuda}"
N_ITER="${N_ITER:-1000}"
OVERWRITE_FLAG="${OVERWRITE_FLAG:-}"

DATASET_IDS=(11 14 16 18 22 37 54 458 1049 1050 1063 1068 1462 1464 1494 1510 40982 40994)
SPLIT_SEEDS=(0 1 2)
CTGAN_SEEDS=(0 1 2 3 4 5 6 7 8 9)

N_DATASETS=${#DATASET_IDS[@]}
N_SPLIT_SEEDS=${#SPLIT_SEEDS[@]}
N_CTGAN_SEEDS=${#CTGAN_SEEDS[@]}
TOTAL=$((N_DATASETS * N_SPLIT_SEEDS * N_CTGAN_SEEDS))

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if (( TASK_ID < 0 || TASK_ID >= TOTAL )); then
  echo "Task id ${TASK_ID} outside [0, $((TOTAL - 1))]"
  exit 1
fi

CTGAN_IDX=$((TASK_ID % N_CTGAN_SEEDS))
TMP=$((TASK_ID / N_CTGAN_SEEDS))
SPLIT_IDX=$((TMP % N_SPLIT_SEEDS))
DATASET_IDX=$((TMP / N_SPLIT_SEEDS))

DATASET_ID="${DATASET_IDS[$DATASET_IDX]}"
SPLIT_SEED="${SPLIT_SEEDS[$SPLIT_IDX]}"
CTGAN_SEED="${CTGAN_SEEDS[$CTGAN_IDX]}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
PYTHON_BIN="${CONDA_PREFIX}/bin/python"

echo "DATASET_ID=${DATASET_ID} SPLIT_SEED=${SPLIT_SEED} CTGAN_SEED=${CTGAN_SEED}"
echo "CONDA_ENV=${CONDA_ENV}"

cd "${REPO_ROOT}"

"${PYTHON_BIN}" scripts/train_ctgan_repeat.py \
  --data-root "${DATA_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --dataset-id "${DATASET_ID}" \
  --split-seed "${SPLIT_SEED}" \
  --ctgan-seed "${CTGAN_SEED}" \
  --n-iter "${N_ITER}" \
  --device "${DEVICE}" \
  ${OVERWRITE_FLAG}
