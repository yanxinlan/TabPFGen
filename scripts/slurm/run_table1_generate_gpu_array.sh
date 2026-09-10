#!/bin/bash
#SBATCH --job-name=tabpfgen_t1_gen
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-377%60
#SBATCH --output=/projects/prjs1237/project/tabpfgen/logs/table1_generate/%A_%a.out
#SBATCH --error=/projects/prjs1237/project/tabpfgen/logs/table1_generate/%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/projects/prjs1237/project/tabpfgen/TabPFGen}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/openml_cc18_tabpfgen}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/table1_generators}"
CONDA_BASE="${CONDA_BASE:-${HOME}/anaconda3}"
TABPFGEN_CONDA_ENV="${TABPFGEN_CONDA_ENV:-tabpfgen}"
SYNTHCITY_CONDA_ENV="${SYNTHCITY_CONDA_ENV:-synthcity}"
DEVICE="${DEVICE:-cuda}"
N_ITER="${N_ITER:-1000}"
TABPFGEN_STEPS="${TABPFGEN_STEPS:-1000}"
TABPFGEN_STEP_SIZE="${TABPFGEN_STEP_SIZE:-0.01}"
TABPFGEN_NOISE_SCALE="${TABPFGEN_NOISE_SCALE:-0.01}"
TABPFGEN_INIT_NOISE_STD="${TABPFGEN_INIT_NOISE_STD:-0.01}"
OVERWRITE_FLAG="${OVERWRITE_FLAG:-}"

DATASET_IDS=(11 14 16 18 22 37 54 458 1049 1050 1063 1068 1462 1464 1494 1510 40982 40994)
SEEDS=(0 1 2)
GENERATORS=(smote ctgan tvae nflow rtvae tabddpm tabpfgen)

N_DATASETS=${#DATASET_IDS[@]}
N_SEEDS=${#SEEDS[@]}
N_GENERATORS=${#GENERATORS[@]}
TOTAL=$((N_DATASETS * N_SEEDS * N_GENERATORS))

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if (( TASK_ID < 0 || TASK_ID >= TOTAL )); then
  echo "Task id ${TASK_ID} outside [0, $((TOTAL - 1))]"
  exit 1
fi

GEN_IDX=$((TASK_ID % N_GENERATORS))
TMP=$((TASK_ID / N_GENERATORS))
SEED_IDX=$((TMP % N_SEEDS))
DATASET_IDX=$((TMP / N_SEEDS))

DATASET_ID="${DATASET_IDS[$DATASET_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
GENERATOR="${GENERATORS[$GEN_IDX]}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
if [[ "${GENERATOR}" == "tabpfgen" ]]; then
  CONDA_ENV="${TABPFGEN_CONDA_ENV}"
else
  CONDA_ENV="${SYNTHCITY_CONDA_ENV}"
fi
conda activate "${CONDA_ENV}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "DATASET_ID=${DATASET_ID} SEED=${SEED} GENERATOR=${GENERATOR}"
echo "CONDA_ENV=${CONDA_ENV}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

cd "${REPO_ROOT}"

"${PYTHON_BIN}" scripts/train_table1_generators.py \
  --data-root "${DATA_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --dataset-id "${DATASET_ID}" \
  --generator "${GENERATOR}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --n-iter "${N_ITER}" \
  --tabpfgen-steps "${TABPFGEN_STEPS}" \
  --tabpfgen-step-size "${TABPFGEN_STEP_SIZE}" \
  --tabpfgen-noise-scale "${TABPFGEN_NOISE_SCALE}" \
  --tabpfgen-init-noise-std "${TABPFGEN_INIT_NOISE_STD}" \
  ${OVERWRITE_FLAG}
