#!/bin/bash
#SBATCH --job-name=tabpfgen_t1_eval
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --array=0-999%120
#SBATCH --output=/projects/prjs1237/project/tabpfgen/logs/table1_evaluate/%A_%a.out
#SBATCH --error=/projects/prjs1237/project/tabpfgen/logs/table1_evaluate/%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/projects/prjs1237/project/tabpfgen/TabPFGen}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/openml_cc18_tabpfgen}"
GENERATOR_ROOT="${GENERATOR_ROOT:-${REPO_ROOT}/outputs/table1_generators}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/table1_downstream}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
OVERWRITE_FLAG="${OVERWRITE_FLAG:-}"
TASK_OFFSET="${TASK_OFFSET:-0}"

DATASET_IDS=(11 14 16 18 22 37 54 458 1049 1050 1063 1068 1462 1464 1494 1510 40982 40994)
SEEDS=(0 1 2)
GENERATORS=(smote ctgan tvae nflow rtvae tabddpm tabpfgen)
DOWNSTREAM_MODELS=(xgb rf lr tabpfn)
MODES=(augmentation replacement)

N_DATASETS=${#DATASET_IDS[@]}
N_SEEDS=${#SEEDS[@]}
N_GENERATORS=${#GENERATORS[@]}
N_DOWNSTREAM=${#DOWNSTREAM_MODELS[@]}
N_MODES=${#MODES[@]}
N_ORIGINAL=$((N_DATASETS * N_SEEDS * N_DOWNSTREAM))
N_SYNTHETIC=$((N_DATASETS * N_SEEDS * N_GENERATORS * N_MODES * N_DOWNSTREAM))
TOTAL=$((N_ORIGINAL + N_SYNTHETIC))

TASK_ID=$((${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required} + TASK_OFFSET))
if (( TASK_ID < 0 || TASK_ID >= TOTAL )); then
  echo "Task id ${TASK_ID} outside [0, $((TOTAL - 1))]; local=${SLURM_ARRAY_TASK_ID} offset=${TASK_OFFSET}"
  exit 1
fi

if (( TASK_ID < N_ORIGINAL )); then
  MODEL_IDX=$((TASK_ID % N_DOWNSTREAM))
  TMP=$((TASK_ID / N_DOWNSTREAM))
  SEED_IDX=$((TMP % N_SEEDS))
  DATASET_IDX=$((TMP / N_SEEDS))
  DATASET_ID="${DATASET_IDS[$DATASET_IDX]}"
  SEED="${SEEDS[$SEED_IDX]}"
  GENERATOR="original"
  MODE="augmentation"
  DOWNSTREAM_MODEL="${DOWNSTREAM_MODELS[$MODEL_IDX]}"
else
  SYN_ID=$((TASK_ID - N_ORIGINAL))
  MODEL_IDX=$((SYN_ID % N_DOWNSTREAM))
  TMP=$((SYN_ID / N_DOWNSTREAM))
  MODE_IDX=$((TMP % N_MODES))
  TMP=$((TMP / N_MODES))
  GEN_IDX=$((TMP % N_GENERATORS))
  TMP=$((TMP / N_GENERATORS))
  SEED_IDX=$((TMP % N_SEEDS))
  DATASET_IDX=$((TMP / N_SEEDS))
  DATASET_ID="${DATASET_IDS[$DATASET_IDX]}"
  SEED="${SEEDS[$SEED_IDX]}"
  GENERATOR="${GENERATORS[$GEN_IDX]}"
  MODE="${MODES[$MODE_IDX]}"
  DOWNSTREAM_MODEL="${DOWNSTREAM_MODELS[$MODEL_IDX]}"
fi

echo "REPO_ROOT=${REPO_ROOT}"
echo "DATASET_ID=${DATASET_ID} SEED=${SEED} GENERATOR=${GENERATOR} MODE=${MODE} DOWNSTREAM_MODEL=${DOWNSTREAM_MODEL}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

cd "${REPO_ROOT}"

"${PYTHON_BIN}" scripts/evaluate_table1_downstream.py \
  --data-root "${DATA_ROOT}" \
  --generator-root "${GENERATOR_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --dataset-id "${DATASET_ID}" \
  --generator "${GENERATOR}" \
  --mode "${MODE}" \
  --downstream-model "${DOWNSTREAM_MODEL}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  ${OVERWRITE_FLAG}
