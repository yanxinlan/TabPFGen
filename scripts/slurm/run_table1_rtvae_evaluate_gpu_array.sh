#!/bin/bash
#SBATCH --job-name=rtvae_only_eval
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --array=0-431%120
#SBATCH --output=/projects/prjs1237/project/tabpfgen/logs/table1_rtvae_evaluate/%A_%a.out
#SBATCH --error=/projects/prjs1237/project/tabpfgen/logs/table1_rtvae_evaluate/%A_%a.err

set -euo pipefail

REPO_ROOT="/projects/prjs1237/project/tabpfgen/TabPFGen"
DATA_ROOT="${REPO_ROOT}/data/openml_cc18_tabpfgen"
GENERATOR_ROOT="${REPO_ROOT}/outputs/table1_generators"
OUTPUT_ROOT="${REPO_ROOT}/outputs/table1_downstream"
CONDA_BASE="${CONDA_BASE:-${HOME}/anaconda3}"
CONDA_ENV="${CONDA_ENV:-tabpfgen}"
DEVICE="${DEVICE:-cuda}"
TABPFN_PAPER_ROOT="${REPO_ROOT}/third_party/tabpfn_paper"
TABPFN_PAPER_CHECKPOINT="${TABPFN_PAPER_ROOT}/tabpfn/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt"
TABPFN_N_ENSEMBLES="${TABPFN_N_ENSEMBLES:-3}"
OVERWRITE_FLAG="${OVERWRITE_FLAG:---overwrite}"

DATASET_IDS=(11 14 16 18 22 37 54 458 1049 1050 1063 1068 1462 1464 1494 1510 40982 40994)
SEEDS=(0 1 2)
MODES=(augmentation replacement)
DOWNSTREAM_MODELS=(xgb rf lr tabpfn)

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
TOTAL=$((${#DATASET_IDS[@]} * ${#SEEDS[@]} * ${#MODES[@]} * ${#DOWNSTREAM_MODELS[@]}))
if (( TASK_ID < 0 || TASK_ID >= TOTAL )); then
  echo "Task id ${TASK_ID} outside [0, $((TOTAL - 1))]" >&2
  exit 1
fi

MODEL_IDX=$((TASK_ID % ${#DOWNSTREAM_MODELS[@]}))
TMP=$((TASK_ID / ${#DOWNSTREAM_MODELS[@]}))
MODE_IDX=$((TMP % ${#MODES[@]}))
TMP=$((TMP / ${#MODES[@]}))
SEED_IDX=$((TMP % ${#SEEDS[@]}))
DATASET_IDX=$((TMP / ${#SEEDS[@]}))

DATASET_ID="${DATASET_IDS[$DATASET_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
MODE="${MODES[$MODE_IDX]}"
DOWNSTREAM_MODEL="${DOWNSTREAM_MODELS[$MODEL_IDX]}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
PYTHON_BIN="${CONDA_PREFIX}/bin/python"

mkdir -p "/projects/prjs1237/project/tabpfgen/logs/table1_rtvae_evaluate"
cd "${REPO_ROOT}"

echo "DATASET_ID=${DATASET_ID} SEED=${SEED} GENERATOR=rtvae MODE=${MODE} DOWNSTREAM_MODEL=${DOWNSTREAM_MODEL}"

"${PYTHON_BIN}" scripts/evaluate_table1_downstream.py \
  --data-root "${DATA_ROOT}" \
  --generator-root "${GENERATOR_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --dataset-id "${DATASET_ID}" \
  --generator rtvae \
  --mode "${MODE}" \
  --downstream-model "${DOWNSTREAM_MODEL}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --tabpfn-source-root "${TABPFN_PAPER_ROOT}" \
  --tabpfn-model-path "${TABPFN_PAPER_CHECKPOINT}" \
  --tabpfn-n-ensembles "${TABPFN_N_ENSEMBLES}" \
  ${OVERWRITE_FLAG}
