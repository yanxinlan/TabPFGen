#!/bin/bash
#SBATCH --job-name=tabpfgen_clsagn_gen
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-53%30
#SBATCH --output=/projects/prjs1237/project/tabpfgen/logs/table1_tabpfgen_class_agnostic_generate/%A_%a.out
#SBATCH --error=/projects/prjs1237/project/tabpfgen/logs/table1_tabpfgen_class_agnostic_generate/%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="/projects/prjs1237/project/tabpfgen/TabPFGen"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/openml_cc18_tabpfgen}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/table1_generators_class_agnostic_init}"
CONDA_BASE="${CONDA_BASE:-${HOME}/anaconda3}"
CONDA_ENV="${CONDA_ENV:-tabpfgen}"
DEVICE="${DEVICE:-cuda}"
TABPFGEN_STEPS="${TABPFGEN_STEPS:-1000}"
TABPFGEN_STEP_SIZE="${TABPFGEN_STEP_SIZE:-0.01}"
TABPFGEN_NOISE_SCALE="${TABPFGEN_NOISE_SCALE:-0.01}"
TABPFGEN_INIT_NOISE_STD="${TABPFGEN_INIT_NOISE_STD:-0.01}"
TABPFGEN_SELECTION_METRIC="${TABPFGEN_SELECTION_METRIC:-energy}"
TABPFGEN_SELECTION_INTERVAL="${TABPFGEN_SELECTION_INTERVAL:-1}"
TABPFGEN_LABEL_STRATEGY="${TABPFGEN_LABEL_STRATEGY:-exact_train_ratio}"
TABPFGEN_SWAPPED_ENERGY_WEIGHT="${TABPFGEN_SWAPPED_ENERGY_WEIGHT:-1.0}"
TABPFGEN_SCALE_FLAG="${TABPFGEN_SCALE_FLAG:-}"
OVERWRITE_FLAG="${OVERWRITE_FLAG:-}"

DATASET_IDS=(11 14 16 18 22 37 54 458 1049 1050 1063 1068 1462 1464 1494 1510 40982 40994)
SEEDS=(0 1 2)

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
DATASET_IDX=$((TASK_ID / ${#SEEDS[@]}))
SEED_IDX=$((TASK_ID % ${#SEEDS[@]}))
DATASET_ID="${DATASET_IDS[$DATASET_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
PYTHON_BIN="${CONDA_PREFIX}/bin/python"

mkdir -p "${REPO_ROOT}/logs/table1_tabpfgen_class_agnostic_generate"
cd "${REPO_ROOT}"

"${PYTHON_BIN}" scripts/train_table1_generators.py \
  --data-root "${DATA_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --dataset-id "${DATASET_ID}" \
  --generator tabpfgen \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --tabpfgen-steps "${TABPFGEN_STEPS}" \
  --tabpfgen-step-size "${TABPFGEN_STEP_SIZE}" \
  --tabpfgen-noise-scale "${TABPFGEN_NOISE_SCALE}" \
  --tabpfgen-init-noise-std "${TABPFGEN_INIT_NOISE_STD}" \
  --tabpfgen-init-strategy class_agnostic \
  --tabpfgen-selection-metric "${TABPFGEN_SELECTION_METRIC}" \
  --tabpfgen-selection-interval "${TABPFGEN_SELECTION_INTERVAL}" \
  --tabpfgen-label-strategy "${TABPFGEN_LABEL_STRATEGY}" \
  --tabpfgen-swapped-energy-weight "${TABPFGEN_SWAPPED_ENERGY_WEIGHT}" \
  ${TABPFGEN_SCALE_FLAG} \
  ${OVERWRITE_FLAG}
