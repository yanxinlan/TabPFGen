#!/bin/bash
#SBATCH --job-name=tabpfgen_grad_health
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --array=0-53%30
#SBATCH --output=/projects/prjs1237/project/tabpfgen/logs/tabpfgen_gradient_health/%A_%a.out
#SBATCH --error=/projects/prjs1237/project/tabpfgen/logs/tabpfgen_gradient_health/%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/projects/prjs1237/project/tabpfgen/TabPFGen}"
CONDA_ENV="${CONDA_ENV:-tabpfgen}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/openml_cc18_tabpfgen}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/gradient_health}"

N_SAMPLES="${N_SAMPLES:-256}"
TABPFGEN_STEPS="${TABPFGEN_STEPS:-1000}"
TABPFGEN_STEP_SIZE="${TABPFGEN_STEP_SIZE:-0.01}"
TABPFGEN_NOISE_SCALE="${TABPFGEN_NOISE_SCALE:-0.01}"
TABPFGEN_INIT_NOISE_STD="${TABPFGEN_INIT_NOISE_STD:-0.01}"
TABPFGEN_GRADIENT_CLIP="${TABPFGEN_GRADIENT_CLIP:-0}"
OVERWRITE_FLAG="${OVERWRITE_FLAG:-}"

DATASET_IDS=(11 14 16 18 22 37 54 458 1049 1050 1063 1068 1462 1464 1494 1510 40982 40994)
SEEDS=(0 1 2)

DATASET_INDEX=$((SLURM_ARRAY_TASK_ID / ${#SEEDS[@]}))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % ${#SEEDS[@]}))
DATASET_ID="${DATASET_IDS[$DATASET_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

mkdir -p "${REPO_ROOT}/logs/tabpfgen_gradient_health" "${OUTPUT_ROOT}"
cd "${REPO_ROOT}"

python scripts/run_tabpfgen_gradient_health.py \
  --data-root "${DATA_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --dataset-id "${DATASET_ID}" \
  --seed "${SEED}" \
  --device cuda \
  --n-samples "${N_SAMPLES}" \
  --steps "${TABPFGEN_STEPS}" \
  --step-size "${TABPFGEN_STEP_SIZE}" \
  --noise-scale "${TABPFGEN_NOISE_SCALE}" \
  --init-noise-std "${TABPFGEN_INIT_NOISE_STD}" \
  --gradient-clip "${TABPFGEN_GRADIENT_CLIP}" \
  ${OVERWRITE_FLAG}
