#!/bin/bash
#SBATCH --job-name=table1_missing_gen
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/projects/prjs1237/project/tabpfgen/logs/table1_missing_generate/%A_%a.out
#SBATCH --error=/projects/prjs1237/project/tabpfgen/logs/table1_missing_generate/%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/openml_cc18_tabpfgen}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/table1_generators}"
CONDA_BASE="${CONDA_BASE:-${HOME}/anaconda3}"
CONDA_ENV="${CONDA_ENV:-tabpfgen}"
DEVICE="${DEVICE:-cuda}"
TABPFGEN_STEPS="${TABPFGEN_STEPS:-1000}"
TABPFGEN_STEP_SIZE="${TABPFGEN_STEP_SIZE:-0.01}"
TABPFGEN_NOISE_SCALE="${TABPFGEN_NOISE_SCALE:-0.01}"
TABPFGEN_INIT_NOISE_STD="${TABPFGEN_INIT_NOISE_STD:-0.01}"

SEEDS=(0 1 2)
SEED="${SEEDS[${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}]}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
PYTHON_BIN="${CONDA_PREFIX}/bin/python"

cd "${REPO_ROOT}"

"${PYTHON_BIN}" scripts/train_table1_generators.py \
  --data-root "${DATA_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --dataset-id 37 \
  --generator tabpfgen \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --tabpfgen-steps "${TABPFGEN_STEPS}" \
  --tabpfgen-step-size "${TABPFGEN_STEP_SIZE}" \
  --tabpfgen-noise-scale "${TABPFGEN_NOISE_SCALE}" \
  --tabpfgen-init-noise-std "${TABPFGEN_INIT_NOISE_STD}" \
  --overwrite
