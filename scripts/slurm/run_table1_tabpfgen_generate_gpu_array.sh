#!/bin/bash
#SBATCH --job-name=tabpfgen_only_gen
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-53%30
#SBATCH --output=/projects/prjs1237/project/tabpfgen/logs/table1_tabpfgen_generate/%A_%a.out
#SBATCH --error=/projects/prjs1237/project/tabpfgen/logs/table1_tabpfgen_generate/%A_%a.err

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
OVERWRITE_FLAG="${OVERWRITE_FLAG:-}"

DATASET_IDS=(11 14 16 18 22 37 54 458 1049 1050 1063 1068 1462 1464 1494 1510 40982 40994)
SEEDS=(0 1 2)

N_DATASETS=${#DATASET_IDS[@]}
N_SEEDS=${#SEEDS[@]}
TOTAL=$((N_DATASETS * N_SEEDS))

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if (( TASK_ID < 0 || TASK_ID >= TOTAL )); then
  echo "Task id ${TASK_ID} outside [0, $((TOTAL - 1))]"
  exit 1
fi

SEED_IDX=$((TASK_ID % N_SEEDS))
DATASET_IDX=$((TASK_ID / N_SEEDS))
DATASET_ID="${DATASET_IDS[$DATASET_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
PYTHON_BIN="${CONDA_PREFIX}/bin/python"
CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/tabpfgen-${SLURM_JOB_ID:-$$}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${CACHE_ROOT}/matplotlib}"
export KEOPS_CACHE_FOLDER="${KEOPS_CACHE_FOLDER:-${CACHE_ROOT}/keops}"
mkdir -p "${MPLCONFIGDIR}" "${KEOPS_CACHE_FOLDER}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found for Conda environment ${CONDA_ENV}: ${PYTHON_BIN}" >&2
  exit 1
fi

echo "REPO_ROOT=${REPO_ROOT}"
echo "DATASET_ID=${DATASET_ID} SEED=${SEED} GENERATOR=tabpfgen"
echo "CONDA_ENV=${CONDA_ENV}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

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
  ${OVERWRITE_FLAG}
