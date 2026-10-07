#!/bin/bash
#SBATCH --job-name=table1_missing_eval
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --array=0-12%13
#SBATCH --output=/projects/prjs1237/project/tabpfgen/logs/table1_missing_evaluate/%A_%a.out
#SBATCH --error=/projects/prjs1237/project/tabpfgen/logs/table1_missing_evaluate/%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/openml_cc18_tabpfgen}"
GENERATOR_ROOT="${GENERATOR_ROOT:-${REPO_ROOT}/outputs/table1_generators}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/table1_downstream}"
CONDA_BASE="${CONDA_BASE:-${HOME}/anaconda3}"
CONDA_ENV="${CONDA_ENV:-tabpfgen}"
DEVICE="${DEVICE:-cuda}"

ROWS=(
  "37 0 tabpfgen augmentation lr"
  "37 0 tabpfgen replacement lr"
  "37 0 tabpfgen replacement tabpfn"
  "37 1 tabpfgen augmentation lr"
  "37 1 tabpfgen replacement lr"
  "37 1 tabpfgen replacement tabpfn"
  "37 2 tabpfgen augmentation lr"
  "37 2 tabpfgen replacement lr"
  "37 2 tabpfgen replacement tabpfn"
  "40994 0 ctgan replacement xgb"
  "40994 0 ctgan replacement rf"
  "40994 0 ctgan replacement lr"
  "40994 0 ctgan replacement tabpfn"
)

read -r DATASET_ID SEED GENERATOR MODE DOWNSTREAM_MODEL <<< "${ROWS[${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}]}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
PYTHON_BIN="${CONDA_PREFIX}/bin/python"

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
  --overwrite
