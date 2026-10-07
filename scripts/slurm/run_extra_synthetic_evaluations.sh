#!/bin/bash
#SBATCH --job-name=extra_synth_eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/prjs1237/project/tabpfgen/logs/extra_synthetic_evaluations/%j.out
#SBATCH --error=/projects/prjs1237/project/tabpfgen/logs/extra_synthetic_evaluations/%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/projects/prjs1237/project/tabpfgen/TabPFGen}"
CONDA_ENV="${CONDA_ENV:-tabpfgen}"
GENERATOR_ROOT="${GENERATOR_ROOT:-${REPO_ROOT}/outputs/table1_generators}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/analysis_extra}"
MAX_ROWS="${MAX_ROWS:-2000}"
RANDOM_STATE="${RANDOM_STATE:-0}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

mkdir -p "${REPO_ROOT}/logs/extra_synthetic_evaluations" "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

python scripts/run_extra_synthetic_evaluations.py \
  --generator-root "${GENERATOR_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-rows "${MAX_ROWS}" \
  --random-state "${RANDOM_STATE}"
