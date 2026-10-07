#!/bin/bash
#SBATCH --job-name=privacy_attacks
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=/projects/prjs1237/project/tabpfgen/logs/privacy_attacks/%j.out
#SBATCH --error=/projects/prjs1237/project/tabpfgen/logs/privacy_attacks/%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/projects/prjs1237/project/tabpfgen/TabPFGen}"
CONDA_ENV="${CONDA_ENV:-tabpfgen}"
GENERATOR_ROOT="${GENERATOR_ROOT:-${REPO_ROOT}/outputs/table1_generators}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/privacy_attacks}"
MAX_ROWS="${MAX_ROWS:-2000}"
RANDOM_STATE="${RANDOM_STATE:-0}"
N_ATTRIBUTE_FEATURES="${N_ATTRIBUTE_FEATURES:-10}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

mkdir -p "${REPO_ROOT}/logs/privacy_attacks" "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

python scripts/run_privacy_attacks.py \
  --generator-root "${GENERATOR_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-rows "${MAX_ROWS}" \
  --random-state "${RANDOM_STATE}" \
  --n-attribute-features "${N_ATTRIBUTE_FEATURES}"
