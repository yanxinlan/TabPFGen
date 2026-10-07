#!/bin/bash
#SBATCH --job-name=tabpfgen_t1_analysis
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/prjs1237/project/tabpfgen/logs/table1_analysis/%j.out
#SBATCH --error=/projects/prjs1237/project/tabpfgen/logs/table1_analysis/%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
GENERATOR_ROOT="${GENERATOR_ROOT:-${REPO_ROOT}/outputs/table1_generators}"
DOWNSTREAM_ROOT="${DOWNSTREAM_ROOT:-${REPO_ROOT}/outputs/table1_downstream}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/analysis}"
CONDA_BASE="${CONDA_BASE:-${HOME}/anaconda3}"
CONDA_ENV="${CONDA_ENV:-tabpfgen}"
SKIP_C2ST_FLAG="${SKIP_C2ST_FLAG:-}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
PYTHON_BIN="${CONDA_PREFIX}/bin/python"

cd "${REPO_ROOT}"

"${PYTHON_BIN}" scripts/analyze_reproduction_results.py \
  --generator-root "${GENERATOR_ROOT}" \
  --downstream-root "${DOWNSTREAM_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  ${SKIP_C2ST_FLAG}
