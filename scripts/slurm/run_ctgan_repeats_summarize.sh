#!/bin/bash
#SBATCH --job-name=ctgan_repeats_summary
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/prjs1237/project/tabpfgen/logs/ctgan_repeats_summary/%j.out
#SBATCH --error=/projects/prjs1237/project/tabpfgen/logs/ctgan_repeats_summary/%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
CONDA_BASE="${CONDA_BASE:-${HOME}/anaconda3}"
CONDA_ENV="${CONDA_ENV:-tabpfgen}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${REPO_ROOT}"
"${CONDA_PREFIX}/bin/python" scripts/summarize_ctgan_repeats.py
