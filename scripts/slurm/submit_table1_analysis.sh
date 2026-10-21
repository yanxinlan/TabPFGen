#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="/projects/prjs1237/project/tabpfgen/TabPFGen"
LOG_ROOT="${LOG_ROOT:-$(cd -- "${REPO_ROOT}/.." && pwd)/logs}"

mkdir -p "${LOG_ROOT}/table1_analysis"

cd "${REPO_ROOT}"

JOB_ID=$(
  sbatch \
    --parsable \
    --export=ALL,REPO_ROOT="${REPO_ROOT}" \
    scripts/slurm/run_table1_analysis.sh
)
echo "Submitted analysis job: ${JOB_ID}"
