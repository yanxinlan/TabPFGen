#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
LOG_ROOT="${LOG_ROOT:-$(cd -- "${REPO_ROOT}/.." && pwd)/logs}"
OVERWRITE_FLAG="${OVERWRITE_FLAG:---overwrite}"

mkdir -p "${LOG_ROOT}/table1_tabpfgen_generate" "${LOG_ROOT}/table1_tabpfgen_evaluate"

cd "${REPO_ROOT}"

GEN_JOB_ID=$(
  sbatch \
    --parsable \
    --export=ALL,REPO_ROOT="${REPO_ROOT}",OVERWRITE_FLAG="${OVERWRITE_FLAG}" \
    scripts/slurm/run_table1_tabpfgen_generate_gpu_array.sh
)
echo "Submitted TabPFGen generation array: ${GEN_JOB_ID}"

EVAL_JOB_ID=$(
  sbatch \
    --parsable \
    --dependency=afterok:${GEN_JOB_ID} \
    --export=ALL,REPO_ROOT="${REPO_ROOT}",OVERWRITE_FLAG="${OVERWRITE_FLAG}" \
    scripts/slurm/run_table1_tabpfgen_evaluate_gpu_array.sh
)
echo "Submitted TabPFGen evaluation array: ${EVAL_JOB_ID}"
