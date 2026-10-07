#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
LOG_ROOT="${LOG_ROOT:-$(cd -- "${REPO_ROOT}/.." && pwd)/logs}"

mkdir -p "${LOG_ROOT}/table1_missing_generate" "${LOG_ROOT}/table1_missing_evaluate"

cd "${REPO_ROOT}"

GEN_JOB_ID=$(
  sbatch \
    --parsable \
    --export=ALL,REPO_ROOT="${REPO_ROOT}" \
    scripts/slurm/run_table1_missing_generate_gpu_array.sh
)
echo "Submitted missing TabPFGen diabetes regeneration: ${GEN_JOB_ID}"

EVAL_JOB_ID=$(
  sbatch \
    --parsable \
    --dependency=afterok:${GEN_JOB_ID} \
    --export=ALL,REPO_ROOT="${REPO_ROOT}" \
    scripts/slurm/run_table1_missing_evaluate_gpu_array.sh
)
echo "Submitted 13 missing downstream evaluations: ${EVAL_JOB_ID}"
