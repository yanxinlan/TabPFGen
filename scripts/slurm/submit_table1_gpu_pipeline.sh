#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/projects/prjs1237/project/tabpfgen/TabPFGen}"
LOG_ROOT="${LOG_ROOT:-/projects/prjs1237/project/tabpfgen/logs}"

mkdir -p "${LOG_ROOT}/table1_generate" "${LOG_ROOT}/table1_evaluate"

cd "${REPO_ROOT}"

GEN_JOB_ID=$(sbatch --parsable scripts/slurm/run_table1_generate_gpu_array.sh)
echo "Submitted generation array: ${GEN_JOB_ID}"

TOTAL_EVAL_TASKS=3240
MAX_ARRAY_TASKS=1000
EVAL_JOB_IDS=()

offset=0
while (( offset < TOTAL_EVAL_TASKS )); do
  remaining=$((TOTAL_EVAL_TASKS - offset))
  chunk_size=${MAX_ARRAY_TASKS}
  if (( remaining < chunk_size )); then
    chunk_size=${remaining}
  fi
  array_end=$((chunk_size - 1))
  eval_job_id=$(
    sbatch \
      --parsable \
      --dependency=afterok:${GEN_JOB_ID} \
      --array=0-${array_end}%120 \
      --export=ALL,TASK_OFFSET=${offset} \
      scripts/slurm/run_table1_evaluate_gpu_array.sh
  )
  EVAL_JOB_IDS+=("${eval_job_id}")
  echo "Submitted evaluation array chunk offset=${offset} size=${chunk_size}: ${eval_job_id}"
  offset=$((offset + chunk_size))
done

echo "Evaluation chunks: ${EVAL_JOB_IDS[*]}"
