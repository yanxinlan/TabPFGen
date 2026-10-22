#!/bin/bash
set -euo pipefail

REPO_ROOT="/projects/prjs1237/project/tabpfgen/TabPFGen"
OVERWRITE_FLAG="--overwrite"

mkdir -p \
  "/projects/prjs1237/project/tabpfgen/logs/table1_generate" \
  "/projects/prjs1237/project/tabpfgen/logs/table1_evaluate_after_rtvae"

cd "${REPO_ROOT}"

# Generic generation task IDs use RTVAE generator index 4 out of 7.
RTVAE_GENERATION_TASKS=()
for ((dataset_idx = 0; dataset_idx < 18; dataset_idx++)); do
  for ((seed_idx = 0; seed_idx < 3; seed_idx++)); do
    RTVAE_GENERATION_TASKS+=("$((dataset_idx * 21 + seed_idx * 7 + 4))")
  done
done
GENERATION_ARRAY=$(IFS=,; echo "${RTVAE_GENERATION_TASKS[*]}")

GEN_JOB_ID=$(
  sbatch \
    --parsable \
    --array="${GENERATION_ARRAY}%30" \
    --export=ALL,OVERWRITE_FLAG="${OVERWRITE_FLAG}" \
    scripts/slurm/run_table1_generate_gpu_array.sh
)
echo "Submitted 54 RTVAE generation tasks: ${GEN_JOB_ID}"

EVAL_JOB_ID=$(
  sbatch \
    --parsable \
    --dependency=afterok:${GEN_JOB_ID} \
    --export=ALL,OVERWRITE_FLAG="${OVERWRITE_FLAG}" \
    scripts/slurm/run_table1_evaluate_all_gpu_array.sh
)
echo "Submitted 810 dependent evaluation tasks covering all 3,240 evaluations: ${EVAL_JOB_ID}"
echo "Total submitted array tasks: 864"
