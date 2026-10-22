#!/bin/bash
set -euo pipefail

REPO_ROOT="/projects/prjs1237/project/tabpfgen/TabPFGen"
OVERWRITE_FLAG="--overwrite"
DEVICE="${DEVICE:-cuda}"
if [[ "${DEVICE}" == "cpu" ]]; then
  DEFAULT_PARTITION="rome"
  DEFAULT_GRES="none"
else
  DEFAULT_PARTITION="gpu_h100"
  DEFAULT_GRES="gpu:1"
fi
TARGET_PARTITION="${TARGET_PARTITION:-${DEFAULT_PARTITION}}"
TARGET_NODE="${TARGET_NODE:-}"
TARGET_GRES="${TARGET_GRES-${DEFAULT_GRES}}"
TARGET_TIME="${TARGET_TIME:-}"

SBATCH_RESOURCE_ARGS=(--partition="${TARGET_PARTITION}")
if [[ -n "${TARGET_NODE}" ]]; then
  SBATCH_RESOURCE_ARGS+=(--nodelist="${TARGET_NODE}")
fi
if [[ -n "${TARGET_GRES}" ]]; then
  SBATCH_RESOURCE_ARGS+=(--gres="${TARGET_GRES}")
fi
if [[ -n "${TARGET_TIME}" ]]; then
  SBATCH_RESOURCE_ARGS+=(--time="${TARGET_TIME}")
fi

mkdir -p \
  "/projects/prjs1237/project/tabpfgen/logs/table1_generate" \
  "/projects/prjs1237/project/tabpfgen/logs/table1_evaluate_after_rtvae"

cd "${REPO_ROOT}"

echo "Submission resources: partition=${TARGET_PARTITION} node=${TARGET_NODE:-scheduler-selected} gres=${TARGET_GRES:-none} device=${DEVICE} time=${TARGET_TIME:-script-default}"

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
    "${SBATCH_RESOURCE_ARGS[@]}" \
    --array="${GENERATION_ARRAY}%30" \
    --export=ALL,OVERWRITE_FLAG="${OVERWRITE_FLAG}",DEVICE="${DEVICE}" \
    scripts/slurm/run_table1_generate_gpu_array.sh
)
echo "Submitted 54 RTVAE generation tasks: ${GEN_JOB_ID}"

EVAL_JOB_ID=$(
  sbatch \
    --parsable \
    "${SBATCH_RESOURCE_ARGS[@]}" \
    --dependency=afterok:${GEN_JOB_ID} \
    --export=ALL,OVERWRITE_FLAG="${OVERWRITE_FLAG}",DEVICE="${DEVICE}" \
    scripts/slurm/run_table1_evaluate_all_gpu_array.sh
)
echo "Submitted 810 dependent evaluation tasks covering all 3,240 evaluations: ${EVAL_JOB_ID}"
echo "Total submitted array tasks: 864"
