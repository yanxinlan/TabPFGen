#!/bin/bash
#SBATCH --job-name=table1_eval_no_rtvae
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-701%120
#SBATCH --output=/projects/prjs1237/project/tabpfgen/logs/table1_non_rtvae_evaluate_cpu/%A_%a.out
#SBATCH --error=/projects/prjs1237/project/tabpfgen/logs/table1_non_rtvae_evaluate_cpu/%A_%a.err

set -euo pipefail

REPO_ROOT="/projects/prjs1237/project/tabpfgen/TabPFGen"
N_ORIGINAL_EVALUATIONS=216
N_TOTAL_EVALUATIONS=3240
N_NON_RTVAE_EVALUATIONS=2808
EVALUATIONS_PER_TASK=4
OVERWRITE_FLAG="${OVERWRITE_FLAG:---overwrite}"

ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
FIRST_COMPACT_ID=$((ARRAY_TASK_ID * EVALUATIONS_PER_TASK))

cd "${REPO_ROOT}"

compact_to_global_id() {
  local compact_id="$1"
  if ((compact_id < N_ORIGINAL_EVALUATIONS)); then
    echo "${compact_id}"
    return
  fi

  # Decode the six-generator compact index, then insert RTVAE's skipped slot
  # at generator index 4 in the seven-generator evaluator ordering.
  local synthetic_id=$((compact_id - N_ORIGINAL_EVALUATIONS))
  local model_idx=$((synthetic_id % 4))
  local tmp=$((synthetic_id / 4))
  local mode_idx=$((tmp % 2))
  tmp=$((tmp / 2))
  local compact_generator_idx=$((tmp % 6))
  tmp=$((tmp / 6))
  local seed_idx=$((tmp % 3))
  local dataset_idx=$((tmp / 3))
  local generator_idx="${compact_generator_idx}"
  if ((generator_idx >= 4)); then
    generator_idx=$((generator_idx + 1))
  fi

  local global_synthetic_id=$((
    ((((dataset_idx * 3) + seed_idx) * 7 + generator_idx) * 2 + mode_idx) * 4
    + model_idx
  ))
  echo "$((N_ORIGINAL_EVALUATIONS + global_synthetic_id))"
}

for ((slot = 0; slot < EVALUATIONS_PER_TASK; slot++)); do
  compact_id=$((FIRST_COMPACT_ID + slot))
  if ((compact_id >= N_NON_RTVAE_EVALUATIONS)); then
    break
  fi

  evaluation_id="$(compact_to_global_id "${compact_id}")"
  if ((evaluation_id < 0 || evaluation_id >= N_TOTAL_EVALUATIONS)); then
    echo "Mapped evaluation id ${evaluation_id} is out of range" >&2
    exit 1
  fi

  echo "Running non-RTVAE evaluation $((compact_id + 1))/${N_NON_RTVAE_EVALUATIONS} (global id ${evaluation_id})"
  SLURM_ARRAY_TASK_ID="${evaluation_id}" \
  TASK_OFFSET=0 \
  DEVICE=cpu \
  OVERWRITE_FLAG="${OVERWRITE_FLAG}" \
    bash scripts/slurm/run_table1_evaluate_gpu_array.sh
done
