#!/bin/bash
#SBATCH --job-name=table1_eval_all
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-809%120
#SBATCH --output=/projects/prjs1237/project/tabpfgen/logs/table1_evaluate_after_rtvae/%A_%a.out
#SBATCH --error=/projects/prjs1237/project/tabpfgen/logs/table1_evaluate_after_rtvae/%A_%a.err

set -euo pipefail

REPO_ROOT="/projects/prjs1237/project/tabpfgen/TabPFGen"
TOTAL_EVALUATIONS=3240
EVALUATIONS_PER_TASK=4
OVERWRITE_FLAG="${OVERWRITE_FLAG:---overwrite}"

ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
FIRST_EVALUATION=$((ARRAY_TASK_ID * EVALUATIONS_PER_TASK))

cd "${REPO_ROOT}"

for ((slot = 0; slot < EVALUATIONS_PER_TASK; slot++)); do
  evaluation_id=$((FIRST_EVALUATION + slot))
  if ((evaluation_id >= TOTAL_EVALUATIONS)); then
    break
  fi

  echo "Running downstream evaluation $((evaluation_id + 1))/${TOTAL_EVALUATIONS}"
  SLURM_ARRAY_TASK_ID="${evaluation_id}" \
  TASK_OFFSET=0 \
  OVERWRITE_FLAG="${OVERWRITE_FLAG}" \
    bash scripts/slurm/run_table1_evaluate_gpu_array.sh
done
