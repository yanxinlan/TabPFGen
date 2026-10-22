#!/bin/bash
set -euo pipefail

REPO_ROOT="/projects/prjs1237/project/tabpfgen/TabPFGen"
LOG_ROOT="/projects/prjs1237/project/tabpfgen/logs/table1_non_rtvae_evaluate_cpu"

mkdir -p "${LOG_ROOT}"
cd "${REPO_ROOT}"

JOB_ID=$(
  sbatch \
    --parsable \
    --partition=rome \
    --gres=none \
    --export=ALL,DEVICE=cpu,OVERWRITE_FLAG=--overwrite \
    scripts/slurm/run_table1_non_rtvae_evaluate_cpu_array.sh
)

echo "Submitted 702 CPU tasks covering 2,808 non-RTVAE evaluations: ${JOB_ID}"
