#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-/projects/prjs1237/project/tabpfgen/TabPFGen}"

mkdir -p "${REPO_ROOT}/logs/tabpfgen_gradient_health"

JOB_ID="$(sbatch --parsable "${SCRIPT_DIR}/run_tabpfgen_gradient_health_gpu_array.sh")"
echo "Submitted TabPFGen gradient-health array: ${JOB_ID}"
