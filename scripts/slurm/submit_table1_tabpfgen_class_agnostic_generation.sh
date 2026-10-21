#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="/projects/prjs1237/project/tabpfgen/TabPFGen"

mkdir -p "${REPO_ROOT}/logs/table1_tabpfgen_class_agnostic_generate"

JOB_ID="$(sbatch --parsable "${SCRIPT_DIR}/run_table1_tabpfgen_class_agnostic_generate_gpu_array.sh")"
echo "Submitted TabPFGen class-agnostic initialization generation array: ${JOB_ID}"
echo "Outputs go to: ${OUTPUT_ROOT:-${REPO_ROOT}/outputs/table1_generators_class_agnostic_init}"
