#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-/projects/prjs1237/project/tabpfgen/TabPFGen}"

mkdir -p \
  "${REPO_ROOT}/logs/extra_synthetic_evaluations" \
  "${REPO_ROOT}/logs/privacy_attacks"

EXTRA_JOB_ID="$(sbatch --parsable "${SCRIPT_DIR}/run_extra_synthetic_evaluations.sh")"
PRIVACY_JOB_ID="$(sbatch --parsable "${SCRIPT_DIR}/run_privacy_attacks.sh")"

echo "Submitted extra synthetic evaluations: ${EXTRA_JOB_ID}"
echo "Submitted privacy attacks: ${PRIVACY_JOB_ID}"
