#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
TABPFN_PAPER_ROOT="${TABPFN_PAPER_ROOT:-${REPO_ROOT}/third_party/tabpfn_paper}"
TABPFN_REPOSITORY="${TABPFN_REPOSITORY:-https://github.com/PriorLabs/TabPFN.git}"
TABPFN_COMMIT="e640ef535e29c8f1b8b148c187744429f847e6d4"
CHECKPOINT_RELATIVE="tabpfn/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt"
CHECKPOINT_SHA256="3c9aadaeddbf51462af8c0ee4b3ca3c697890f77e92318abbb0821b75261c392"

if [[ ! -d "${TABPFN_PAPER_ROOT}/.git" ]]; then
  if [[ -e "${TABPFN_PAPER_ROOT}" ]] && [[ -n "$(ls -A "${TABPFN_PAPER_ROOT}")" ]]; then
    echo "Refusing to replace non-empty path: ${TABPFN_PAPER_ROOT}" >&2
    exit 1
  fi
  mkdir -p "$(dirname -- "${TABPFN_PAPER_ROOT}")"
  git clone --filter=blob:none --no-checkout "${TABPFN_REPOSITORY}" "${TABPFN_PAPER_ROOT}"
fi

git -C "${TABPFN_PAPER_ROOT}" fetch --depth 1 origin "${TABPFN_COMMIT}"
git -C "${TABPFN_PAPER_ROOT}" checkout --detach "${TABPFN_COMMIT}"
printf '%s\n' "${TABPFN_COMMIT}" > "${TABPFN_PAPER_ROOT}/.tabpfgen-paper-commit"

CHECKPOINT_PATH="${TABPFN_PAPER_ROOT}/${CHECKPOINT_RELATIVE}"
if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Paper checkpoint was not present after checkout: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

ACTUAL_SHA256="$(shasum -a 256 "${CHECKPOINT_PATH}" | awk '{print $1}')"
if [[ "${ACTUAL_SHA256}" != "${CHECKPOINT_SHA256}" ]]; then
  echo "Checkpoint checksum mismatch: ${ACTUAL_SHA256}" >&2
  exit 1
fi

echo "Paper TabPFN source: ${TABPFN_PAPER_ROOT}"
echo "Pinned commit: ${TABPFN_COMMIT}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "SHA256: ${ACTUAL_SHA256}"
