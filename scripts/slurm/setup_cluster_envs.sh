#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/projects/prjs1237/project/tabpfgen/TabPFGen}"
CONDA_BASE="${CONDA_BASE:-${HOME}/anaconda3}"
TABPFGEN_ENV="${TABPFGEN_ENV:-tabpfgen}"
SYNTHCITY_ENV="${SYNTHCITY_ENV:-synthcity}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -Fxq "${SYNTHCITY_ENV}"; then
  conda create -y -n "${SYNTHCITY_ENV}" python=3.10
fi

conda activate "${SYNTHCITY_ENV}"
python -m pip install --upgrade pip
python -m pip install -r "${REPO_ROOT}/requirements-synthcity.txt"
python - <<'PY'
import torch
from packaging.version import Version
from synthcity.plugins import Plugins

version = Version(torch.__version__.split("+", 1)[0])
assert Version("2.1") <= version < Version("2.3"), torch.__version__
print(f"synthcity env torch={torch.__version__}")
print("synthcity plugins import OK")
print(Plugins().list())
PY

conda deactivate
conda activate "${TABPFGEN_ENV}"
python - <<'PY'
import torch
from packaging.version import Version

version = Version(torch.__version__.split("+", 1)[0])
assert version >= Version("2.5"), (
    f"TabPFN/TabPFGen env needs torch>=2.5, found {torch.__version__}"
)
print(f"tabpfgen env torch={torch.__version__}")
PY

echo "Environment setup/check complete."
