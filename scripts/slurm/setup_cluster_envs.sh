#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
CONDA_BASE="${CONDA_BASE:-${HOME}/anaconda3}"
TABPFGEN_ENV="${TABPFGEN_ENV:-tabpfgen}"
SYNTHCITY_ENV="${SYNTHCITY_ENV:-synthcity}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"

ensure_env() {
  local env_name="$1"
  local python_version="$2"
  if ! conda env list | awk '{print $1}' | grep -Fxq "${env_name}"; then
    conda create -y -n "${env_name}" "python=${python_version}" pip
  fi
}

ensure_env "${SYNTHCITY_ENV}" "3.10"
ensure_env "${TABPFGEN_ENV}" "3.12"

conda activate "${SYNTHCITY_ENV}"
python -m pip install --upgrade pip
# SynthCity declares dependencies for every optional plugin family. Installing it
# without dependencies avoids pulling image, survival, and LLM stacks that the
# Table 1 generic generators never use; their runtime dependencies are pinned in
# requirements-synthcity.txt instead.
python -m pip install --no-deps "synthcity==0.2.12"
python -m pip install -r "${REPO_ROOT}/requirements-synthcity.txt"
# SynthCity eagerly imports metrics and models from plugin families that are not
# used here. Their top-level modules must therefore exist, but installing their
# transitive stacks would re-resolve Torch and can break the generic plugins.
python -m pip install --no-deps \
  "arfpy==0.1.1" \
  "be-great==0.0.14" \
  "decaf-synthetic-data==0.1.7" \
  "fastai==2.7.19" \
  "fastcore==1.7.29" \
  "fflows==0.0.3" \
  "geomloss==0.3.1" \
  "importlib-metadata>=7" \
  "lifelines==0.29.0" \
  "monai==1.4.0" \
  "pgmpy==0.1.26" \
  "pycox==0.3.0" \
  "shap==0.49.1" \
  "tsai==0.3.9" \
  "xgbse==0.3.3"
python - <<'PY'
import torch
from packaging.version import Version
from synthcity.plugins import Plugins

version = Version(torch.__version__.split("+", 1)[0])
assert Version("2.1") <= version < Version("2.3"), torch.__version__
print(f"synthcity env torch={torch.__version__}")
print("synthcity plugins import OK")
available = set(Plugins(categories=["generic"]).list())
required = {"ctgan", "tvae", "rtvae"}
missing = required - available
assert not missing, f"Missing required SynthCity plugins: {sorted(missing)}"
assert {"nflow", "nf"} & available, "Missing nflow/nf plugin"
assert {"tabddpm", "ddpm"} & available, "Missing tabddpm/ddpm plugin"
print(f"required plugins available: {sorted(required)}")
PY

conda deactivate
conda activate "${TABPFGEN_ENV}"
python -m pip install --upgrade pip
python -m pip install -r "${REPO_ROOT}/requirements-tabpfn.txt"
python -m pip install --no-deps -e "${REPO_ROOT}"
python - <<'PY'
import torch
from packaging.version import Version
from tabpfgen import TabPFGen
from tabpfn import TabPFNClassifier

version = Version(torch.__version__.split("+", 1)[0])
assert version >= Version("2.5"), (
    f"TabPFN/TabPFGen env needs torch>=2.5, found {torch.__version__}"
)
print(f"tabpfgen env torch={torch.__version__}")
print(f"TabPFGen import OK: {TabPFGen.__module__}")
print(f"TabPFNClassifier import OK: {TabPFNClassifier.__module__}")
PY

echo "Environment setup/check complete."
