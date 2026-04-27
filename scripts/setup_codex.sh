#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
pip install -r requirements.txt

python - <<'PY'
import numpy, pandas, scipy, sklearn, matplotlib
print("Environment OK")
PY
