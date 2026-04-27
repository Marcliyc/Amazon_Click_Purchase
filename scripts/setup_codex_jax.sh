#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Optional GPU wheel installation.
# Usage examples:
#   JAX_ACCELERATOR=cpu bash setup_codex_jax.sh
#   JAX_ACCELERATOR=cuda13 bash setup_codex_jax.sh
#   JAX_ACCELERATOR=cuda12 bash setup_codex_jax.sh
JAX_ACCELERATOR="${JAX_ACCELERATOR:-cuda12}"

case "${JAX_ACCELERATOR}" in
  cpu)
    echo "Using portable CPU JAX from requirements.txt"
    ;;
  cuda13)
    python -m pip install -U "jax[cuda13]"
    ;;
  cuda12)
    python -m pip install -U "jax[cuda12]"
    ;;
  *)
    echo "Unknown JAX_ACCELERATOR=${JAX_ACCELERATOR}; expected cpu, cuda12, or cuda13" >&2
    exit 2
    ;;
esac

python - <<'PY'
import jax
print('JAX version:', jax.__version__)
print('JAX devices:', jax.devices())
PY
