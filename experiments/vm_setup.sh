#!/usr/bin/env bash
# One-time setup on a Google Cloud GPU VM (Deep Learning VM image). Safe to re-run.
# Usage:  bash ~/coms4705_setfit/experiments/vm_setup.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${SETFIT_VENV:-$HOME/setfit-venv}"

echo "== 1/4 GPU driver"
if ! nvidia-smi >/dev/null 2>&1; then
  cat <<'MSG'
No working NVIDIA driver (nvidia-smi failed).
On a Deep Learning VM, run:   sudo /opt/deeplearning/install-driver.sh
(or close this SSH window, open a new one and answer "y" to the driver prompt),
then run this script again.
MSG
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo "== 2/4 System packages"
pkgs=""
command -v tmux >/dev/null 2>&1 || pkgs="$pkgs tmux"
python3 -c "import ensurepip" >/dev/null 2>&1 || pkgs="$pkgs python3-venv"
if [ -n "$pkgs" ]; then
  echo "installing:$pkgs"
  sudo apt-get update -qq
  sudo apt-get install -y -qq $pkgs
else
  echo "tmux and python3-venv present"
fi

echo "== 3/4 Python environment in $VENV (a few minutes the first time)"
[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install -q --upgrade pip
"$VENV/bin/python" -m pip install -q -r "$HERE/requirements.txt"
"$VENV/bin/python" - <<'PY'
import torch, transformers, sklearn
assert torch.cuda.is_available(), "PyTorch is installed but cannot see the GPU (check nvidia-smi)"
print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__,
      "| transformers", transformers.__version__, "| scikit-learn", sklearn.__version__)
PY

echo "== 4/4 Offline tests (synthetic data, CPU)"
(cd "$HERE" && "$VENV/bin/python" -m pytest -q tests)

cat <<MSG

Setup done. Next:
  tmux new -s setfit
  $HERE/vm_run.sh --stage smoke
MSG
