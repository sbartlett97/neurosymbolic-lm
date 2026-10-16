#!/usr/bin/env bash
# One-time (idempotent) setup on a RunPod PyTorch template pod.
#
#   bash scripts/runpod_setup.sh            # training + GLiNER2 curation env
#   WITH_VLLM=1 bash scripts/runpod_setup.sh  # also the separate vLLM env
#
# Everything lives on /workspace so it survives a pod stop. Re-running is
# safe. See docs/RUNPOD.md for the full runbook.
set -euo pipefail

WS=${WORKSPACE:-/workspace}
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV=$WS/venv-train
VLLM_VENV=$WS/venv-vllm

echo "== repo: $REPO_DIR"
echo "== workspace: $WS"

# --- caches on the volume ------------------------------------------------
mkdir -p "$WS/hf-cache" "$WS/pip-cache" "$WS/uv-cache" "$WS/data" "$WS/checkpoints" "$WS/runs" "$WS/logs"
ENV_FILE=$WS/neurosym.env
cat > "$ENV_FILE" <<EOF
export HF_HOME=$WS/hf-cache
export PIP_CACHE_DIR=$WS/pip-cache
export UV_CACHE_DIR=$WS/uv-cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EOF
# shellcheck disable=SC1090
. "$ENV_FILE"
grep -q "neurosym.env" ~/.bashrc 2>/dev/null || echo ". $ENV_FILE" >> ~/.bashrc

# --- system tools ----------------------------------------------------------
if ! command -v tmux >/dev/null 2>&1; then
  DEBIAN_FRONTEND=noninteractive apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux git-lfs >/dev/null
fi
command -v uv >/dev/null 2>&1 || pip install -q uv

# --- training env: reuse the template's CUDA torch ------------------------
SYS_TORCH=$(python - <<'EOF'
import sys, torch
print(f"== system python {sys.version.split()[0]}, torch {torch.__version__}, "
      f"cuda {torch.version.cuda}, gpu available: {torch.cuda.is_available()}", file=sys.stderr)
assert sys.version_info >= (3, 11), "Python >= 3.11 required"
# transformers 5.13 needs torch >= 2.4; >= 2.6 for torch.load of .bin weights
major, minor = (int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
assert (major, minor) >= (2, 6), \
    "Template torch < 2.6: pick a newer RunPod PyTorch template"
print(torch.__version__)
EOF
)
if [ ! -x "$VENV/bin/python" ]; then
  # --system-site-packages: inherit the template torch instead of
  # downloading another multi-GB build
  uv venv --system-site-packages "$VENV"
fi
uv pip install --python "$VENV/bin/python" -r "$REPO_DIR/requirements.txt"

# The venv must still use the template torch (a dependency that needs a
# newer torch would silently install a second, possibly CPU-only, build)
VENV_TORCH=$("$VENV/bin/python" -c "import torch; print(torch.__version__)")
if [ "$VENV_TORCH" != "$SYS_TORCH" ]; then
  echo "ERROR: venv torch $VENV_TORCH differs from template torch $SYS_TORCH" >&2
  exit 1
fi
"$VENV/bin/python" -c "import torch; assert torch.cuda.is_available(), 'venv torch sees no CUDA'"

"$VENV/bin/python" - <<'EOF'
import torch, transformers, peft, gliner2  # noqa: F401
print(f"== train env OK: torch {torch.__version__}, transformers {transformers.__version__}, "
      f"peft {peft.__version__}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"== GPU: {p.name}, {p.total_memory / 1e9:.0f} GB, bf16: {torch.cuda.is_bf16_supported()}")
EOF

# --- optional vLLM env (hybrid QA curation) --------------------------------
if [ "${WITH_VLLM:-0}" = "1" ]; then
  [ -x "$VLLM_VENV/bin/python" ] || uv venv "$VLLM_VENV"
  uv pip install --python "$VLLM_VENV/bin/python" -r "$REPO_DIR/requirements-vllm.txt"
  "$VLLM_VENV/bin/python" -c "import vllm; print('== vllm env OK:', vllm.__version__)"
fi

# --- offline test suite (no downloads beyond the Qwen tokenizer) ----------
(cd "$REPO_DIR" && "$VENV/bin/python" -m pytest -q 2>&1 | tail -3)

if [ -z "${HF_TOKEN:-}" ]; then
  echo "== NOTE: HF_TOKEN is not set. Gated checkpoints (Llama 3.x, Gemma 3) need it:"
  echo "   export HF_TOKEN=hf_...   (or set it as a pod env var / RunPod secret)"
fi
echo "== done. Activate with:  . $ENV_FILE && . $VENV/bin/activate"
