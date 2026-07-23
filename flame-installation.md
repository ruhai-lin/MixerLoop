# Flame environment

MixerLoop uses the same CUDA extension stack as Flame. Install PyTorch first,
then install the repository and the extensions against that PyTorch build.

The local release was validated on Ubuntu, Python 3.10.12, two RTX 3090 Ti
GPUs, CUDA 13.0, and PyTorch 2.13.0+cu130. Python 3.11 is preferred for a fresh
environment when compatible wheels are available.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel ninja packaging

# Choose the PyTorch index matching the installed CUDA driver.
python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu130

export CUDA_HOME=/usr/local/cuda
python -m pip install -e .
```

The four configurations in this repository use Flash Linear Attention's GDN
kernels and do not require the separate `flash-attn` or Transformer Engine
packages.

If extension compilation cannot find NCCL, expose the headers and libraries
shipped with the PyTorch environment before installing the extensions:

```bash
export NCCL_HOME="$VIRTUAL_ENV/lib/python$(python -c \
  'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages/nvidia/nccl"
export CPATH="$NCCL_HOME/include:${CPATH:-}"
export CPLUS_INCLUDE_PATH="$NCCL_HOME/include:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="$NCCL_HOME/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$NCCL_HOME/lib:${LD_LIBRARY_PATH:-}"
```

Verify the complete environment from the repository root:

```bash
python -m pip check
python - <<'PY'
import torch
import fla
import torchtitan
import custom_models

print("torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("custom models:", custom_models.__file__)
PY

python -m pytest -q
```

The `.venv` directory is intentionally ignored by Git. It is machine-local and
must not be copied into a binary release or committed.
