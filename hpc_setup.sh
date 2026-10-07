#!/bin/bash
# Set up the Python environment for the deepmd LES work on this pod.
#
# Mirrors the originating host (py39, torch 2.8.0+cu128, numpy 2.0.2) as closely
# as this pod's python 3.10 allows. The one deliberate deviation:
# CACE declares numpy<2 and ase<=3.22.1, but the originating host runs numpy
# 2.0.2 and ase 3.26.0 with CACE installed --no-deps, and that combination is
# what produced the validated results. Installing CACE's declared pins here
# would downgrade numpy and break deepmd, so we install it --no-deps too.
set -euo pipefail

PY=/root/venv310/bin/python
PIP=/root/venv310/bin/pip

echo "=== python ==="
$PY -V

echo "=== deepmd runtime deps ==="
$PIP install -q numpy scipy pyyaml 'dargs>=0.4.7' h5py wcmatch packaging \
    ml_dtypes mendeleev array-api-compat ase

echo "=== deepmd-kit editable, PyTorch backend, no TensorFlow, no C++ ops ==="
# The originating host also has ENABLE_CUSTOMIZED_OP=False: deepmd.pt is pure
# Python and falls back cleanly when deepmd_op_pt is absent. Leaving both
# backends off skips the CMake op build entirely and matches that install.
cd /root/app/deepmd-les/deepmd-les/deepmd-kit-3.1.2
DP_ENABLE_TENSORFLOW=0 DP_VARIANT=cpu $PIP install -e . 2>&1 | tail -5

echo "=== les editable ==="
# --no-deps: les would otherwise pull deepmd-kit[torch] from PyPI over the
# local editable checkout.
cd /root/app/deepmd-les/deepmd-les/les
$PIP install -e . --no-deps 2>&1 | tail -3

echo "=== cace editable ==="
cd /root/app/cace
$PIP install -e . --no-deps 2>&1 | tail -3
$PIP install -q matscipy

echo "=== versions ==="
$PY - <<'EOF'
import torch, numpy
print("torch   ", torch.__version__, "cuda_build", torch.version.cuda, "avail", torch.cuda.is_available())
print("numpy   ", numpy.__version__)
import deepmd
print("deepmd  ", deepmd.__version__, deepmd.__file__)
from deepmd.pt.cxx_op import ENABLE_CUSTOMIZED_OP
print("customized_op", ENABLE_CUSTOMIZED_OP)
import les
print("les     ", les.__file__)
import cace
print("cace    ", cace.__file__)
EOF
echo "SETUP DONE"
