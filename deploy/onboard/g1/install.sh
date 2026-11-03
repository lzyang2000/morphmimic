#!/usr/bin/env bash
# On the G1's Jetson, after the first push (and after a push that changes the SDK
# wrapper): the deploy env, .venv, with only what the policy node and g1_node import, and
# the Unitree SDK binding built against it. Not `uv sync`: the project pulls torch and
# mjlab. No sudo: the binding's apt packages are on the Jetson already; the system's
# pybind11 (2.4) is too old for Python 3.11, so the build uses the env's.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PATH="$REPO/.venv/bin:$HOME/.local/bin:$PATH"
cd "$REPO"

[ -x .venv/bin/python ] || uv venv --python 3.11 .venv
# The repo's locked versions (uv.lock).
VIRTUAL_ENV=.venv uv pip install numpy==2.4.6 onnxruntime==1.31.0 onnx==1.23.2 \
  pyzmq==27.2.0 msgpack==1.2.3 scipy==1.17.1 mujoco==3.11.0 pybind11 pybind11-stubgen

export pybind11_DIR="$(python -m pybind11 --cmakedir)"
(cd third_party/unitree_sdk2_wrapper/python_binding \
  && bash build.sh --sdk-path "$PWD/.." --python-version 3.11 > "$REPO/.venv/unitree_build.log" 2>&1) \
  || { echo "[install] the binding build failed: .venv/unitree_build.log"; exit 1; }
cp third_party/unitree_sdk2_wrapper/python_binding/build/lib/unitree_interface.cpython-311-*.so \
  "$(python -c 'import site; print(site.getsitepackages()[0])')/unitree_interface.so"
python -c "import unitree_interface, onnx, onnxruntime, mujoco, zmq; print('[install] OK')"
