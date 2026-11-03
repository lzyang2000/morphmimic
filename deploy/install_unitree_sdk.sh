#!/usr/bin/env bash
# Build the Unitree SDK's python binding (`unitree_interface`) into this repo's uv env,
# for the G1 hardware node (deploy/hardware/g1_node.py). `third_party/unitree_sdk2_wrapper`
# is a fork of Unitree's unitree_sdk2 (prebuilt libunitree_sdk2.a for x86_64 and aarch64,
# CycloneDDS, the headers) with a pybind11 binding on top, the one
# humanoid_dodgeball builds. Run it on the machine that talks to the robot (the
# workstation on the robot's network, or the robot's own computer); it needs sudo for the
# system build packages.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WRAPPER="$REPO/third_party/unitree_sdk2_wrapper"
echo "[install] arch=$(uname -m)  wrapper=$WRAPPER"

# 1. System build packages.
sudo apt-get update
sudo apt-get install -y build-essential cmake python3-dev pybind11-dev \
  libspdlog-dev libboost-all-dev libyaml-cpp-dev libfmt-dev

# 2. The submodule, and the env's build-time python packages.
git -C "$REPO" submodule update --init third_party/unitree_sdk2_wrapper
cd "$REPO"
uv sync --group export --inexact
uv pip install pybind11

# 3. Build the binding against the bundled SDK, with the env's interpreter.
PY_VERSION="$(uv run python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
cd "$WRAPPER/python_binding"
export UNITREE_SDK2_PATH="$WRAPPER"
uv run --project "$REPO" bash build.sh --sdk-path "$UNITREE_SDK2_PATH" --python-version "$PY_VERSION"

# 4. Into the env's site-packages (a plain `uv sync` would remove it: use --inexact).
SITE="$(cd "$REPO" && uv run python -c 'import site; print(site.getsitepackages()[0])')"
cp build/lib/unitree_interface.cpython-*-linux-gnu.so "$SITE/unitree_interface.so"
cd "$REPO" && uv run python -c "import unitree_interface; print('unitree_interface OK')"
