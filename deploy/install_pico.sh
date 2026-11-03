#!/usr/bin/env bash
# Install the XRoboToolkit PICO SDK (`xrobotoolkit_sdk`) into this repo's .venv.
#
# Run from anywhere; it works on the repo that contains this script:
#   bash deploy/install_pico.sh
#
# It touches only this repo: the clone goes to deploy/third_party/ (git-ignored) and the
# package into .venv (Python 3.11, uv). It does not use sudo and does not install the
# XRoboToolkit PC service .deb; do that yourself (README, "Teleop").
#
# Why this differs from SONIC's install_pico.sh: SONIC builds an older pybind11 +
# CMake version of the bindings against a libPXREARobotSDK.so it vendors (the x86_64 one
# is a git-lfs pointer in a plain clone). The upstream repo has since moved to
# scikit-build-core (CPython stable ABI, no pybind11) and ships the prebuilt x86_64 and
# aarch64 libPXREARobotSDK.so in vendor/. So the build needs only a C++17 compiler;
# CMake >= 3.26 is fetched by the build backend if missing.
#
# `uv sync` (without --inexact) removes packages that are not in the lockfile, which
# includes xrobotoolkit_sdk. After this script, use `uv sync --group teleop --inexact`
# (or `uv run`, which is inexact).
#
# XRT_REF pins the upstream commit; override it to try a newer one.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
URL="https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind"
XRT_REF="${XRT_REF:-8941e1dbbf4313fe83846f82984eaa8c992690bf}"
CLONE="$REPO_ROOT/deploy/third_party/XRoboToolkit-PC-Service-Pybind"

cd "$REPO_ROOT"

command -v uv >/dev/null || { echo "[ERROR] uv not found; install it first (https://docs.astral.sh/uv/)"; exit 1; }
command -v g++ >/dev/null || { echo "[ERROR] no C++ compiler (g++) found; the extension is built from source"; exit 1; }

# 1. Clone (or update) the bindings at the pinned commit.
mkdir -p "$(dirname "$CLONE")"
if [ ! -d "$CLONE/.git" ]; then
  echo "[INFO] cloning $URL"
  git clone "$URL" "$CLONE"
fi
git -C "$CLONE" fetch --quiet origin "$XRT_REF" 2>/dev/null || git -C "$CLONE" fetch --quiet origin
git -C "$CLONE" checkout --quiet "$XRT_REF"
echo "[OK] bindings at $(git -C "$CLONE" rev-parse --short HEAD)"

# 2. Make sure .venv exists with the teleop group (pyzmq, msgpack).
if [ ! -x .venv/bin/python ]; then
  echo "[INFO] creating .venv"
  uv sync --group teleop --inexact
fi
echo "[OK] $(.venv/bin/python --version) in .venv"

# 3. Build and install the SDK into .venv. The package is built in an isolated build
#    environment (scikit-build-core + cmake), so nothing else is installed into .venv.
echo "[INFO] building xrobotoolkit_sdk"
uv pip install --python .venv/bin/python "$CLONE"

# 4. Re-sync without removing the SDK, then check the import.
uv sync --group teleop --inexact
.venv/bin/python -c "import xrobotoolkit_sdk; print('[OK] xrobotoolkit_sdk imported from', xrobotoolkit_sdk.__file__)"
