#!/usr/bin/env bash
# Run Python scripts with the Newton simulator environment.
#
# Newton/Warp are built against CUDA 12.6, so the CUDA paths have to be
# redirected before the interpreter starts. Otherwise the loader picks up the
# default /usr/local/cuda runtime and Warp fails to initialise.
#
# Usage:
#   ./run_newton.sh rl_integration.py --num_envs 16 --max_iterations 10
#   ./run_newton.sh tests/load_urdf.py --viewer null --test

set -euo pipefail

export PATH=$(echo "$PATH" | sed 's|/usr/local/cuda|/usr/local/cuda-12.6|g')
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | sed 's|/usr/local/cuda|/usr/local/cuda-12.6|g')

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

exec /home/user/miniconda3/envs/newton/bin/python "$@"
