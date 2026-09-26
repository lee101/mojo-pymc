#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo_dir/dist"

# The shared toolchain exports MODULAR_HOME but no CONDA_PREFIX; cblas lives in
# the same prefix as the compiler, so derive it when the caller did not set one.
: "${CONDA_PREFIX:=${MODULAR_HOME%/share/max}}"

mojo build --emit shared-lib "$repo_dir/src/kernels.mojo" \
    -Xlinker "-L$CONDA_PREFIX/lib" -Xlinker -lcblas \
    -o "$repo_dir/dist/libmojo-pymc.so"
