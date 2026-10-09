#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build-vector-report"
rm -rf "$BUILD"
cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-fopt-info-vec-optimized" 2>&1 | tee "$BUILD-configure.log"
cmake --build "$BUILD" --parallel 2>&1 | tee "$BUILD-build.log"
if grep -E "optimized: loop vectorized|vectorized" "$BUILD-build.log" "$BUILD-configure.log" >/dev/null; then
  echo "Vectorization diagnostics found."
else
  echo "No vectorization diagnostic found; inspect build logs and generated assembly." >&2
  exit 1
fi
