#!/usr/bin/env bash
set -Eeuo pipefail

# -----------------------------------------------------------------------------
# BuildLinux.sh
#
# Configure, build, install and optionally run LichtFeld-Studio on Linux.
#
# Normal use:
#   ./BuildLinux.sh
#
# Useful variants:
#   ./BuildLinux.sh --clean --no-run
#   UPDATE_VCPKG=1 ./BuildLinux.sh --clean --no-run
#   BUILD_TYPE=Debug ./BuildLinux.sh --clean --no-run
# -----------------------------------------------------------------------------

die() {
    echo ""
    echo "ERROR: $*" >&2
    echo ""
    exit 1
}

warn() {
    echo "WARNING: $*" >&2
}

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "Missing command: $1"
}

usage() {
    cat <<'USAGE'
Usage:
  ./BuildLinux.sh [options]

Options:
  --clean       Remove LinuxBuild and dist before configuring.
  --no-run      Build and install, but do not launch the app.
  --run         Launch the app after build/install. Default.
  -h, --help    Show this help.

Environment variables:
  CUDA_HOME             Default: /usr/local/cuda-12.8
  VCPKG_ROOT            Default: /home/victor/Documentos/SMN/software/vcpkg
  BUILD_TYPE            Default: Release
  BUILD_PORTABLE        Default: ON
  BUILD_CUDA_FATBIN     Default: ON
  UPDATE_VCPKG          Default: 0
  RUN_AFTER_BUILD       Default: 1
  CMAKE_GENERATOR       Default: Ninja

Examples:
  ./BuildLinux.sh --clean --no-run
  UPDATE_VCPKG=1 ./BuildLinux.sh --clean --no-run
  BUILD_TYPE=Debug ./BuildLinux.sh --clean --no-run
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$SCRIPT_DIR/LinuxBuild}"
DIST_DIR="${DIST_DIR:-$SCRIPT_DIR/dist}"

# -----------------------------------------------------------------------------
# Load ~/.bashrc softly.
# Some .bashrc files contain interactive-only commands, so do not let that break.
# -----------------------------------------------------------------------------
if [[ -f "$HOME/.bashrc" ]]; then
    set +e +u
    # shellcheck disable=SC1090
    source "$HOME/.bashrc" >/dev/null 2>&1
    set -Eeuo pipefail
fi

# -----------------------------------------------------------------------------
# Defaults. User environment can override these.
# -----------------------------------------------------------------------------
: "${CUDA_HOME:=/usr/local/cuda-12.8}"
: "${VCPKG_ROOT:=/home/victor/Documentos/SMN/software/vcpkg}"
: "${BUILD_TYPE:=Release}"
: "${BUILD_PORTABLE:=ON}"
: "${BUILD_CUDA_FATBIN:=ON}"
: "${UPDATE_VCPKG:=0}"
: "${RUN_AFTER_BUILD:=1}"
: "${CMAKE_GENERATOR:=Ninja}"

CLEAN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)
            CLEAN=1
            ;;
        --no-run)
            RUN_AFTER_BUILD=0
            ;;
        --run)
            RUN_AFTER_BUILD=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "Unknown argument: $1. Use --help."
            ;;
    esac
    shift
done

export CUDA_HOME
export VCPKG_ROOT
export PATH="$CUDA_HOME/bin:$VCPKG_ROOT:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# -----------------------------------------------------------------------------
# Required tools
# -----------------------------------------------------------------------------
need_cmd git
need_cmd cmake
need_cmd python3
need_cmd nvcc
need_cmd cc
need_cmd c++

if [[ "$CMAKE_GENERATOR" == "Ninja" ]]; then
    need_cmd ninja
fi

[[ -d "$CUDA_HOME" ]] || die "CUDA_HOME does not exist: $CUDA_HOME"
[[ -d "$VCPKG_ROOT" ]] || die "VCPKG_ROOT does not exist: $VCPKG_ROOT"
[[ -d "$VCPKG_ROOT/.git" ]] || die "VCPKG_ROOT is not a git repository: $VCPKG_ROOT"
[[ -f "$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" ]] || die "vcpkg CMake toolchain not found."

if [[ ! -x "$VCPKG_ROOT/vcpkg" ]]; then
    [[ -x "$VCPKG_ROOT/bootstrap-vcpkg.sh" ]] || die "bootstrap-vcpkg.sh not found or not executable."
    echo "-- Bootstrapping vcpkg"
    "$VCPKG_ROOT/bootstrap-vcpkg.sh"
fi

echo "-- Using project: $SCRIPT_DIR"
echo "-- Using build dir: $BUILD_DIR"
echo "-- Using dist dir: $DIST_DIR"
echo "-- Using CUDA_HOME: $CUDA_HOME"
echo "-- Using VCPKG_ROOT: $VCPKG_ROOT"
echo "-- Using generator: $CMAKE_GENERATOR"
echo "-- Using build type: $BUILD_TYPE"
echo "-- nvcc: $(nvcc --version | tail -n 1)"
echo "-- $(cmake --version | head -n 1)"

if [[ "$CMAKE_GENERATOR" == "Ninja" ]]; then
    echo "-- ninja: $(ninja --version)"
fi

# -----------------------------------------------------------------------------
# Optional vcpkg update.
#
# This updates the local vcpkg registry/tool, but does NOT mutate the project's
# builtin-baseline. For that, use UpdateVcpkgBaselineLinux.sh explicitly.
# -----------------------------------------------------------------------------
if [[ "$UPDATE_VCPKG" == "1" ]]; then
    echo "-- Updating local vcpkg registry"
    git -C "$VCPKG_ROOT" pull --ff-only
    "$VCPKG_ROOT/bootstrap-vcpkg.sh"
fi

# -----------------------------------------------------------------------------
# Validate vcpkg builtin-baseline when vcpkg.json exists.
# This catches the common case where the baseline commit has not been fetched.
# -----------------------------------------------------------------------------
if [[ -f "$SCRIPT_DIR/vcpkg.json" ]]; then
    BASELINE="$(
        python3 - "$SCRIPT_DIR/vcpkg.json" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    data = json.load(f)

print(data.get("builtin-baseline", ""))
PY
    )"

    if [[ -n "$BASELINE" ]]; then
        echo "-- vcpkg builtin-baseline: $BASELINE"

        if ! git -C "$VCPKG_ROOT" cat-file -e "$BASELINE^{commit}" 2>/dev/null; then
            echo "-- Baseline commit not present locally; fetching it"
            git -C "$VCPKG_ROOT" fetch origin "$BASELINE"
        fi

        git -C "$VCPKG_ROOT" show "$BASELINE:versions/baseline.json" >/dev/null 2>&1 \
            || die "vcpkg baseline exists but does not contain versions/baseline.json: $BASELINE"
    else
        warn "vcpkg.json has no builtin-baseline. Builds may be less reproducible."
    fi
fi

# -----------------------------------------------------------------------------
# Clean
# -----------------------------------------------------------------------------
if [[ "$CLEAN" == "1" ]]; then
    echo "-- Cleaning build/install directories"
    rm -rf "$BUILD_DIR" "$DIST_DIR"
fi

# -----------------------------------------------------------------------------
# Configure
# -----------------------------------------------------------------------------
cmake \
    -S "$SCRIPT_DIR" \
    -B "$BUILD_DIR" \
    -G "$CMAKE_GENERATOR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
    -DBUILD_PORTABLE="$BUILD_PORTABLE" \
    -DBUILD_CUDA_FATBIN="$BUILD_CUDA_FATBIN"

# -----------------------------------------------------------------------------
# Build and install
# -----------------------------------------------------------------------------
cmake --build "$BUILD_DIR" --parallel

if [[ "$BUILD_PORTABLE" == "ON" ]]; then
    cmake --install "$BUILD_DIR" --prefix "$DIST_DIR"
else
    warn "BUILD_PORTABLE is OFF; skipping portable install into dist."
fi

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
if [[ "$RUN_AFTER_BUILD" == "1" ]]; then
    if [[ "$BUILD_PORTABLE" != "ON" ]]; then
        warn "RUN_AFTER_BUILD=1 but BUILD_PORTABLE is OFF; no dist executable expected."
        exit 0
    fi

    APP="$DIST_DIR/bin/LichtFeld-Studio"
    [[ -x "$APP" ]] || die "Expected executable not found: $APP"

    echo "-- Running $APP"
    "$APP"
fi