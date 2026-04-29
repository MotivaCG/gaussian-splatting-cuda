#!/usr/bin/env bash
set -Eeuo pipefail

# -----------------------------------------------------------------------------
# SMNBuildLinux.sh
#
# Configure, build, install and optionally run LichtFeld-Studio on Linux.
#
# Normal use:
#   ./SMNBuildLinux.sh
#
# Useful variants:
#   ./SMNBuildLinux.sh --clean --no-run
#   UPDATE_VCPKG=1 ./SMNBuildLinux.sh --clean --no-run
#   BUILD_TYPE=Debug ./SMNBuildLinux.sh --clean --no-run
#   REFRESH_PYTHON_STUBS=0 ./SMNBuildLinux.sh --no-run
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
  ./SMNBuildLinux.sh [options]

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
  REFRESH_PYTHON_STUBS  Default: 1

Examples:
  ./SMNBuildLinux.sh --clean --no-run
  UPDATE_VCPKG=1 ./SMNBuildLinux.sh --clean --no-run
  BUILD_TYPE=Debug ./SMNBuildLinux.sh --clean --no-run
  REFRESH_PYTHON_STUBS=0 ./SMNBuildLinux.sh --no-run
USAGE
}

patch_cmake_runtime_dependency_dirs() {
    local build_dir="$1"
    local vcpkg_installed="$2"
    local cuda_home="$3"

    local dirs=()

    [[ -d "$vcpkg_installed/lib" ]] && dirs+=("$vcpkg_installed/lib")
    [[ -d "$vcpkg_installed/bin" ]] && dirs+=("$vcpkg_installed/bin")
    [[ -d "$cuda_home/lib64" ]] && dirs+=("$cuda_home/lib64")

    if [[ "${#dirs[@]}" -eq 0 ]]; then
        warn "No runtime dependency directories found to patch."
        return 0
    fi

    echo "-- Patching CMake runtime dependency resolver directories"

    python3 - "$build_dir" "${dirs[@]}" <<'PY'
import pathlib
import re
import sys

build_dir = pathlib.Path(sys.argv[1])
dirs = sys.argv[2:]

def patch_block(block: str):
    quoted_dirs = [f'    "{d}"' for d in dirs if f'"{d}"' not in block]

    if not quoted_dirs:
        return block, False

    addition = "\n" + "\n".join(quoted_dirs)

    m = re.search(r"\bDIRECTORIES\b", block)
    if m:
        insert_at = m.end()
        return block[:insert_at] + addition + block[insert_at:], True

    return block[:-1] + "  DIRECTORIES" + addition + "\n" + block[-1:], True

patched_files = []

for path in build_dir.rglob("cmake_install.cmake"):
    text = path.read_text(encoding="utf-8")
    out = []
    i = 0
    changed = False

    while True:
        start = text.find("file(GET_RUNTIME_DEPENDENCIES", i)

        if start == -1:
            out.append(text[i:])
            break

        out.append(text[i:start])

        depth = 0
        in_quote = False
        escape = False
        end = None
        j = start

        while j < len(text):
            c = text[j]

            if in_quote:
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == '"':
                    in_quote = False
            else:
                if c == '"':
                    in_quote = True
                elif c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                    if depth == 0:
                        end = j + 1
                        break

            j += 1

        if end is None:
            out.append(text[start:])
            break

        block = text[start:end]
        block, block_changed = patch_block(block)
        changed = changed or block_changed
        out.append(block)
        i = end

    if changed:
        path.write_text("".join(out), encoding="utf-8")
        patched_files.append(path)

print(f"Patched {len(patched_files)} cmake_install.cmake file(s).")
for path in patched_files:
    print(f"  {path}")
PY
}

write_portable_launcher() {
    local dist_dir="$1"

    echo "-- Writing portable launcher"

    cat > "$dist_dir/LichtFeld-Studio.sh" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LD_LIBRARY_PATH="$HERE/lib:$HERE/lib/extensions:$HERE/bin:${LD_LIBRARY_PATH:-}"
export PXR_PLUGINPATH_NAME="$HERE/lib/usd:${PXR_PLUGINPATH_NAME:-}"

exec "$HERE/bin/LichtFeld-Studio" "$@"
EOF

    chmod +x "$dist_dir/LichtFeld-Studio.sh"
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
: "${REFRESH_PYTHON_STUBS:=1}"

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

VCPKG_INSTALLED_DIR_LINUX="$BUILD_DIR/vcpkg_installed/x64-linux"

# Runtime search paths used by build-time tools.
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

if [[ -d "$VCPKG_INSTALLED_DIR_LINUX/lib" ]]; then
    export LD_LIBRARY_PATH="$VCPKG_INSTALLED_DIR_LINUX/lib:$LD_LIBRARY_PATH"
fi

if [[ -d "$VCPKG_INSTALLED_DIR_LINUX/bin" ]]; then
    export LD_LIBRARY_PATH="$VCPKG_INSTALLED_DIR_LINUX/bin:$LD_LIBRARY_PATH"
fi

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

# vcpkg_installed exists after configure.
VCPKG_INSTALLED_DIR_LINUX="$BUILD_DIR/vcpkg_installed/x64-linux"

if [[ -d "$VCPKG_INSTALLED_DIR_LINUX/lib" ]]; then
    export LD_LIBRARY_PATH="$VCPKG_INSTALLED_DIR_LINUX/lib:$LD_LIBRARY_PATH"
fi

if [[ -d "$VCPKG_INSTALLED_DIR_LINUX/bin" ]]; then
    export LD_LIBRARY_PATH="$VCPKG_INSTALLED_DIR_LINUX/bin:$LD_LIBRARY_PATH"
fi

# -----------------------------------------------------------------------------
# Refresh Python stubs before full build.
#
# This target updates committed .pyi files when the native Python API changes.
# It may modify files under src/python/stubs.
# -----------------------------------------------------------------------------
if [[ "$REFRESH_PYTHON_STUBS" == "1" ]]; then
    if cmake --build "$BUILD_DIR" --target help | grep -q "refresh_python_stubs"; then
        echo "-- Refreshing Python stubs"
        cmake --build "$BUILD_DIR" --target refresh_python_stubs

        if git -C "$SCRIPT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
            if ! git -C "$SCRIPT_DIR" diff --quiet -- src/python/stubs 2>/dev/null; then
                warn "Python stubs changed. Review and commit them if the API change is intentional:"
                git -C "$SCRIPT_DIR" --no-pager diff --stat -- src/python/stubs || true
            fi
        fi
    else
        warn "refresh_python_stubs target not found; skipping stub refresh."
    fi
else
    echo "-- Python stub refresh disabled"
fi

# -----------------------------------------------------------------------------
# Build
# -----------------------------------------------------------------------------
cmake --build "$BUILD_DIR" --parallel

# -----------------------------------------------------------------------------
# Install portable dist
# -----------------------------------------------------------------------------
if [[ "$BUILD_PORTABLE" == "ON" ]]; then
    patch_cmake_runtime_dependency_dirs \
        "$BUILD_DIR" \
        "$VCPKG_INSTALLED_DIR_LINUX" \
        "$CUDA_HOME"

    cmake --install "$BUILD_DIR" --prefix "$DIST_DIR"

    write_portable_launcher "$DIST_DIR"
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

    APP="$DIST_DIR/LichtFeld-Studio.sh"
    [[ -x "$APP" ]] || die "Expected portable launcher not found: $APP"

    echo "-- Running $APP"
    "$APP"
fi