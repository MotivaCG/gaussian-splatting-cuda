#!/usr/bin/env bash
set -Eeuo pipefail

# -----------------------------------------------------------------------------
# SMNUpdateVcpkgBaselineLinux.sh
#
# Intentionally update the project's vcpkg builtin-baseline on Linux.
# This script does not configure, build, install, or run LichtFeld Studio.
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
  ./SMNUpdateVcpkgBaselineLinux.sh [options]

Options:
  --allow-dirty  Allow running with local project changes.
  -h, --help     Show this help.

Environment variables:
  VCPKG_ROOT     Default: /home/victor/Documentos/SMN/software/vcpkg

What it does:
  1. Validates VCPKG_ROOT.
  2. Refuses a dirty project tree unless --allow-dirty is passed.
  3. Updates and bootstraps the local vcpkg checkout.
  4. Runs vcpkg x-update-baseline for this project.
  5. Shows the manifest diff for review.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${VCPKG_ROOT:=/home/victor/Documentos/SMN/software/vcpkg}"

ALLOW_DIRTY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --allow-dirty)
            ALLOW_DIRTY=1
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

need_cmd git

[[ -d "$VCPKG_ROOT" ]] || die "VCPKG_ROOT does not exist: $VCPKG_ROOT"
[[ -d "$VCPKG_ROOT/.git" ]] || die "VCPKG_ROOT is not a git repository: $VCPKG_ROOT"
[[ -x "$VCPKG_ROOT/bootstrap-vcpkg.sh" ]] || die "bootstrap-vcpkg.sh not found or not executable: $VCPKG_ROOT/bootstrap-vcpkg.sh"

if [[ "$ALLOW_DIRTY" != "1" ]]; then
    if git -C "$SCRIPT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        STATUS="$(git -C "$SCRIPT_DIR" status --porcelain --untracked-files=all -- . ':!LinuxBuild' ':!dist')"
        if [[ -n "$STATUS" ]]; then
            echo "$STATUS" >&2
            die "Project tree has local changes. Commit/stash them, or rerun with --allow-dirty."
        fi
    else
        warn "Project is not a Git work tree; dirty-tree check skipped."
    fi
fi

echo "-- Using project: $SCRIPT_DIR"
echo "-- Using VCPKG_ROOT: $VCPKG_ROOT"

echo "-- Updating local vcpkg registry"
git -C "$VCPKG_ROOT" pull --ff-only

echo "-- Bootstrapping vcpkg"
"$VCPKG_ROOT/bootstrap-vcpkg.sh"

[[ -x "$VCPKG_ROOT/vcpkg" ]] || die "vcpkg executable not found after bootstrap: $VCPKG_ROOT/vcpkg"

echo "-- Updating project vcpkg builtin-baseline"
"$VCPKG_ROOT/vcpkg" x-update-baseline --x-manifest-root="$SCRIPT_DIR"

if git -C "$SCRIPT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "-- Manifest diff:"
    git -C "$SCRIPT_DIR" --no-pager diff -- vcpkg.json vcpkg-configuration.json || true
else
    warn "Project is not a Git work tree; cannot show manifest diff."
fi

echo "-- Done. Review vcpkg manifest changes before committing."
