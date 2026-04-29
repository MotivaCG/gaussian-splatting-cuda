#!/bin/bash
set -e

source ~/.bashrc

# Check required environment variables
if [ -z "$VCPKG_ROOT" ] || [ -z "$CUDA_HOME" ]; then
    echo ""
    echo "ERROR: Required environment variables are not set."
    echo ""
    echo "-- FIRST TIME ONLY --"
    echo "nano ~/.bashrc"
    echo "# add at the end of the file"
    echo "export CUDA_HOME=/usr/local/cuda-12.8"
    echo "export PATH=\$CUDA_HOME/bin:\$PATH"
    echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
    echo "export VCPKG_ROOT=/home/victor/Documentos/SMN/software/vcpkg"
    echo "export PATH=\$VCPKG_ROOT:\$PATH"
    echo ""
    exit 1
fi

# Configure
cmake -B LinuxBuild -G Ninja -DBUILD_CUDA_FATBIN=ON

# -----------------------------------------------------------------------------
# Build options (uncomment the one you want):
# -----------------------------------------------------------------------------

# Option 1: Full build
# cmake --build LinuxBuild

# Option 2: Portable install with bundled dependencies into ./dist
cmake -B LinuxBuild -G Ninja -DBUILD_PORTABLE=ON -DBUILD_CUDA_FATBIN=ON
cmake --build LinuxBuild
cmake --install LinuxBuild --prefix ./dist
./dist/bin/LichtFeld-Studio