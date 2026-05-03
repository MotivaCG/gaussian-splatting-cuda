# SMN Linux build notes

This document explains the Linux build scripts used by this project.

Scripts:

- `SMNBuildLinux.sh`
- `SMNUpdateVcpkgBaselineLinux.sh`

The important separation is:

- `SMNBuildLinux.sh` validates the environment and builds the project.
- `SMNUpdateVcpkgBaselineLinux.sh` intentionally updates the vcpkg baseline used by the project.

Do not update the vcpkg baseline automatically in every build. That changes dependency resolution and can introduce new dependency versions unexpectedly.

---

## Expected layout

The examples assume this layout:

```text
/home/victor/Documentos/SMN/software/
|-- gaussian-splatting-cuda/
|   |-- SMNBuildLinux.sh
|   |-- SMNUpdateVcpkgBaselineLinux.sh
|   |-- SMNLinux.md
|   |-- CMakeLists.txt
|   `-- vcpkg.json
`-- vcpkg/
```

Default paths used by the scripts:

```bash
CUDA_HOME=/usr/local/cuda-12.8
VCPKG_ROOT=/home/victor/Documentos/SMN/software/vcpkg
```

These can be overridden through environment variables.

---

## System prerequisites

Install common Linux build tools:

```bash
sudo apt update
sudo apt install -y git cmake ninja-build build-essential pkg-config python3
```

CUDA must be installed and `nvcc` must be available:

```bash
which nvcc
nvcc --version
```

Expected CUDA environment:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

vcpkg environment:

```bash
export VCPKG_ROOT=/home/victor/Documentos/SMN/software/vcpkg
export PATH=$VCPKG_ROOT:$PATH
```

You can add these lines at the end of `~/.bashrc`.

---

## First-time vcpkg setup

```bash
cd /home/victor/Documentos/SMN/software

git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh
```

If `vcpkg` already exists:

```bash
cd /home/victor/Documentos/SMN/software/vcpkg
git pull --ff-only
./bootstrap-vcpkg.sh
```

---

## Make scripts executable

From the project root:

```bash
cd /home/victor/Documentos/SMN/software/gaussian-splatting-cuda

chmod +x SMNBuildLinux.sh SMNUpdateVcpkgBaselineLinux.sh
```

---

## Normal build

```bash
./SMNBuildLinux.sh
```

This will:

1. Load `~/.bashrc` softly.
2. Validate `CUDA_HOME`.
3. Validate `VCPKG_ROOT`.
4. Check required tools: `git`, `cmake`, `ninja`, `python3`, `nvcc`, `cc`, `c++`.
5. Validate the vcpkg `builtin-baseline` if `vcpkg.json` exists.
6. Configure CMake with the vcpkg toolchain.
7. Optionally refresh Python stubs.
8. Build.
9. Install into `./dist`.
10. Run `./dist/LichtFeld-Studio.sh`.

The portable launcher sets these runtime paths before starting the binary:

```bash
LD_LIBRARY_PATH=./dist/lib:./dist/lib/extensions:./dist/bin:$LD_LIBRARY_PATH
PXR_PLUGINPATH_NAME=./dist/lib/usd:$PXR_PLUGINPATH_NAME
```

The launcher then executes `./dist/bin/LichtFeld-Studio`.

---

## Python stub refresh

`SMNBuildLinux.sh` defaults to `REFRESH_PYTHON_STUBS=1`. The `refresh_python_stubs` target can modify committed files under `src/python/stubs` when the native Python API changes.

For faster local builds, or when you do not want stub files touched:

```bash
REFRESH_PYTHON_STUBS=0 ./SMNBuildLinux.sh --no-run
```

---

## Clean build without running the app

```bash
./SMNBuildLinux.sh --clean --no-run
```

This removes:

```bash
LinuxBuild/
dist/
```

Then configures, builds and installs again.

---

## Update local vcpkg before building

Sometimes vcpkg reports errors like:

```text
error: no version database entry for openexr at 3.4.7
error: no version database entry for openssl at 3.6.1#3
```

That usually means the local vcpkg repository is behind the baseline used by the project.

Use:

```bash
UPDATE_VCPKG=1 ./SMNBuildLinux.sh --clean --no-run
```

This runs:

```bash
git -C "$VCPKG_ROOT" pull --ff-only
"$VCPKG_ROOT/bootstrap-vcpkg.sh"
```

It does not change `vcpkg.json`.

---

## Intentionally update the project vcpkg baseline

Use this only when you want to move the project to newer vcpkg dependency versions:

```bash
./SMNUpdateVcpkgBaselineLinux.sh
```

This will:

1. Validate `VCPKG_ROOT`.
2. Refuse a dirty project tree unless `--allow-dirty` is passed.
3. Update the local vcpkg repository.
4. Bootstrap the vcpkg executable.
5. Run:

```bash
vcpkg x-update-baseline --x-manifest-root=<project-root>
```

6. Show the resulting diff for `vcpkg.json` / `vcpkg-configuration.json`.

Review the diff before committing.

If you have local changes and still want to run it:

```bash
./SMNUpdateVcpkgBaselineLinux.sh --allow-dirty
```

---

## Useful build variants

Debug build:

```bash
BUILD_TYPE=Debug ./SMNBuildLinux.sh --clean --no-run
```

Do not run app after build:

```bash
RUN_AFTER_BUILD=0 ./SMNBuildLinux.sh
```

Disable portable install:

```bash
BUILD_PORTABLE=OFF ./SMNBuildLinux.sh --clean --no-run
```

Disable CUDA fatbin:

```bash
BUILD_CUDA_FATBIN=OFF ./SMNBuildLinux.sh --clean --no-run
```

Use a different vcpkg:

```bash
VCPKG_ROOT=/path/to/vcpkg ./SMNBuildLinux.sh --clean --no-run
```

Use a different CUDA:

```bash
CUDA_HOME=/usr/local/cuda-12.8 ./SMNBuildLinux.sh --clean --no-run
```

---

## Troubleshooting

### Ninja not found

Error:

```text
CMake Error: CMake was unable to find a build program corresponding to "Ninja"
```

Fix:

```bash
sudo apt update
sudo apt install -y ninja-build
```

---

### nvcc not found

Error:

```text
ERROR: Missing command: nvcc
```

Check:

```bash
echo $CUDA_HOME
ls "$CUDA_HOME/bin/nvcc"
```

Fix your environment:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

---

### vcpkg baseline commit not found

The build script tries to fetch the baseline commit automatically.

Manual check:

```bash
cd /home/victor/Documentos/SMN/software/vcpkg

git show <baseline-commit>:versions/baseline.json | head
```

If that fails, update vcpkg:

```bash
git pull --ff-only
./bootstrap-vcpkg.sh
```

Then build again:

```bash
cd /home/victor/Documentos/SMN/software/gaussian-splatting-cuda
./SMNBuildLinux.sh --clean --no-run
```

---

### vcpkg version database entry missing

Error example:

```text
error: no version database entry for openexr at 3.4.7
```

Fix:

```bash
UPDATE_VCPKG=1 ./SMNBuildLinux.sh --clean --no-run
```

If that still fails, update the baseline intentionally:

```bash
./SMNUpdateVcpkgBaselineLinux.sh
./SMNBuildLinux.sh --clean --no-run
```

---

## Recommended workflow

For everyday development:

```bash
./SMNBuildLinux.sh --clean --no-run
```

When vcpkg local registry is behind:

```bash
UPDATE_VCPKG=1 ./SMNBuildLinux.sh --clean --no-run
```

When you intentionally want to move dependency versions forward:

```bash
./SMNUpdateVcpkgBaselineLinux.sh
./SMNBuildLinux.sh --clean --no-run
```

Then review and commit the vcpkg manifest changes.
