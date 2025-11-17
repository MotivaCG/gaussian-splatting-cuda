# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# lfs_training - Training module (header-only for now, just dataset.hpp)
add_library(lfs_training INTERFACE)

target_include_directories(lfs_training
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

# Link dependencies
target_link_libraries(lfs_training
    INTERFACE
        lfs_core    # Depends on new core module
)

# Set C++ standard
target_compile_features(lfs_training INTERFACE cxx_std_23)

message(STATUS "╔════════════════════════════════════════════════════════════════╗")
message(STATUS "║  lfs_training - Training Dataset Module (Header-only)         ║")
message(STATUS "╚════════════════════════════════════════════════════════════════╝")
message(STATUS "  • Contains: dataset.hpp")
message(STATUS "  • Namespace: lfs::training")
