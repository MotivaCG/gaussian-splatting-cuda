/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <expected>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace lfs::training {

    class PPISP;
    class PPISPControllerPool;

    constexpr uint32_t PPISP_FILE_MAGIC = 0x50505349; // "PPIS"
    constexpr uint32_t PPISP_FILE_VERSION = 1;

    struct PPISPFileHeader {
        uint32_t magic = PPISP_FILE_MAGIC;
        uint32_t version = PPISP_FILE_VERSION;
        uint32_t num_cameras = 0;
        uint32_t num_frames = 0;
        uint32_t flags = 0;
        uint32_t reserved[3] = {0, 0, 0};
    };

    enum class PPISPFileFlags : uint32_t {
        NONE = 0,
        HAS_CONTROLLER = 1 << 0,
    };

    inline PPISPFileFlags operator|(PPISPFileFlags a, PPISPFileFlags b) {
        return static_cast<PPISPFileFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
    }

    inline bool has_flag(uint32_t flags, PPISPFileFlags flag) {
        return (flags & static_cast<uint32_t>(flag)) != 0;
    }

    /// Save PPISP (and optionally controller pool) to standalone file
    /// @param path Output file path (typically .ppisp extension)
    /// @param ppisp Required - the trained PPISP module
    /// @param controller_pool Optional - controller pool for novel views
    [[nodiscard]] std::expected<void, std::string> save_ppisp_file(
        const std::filesystem::path& path,
        const PPISP& ppisp,
        const PPISPControllerPool* controller_pool = nullptr);

    /// Load PPISP state from standalone file
    /// @param path Input file path
    /// @param ppisp PPISP instance to load into (must be pre-constructed with matching dimensions)
    /// @param controller_pool Optional controller pool to load into
    [[nodiscard]] std::expected<void, std::string> load_ppisp_file(
        const std::filesystem::path& path,
        PPISP& ppisp,
        PPISPControllerPool* controller_pool = nullptr);

    /// Check if a PPISP companion file exists for a given PLY/splat file
    /// Returns the companion path if it exists, empty path otherwise
    std::filesystem::path find_ppisp_companion(const std::filesystem::path& splat_path);

    /// Get the companion file path for a splat file (doesn't check existence)
    std::filesystem::path get_ppisp_companion_path(const std::filesystem::path& splat_path);

} // namespace lfs::training
