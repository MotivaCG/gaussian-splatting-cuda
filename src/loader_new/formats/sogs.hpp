/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core_new/splat_data.hpp"
#include <expected>
#include <filesystem>
#include <string>

namespace lfs::loader {

    // Import types from lfs::core for convenience
    using lfs::core::SplatData;

    /**
     * @brief Load SOG file and return SplatData
     * @param filepath Path to the SOG file or directory
     * @return SplatData on success, error string on failure
     */
    std::expected<SplatData, std::string> load_sog(const std::filesystem::path& filepath);

} // namespace lfs::loader