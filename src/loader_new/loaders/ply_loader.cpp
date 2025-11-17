/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "ply_loader.hpp"
#include "core_new/logger.hpp"
#include "core_new/splat_data.hpp"
#include "formats/ply.hpp"
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>

namespace lfs::loader {

    // Import types from lfs::core for convenience
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::SplatData;
    using lfs::core::Tensor;

    std::expected<LoadResult, std::string> PLYLoader::load(
        const std::filesystem::path& path,
        const LoadOptions& options) {

        LOG_TIMER("PLY Loading");
        auto start_time = std::chrono::high_resolution_clock::now();

        // Report progress if callback provided
        if (options.progress) {
            options.progress(0.0f, "Loading PLY file...");
        }

        // Validate file exists
        if (!std::filesystem::exists(path)) {
            std::string error_msg = std::format("PLY file does not exist: {}", path.string());
            LOG_ERROR("{}", error_msg);
            return std::unexpected(error_msg);
        }

        if (!std::filesystem::is_regular_file(path)) {
            std::string error_msg = std::format("Path is not a regular file: {}", path.string());
            LOG_ERROR("{}", error_msg);
            return std::unexpected(error_msg);
        }

        // Validation only mode
        if (options.validate_only) {
            LOG_DEBUG("Validation only mode for PLY: {}", path.string());
            // Basic validation - check if it's a PLY file
            std::ifstream file(path, std::ios::binary);
            if (!file) {
                std::string error_msg = std::format("Cannot open file for reading: {}", path.string());
                LOG_ERROR("{}", error_msg);
                return std::unexpected(error_msg);
            }

            std::string header;
            std::getline(file, header);
            if (header != "ply" && header != "ply\r") {
                std::string error_msg = std::format("File does not start with 'ply' header: {}", path.string());
                LOG_ERROR("{}", error_msg);
                return std::unexpected(error_msg);
            }

            if (options.progress) {
                options.progress(100.0f, "PLY validation complete");
            }

            LOG_DEBUG("PLY validation successful");

            // Return empty result for validation only
            LoadResult result;
            result.data = std::shared_ptr<SplatData>{}; // Empty shared_ptr
            result.scene_center = Tensor::zeros({3}, Device::CPU);
            result.loader_used = name();
            result.load_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start_time);
            result.warnings = {};

            return result;
        }

        // Load the PLY file using the new tensor-based implementation
        if (options.progress) {
            options.progress(50.0f, "Parsing PLY data...");
        }

        LOG_INFO("Loading PLY file: {}", path.string());
        auto splat_result = load_ply(path);
        if (!splat_result) {
            std::string error_msg = splat_result.error();
            LOG_ERROR("Failed to load PLY: {}", error_msg);
            return std::unexpected(error_msg);
        }

        if (options.progress) {
            options.progress(100.0f, "PLY loading complete");
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);

        LoadResult result{
            .data = std::make_shared<SplatData>(std::move(*splat_result)),
            .scene_center = Tensor::zeros({3}, Device::CPU),
            .loader_used = name(),
            .load_time = load_time,
            .warnings = {}};

        LOG_INFO("PLY loaded successfully in {}ms", load_time.count());

        return result;
    }

    bool PLYLoader::canLoad(const std::filesystem::path& path) const {
        if (!std::filesystem::exists(path) || std::filesystem::is_directory(path)) {
            return false;
        }

        auto ext = path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        return ext == ".ply";
    }

    std::string PLYLoader::name() const {
        return "PLY";
    }

    std::vector<std::string> PLYLoader::supportedExtensions() const {
        return {".ply", ".PLY"};
    }

    int PLYLoader::priority() const {
        return 10; // Higher priority for PLY files
    }

} // namespace lfs::loader