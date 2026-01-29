/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "ppisp_file.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "ppisp.hpp"
#include "ppisp_controller_pool.hpp"
#include <fstream>

namespace lfs::training {

    std::expected<void, std::string> save_ppisp_file(
        const std::filesystem::path& path,
        const PPISP& ppisp,
        const PPISPControllerPool* controller_pool) {

        try {
            std::ofstream file;
            if (!lfs::core::open_file_for_write(path, std::ios::binary, file)) {
                return std::unexpected("Failed to open file for writing: " + lfs::core::path_to_utf8(path));
            }

            PPISPFileHeader header{};
            header.num_cameras = static_cast<uint32_t>(ppisp.num_cameras());
            header.num_frames = static_cast<uint32_t>(ppisp.num_frames());
            header.flags = 0;
            if (controller_pool) {
                header.flags |= static_cast<uint32_t>(PPISPFileFlags::HAS_CONTROLLER);
            }

            file.write(reinterpret_cast<const char*>(&header), sizeof(header));

            ppisp.serialize_inference(file);

            // Save controller pool
            if (controller_pool) {
                controller_pool->serialize_inference(file);
            }

            LOG_INFO("PPISP file saved: {} ({} cameras, {} frames{})",
                     lfs::core::path_to_utf8(path),
                     header.num_cameras,
                     header.num_frames,
                     controller_pool
                         ? ", +controller_pool(" + std::to_string(controller_pool->num_cameras()) + ")"
                         : "");

            return {};

        } catch (const std::exception& e) {
            return std::unexpected(std::string("Failed to save PPISP file: ") + e.what());
        }
    }

    std::expected<void, std::string> load_ppisp_file(
        const std::filesystem::path& path,
        PPISP& ppisp,
        PPISPControllerPool* controller_pool) {

        try {
            std::ifstream file;
            if (!lfs::core::open_file_for_read(path, std::ios::binary, file)) {
                return std::unexpected("Failed to open file for reading: " + lfs::core::path_to_utf8(path));
            }

            PPISPFileHeader header{};
            file.read(reinterpret_cast<char*>(&header), sizeof(header));

            if (header.magic != PPISP_FILE_MAGIC) {
                return std::unexpected("Invalid PPISP file: wrong magic number");
            }

            if (header.version > PPISP_FILE_VERSION) {
                return std::unexpected("Unsupported PPISP file version: " + std::to_string(header.version));
            }

            const bool is_inference_load = ppisp.num_cameras() == 0 && ppisp.num_frames() == 0;
            if (!is_inference_load &&
                (static_cast<int>(header.num_cameras) != ppisp.num_cameras() ||
                 static_cast<int>(header.num_frames) != ppisp.num_frames())) {
                return std::unexpected(
                    "PPISP dimension mismatch: file has " +
                    std::to_string(header.num_cameras) + " cameras, " +
                    std::to_string(header.num_frames) + " frames; expected " +
                    std::to_string(ppisp.num_cameras()) + " cameras, " +
                    std::to_string(ppisp.num_frames()) + " frames");
            }

            ppisp.deserialize_inference(file);

            if (has_flag(header.flags, PPISPFileFlags::HAS_CONTROLLER)) {
                if (controller_pool) {
                    controller_pool->deserialize_inference(file);
                    LOG_INFO("PPISP file loaded: {} ({} cameras, {} frames, +controller_pool({}))",
                             lfs::core::path_to_utf8(path), header.num_cameras, header.num_frames,
                             controller_pool->num_cameras());
                } else {
                    LOG_WARN("PPISP file has controller pool but none provided - skipping");
                    // Skip controller pool data by reading into a temporary
                    constexpr uint32_t INFERENCE_MAGIC = 0x4C464349;
                    uint32_t magic, version;
                    int num_cameras;
                    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
                    file.read(reinterpret_cast<char*>(&version), sizeof(version));
                    file.read(reinterpret_cast<char*>(&num_cameras), sizeof(num_cameras));
                    // Create temporary pool to skip data
                    PPISPControllerPool temp(num_cameras, 1);
                    file.seekg(-static_cast<std::streamoff>(sizeof(magic) + sizeof(version) + sizeof(num_cameras)),
                               std::ios::cur);
                    temp.deserialize_inference(file);
                    LOG_INFO("PPISP file loaded: {} ({} cameras, {} frames)",
                             lfs::core::path_to_utf8(path), header.num_cameras, header.num_frames);
                }
            } else {
                if (controller_pool) {
                    LOG_WARN("Controller pool requested but not present in PPISP file");
                }
                LOG_INFO("PPISP file loaded: {} ({} cameras, {} frames)",
                         lfs::core::path_to_utf8(path), header.num_cameras, header.num_frames);
            }

            return {};

        } catch (const std::exception& e) {
            return std::unexpected(std::string("Failed to load PPISP file: ") + e.what());
        }
    }

    std::filesystem::path find_ppisp_companion(const std::filesystem::path& splat_path) {
        auto companion = get_ppisp_companion_path(splat_path);
        if (std::filesystem::exists(companion)) {
            return companion;
        }
        return {};
    }

    std::filesystem::path get_ppisp_companion_path(const std::filesystem::path& splat_path) {
        auto path = splat_path;
        path.replace_extension(".ppisp");
        return path;
    }

} // namespace lfs::training
