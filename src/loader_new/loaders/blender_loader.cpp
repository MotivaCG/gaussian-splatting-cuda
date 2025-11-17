/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "loader_new/loaders/blender_loader.hpp"
#include "core_new/camera.hpp"
#include "core_new/logger.hpp"
#include "core_new/point_cloud.hpp"
#include "formats/transforms.hpp"
#include "training_new/dataset.hpp"
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <nlohmann/json.hpp>

namespace lfs::loader {

    // Import types from lfs::core for convenience
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::PointCloud;
    using lfs::core::Tensor;

    std::expected<LoadResult, std::string> BlenderLoader::load(
        const std::filesystem::path& path,
        const LoadOptions& options) {

        LOG_TIMER("Blender/NeRF Loading");
        auto start_time = std::chrono::high_resolution_clock::now();

        // Validate path exists
        if (!std::filesystem::exists(path)) {
            std::string error_msg = std::format("Path does not exist: {}", path.string());
            LOG_ERROR("{}", error_msg);
            return std::unexpected(error_msg);
        }

        // Report initial progress
        if (options.progress) {
            options.progress(0.0f, "Loading Blender/NeRF dataset...");
        }

        // Determine transforms file path
        std::filesystem::path transforms_file;

        if (std::filesystem::is_directory(path)) {
            // Look for transforms files in directory
            if (std::filesystem::exists(path / "transforms_train.json")) {
                transforms_file = path / "transforms_train.json";
                LOG_DEBUG("Found transforms_train.json");
            } else if (std::filesystem::exists(path / "transforms.json")) {
                transforms_file = path / "transforms.json";
                LOG_DEBUG("Found transforms.json");
            } else {
                std::string error_msg = "No transforms file found (expected 'transforms.json' or 'transforms_train.json')";
                LOG_ERROR("{}", error_msg);
                return std::unexpected(error_msg);
            }
        } else if (path.extension() == ".json") {
            // Direct path to transforms file
            transforms_file = path;
            LOG_DEBUG("Using direct transforms file: {}", transforms_file.string());
        } else {
            std::string error_msg = "Path must be a directory or a JSON file";
            LOG_ERROR("{}", error_msg);
            return std::unexpected(error_msg);
        }

        // Validation only mode
        if (options.validate_only) {
            LOG_DEBUG("Validation only mode for Blender/NeRF: {}", transforms_file.string());
            // Check if the transforms file is valid JSON
            std::ifstream file(transforms_file);
            if (!file) {
                std::string error_msg = std::format("Cannot open transforms file: {}", transforms_file.string());
                LOG_ERROR("{}", error_msg);
                return std::unexpected(error_msg);
            }

            // Try to parse as JSON (basic validation)
            try {
                nlohmann::json j;
                file >> j;

                if (!j.contains("frames") || !j["frames"].is_array()) {
                    std::string error_msg = "Invalid transforms file: missing 'frames' array";
                    LOG_ERROR("{}", error_msg);
                    return std::unexpected(error_msg);
                }
            } catch (const std::exception& e) {
                std::string error_msg = std::format("Invalid JSON: {}", e.what());
                LOG_ERROR("{}", error_msg);
                return std::unexpected(error_msg);
            }

            if (options.progress) {
                options.progress(100.0f, "Blender/NeRF validation complete");
            }

            LOG_DEBUG("Blender/NeRF validation successful");

            auto end_time = std::chrono::high_resolution_clock::now();
            return LoadResult{
                .data = LoadedScene{
                    .cameras = nullptr,
                    .point_cloud = nullptr},
                .scene_center = Tensor::zeros({3}, Device::CPU, DataType::Float32),
                .loader_used = name(),
                .load_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time),
                .warnings = {"Validation mode - point cloud not loaded"}};
        }

        // Load the dataset
        if (options.progress) {
            options.progress(20.0f, "Reading transforms file...");
        }

        try {
            LOG_INFO("Loading Blender/NeRF dataset from: {}", transforms_file.string());

            // Read transforms and create cameras
            auto [camera_infos, scene_center, train_val_split] = read_transforms_cameras_and_images(transforms_file);

            if (options.progress) {
                options.progress(40.0f, std::format("Creating {} cameras...", camera_infos.size()));
            }

            LOG_DEBUG("Creating {} camera objects", camera_infos.size());

            // Convert CameraData to Camera objects
            std::vector<std::shared_ptr<lfs::core::Camera>> cameras;
            cameras.reserve(camera_infos.size());

            for (size_t i = 0; i < camera_infos.size(); ++i) {
                const auto& info = camera_infos[i];

                try {
                    auto cam = std::make_shared<lfs::core::Camera>(
                        info._R,
                        info._T,
                        info._focal_x,
                        info._focal_y,
                        info._center_x,
                        info._center_y,
                        info._radial_distortion,
                        info._tangential_distortion,
                        info._camera_model_type,
                        info._image_name,
                        info._image_path,
                        info._width,
                        info._height,
                        static_cast<int>(i));

                    cameras.push_back(cam);
                } catch (const std::exception& e) {
                    LOG_ERROR("Failed to create camera {}: {}", i, e.what());
                    throw;
                }
            }

            // Create dataset configuration
            lfs::core::param::DatasetConfig dataset_config;
            dataset_config.data_path = path;
            dataset_config.images = options.images_folder;
            dataset_config.resize_factor = options.resize_factor;
            dataset_config.max_width = options.max_width;

            // Create dataset with ALL images
            auto dataset = std::make_shared<lfs::training::CameraDataset>(
                std::move(cameras), dataset_config, lfs::training::CameraDataset::Split::ALL);

            if (options.progress) {
                options.progress(60.0f, "Generating initialization point cloud...");
            }

            // Generate random point cloud for initialization
            // Note: Blender/NeRF datasets don't typically include sparse point clouds
            // If a pointcloud.ply exists, users should load it separately via the PLY loader
            LOG_DEBUG("Generating random point cloud for initialization");
            auto random_pc = generate_random_point_cloud();
            auto point_cloud = std::make_shared<PointCloud>(std::move(random_pc));
            LOG_INFO("Generated random point cloud with {} points", point_cloud->size());

            if (options.progress) {
                options.progress(100.0f, "Blender/NeRF loading complete");
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time);

            // Get scene center values for logging
            auto scene_center_cpu = scene_center.cpu();
            const float* sc_ptr = scene_center_cpu.template ptr<float>();

            // Save dataset size before moving
            size_t num_cameras = dataset->size();

            LoadResult result{
                .data = LoadedScene{
                    .cameras = std::move(dataset),
                    .point_cloud = std::move(point_cloud)},
                .scene_center = scene_center,
                .loader_used = name(),
                .load_time = load_time,
                .warnings = {"Using random point cloud initialization"}};

            LOG_INFO("Blender/NeRF dataset loaded successfully in {}ms", load_time.count());
            LOG_INFO("  - {} cameras", num_cameras);
            LOG_DEBUG("  - Scene center: [{:.3f}, {:.3f}, {:.3f}]",
                      sc_ptr[0], sc_ptr[1], sc_ptr[2]);

            return result;

        } catch (const std::exception& e) {
            std::string error_msg = std::format("Failed to load Blender/NeRF dataset: {}", e.what());
            LOG_ERROR("{}", error_msg);
            return std::unexpected(error_msg);
        }
    }

    bool BlenderLoader::canLoad(const std::filesystem::path& path) const {
        if (!std::filesystem::exists(path)) {
            return false;
        }

        if (std::filesystem::is_directory(path)) {
            // Check for transforms files in directory
            return std::filesystem::exists(path / "transforms.json") ||
                   std::filesystem::exists(path / "transforms_train.json");
        } else {
            // Check if it's a JSON file
            return path.extension() == ".json";
        }
    }

    std::string BlenderLoader::name() const {
        return "Blender/NeRF";
    }

    std::vector<std::string> BlenderLoader::supportedExtensions() const {
        return {".json"}; // Can load JSON files directly
    }

    int BlenderLoader::priority() const {
        return 5; // Medium priority
    }

} // namespace lfs::loader