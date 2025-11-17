/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#ifdef _WIN32
#define NOMINMAX
#endif

#include "core_new/sogs.hpp"
#include "core_new/logger.hpp"
#include "core_new/splat_data.hpp"
#include "kernels/kmeans_new.cuh"
#include "kernels/morton_encoding_new.cuh"
#include <algorithm>
#include <archive.h>
#include <archive_entry.h>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>
#include <vector>
#include <webp/encode.h>

namespace lfs::core {

    namespace {

#ifdef _WIN32
        using ssize_t = std::ptrdiff_t;
#endif

        // Apply log transform for better quantization
        float log_transform(float value) {
            return std::copysign(std::log(std::abs(value) + 1.0f), value);
        }

        // Pack quaternion into 8-bit values
        std::array<uint8_t, 4> pack_quaternion(float w, float x, float y, float z) {
            // Normalize
            float len = std::sqrt(w * w + x * x + y * y + z * z);
            if (len > 0) {
                w /= len;
                x /= len;
                y /= len;
                z /= len;
            } else {
                // Handle zero-length quaternion: set to identity quaternion
                w = 1.0f;
                x = 0.0f;
                y = 0.0f;
                z = 0.0f;
                LOG_WARN("pack_quaternion: Zero-length quaternion encountered, replaced with identity quaternion.");
            }

            // Find largest component (in absolute value)
            float max_val = std::abs(w);
            int max_idx = 0; // 0 = w, 1 = x, 2 = y, 3 = z

            if (std::abs(x) > max_val) {
                max_val = std::abs(x);
                max_idx = 1;
            }
            if (std::abs(y) > max_val) {
                max_val = std::abs(y);
                max_idx = 2;
            }
            if (std::abs(z) > max_val) {
                max_val = std::abs(z);
                max_idx = 3;
            }

            // Ensure largest component is positive
            if ((max_idx == 0 && w < 0) ||
                (max_idx == 1 && x < 0) ||
                (max_idx == 2 && y < 0) ||
                (max_idx == 3 && z < 0)) {
                w = -w;
                x = -x;
                y = -y;
                z = -z;
            }

            // Scale the quaternion components by sqrt(2)
            constexpr float sqrt2 = 1.41421356237f;
            w *= sqrt2;
            x *= sqrt2;
            y *= sqrt2;
            z *= sqrt2;

            // Pack the other 3 components based on which is largest
            std::array<uint8_t, 4> result;

            if (max_idx == 0) {
                // w is largest, store x, y, z
                result[0] = static_cast<uint8_t>(std::clamp((x * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
                result[1] = static_cast<uint8_t>(std::clamp((y * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
                result[2] = static_cast<uint8_t>(std::clamp((z * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
            } else if (max_idx == 1) {
                // x is largest, store w, y, z
                result[0] = static_cast<uint8_t>(std::clamp((w * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
                result[1] = static_cast<uint8_t>(std::clamp((y * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
                result[2] = static_cast<uint8_t>(std::clamp((z * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
            } else if (max_idx == 2) {
                // y is largest, store w, x, z
                result[0] = static_cast<uint8_t>(std::clamp((w * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
                result[1] = static_cast<uint8_t>(std::clamp((x * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
                result[2] = static_cast<uint8_t>(std::clamp((z * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
            } else {
                // z is largest, store w, x, y
                result[0] = static_cast<uint8_t>(std::clamp((w * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
                result[1] = static_cast<uint8_t>(std::clamp((x * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
                result[2] = static_cast<uint8_t>(std::clamp((y * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
            }

            // Store which component was largest
            result[3] = 252 + max_idx;

            return result;
        }

        // Write WebP image
        bool write_webp_image(const std::filesystem::path& path,
                              const uint8_t* data,
                              int width,
                              int height,
                              int channels = 4) {

            if (!data || width <= 0 || height <= 0) {
                LOG_ERROR("Invalid input to write_webp_image: data={}, width={}, height={}",
                          (void*)data, width, height);
                return false;
            }

            uint8_t* output = nullptr;
            size_t output_size = 0;

            std::vector<uint8_t> rgba_buffer;

            if (channels == 4) {
                rgba_buffer.resize(width * height * 4);
                std::memcpy(rgba_buffer.data(), data, width * height * 4);

                output_size = WebPEncodeLosslessRGBA(
                    rgba_buffer.data(),
                    width,
                    height,
                    width * 4,
                    &output);
            } else if (channels == 3) {
                rgba_buffer.resize(width * height * 4);
                for (int i = 0; i < width * height; ++i) {
                    rgba_buffer[i * 4 + 0] = data[i * 3 + 0];
                    rgba_buffer[i * 4 + 1] = data[i * 3 + 1];
                    rgba_buffer[i * 4 + 2] = data[i * 3 + 2];
                    rgba_buffer[i * 4 + 3] = 255;
                }

                output_size = WebPEncodeLosslessRGBA(
                    rgba_buffer.data(),
                    width,
                    height,
                    width * 4,
                    &output);
            } else {
                LOG_ERROR("Unsupported number of channels: {}", channels);
                return false;
            }

            if (output_size == 0 || output == nullptr) {
                LOG_ERROR("WebP encoding failed for {} (size={})", path.string(), output_size);
                if (output)
                    WebPFree(output);
                return false;
            }

            std::ofstream file(path, std::ios::binary);
            if (!file) {
                WebPFree(output);
                LOG_ERROR("Failed to open file: {}", path.string());
                return false;
            }

            file.write(reinterpret_cast<const char*>(output), output_size);
            WebPFree(output);

            if (!file.good()) {
                LOG_ERROR("Failed to write file: {}", path.string());
                return false;
            }

            LOG_DEBUG("Successfully wrote WebP: {} ({}x{}, {} bytes)",
                      path.string(), width, height, output_size);
            return true;
        }

        // Create a ZIP archive for .sog bundle
        class SogArchive {
            struct archive* a;
            std::filesystem::path path;

        public:
            SogArchive(const std::filesystem::path& output_path) : path(output_path) {
                a = archive_write_new();
                archive_write_set_format_zip(a);
                archive_write_open_filename(a, path.string().c_str());
            }

            ~SogArchive() {
                if (a) {
                    archive_write_close(a);
                    archive_write_free(a);
                }
            }

            bool add_file(const std::string& filename, const void* data, size_t size) {
                struct archive_entry* entry = archive_entry_new();

                auto now = std::chrono::system_clock::now();
                auto time_t = std::chrono::system_clock::to_time_t(now);

                archive_entry_set_pathname(entry, filename.c_str());
                archive_entry_set_size(entry, size);
                archive_entry_set_filetype(entry, AE_IFREG);
                archive_entry_set_perm(entry, 0644);
                archive_entry_set_mtime(entry, time_t, 0);

                if (archive_write_header(a, entry) != ARCHIVE_OK) {
                    archive_entry_free(entry);
                    LOG_ERROR("Failed to write archive header: {}", archive_error_string(a));
                    return false;
                }

                if (archive_write_data(a, data, size) != static_cast<ssize_t>(size)) {
                    archive_entry_free(entry);
                    LOG_ERROR("Failed to write archive data: {}", archive_error_string(a));
                    return false;
                }

                archive_entry_free(entry);
                return true;
            }

            bool add_webp(const std::string& filename, const uint8_t* data,
                          int width, int height, int channels = 4) {

                if (!data || width <= 0 || height <= 0) {
                    LOG_ERROR("Invalid input to add_webp: data={}, width={}, height={}",
                              (void*)data, width, height);
                    return false;
                }

                uint8_t* output = nullptr;
                size_t output_size = 0;

                std::vector<uint8_t> rgba_buffer;

                if (channels == 4) {
                    rgba_buffer.resize(width * height * 4);
                    std::memcpy(rgba_buffer.data(), data, width * height * 4);

                    output_size = WebPEncodeLosslessRGBA(
                        rgba_buffer.data(),
                        width,
                        height,
                        width * 4,
                        &output);
                } else if (channels == 3) {
                    rgba_buffer.resize(width * height * 4);
                    for (int i = 0; i < width * height; ++i) {
                        rgba_buffer[i * 4 + 0] = data[i * 3 + 0];
                        rgba_buffer[i * 4 + 1] = data[i * 3 + 1];
                        rgba_buffer[i * 4 + 2] = data[i * 3 + 2];
                        rgba_buffer[i * 4 + 3] = 255;
                    }

                    output_size = WebPEncodeLosslessRGBA(
                        rgba_buffer.data(),
                        width,
                        height,
                        width * 4,
                        &output);
                } else {
                    LOG_ERROR("Unsupported number of channels: {}", channels);
                    return false;
                }

                if (output_size == 0 || output == nullptr) {
                    LOG_ERROR("WebP encoding failed for {} in archive", filename);
                    if (output)
                        WebPFree(output);
                    return false;
                }

                bool result = add_file(filename, output, output_size);
                WebPFree(output);

                if (result) {
                    LOG_DEBUG("Added {} to archive ({}x{}, {} bytes)",
                              filename, width, height, output_size);
                }

                return result;
            }
        };

        // Identity layout function - matches TypeScript
        int identity_layout(int index, int width) {
            return index;
        }

    } // anonymous namespace

    std::expected<void, std::string> write_sog(
        const SplatData& splat_data,
        const SogWriteOptions& options) {

        try {
            LOG_INFO("Writing SOG format (new) to: {}", options.output_path.string());

            const int64_t num_splats = splat_data.size();
            if (num_splats == 0) {
                return std::unexpected("No splats to write");
            }

            // Calculate texture dimensions (multiple of 4) - matches TypeScript
            const int width = ((int)std::ceil(std::sqrt(num_splats) / 4.0)) * 4;
            const int height = ((int)std::ceil(num_splats / (float)width / 4.0)) * 4;
            const int channels = 4; // Always use RGBA

            LOG_DEBUG("SOG texture dimensions: {}x{} for {} splats", width, height, num_splats);

            // Get data tensors - already on CUDA
            auto means = splat_data.means_raw().cuda();
            auto scales = splat_data.scaling_raw().cuda();
            auto rotations = splat_data.rotation_raw().cuda();
            auto opacities = splat_data.get_opacity().cuda(); // Apply sigmoid
            auto sh0 = splat_data.sh0_raw().cuda();
            auto shN = splat_data.shN_raw().cuda();

            // Determine SH degree from shN shape
            int sh_degree = splat_data.get_max_sh_degree();
            LOG_DEBUG("Detected SH degree: {}", sh_degree);

            // Morton encoding for spatial coherence
            auto morton_codes = morton_encode_new(means);
            auto indices = morton_sort_indices_new(morton_codes).cpu();

            // Check if output is .sog bundle or individual files
            bool is_bundle = options.output_path.extension() == ".sog";
            std::unique_ptr<SogArchive> archive;
            std::filesystem::path base_path;

            if (is_bundle) {
                archive = std::make_unique<SogArchive>(options.output_path);
                base_path = options.output_path.parent_path();
            } else {
                base_path = options.output_path.parent_path();
                std::filesystem::create_directories(base_path);
            }

            // Helper lambda to write images
            auto write_image = [&](const std::string& filename,
                                   const uint8_t* data,
                                   int w = -1, int h = -1) -> bool {
                if (w == -1)
                    w = width;
                if (h == -1)
                    h = height;

                if (!data) {
                    LOG_ERROR("Null data pointer for {}", filename);
                    return false;
                }

                if (archive) {
                    LOG_DEBUG("Adding {} to archive ({}x{})", filename, w, h);
                    return archive->add_webp(filename, data, w, h, channels);
                } else {
                    auto file_path = base_path / filename;
                    auto webp_path = file_path;
                    if (webp_path.extension() != ".webp") {
                        webp_path.replace_extension(".webp");
                    }
                    LOG_DEBUG("Writing {} ({}x{})", webp_path.string(), w, h);
                    return write_webp_image(webp_path, data, w, h, channels);
                }
            };

            LOG_DEBUG("Processing positions with log transform");

            // 1. Process positions with log transform
            std::vector<uint8_t> means_l(width * height * channels, 255);
            std::vector<uint8_t> means_u(width * height * channels, 255);

            // Apply log transform and find min/max
            auto means_cpu = means.cpu();
            Tensor means_log = Tensor::empty(means.shape(), Device::CPU, DataType::Float32);

            // Manual log transform
            for (int64_t i = 0; i < num_splats; ++i) {
                for (int j = 0; j < 3; ++j) {
                    float val = means_cpu[i][j]; // Already returns float for 2D access
                    means_log[i][j] = log_transform(val);
                }
            }

            // Find min/max per dimension
            auto means_min = means_log.min(0, true);
            auto means_max = means_log.max(0, true);

            // Quantize to 16-bit
            for (int64_t i = 0; i < num_splats; ++i) {
                int64_t idx = indices[i].item_as<int64_t>();
                int ti = identity_layout(i, width);

                float x = (means_log[idx][0] - means_min[0].item()) /
                          (means_max[0].item() - means_min[0].item() + 1e-10f);
                float y = (means_log[idx][1] - means_min[1].item()) /
                          (means_max[1].item() - means_min[1].item() + 1e-10f);
                float z = (means_log[idx][2] - means_min[2].item()) /
                          (means_max[2].item() - means_min[2].item() + 1e-10f);

                uint16_t x16 = static_cast<uint16_t>(65535 * std::clamp(x, 0.0f, 1.0f));
                uint16_t y16 = static_cast<uint16_t>(65535 * std::clamp(y, 0.0f, 1.0f));
                uint16_t z16 = static_cast<uint16_t>(65535 * std::clamp(z, 0.0f, 1.0f));

                means_l[ti * 4 + 0] = x16 & 0xff;
                means_l[ti * 4 + 1] = y16 & 0xff;
                means_l[ti * 4 + 2] = z16 & 0xff;

                means_u[ti * 4 + 0] = (x16 >> 8) & 0xff;
                means_u[ti * 4 + 1] = (y16 >> 8) & 0xff;
                means_u[ti * 4 + 2] = (z16 >> 8) & 0xff;
            }

            if (!write_image("means_l.webp", means_l.data())) {
                return std::unexpected("Failed to write means_l.webp");
            }
            if (!write_image("means_u.webp", means_u.data())) {
                return std::unexpected("Failed to write means_u.webp");
            }

            LOG_DEBUG("Processing quaternions");

            // 2. Process quaternions
            std::vector<uint8_t> quats(width * height * channels, 255);
            auto rotations_cpu = rotations.cpu();

            for (int64_t i = 0; i < num_splats; ++i) {
                int64_t idx = indices[i].item_as<int64_t>();
                int ti = identity_layout(i, width);

                auto quat = pack_quaternion(
                    rotations_cpu[idx][0],
                    rotations_cpu[idx][1],
                    rotations_cpu[idx][2],
                    rotations_cpu[idx][3]);

                quats[ti * 4 + 0] = quat[0];
                quats[ti * 4 + 1] = quat[1];
                quats[ti * 4 + 2] = quat[2];
                quats[ti * 4 + 3] = quat[3];
            }

            if (!write_image("quats.webp", quats.data())) {
                return std::unexpected("Failed to write quats.webp");
            }

            // 3. Cluster scales using k-means
            LOG_DEBUG("Clustering scales with k=256, iterations={}", options.iterations);

            // Flatten scales in column-major order
            auto scales_flat = Tensor::empty({static_cast<size_t>(num_splats * 3)}, Device::CUDA, DataType::Float32);
            auto scales_cpu = scales.cpu();

            for (int64_t i = 0; i < num_splats; ++i) {
                scales_flat[i] = scales_cpu[i][0];
                scales_flat[num_splats + i] = scales_cpu[i][1];
                scales_flat[2 * num_splats + i] = scales_cpu[i][2];
            }

            auto [scales_centroids, scales_labels] = cuda::kmeans_1d_new(
                scales_flat, 256, options.iterations);

            std::vector<uint8_t> scales_data(width * height * channels, 255);
            auto scales_labels_cpu = scales_labels.cpu();

            for (int64_t i = 0; i < num_splats; ++i) {
                int64_t idx = indices[i].item_as<int64_t>();
                int ti = identity_layout(i, width);

                scales_data[ti * 4 + 0] = static_cast<uint8_t>(scales_labels_cpu[idx].item_as<int>());
                scales_data[ti * 4 + 1] = static_cast<uint8_t>(scales_labels_cpu[num_splats + idx].item_as<int>());
                scales_data[ti * 4 + 2] = static_cast<uint8_t>(scales_labels_cpu[2 * num_splats + idx].item_as<int>());
            }

            if (!write_image("scales.webp", scales_data.data())) {
                return std::unexpected("Failed to write scales.webp");
            }

            // 4. Cluster colors using k-means
            LOG_DEBUG("Clustering colors with k=256, iterations={}", options.iterations);

            auto sh0_reshaped = sh0.reshape({static_cast<int>(num_splats), 3});

            // Create concatenated 1D tensor in column-major order
            Tensor colors_1d = Tensor::empty({static_cast<size_t>(num_splats * 3)}, Device::CUDA, DataType::Float32);
            auto sh0_cpu = sh0_reshaped.cpu();

            for (int64_t i = 0; i < num_splats; ++i) {
                colors_1d[i] = sh0_cpu[i][0];
                colors_1d[num_splats + i] = sh0_cpu[i][1];
                colors_1d[2 * num_splats + i] = sh0_cpu[i][2];
            }

            auto [colors_centroids, colors_labels] = cuda::kmeans_1d_new(
                colors_1d, 256, options.iterations);

            std::vector<uint8_t> sh0_data(width * height * channels, 0);
            auto colors_labels_cpu = colors_labels.cpu();
            auto opacities_cpu = opacities.cpu();

            for (int64_t i = 0; i < num_splats; ++i) {
                int64_t idx = indices[i].item_as<int64_t>();
                int ti = identity_layout(i, width);

                sh0_data[ti * 4 + 0] = static_cast<uint8_t>(colors_labels_cpu[idx].item_as<int>());
                sh0_data[ti * 4 + 1] = static_cast<uint8_t>(colors_labels_cpu[num_splats + idx].item_as<int>());
                sh0_data[ti * 4 + 2] = static_cast<uint8_t>(colors_labels_cpu[2 * num_splats + idx].item_as<int>());

                float opacity = opacities_cpu[idx].item();
                sh0_data[ti * 4 + 3] = static_cast<uint8_t>(255 * opacity);
            }

            if (!write_image("sh0.webp", sh0_data.data())) {
                return std::unexpected("Failed to write sh0.webp");
            }

            // Create meta.json
            nlohmann::json meta;
            meta["version"] = 2;
            meta["count"] = num_splats;
            meta["width"] = width;
            meta["height"] = height;

            // Store means min/max
            meta["means"]["mins"] = {
                means_min[0].item(),
                means_min[1].item(),
                means_min[2].item()};
            meta["means"]["maxs"] = {
                means_max[0].item(),
                means_max[1].item(),
                means_max[2].item()};
            meta["means"]["files"] = {"means_l.webp", "means_u.webp"};

            // Convert scale centroids to vector
            std::vector<float> scale_codebook;
            auto scales_centroids_cpu = scales_centroids.cpu();
            for (int i = 0; i < scales_centroids.size(0); ++i) {
                scale_codebook.push_back(scales_centroids_cpu[i][0]);
            }
            meta["scales"]["codebook"] = scale_codebook;
            meta["scales"]["files"] = {"scales.webp"};

            meta["quats"]["files"] = {"quats.webp"};

            // Convert color centroids to vector
            std::vector<float> color_codebook;
            auto colors_centroids_cpu = colors_centroids.cpu();
            for (int i = 0; i < colors_centroids.size(0); ++i) {
                color_codebook.push_back(colors_centroids_cpu[i][0]);
            }
            meta["sh0"]["codebook"] = color_codebook;
            meta["sh0"]["files"] = {"sh0.webp"};

            // Handle higher-order spherical harmonics if present
            if (sh_degree > 0 && shN.is_valid() && shN.numel() > 0) {
                LOG_DEBUG("Processing spherical harmonics bands (degree {})", sh_degree);

                const int sh_coeffs = shN.size(2);

                // Flatten SH coefficients for clustering
                auto shN_reshaped = shN.reshape({static_cast<int>(num_splats), sh_coeffs * 3});

                // Calculate palette size
                int palette_size = std::min(64,
                                            std::max(1, static_cast<int>(std::pow(2, std::floor(std::log2(num_splats / 1024.0)))) * 1024));
                palette_size = std::min(palette_size, static_cast<int>(num_splats));

                LOG_DEBUG("Clustering SH with palette_size={}, sh_coeffs={}", palette_size, sh_coeffs);

                // Cluster SH coefficients
                auto [sh_centroids, sh_labels] = cuda::kmeans_new(
                    shN_reshaped, palette_size, options.iterations);

                if (sh_centroids.size(0) == 0) {
                    LOG_WARN("SH clustering returned empty centroids, skipping SH compression");
                } else {
                    int actual_palette_size = sh_centroids.size(0);
                    LOG_DEBUG("SH clustering complete, actual_palette_size={}", actual_palette_size);

                    // Further cluster the centroids to create codebook
                    auto [codebook_centroids, codebook_labels] = cuda::kmeans_1d_new(
                        sh_centroids.flatten(), 256, options.iterations);

                    // Calculate dimensions for centroids texture
                    const int centroids_width = 64 * sh_coeffs;
                    const int centroids_height = (actual_palette_size + 63) / 64;

                    LOG_DEBUG("Writing SH centroids with dimensions {}x{}",
                              centroids_width, centroids_height);

                    // Write centroids with proper band-major ordering
                    std::vector<uint8_t> centroids_buf(centroids_width * centroids_height * channels, 255);
                    auto codebook_labels_cpu = codebook_labels.cpu();

                    for (int i = 0; i < actual_palette_size; ++i) {
                        for (int j = 0; j < sh_coeffs; ++j) {
                            int pixel_idx = i * sh_coeffs + j;

                            if (pixel_idx < centroids_width * centroids_height) {
                                for (int c = 0; c < 3; ++c) {
                                    int coeff_idx = j + c * sh_coeffs;
                                    int centroid_idx = i * (sh_coeffs * 3) + coeff_idx;

                                    if (centroid_idx < codebook_labels.size(0)) {
                                        centroids_buf[pixel_idx * 4 + c] =
                                            static_cast<uint8_t>(codebook_labels_cpu[centroid_idx].item_as<int>());
                                    }
                                }
                            }
                        }
                    }

                    if (!write_image("shN_centroids.webp", centroids_buf.data(), centroids_width, centroids_height)) {
                        return std::unexpected("Failed to write shN_centroids.webp");
                    }

                    LOG_DEBUG("Writing SH labels");

                    // Write labels
                    std::vector<uint8_t> labels_buf(width * height * channels, 255);
                    auto sh_labels_cpu = sh_labels.cpu();

                    for (int64_t i = 0; i < num_splats; ++i) {
                        int64_t idx = indices[i].item_as<int64_t>();
                        int32_t label = sh_labels_cpu[idx].item_as<int>();
                        int ti = identity_layout(i, width);

                        labels_buf[ti * 4 + 0] = label & 0xff;
                        labels_buf[ti * 4 + 1] = (label >> 8) & 0xff;
                        labels_buf[ti * 4 + 2] = 0;
                    }

                    if (!write_image("shN_labels.webp", labels_buf.data())) {
                        return std::unexpected("Failed to write shN_labels.webp");
                    }

                    // Add to meta.json
                    std::vector<float> sh_codebook;
                    auto codebook_centroids_cpu = codebook_centroids.cpu();
                    int codebook_size = std::min(256, static_cast<int>(codebook_centroids.size(0)));
                    for (int i = 0; i < codebook_size; ++i) {
                        sh_codebook.push_back(codebook_centroids_cpu[i][0]);
                    }

                    meta["shN"]["codebook"] = sh_codebook;
                    meta["shN"]["palette_size"] = actual_palette_size;
                    meta["shN"]["bands"] = sh_degree;
                    meta["shN"]["coeffs"] = sh_coeffs;
                    meta["shN"]["files"] = {"shN_centroids.webp", "shN_labels.webp"};

                    LOG_DEBUG("SH processing complete - codebook size: {}, palette: {}, bands: {}, coeffs: {}",
                              sh_codebook.size(), actual_palette_size, sh_degree, sh_coeffs);
                }
            }

            // Write meta.json
            std::string meta_json = meta.dump(2);

            if (archive) {
                LOG_INFO("Writing meta.json to archive");
                if (!archive->add_file("meta.json", meta_json.c_str(), meta_json.size())) {
                    return std::unexpected("Failed to write meta.json to archive");
                }
                LOG_INFO("Successfully wrote SOG bundle: {}", options.output_path.string());
            } else {
                auto meta_path = options.output_path;
                if (meta_path.extension() != ".json") {
                    meta_path = base_path / "meta.json";
                }

                LOG_INFO("Writing meta.json to: {}", meta_path.string());
                std::ofstream meta_file(meta_path);
                if (!meta_file) {
                    LOG_ERROR("Failed to open meta.json for writing at: {}", meta_path.string());
                    return std::unexpected("Failed to open meta.json for writing");
                }
                meta_file << meta_json;
                meta_file.close();

                if (!meta_file) {
                    LOG_ERROR("Failed to write meta.json");
                    return std::unexpected("Failed to write meta.json");
                }

                LOG_INFO("Successfully wrote SOG format as individual files to: {}", base_path.string());
            }

            LOG_INFO("Successfully completed SOG write with {} splats", num_splats);

            return {};

        } catch (const std::exception& e) {
            LOG_ERROR("Exception in write_sog: {}", e.what());
            return std::unexpected(std::format("Failed to write SOG: {}", e.what()));
        }
    }
} // namespace lfs::core