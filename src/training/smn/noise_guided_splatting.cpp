/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "noise_guided_splatting.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <format>

namespace lfs::training {

    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::Tensor;
    using lfs::core::TensorShape;

    // Forward declaration of CUDA kernel launcher (defined in .cu file)
    void launch_ngs_randomize_colors_seed(float* sh0, int n, int stride_floats, uint32_t seed);

    // ============================================================================
    // NoiseGaussians Implementation
    // ============================================================================

    std::expected<void, std::string> NoiseGaussians::load(const std::filesystem::path& ply_path) {
        if (ply_path.empty()) {
            return std::unexpected("NGS noise PLY path is empty");
        }

        if (!std::filesystem::exists(ply_path)) {
            return std::unexpected(std::format(
                "NGS noise PLY not found: {}",
                lfs::core::path_to_utf8(ply_path)));
        }

        LOG_INFO("Loading NGS noise from: {}", lfs::core::path_to_utf8(ply_path));

        auto result = load_noise_ply(ply_path, kNgsInitialOpacity);
        if (!result) {
            return std::unexpected(result.error());
        }

        auto [means, scaling, rotation, opacity] = std::move(*result);
        means_ = std::move(means);
        scaling_ = std::move(scaling);
        rotation_ = std::move(rotation);
        opacity_ = std::move(opacity);

        // Allocate SH0 for colors
        sh0_ = Tensor::zeros(TensorShape{size(), 1, 3}, Device::CUDA, DataType::Float32);
        randomize_colors();

        LOG_INFO("Loaded {} noise Gaussians", size());
        return {};
    }

    void NoiseGaussians::randomize_colors() {
        if (!is_valid()) return;

        const int n = static_cast<int>(size());
        
        // Deterministic seed from RNG (no per-iteration allocations)
        const uint32_t seed = static_cast<uint32_t>(rng_());

        // SH0 is contiguous; stride = floats per gaussian
        const int stride = static_cast<int>(sh0_.numel() / sh0_.shape()[0]);
        if (stride < 3) return;

        
        launch_ngs_randomize_colors_seed(sh0_.ptr<float>(), n, stride, seed);
    }

    void ngs_randomize_sh0_range_inplace(lfs::core::Tensor& sh0, size_t start, size_t count, uint32_t seed) {
        if (!sh0.is_valid() || sh0.numel() == 0 || count == 0) return;
        if (sh0.device() != Device::CUDA || sh0.dtype() != DataType::Float32) {
            LOG_WARN("NGS: sh0 randomization requires CUDA float32 tensor");
            return;
        }
        const size_t N = sh0.shape()[0];
        if (start >= N) return;
        
        const size_t end = std::min(N, start + count);
        const size_t n = end - start;
        if (n == 0) return;

        const int stride = static_cast<int>(sh0.numel() / sh0.shape()[0]);
        if (stride < 3) {
            LOG_WARN("NGS: sh0 stride < 3, cannot randomize colors");
            return;
        }

        float* base = sh0.ptr<float>() + static_cast<size_t>(stride) * start;
        launch_ngs_randomize_colors_seed(base, static_cast<int>(n), stride, seed);
    }

    // ============================================================================
    // NGSPhaseManager Implementation
    // ============================================================================

    NGSPhaseManager::NGSPhaseManager(const NGSConfig& config, int total_iterations)
        : config_(config) {
        noise_start_iter_ = static_cast<int>(kNgsInjectionStart * total_iterations);
        noise_end_iter_ = static_cast<int>(kNgsRemovalPoint * total_iterations);

        LOG_INFO("NGS phases: Standard[0-{}), WithNoise[{}-{}), Cleanup[{}-{})",
                 noise_start_iter_, noise_start_iter_, noise_end_iter_,
                 noise_end_iter_, total_iterations);
    }

    NGSPhase NGSPhaseManager::get_phase(int iteration) const {
        if (iteration < noise_start_iter_) {
            return NGSPhase::StandardTraining;
        } else if (iteration < noise_end_iter_) {
            return NGSPhase::WithNoise;
        } else {
            return NGSPhase::Cleanup;
        }
    }

    bool NGSPhaseManager::phase_changed(int iteration, int prev_iteration) const {
        return get_phase(iteration) != get_phase(prev_iteration);
    }

    // ============================================================================
    // LearningRateSnapshot Implementation
    // ============================================================================

    LearningRateSnapshot LearningRateSnapshot::capture(const AdamOptimizer& optimizer) {
        LearningRateSnapshot s;
        s.means_lr = optimizer.get_param_lr(ParamType::Means);
        s.sh0_lr = optimizer.get_param_lr(ParamType::Sh0);
        s.shN_lr = optimizer.get_param_lr(ParamType::ShN);
        s.scaling_lr = optimizer.get_param_lr(ParamType::Scaling);
        s.rotation_lr = optimizer.get_param_lr(ParamType::Rotation);
        s.opacity_lr = optimizer.get_param_lr(ParamType::Opacity);
        return s;
    }

    void LearningRateSnapshot::restore(AdamOptimizer& optimizer) const {
        optimizer.set_param_lr(ParamType::Means, means_lr);
        optimizer.set_param_lr(ParamType::Sh0, sh0_lr);
        optimizer.set_param_lr(ParamType::ShN, shN_lr);
        optimizer.set_param_lr(ParamType::Scaling, scaling_lr);
        optimizer.set_param_lr(ParamType::Rotation, rotation_lr);
        optimizer.set_param_lr(ParamType::Opacity, opacity_lr);
    }

    // ============================================================================
    // PLY Loading
    // ============================================================================

    namespace {
        struct NoisePlyLayout {
            size_t vertex_count = 0;
            size_t vertex_stride = 0;
            size_t pos_x = SIZE_MAX, pos_y = SIZE_MAX, pos_z = SIZE_MAX;
            size_t opacity = SIZE_MAX;
            size_t scale[3] = {SIZE_MAX, SIZE_MAX, SIZE_MAX};
            size_t rot[4] = {SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX};
        };

        std::expected<std::pair<size_t, NoisePlyLayout>, std::string>
        parse_header(const char* data, size_t file_size) {
            if (file_size < 10 || std::strncmp(data, "ply", 3) != 0) {
                return std::unexpected("Invalid PLY file");
            }

            NoisePlyLayout layout{};
            const char* ptr = data;
            const char* end = data + file_size;
            bool is_binary = false, in_vertex = false;

            // Skip first line
            while (ptr < end && *ptr != '\n') ++ptr;
            if (ptr < end) {
                ++ptr;
            }

            while (ptr < end) {
                const char* line_start = ptr;
                while (ptr < end && *ptr != '\n' && *ptr != '\r') ++ptr;
                std::string_view line(line_start, ptr - line_start);
                while (ptr < end && (*ptr == '\n' || *ptr == '\r')) ++ptr;

                if (line.starts_with("format binary_little_endian")) {
                    is_binary = true;
                } else if (line.starts_with("element vertex ")) {
                    layout.vertex_count = std::stoull(std::string(line.substr(15)));
                    in_vertex = true;
                } else if (line.starts_with("element ")) {
                    in_vertex = false;
                } else if (line.starts_with("property float ") && in_vertex) {
                    auto name = line.substr(15);
                    if (name == "x") layout.pos_x = layout.vertex_stride;
                    else if (name == "y") layout.pos_y = layout.vertex_stride;
                    else if (name == "z") layout.pos_z = layout.vertex_stride;
                    else if (name == "opacity") layout.opacity = layout.vertex_stride;
                    else if (name == "scale_0") layout.scale[0] = layout.vertex_stride;
                    else if (name == "scale_1") layout.scale[1] = layout.vertex_stride;
                    else if (name == "scale_2") layout.scale[2] = layout.vertex_stride;
                    else if (name == "rot_0") layout.rot[0] = layout.vertex_stride;
                    else if (name == "rot_1") layout.rot[1] = layout.vertex_stride;
                    else if (name == "rot_2") layout.rot[2] = layout.vertex_stride;
                    else if (name == "rot_3") layout.rot[3] = layout.vertex_stride;
                    layout.vertex_stride += 4;
                } else if (line.starts_with("end_header")) {
                    if (!is_binary) return std::unexpected("Only binary PLY supported");
                    return std::make_pair(static_cast<size_t>(ptr - data), layout);
                }
            }
            return std::unexpected("No end_header found");
        }
    } // anonymous namespace

    std::expected<std::tuple<Tensor, Tensor, Tensor, Tensor>, std::string>
    load_noise_ply(const std::filesystem::path& filepath, float initial_opacity) {
        std::ifstream file;
        if (!lfs::core::open_file_for_read(filepath, std::ios::binary, file)) {
            return std::unexpected("Cannot open file");
        }

        file.seekg(0, std::ios::end);
        const std::streampos end_pos = file.tellg();
        if (end_pos <= 0) {
            return std::unexpected("Cannot determine PLY file size");
        }

        std::vector<char> buffer(static_cast<size_t>(end_pos));
        file.seekg(0);
        if (!file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()))) {
            return std::unexpected("Failed to read complete PLY file");
        }

        auto header = parse_header(buffer.data(), buffer.size());
        if (!header) return std::unexpected(header.error());

        auto [offset, layout] = *header;
        const size_t pos_offsets[] = {layout.pos_x, layout.pos_y, layout.pos_z};
        if (!std::all_of(std::begin(pos_offsets), std::end(pos_offsets),
                         [](size_t value) { return value != SIZE_MAX; })) {
            return std::unexpected("Missing position data");
        }

        const char* vdata = buffer.data() + offset;
        const size_t N = layout.vertex_count;
        const size_t stride = layout.vertex_stride;
        if (stride == 0) {
            return std::unexpected("PLY vertex stride is zero");
        }
        if (N > (buffer.size() - offset) / stride) {
            return std::unexpected("PLY file is truncated for declared vertex count");
        }

        const auto has_all = [](const size_t* offsets, size_t count) {
            return std::all_of(offsets, offsets + count, [](size_t value) { return value != SIZE_MAX; });
        };

        auto read_float = [&](size_t i, size_t off) {
            float value = 0.0f;
            std::memcpy(&value, vdata + i * stride + off, sizeof(float));
            return value;
        };

        std::vector<float> means(N * 3), scaling(N * 3), rotation(N * 4), opacity(N);

        for (size_t i = 0; i < N; ++i) {
            means[i * 3 + 0] = read_float(i, layout.pos_x);
            means[i * 3 + 1] = read_float(i, layout.pos_y);
            means[i * 3 + 2] = read_float(i, layout.pos_z);

            if (has_all(layout.scale, 3)) {
                scaling[i * 3 + 0] = read_float(i, layout.scale[0]);
                scaling[i * 3 + 1] = read_float(i, layout.scale[1]);
                scaling[i * 3 + 2] = read_float(i, layout.scale[2]);
            } else {
                scaling[i * 3 + 0] = scaling[i * 3 + 1] = scaling[i * 3 + 2] = -5.0f;
            }

            if (has_all(layout.rot, 4)) {
                rotation[i * 4 + 0] = read_float(i, layout.rot[0]);
                rotation[i * 4 + 1] = read_float(i, layout.rot[1]);
                rotation[i * 4 + 2] = read_float(i, layout.rot[2]);
                rotation[i * 4 + 3] = read_float(i, layout.rot[3]);
            } else {
                rotation[i * 4 + 0] = 1.0f;
                rotation[i * 4 + 1] = rotation[i * 4 + 2] = rotation[i * 4 + 3] = 0.0f;
            }

            if (layout.opacity != SIZE_MAX) {
                opacity[i] = read_float(i, layout.opacity);
            } else {
                // Convert initial_opacity to logit
                float p = std::clamp(initial_opacity, 0.001f, 0.999f);
                opacity[i] = std::log(p / (1.0f - p));
            }
        }

        return std::make_tuple(
            Tensor::from_vector(means, TensorShape{N, 3}, Device::CUDA),
            Tensor::from_vector(scaling, TensorShape{N, 3}, Device::CUDA),
            Tensor::from_vector(rotation, TensorShape{N, 4}, Device::CUDA),
            Tensor::from_vector(opacity, TensorShape{N, 1}, Device::CUDA));
    }

} // namespace lfs::training
