/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

/**
 * @file noise_guided_splatting.hpp
 * @brief Noise Guided Splatting (NGS) implementation for fixing false transparency
 *
 * Based on: "Fix False Transparency by Noise Guided Splatting" (NeurIPS 2025)
 *
 * NGS injects opaque noise Gaussians inside object volumes to block line-of-sight
 * between front and back surfaces, forcing the optimizer to make surfaces opaque.
 *
 * Simplified training phases (no freeze needed with focused mode):
 * 1. Standard training: [0%, 15%) - normal training without noise
 * 2. With noise: [15%, 80%) - surface + noise train together
 * 3. Cleanup: [80%, 100%] - noise removed, surface continues alone
 */

#include "core/tensor.hpp"
#include "optimizer/adam_optimizer.hpp"
#include <expected>
#include <cstdint>
#include <filesystem>
#include <random>

namespace lfs::training {

    // ========== NGS Timing Constants ==========
    constexpr float kNgsInjectionStart = 0.15f;       // Start noise injection at 15%
    constexpr float kNgsOpacityUnfreezePoint = 0.30f; // Unfreeze noise opacity at 30%
    constexpr float kNgsRemovalPoint = 0.80f;         // Remove noise at 80%
    constexpr float kNgsInitialOpacity = 0.9f;       // Initial opacity (high to block light)

    /**
     * @brief Configuration for Noise Guided Splatting
     */
    struct NGSConfig {
        std::filesystem::path noise_ply_path;  // Path to noise Gaussians PLY (empty = NGS disabled)
    };

    /**
     * @brief Training phase for NGS
     */
    enum class NGSPhase {
        StandardTraining,  // Before injection: normal training, no noise
        WithNoise,         // During: surface + noise train together
        Cleanup            // After removal: noise removed, surface continues
    };

    /**
     * @brief Manages noise Gaussians for NGS training
     */
    class NoiseGaussians {
    public:
        NoiseGaussians() = default;
        ~NoiseGaussians() = default;

        NoiseGaussians(const NoiseGaussians&) = delete;
        NoiseGaussians& operator=(const NoiseGaussians&) = delete;
        NoiseGaussians(NoiseGaussians&&) = default;
        NoiseGaussians& operator=(NoiseGaussians&&) = default;

        /**
         * @brief Load noise Gaussians from PLY file
         */
        std::expected<void, std::string> load(const std::filesystem::path& ply_path);

        bool is_valid() const { return means_.is_valid() && means_.numel() > 0; }
        size_t size() const { return means_.is_valid() ? means_.shape()[0] : 0; }

        const lfs::core::Tensor& means() const { return means_; }
        const lfs::core::Tensor& scaling_raw() const { return scaling_; }
        const lfs::core::Tensor& rotation_raw() const { return rotation_; }
        const lfs::core::Tensor& opacity_raw() const { return opacity_; }
        const lfs::core::Tensor& sh0() const { return sh0_; }

        /**
         * @brief Randomize noise colors (call each iteration during noise phase)
         */
        void randomize_colors();

    private:
        lfs::core::Tensor means_;    // [N, 3]
        lfs::core::Tensor scaling_;  // [N, 3]
        lfs::core::Tensor rotation_; // [N, 4]
        lfs::core::Tensor opacity_;  // [N, 1]
        lfs::core::Tensor sh0_;      // [N, 1, 3] - randomized colors

        std::mt19937 rng_{std::random_device{}()};
    };

    /**
     * @brief NGS phase manager - handles training phase transitions
     */
    class NGSPhaseManager {
    public:
        NGSPhaseManager(const NGSConfig& config, int total_iterations);

        NGSPhase get_phase(int iteration) const;
        bool phase_changed(int iteration, int prev_iteration) const;

        int noise_start() const { return noise_start_iter_; }
        int noise_end() const { return noise_end_iter_; }
        const NGSConfig& config() const { return config_; }

    private:
        NGSConfig config_;
        int noise_start_iter_;
        int noise_end_iter_;
    };

    /**
     * @brief Save/restore learning rates (kept for potential future use)
     */
    struct LearningRateSnapshot {
        double means_lr = 0.0;
        double sh0_lr = 0.0;
        double shN_lr = 0.0;
        double scaling_lr = 0.0;
        double rotation_lr = 0.0;
        double opacity_lr = 0.0;

        static LearningRateSnapshot capture(const AdamOptimizer& optimizer);
        void restore(AdamOptimizer& optimizer) const;
    };

 
    /**
     * @brief Randomize SH0 colors in-place for a contiguous gaussian range.
     *
     * Writes the first 3 SH0 coeffs (RGB) using a deterministic GPU hash from `seed`.
     * This is intended for NGS noise Gaussians, and avoids any per-iteration allocations.
     *
     * @param sh0         SH0 tensor (CUDA, float32) with shape [N, ...]
     * @param start       Start gaussian index in sh0
     * @param count       Number of gaussians to randomize
     * @param seed        Deterministic seed (e.g. iteration number)
     */
    void ngs_randomize_sh0_range_inplace(lfs::core::Tensor& sh0, size_t start, size_t count, uint32_t seed);

    // PLY loading utility
    std::expected<
        std::tuple<lfs::core::Tensor, lfs::core::Tensor, lfs::core::Tensor, lfs::core::Tensor>,
        std::string>
    load_noise_ply(const std::filesystem::path& filepath, float initial_opacity);

} // namespace lfs::training