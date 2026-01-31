/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "strategy_utils.hpp"
#include "core/logger.hpp"
#include <algorithm>
#include <cmath>

namespace lfs::training {

    void initialize_gaussians(lfs::core::SplatData& splat_data, int max_cap) {
        // Tensors are already on GPU in the new framework (created with Device::CUDA by default)
        // Gradients are now owned by AdamOptimizer, not SplatData

        // Pre-allocate tensor capacity to avoid reallocations during MCMC operations
        // This eliminates memory fragmentation from varying tensor sizes
        if (max_cap > 0) {
            const size_t capacity = static_cast<size_t>(max_cap);
            LOG_INFO("Pre-allocating tensor capacity for {} Gaussians (parameters)", capacity);

            // Reserve capacity for all parameters (skip empty tensors like shN at sh_degree=0)
            auto reserve_if_valid = [capacity](lfs::core::Tensor& t) {
                if (t.is_valid() && t.numel() > 0)
                    t.reserve(capacity);
            };
            reserve_if_valid(splat_data.means());
            reserve_if_valid(splat_data.sh0());
            reserve_if_valid(splat_data.shN());
            reserve_if_valid(splat_data.scaling_raw());
            reserve_if_valid(splat_data.rotation_raw());
            reserve_if_valid(splat_data.opacity_raw());
        }
    }

    std::unique_ptr<AdamOptimizer> create_optimizer(
        lfs::core::SplatData& splat_data,
        const lfs::core::param::OptimizationParameters& params) {

        // Create Adam config with per-parameter learning rates
        AdamConfig config;
        config.lr = params.means_lr * splat_data.get_scene_scale(); // Default LR (for means)
        // Use double literals (not float!) to match legacy precision
        config.beta1 = 0.9;
        config.beta2 = 0.999;
        config.eps = 1e-15;

        // Set per-parameter learning rates (matching legacy MCMC strategy)
        config.param_lrs["means"] = params.means_lr * splat_data.get_scene_scale();
        config.param_lrs["sh0"] = params.shs_lr;
        config.param_lrs["shN"] = params.shs_lr / 20.0f; // ShN uses reduced LR (1/20 of SH0)
        config.param_lrs["scaling"] = params.scaling_lr;
        config.param_lrs["rotation"] = params.rotation_lr;
        config.param_lrs["opacity"] = params.opacity_lr;

        // Pre-allocate optimizer state capacity to avoid reallocations during training
        // This dramatically reduces peak memory usage by avoiding double-buffering during growth
        if (params.max_cap > 0) {
            config.initial_capacity = static_cast<size_t>(params.max_cap);
            config.growth_factor = 1.5f; // Still allow growth beyond max_cap if needed
            LOG_INFO("AdamOptimizer: pre-allocating capacity for {} Gaussians (optimizer states)", config.initial_capacity);
        }

        LOG_DEBUG("Creating optimizer with per-parameter LRs:");
        LOG_DEBUG("  means: {:.2e}", config.param_lrs["means"]);
        LOG_DEBUG("  sh0: {:.2e}", config.param_lrs["sh0"]);
        LOG_DEBUG("  shN: {:.2e}", config.param_lrs["shN"]);
        LOG_DEBUG("  scaling: {:.2e}", config.param_lrs["scaling"]);
        LOG_DEBUG("  rotation: {:.2e}", config.param_lrs["rotation"]);
        LOG_DEBUG("  opacity: {:.2e}", config.param_lrs["opacity"]);

        auto optimizer = std::make_unique<AdamOptimizer>(splat_data, config);

        return optimizer;
    }

    std::unique_ptr<ExponentialLR> create_scheduler(
        const lfs::core::param::OptimizationParameters& params,
        AdamOptimizer& optimizer) {

        // Python: gamma = 0.01^(1/max_steps)
        // This means after max_steps, lr will be 0.01 * initial_lr
        const double gamma = std::pow(0.01, 1.0 / params.iterations);

        return std::make_unique<ExponentialLR>(optimizer, gamma, std::vector<ParamType>{ParamType::Means});
    }

    void update_param_with_optimizer(
        const ParamUpdateFn& param_fn,
        const OptimizerUpdateFn& optimizer_fn,
        std::unique_ptr<AdamOptimizer>& optimizer,
        lfs::core::SplatData& splat_data,
        std::vector<size_t> param_idxs) {

        // CRITICAL: Ensure CUDA device is set for this thread
        // Some operations might spawn TBB threads, and those need CUDA context
        cudaSetDevice(0);

        // Map param index to ParamType
        auto index_to_param_type = [](size_t idx) -> ParamType {
            switch (idx) {
            case 0: return ParamType::Means;
            case 1: return ParamType::Sh0;
            case 2: return ParamType::ShN;
            case 3: return ParamType::Scaling;
            case 4: return ParamType::Rotation;
            case 5: return ParamType::Opacity;
            default:
                LOG_ERROR("Invalid parameter index: {}", idx);
                return ParamType::Means;
            }
        };

        // Get references to all parameters
        // (Gradients are now owned by AdamOptimizer, not SplatData)
        std::array<lfs::core::Tensor*, 6> params = {
            &splat_data.means(),
            &splat_data.sh0(),
            &splat_data.shN(),
            &splat_data.scaling_raw(),
            &splat_data.rotation_raw(),
            &splat_data.opacity_raw()};

        std::array<lfs::core::Tensor, 6> new_params;

        // First pass: Compute new parameters and update optimizer state
        for (auto i : param_idxs) {
            auto param = params[i];
            cudaError_t err_before = cudaGetLastError();
            if (err_before != cudaSuccess) {
                LOG_ERROR("CUDA error before param_fn: {}", cudaGetErrorString(err_before));
            }

            auto param_type = index_to_param_type(i);
            LOG_DEBUG("Calling param_fn for param {}", i);

            auto new_param = param_fn(i, *param);

            cudaError_t err_after = cudaGetLastError();
            if (err_after != cudaSuccess) {
                LOG_ERROR("CUDA error after param_fn({}) [param_type={}]: {}", i, static_cast<int>(param_type), cudaGetErrorString(err_after));
                throw std::runtime_error(std::string("CUDA error in param_fn (param ") + std::to_string(i) + "): " + cudaGetErrorString(err_after));
            }
            new_params[i] = new_param;

            // Modify state in-place (preserves capacity)
            AdamParamState* state = optimizer->get_state_mutable(param_type);
            if (state) {
                optimizer_fn(*state, new_param);
            }
        }

        // Second pass: Update parameters in SplatData
        // (Gradient updates are handled by the optimizer_fn callback which updates optimizer state)
        for (auto i : param_idxs) {
            if (i == 0) {
                splat_data.means() = new_params[i];
            } else if (i == 1) {
                splat_data.sh0() = new_params[i];
            } else if (i == 2) {
                splat_data.shN() = new_params[i];
            } else if (i == 3) {
                splat_data.scaling_raw() = new_params[i];
            } else if (i == 4) {
                splat_data.rotation_raw() = new_params[i];
            } else if (i == 5) {
                splat_data.opacity_raw() = new_params[i];
            }
        }
    }
    
    bool means_lr_multiplier_enabled(const lfs::core::param::OptimizationParameters& params) {
        // Enabled if multipliers deviate from identity, or if CLI explicitly requested an override.
        return params.means_lr_mul_cli_override ||
               (params.means_lr_mul_start != 1.0f) ||
               (params.means_lr_mul_end != 1.0f);
    }

    namespace {
        inline double compute_means_lr_multiplier(const lfs::core::param::OptimizationParameters& params, const int iter) {
            const double m0 = static_cast<double>(params.means_lr_mul_start);
            const double m1 = static_cast<double>(params.means_lr_mul_end);
            const double p = std::max(1e-8, static_cast<double>(params.means_lr_mul_power));

            if (params.iterations <= 1) {
                return m1;
            }

            const double t = std::clamp(static_cast<double>(iter) / static_cast<double>(params.iterations - 1), 0.0, 1.0);
            const double one_minus_t = 1.0 - t;
            return m1 + (m0 - m1) * std::pow(one_minus_t, p);
        }

        inline double compute_gamma(const lfs::core::param::OptimizationParameters& params) {
            // Matches create_scheduler(): gamma = 0.01^(1/iterations)
            return std::pow(0.01, 1.0 / static_cast<double>(params.iterations));
        }
    } // namespace

    void apply_means_lr_schedule(
        AdamOptimizer& optimizer,
        const lfs::core::param::OptimizationParameters& params,
        const lfs::core::SplatData& splat_data,
        const int iter) {

        // Base LR (same as create_optimizer)
        const double base_lr0 = static_cast<double>(params.means_lr) * static_cast<double>(splat_data.get_scene_scale());

        // Standard exponential decay to 1% over `iterations`
        const double gamma = compute_gamma(params);
        const double decay = std::pow(gamma, static_cast<double>(std::max(0, iter)));

        // Optional multiplier curve
        const double mult = compute_means_lr_multiplier(params, iter);

        const double lr = base_lr0 * decay * mult;

        // Means uses per-parameter LR, but keep global LR in sync for consistency/logging.
        optimizer.set_lr(static_cast<float>(lr));
        optimizer.set_param_lr(ParamType::Means, lr);
        if (iter == 0 || iter == (params.iterations / 2) || iter == (params.iterations - 1) || (iter % 1000) == 0) {
            LOG_INFO("[MeansLR] iter {}/{}  base_lr0={}  mult={}  effective_lr={}\n",
                     iter, params.iterations,
                     static_cast<float>(base_lr0),
                     static_cast<float>(mult),
                     static_cast<float>(lr));
        }
    }

} // namespace lfs::training
