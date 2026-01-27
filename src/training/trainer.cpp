/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "trainer.hpp"
#include "components/bilateral_grid.hpp"
#include "components/sparsity_optimizer.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/events.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/splat_data_export.hpp"
#include "core/splat_data_transform.hpp"
#include "core/tensor/internal/memory_pool.hpp"
#include "io/cache_image_loader.hpp"
#include "io/exporter.hpp"
#include "io/filesystem_utils.hpp"
#include "lfs/kernels/ssim.cuh"
#include "losses/losses.hpp"
#include "optimizer/adam_optimizer.hpp"
#include "rasterization/fast_rasterizer.hpp"
#include "rasterization/gsplat_rasterizer.hpp"
#include "strategies/adc.hpp"
#include "strategies/mcmc.hpp"
#include "training/kernels/grad_alpha.hpp"
#include "visualizer/scene/scene.hpp"

#include <atomic>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <expected>
#include <memory>
#include <nvtx3/nvToolsExt.h>

#include "smn/mask_penalty.hpp" //matting modes
#include "smn/mask_pruning.hpp"
#include "smn/mask_pruning_visualizer.hpp"

namespace lfs::training {

    // Tile configuration for memory-efficient training
    enum class TileMode {
        One = 1, // 1 tile  - 1x1 - Render full image (no tiling)
        Two = 2, // 2 tiles - 2x1 - Two horizontal tiles
        Four = 4 // 4 tiles - 2x2 - Four tiles in a grid
    };

    void Trainer::cleanup() {
        LOG_DEBUG("Cleaning up trainer for re-initialization");

        // Stop any ongoing operations
        stop_requested_ = true;

        // Sync callback stream to avoid race conditions
        if (callback_stream_) {
            cudaStreamSynchronize(callback_stream_);
        }
        callback_busy_ = false;

        // Reset all components
        progress_.reset();
        bilateral_grid_.reset();
        sparsity_optimizer_.reset();
        evaluator_.reset();

        // Clear datasets (will be recreated)
        train_dataset_.reset();
        val_dataset_.reset();

        // Reset flags
        pause_requested_ = false;
        save_requested_ = false;
        stop_requested_ = false;
        is_paused_ = false;
        is_running_ = false;
        training_complete_ = false;
        ready_to_start_ = false;
        current_iteration_ = 0;
        current_loss_ = 0.0f;

        // NGS
        ngs_noise_.reset();
        ngs_phase_manager_.reset();
        ngs_enabled_ = false;

        LOG_DEBUG("Trainer cleanup complete");
    }

    std::expected<void, std::string> Trainer::initialize_bilateral_grid() {
        if (!params_.optimization.use_bilateral_grid) {
            return {};
        }

        try {
            BilateralGrid::Config config;
            config.lr = params_.optimization.bilateral_grid_lr;

            bilateral_grid_ = std::make_unique<BilateralGrid>(
                static_cast<int>(train_dataset_size_),
                params_.optimization.bilateral_grid_X,
                params_.optimization.bilateral_grid_Y,
                params_.optimization.bilateral_grid_W,
                params_.optimization.iterations,
                config);

            LOG_INFO("Bilateral grid initialized: {}x{}x{} for {} images",
                     params_.optimization.bilateral_grid_X,
                     params_.optimization.bilateral_grid_Y,
                     params_.optimization.bilateral_grid_W,
                     train_dataset_size_);

            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to init bilateral grid: {}", e.what()));
        }
    }

    // Compute photometric loss AND gradient manually
    std::expected<std::pair<lfs::core::Tensor, lfs::core::Tensor>, std::string> Trainer::compute_photometric_loss_with_gradient(
        const lfs::core::Tensor& rendered,
        const lfs::core::Tensor& gt_image,
        const lfs::core::param::OptimizationParameters& opt_params) {
        lfs::training::losses::PhotometricLoss photometric_loss;
        lfs::training::losses::PhotometricLoss::Params params{.lambda_dssim = opt_params.lambda_dssim};
        auto result = photometric_loss.forward(rendered, gt_image, params);
        if (!result) {
            return std::unexpected(result.error());
        }
        auto [loss_tensor, ctx] = *result;
        return std::make_pair(loss_tensor, ctx.grad_image);
    }

    std::expected<void, std::string> Trainer::validate_masks() {
        const auto& opt = params_.optimization;
        if (opt.mask_mode == lfs::core::param::MaskMode::None) {
            return {};
        }

        size_t masks_found = 0;
        for (const auto& cam : train_dataset_->get_cameras()) {
            if (cam && cam->has_mask()) {
                ++masks_found;
            }
        }

        if (masks_found == 0) {
            return std::unexpected(std::format(
                "Mask mode enabled but no masks found in {}/masks/",
                lfs::core::path_to_utf8(params_.dataset.data_path)));
        }

        LOG_INFO("Found {} masks{}", masks_found, opt.invert_masks ? " (inverted)" : "");
        return {};
    }

    
std::expected<Trainer::MaskLossResult, std::string> Trainer::compute_photometric_loss_with_mask(
        const lfs::core::Tensor& rendered,
        const lfs::core::Tensor& gt_image,
        const lfs::core::Tensor& mask,
        const lfs::core::Tensor& alpha,
        const lfs::core::param::OptimizationParameters& opt_params,
        const lfs::core::Tensor* fg_core_2d,
        const lfs::core::Tensor* bg_core_2d) {

        using namespace lfs::core;
        constexpr float EPSILON = 1e-8f;
        constexpr float ALPHA_CONSISTENCY_WEIGHT = 10.0f;

        const auto mode = opt_params.mask_mode;
        const Tensor mask_2d = mask.ndim() == 3 ? mask.squeeze(0) : mask;

        Tensor loss, grad, grad_alpha;

        if (mode == param::MaskMode::Segment || mode == param::MaskMode::Ignore) {
            if (opt_params.lambda_dssim > 0.0f) {
                // Use FUSED masked L1+SSIM kernel
                auto [loss_tensor, ctx] = lfs::training::kernels::masked_fused_l1_ssim_forward(
                    rendered, gt_image, mask_2d, opt_params.lambda_dssim, masked_fused_workspace_);

                grad = lfs::training::kernels::masked_fused_l1_ssim_backward(ctx, masked_fused_workspace_);
                loss = loss_tensor;

                // Squeeze gradient to match input dimensions
                if (grad.ndim() == 4 && rendered.ndim() == 3) {
                    grad = grad.squeeze(0);
                }
            } else {
                // Pure L1 with mask (no SSIM)
                const Tensor mask_expanded = mask_2d.unsqueeze(0).expand({static_cast<int>(rendered.shape()[0]),
                                                                          static_cast<int>(mask_2d.shape()[0]),
                                                                          static_cast<int>(mask_2d.shape()[1])});
                const Tensor mask_sum = mask_expanded.sum() + EPSILON;

                const Tensor l1_diff = (rendered - gt_image).abs();
                const Tensor masked_l1 = (l1_diff * mask_expanded).sum() / mask_sum;
                const Tensor sign_diff = (rendered - gt_image).sign();
                grad = sign_diff * mask_expanded / mask_sum;
                loss = masked_l1;
            }

            // Segment: opacity penalty for background
            if (mode == param::MaskMode::Segment && alpha.is_valid()) {
                const Tensor alpha_2d = alpha.ndim() == 3 ? alpha.squeeze(0) : alpha;
                const Tensor bg_mask = Tensor::full(mask_2d.shape(), 1.0f, mask_2d.device()) - mask_2d;
                const Tensor penalty_weights = bg_mask.pow(opt_params.mask_opacity_penalty_power);
                const Tensor penalty = (alpha_2d * penalty_weights).mean() * opt_params.mask_opacity_penalty_weight;

                const float inv_pixels = opt_params.mask_opacity_penalty_weight / static_cast<float>(alpha_2d.numel());
                grad_alpha = penalty_weights * inv_pixels;
                loss = loss + penalty;
            }

        } else if (mode == param::MaskMode::AlphaConsistent) {
            // Standard photometric loss
            lfs::training::losses::PhotometricLoss photo_loss_fn;
            const lfs::training::losses::PhotometricLoss::Params params{.lambda_dssim = opt_params.lambda_dssim};
            auto result = photo_loss_fn.forward(rendered, gt_image, params);
            if (!result) {
                return std::unexpected(result.error());
            }
            auto [photo_loss, ctx] = *result;
            loss = photo_loss;
            grad = ctx.grad_image;

            // Alpha should match mask
            if (alpha.is_valid()) {
                const Tensor alpha_2d = alpha.ndim() == 3 ? alpha.squeeze(0) : alpha;
                const Tensor alpha_loss = (alpha_2d - mask_2d).abs().mean() * ALPHA_CONSISTENCY_WEIGHT;
                loss = loss + alpha_loss;
                grad_alpha = (alpha_2d - mask_2d).sign() * (ALPHA_CONSISTENCY_WEIGHT / static_cast<float>(alpha_2d.numel()));
            }

        } else if (mode == param::MaskMode::HardMatting || mode == param::MaskMode::SoftMatting) {
            // =====================================================================
            // OPTIMIZED SOFTMATTING - SINGLE PASS APPROACH (like old attention-mask)
            // =====================================================================
            // Key optimization: ONE photometric pass with weighted mask instead of TWO separate passes

            const bool use_cores = (mode == param::MaskMode::SoftMatting);

            if (use_cores) {
                if (!fg_core_2d || !bg_core_2d || !fg_core_2d->is_valid() || !bg_core_2d->is_valid()) {
                    return std::unexpected(
                        "MaskMode::SoftMatting requires fg_core_2d and bg_core_2d (eroded cores).");
                }
            }

            // Build weight map: full weight inside mask + reduced weight outside
            // This mimics the old attention-mask behavior where background gets some supervision
            constexpr float kBgPhotometricRatio = 0.05f;

            Tensor weight_map;
            {
                // weight_map = mask + (1 - mask) * kBgPhotometricRatio
                // This gives 1.0 inside mask, ~0.05 outside mask
                static thread_local Tensor bg_mask_2d;
                if (bg_mask_2d.is_empty() || bg_mask_2d.shape() != mask_2d.shape() || bg_mask_2d.device() != mask_2d.device()) {
                    bg_mask_2d = Tensor::empty(mask_2d.shape(), mask_2d.device(), DataType::Float32);
                }
                bg_mask_2d.copy_(mask_2d);
                bg_mask_2d.mul_(-1.0f);
                bg_mask_2d.add_(1.0f);

                weight_map = mask_2d + bg_mask_2d * kBgPhotometricRatio;
            }

            // SINGLE photometric pass with weighted mask
            if (opt_params.lambda_dssim > 0.0f) {
                auto [loss_tensor, ctx] = lfs::training::kernels::masked_fused_l1_ssim_forward(
                    rendered, gt_image, weight_map, opt_params.lambda_dssim, masked_fused_workspace_);

                grad = lfs::training::kernels::masked_fused_l1_ssim_backward(ctx, masked_fused_workspace_);
                loss = loss_tensor;

                if (grad.ndim() == 4 && rendered.ndim() == 3) {
                    grad = grad.squeeze(0);
                }
            } else {
                // Pure L1 with weighted mask
                const Tensor weight_exp = weight_map.unsqueeze(0).expand({static_cast<int>(rendered.shape()[0]),
                                                                          static_cast<int>(mask_2d.shape()[0]),
                                                                          static_cast<int>(mask_2d.shape()[1])});
                const Tensor weight_sum = weight_exp.sum() + EPSILON;

                const Tensor diff = rendered - gt_image;
                const Tensor l1 = diff.abs();
                const Tensor sign = diff.sign();

                loss = (l1 * weight_exp).sum() / weight_sum;
                grad = sign * weight_exp / weight_sum;
            }

            // Simple alpha regularization (fast, like old trainer)
            if (alpha.is_valid()) {
                const Tensor alpha_2d = (alpha.ndim() == 3) ? alpha.squeeze(0) : alpha;

                // Use pixelBasedOpacityPenalty approach from old trainer
                // This is much faster than the complex multi-term approach
                const float penalty_weight = opt_params.mask_opacity_penalty_weight;

                if (penalty_weight > 0.0f) {
                    constexpr float kIn = 0.667f;  // Weight for penalizing transparency inside mask
                    constexpr float kOut = 0.333f; // Weight for penalizing opacity outside mask

                    // Binary mask for inside/outside
                    const Tensor fg_mask = mask_2d.gt(0.5f).to(DataType::Float32);
                    const Tensor bg_mask = Tensor::full(mask_2d.shape(), 1.0f, mask_2d.device()) - fg_mask;

                    // Penalty computation
                    const float num_pixels = static_cast<float>(alpha_2d.numel());
                    const Tensor inside_penalty_map = (Tensor::full(alpha_2d.shape(), 1.0f, alpha_2d.device()) - alpha_2d) * fg_mask;
                    const Tensor outside_penalty_map = alpha_2d * bg_mask;

                    const Tensor total_penalty = (inside_penalty_map.sum() * kIn + outside_penalty_map.sum() * kOut) / num_pixels;
                    const float penalty_scalar = total_penalty.item<float>() * penalty_weight;

                    // Add penalty to loss (multiplicative + small additive component)
                    loss = loss * (1.0f + penalty_scalar) + total_penalty * 0.01f * penalty_weight;

                    // Gradient w.r.t. alpha
                    const float grad_scale = penalty_weight / num_pixels;
                    grad_alpha = bg_mask * (kOut * grad_scale) - fg_mask * (kIn * grad_scale);

                    // Scale image gradient by penalty factor for consistency
                    grad = grad * (1.0f + penalty_scalar);
                }
            }

        } else {
            auto fallback = compute_photometric_loss_with_gradient(rendered, gt_image, opt_params);
            if (!fallback) {
                return std::unexpected(fallback.error());
            }
            return MaskLossResult{.loss = fallback->first, .grad_image = fallback->second, .grad_alpha = {}};
        }

        return MaskLossResult{.loss = loss, .grad_image = grad, .grad_alpha = grad_alpha};
    }

    // Returns GPU tensor for loss - NO SYNC!
    std::expected<lfs::core::Tensor, std::string> Trainer::compute_scale_reg_loss(
        lfs::core::SplatData& splatData,
        AdamOptimizer& optimizer,
        const lfs::core::param::OptimizationParameters& opt_params) {
        lfs::training::losses::ScaleRegularization::Params params{.weight = opt_params.scale_reg};
        return lfs::training::losses::ScaleRegularization::forward(splatData.scaling_raw(), optimizer.get_grad(ParamType::Scaling), params);
    }

    // Returns GPU tensor for loss - NO SYNC!
    std::expected<lfs::core::Tensor, std::string> Trainer::compute_opacity_reg_loss(
        lfs::core::SplatData& splatData,
        AdamOptimizer& optimizer,
        const lfs::core::param::OptimizationParameters& opt_params) {
        lfs::training::losses::OpacityRegularization::Params params{.weight = opt_params.opacity_reg};
        return lfs::training::losses::OpacityRegularization::forward(splatData.opacity_raw(), optimizer.get_grad(ParamType::Opacity), params);
    }

    std::expected<std::pair<lfs::core::Tensor, SparsityLossContext>, std::string>
    Trainer::compute_sparsity_loss_forward(const int iter, const lfs::core::SplatData& splat_data) {
        if (!sparsity_optimizer_ || !sparsity_optimizer_->should_apply_loss(iter)) {
            auto zero = lfs::core::Tensor::zeros({1}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
            return std::make_pair(std::move(zero), SparsityLossContext{});
        }

        if (!sparsity_optimizer_->is_initialized()) {
            if (auto result = sparsity_optimizer_->initialize(splat_data.opacity_raw()); !result) {
                return std::unexpected(result.error());
            }
            LOG_DEBUG("Sparsity optimizer initialized at iteration {}", iter);
        }

        return sparsity_optimizer_->compute_loss_forward(splat_data.opacity_raw());
    }

    std::expected<void, std::string> Trainer::handle_sparsity_update(const int iter, lfs::core::SplatData& splat_data) {
        if (!sparsity_optimizer_ || !sparsity_optimizer_->should_update(iter)) {
            return {};
        }
        return sparsity_optimizer_->update_state(splat_data.opacity_raw());
    }

    std::expected<void, std::string> Trainer::apply_sparsity_pruning(const int iter, lfs::core::SplatData& splat_data) {
        if (!sparsity_optimizer_ || !sparsity_optimizer_->should_prune(iter)) {
            return {};
        }

        auto mask_result = sparsity_optimizer_->get_prune_mask(splat_data.opacity_raw());
        if (!mask_result) {
            return std::unexpected(mask_result.error());
        }

        const int n_before = static_cast<int>(splat_data.size());
        strategy_->remove_gaussians(*mask_result);
        const int n_after = static_cast<int>(splat_data.size());

        LOG_INFO("Sparsity pruning: {} -> {} Gaussians ({}% reduction)",
                 n_before, n_after, static_cast<int>(100.0f * (n_before - n_after) / n_before));

        sparsity_optimizer_.reset();
        return {};
    }

    Trainer::Trainer(std::shared_ptr<CameraDataset> dataset,
                     std::unique_ptr<IStrategy> strategy,
                     std::optional<std::tuple<std::vector<std::string>, std::vector<std::string>>> provided_splits)
        : base_dataset_(std::move(dataset)),
          strategy_(std::move(strategy)),
          provided_splits_(std::move(provided_splits)) {
        // Check CUDA availability
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess || device_count == 0) {
            throw std::runtime_error("CUDA is not available – aborting.");
        }

        cudaStreamCreateWithFlags(&callback_stream_, cudaStreamNonBlocking);

        LOG_DEBUG("Trainer constructed with {} cameras", base_dataset_->get_cameras().size());
    }

    // New constructor - Scene owns all data
    Trainer::Trainer(lfs::vis::Scene& scene)
        : scene_(&scene) {
        // Check CUDA availability
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess || device_count == 0) {
            throw std::runtime_error("CUDA is not available – aborting.");
        }

        cudaStreamCreateWithFlags(&callback_stream_, cudaStreamNonBlocking);

        // Datasets will be created in initialize() from Scene cameras
        if (!scene.getTrainCameras()) {
            throw std::runtime_error("Scene has no train cameras");
        }

        LOG_DEBUG("Trainer constructed from Scene with {} cameras", scene.getTrainCameras()->get_cameras().size());
    }

    std::expected<void, std::string> Trainer::initialize(const lfs::core::param::TrainingParameters& params) {
        // Thread-safe initialization using mutex
        std::lock_guard<std::mutex> lock(init_mutex_);

        // Check again after acquiring lock (double-checked locking pattern)
        if (initialized_.load()) {
            LOG_INFO("Re-initializing trainer with new parameters");
            // Clean up existing state for re-initialization
            cleanup();
        }

        LOG_INFO("Initializing trainer with {} iterations", params.optimization.iterations);

        try {
            params_ = params;

            // Create DatasetConfig for lfs::training::CameraDataset
            lfs::training::DatasetConfig dataset_config;
            dataset_config.resize_factor = params.dataset.resize_factor;
            dataset_config.max_width = params.dataset.max_width;
            dataset_config.test_every = params.dataset.test_every;

            // Get source cameras - from Scene (new mode) or base_dataset_ (legacy mode)
            std::vector<std::shared_ptr<lfs::core::Camera>> source_cameras;
            if (scene_) {
                // Scene mode: get cameras from Scene
                auto scene_dataset = scene_->getTrainCameras();
                if (!scene_dataset) {
                    return std::unexpected("Scene has no train cameras");
                }
                source_cameras = scene_dataset->get_cameras();
            } else if (base_dataset_) {
                // Legacy mode: use base_dataset_
                source_cameras = base_dataset_->get_cameras();
            } else {
                return std::unexpected("No camera source available");
            }

            // Handle dataset split based on evaluation flag
            if (params.optimization.enable_eval) {
                // Create train/val split
                train_dataset_ = std::make_shared<CameraDataset>(
                    source_cameras, dataset_config, CameraDataset::Split::TRAIN,
                    provided_splits_ ? std::make_optional(std::get<0>(*provided_splits_)) : std::nullopt);
                val_dataset_ = std::make_shared<CameraDataset>(
                    source_cameras, dataset_config, CameraDataset::Split::VAL,
                    provided_splits_ ? std::make_optional(std::get<1>(*provided_splits_)) : std::nullopt);

                LOG_INFO("Created train/val split: {} train, {} val images",
                         train_dataset_->size(),
                         val_dataset_->size());
            } else {
                // Use all images for training
                train_dataset_ = std::make_shared<CameraDataset>(
                    source_cameras, dataset_config, CameraDataset::Split::ALL);
                val_dataset_ = nullptr;

                LOG_INFO("Using all {} images for training (no evaluation)",
                         train_dataset_->size());
            }

            train_dataset_size_ = train_dataset_->size();

            // If using Scene mode and no strategy yet, create one
            if (scene_ && !strategy_) {
                auto* model = scene_->getTrainingModel();
                if (!model) {
                    return std::unexpected("Scene has no training model set");
                }

                if (params.optimization.strategy == "mcmc") {
                    strategy_ = std::make_unique<MCMC>(*model);
                    LOG_DEBUG("Created MCMC strategy from Scene model");
                } else {
                    strategy_ = std::make_unique<ADC>(*model);
                    LOG_DEBUG("Created ADC strategy from Scene model");
                }
            }

            auto& splat = strategy_->get_model();

            int max_cap = params.optimization.max_cap;
            if (max_cap < splat.size()) {
                LOG_WARN("Max cap is less than to {} initial splats {}. Choosing randomly {} splats", max_cap, splat.size(), max_cap);
                lfs::core::random_choose(splat, max_cap);
            }

            // Re-initialize strategy with new parameters
            strategy_->initialize(params.optimization);
            LOG_DEBUG("Strategy initialized");

            // Initialize bilateral grid if enabled
            if (auto result = initialize_bilateral_grid(); !result) {
                return std::unexpected(result.error());
            }

            // Validate masks if mask mode is enabled
            if (auto result = validate_masks(); !result) {
                return std::unexpected(result.error());
            }

            // Initialize sparsity optimizer
            if (params.optimization.enable_sparsity) {
                constexpr int UPDATE_INTERVAL = 50;
                const int sparsify_steps = params.optimization.sparsify_steps;
                const int stored_iters = static_cast<int>(params.optimization.iterations);

                // Checkpoint already has total iterations; fresh start needs sparsify_steps added
                const bool is_resume = params.resume_checkpoint.has_value();
                const int base_iters = is_resume ? (stored_iters - sparsify_steps) : stored_iters;

                if (!is_resume) {
                    params_.optimization.iterations = static_cast<size_t>(base_iters + sparsify_steps);
                }

                const ADMMSparsityOptimizer::Config config{
                    .sparsify_steps = sparsify_steps,
                    .init_rho = params.optimization.init_rho,
                    .prune_ratio = params.optimization.prune_ratio,
                    .update_every = UPDATE_INTERVAL,
                    .start_iteration = base_iters};

                sparsity_optimizer_ = SparsityOptimizerFactory::create("admm", config);
                if (sparsity_optimizer_) {
                    LOG_INFO("Sparsity: base={}, steps={}, prune={:.0f}%",
                             base_iters, sparsify_steps, params.optimization.prune_ratio * 100);
                }
            }

            // Initialize background color tensor from params
            {
                const auto& bg_color = params.optimization.bg_color;
                background_ = lfs::core::Tensor::empty({3}, lfs::core::Device::CPU, lfs::core::DataType::Float32);
                auto* bg_ptr = background_.ptr<float>();
                bg_ptr[0] = bg_color[0];
                bg_ptr[1] = bg_color[1];
                bg_ptr[2] = bg_color[2];
                background_ = background_.to(lfs::core::Device::CUDA);
                LOG_INFO("Background color set to RGB({:.2f}, {:.2f}, {:.2f})", bg_color[0], bg_color[1], bg_color[2]);
            }

            // Load background image if specified
            if (params.optimization.bg_mode == lfs::core::param::BackgroundMode::Image &&
                !params.optimization.bg_image_path.empty() &&
                std::filesystem::exists(params.optimization.bg_image_path)) {
                try {
                    auto& loader = lfs::io::CacheLoader::getInstance();
                    lfs::io::LoadParams load_params{
                        .resize_factor = 1,
                        .max_width = 0, // No max width limit
                        .cuda_stream = nullptr};
                    bg_image_base_ = loader.load_cached_image(params.optimization.bg_image_path, load_params);
                    if (bg_image_base_.device() != lfs::core::Device::CUDA) {
                        bg_image_base_ = bg_image_base_.to(lfs::core::Device::CUDA);
                    }
                    if (bg_image_base_.shape()[0] != 3) {
                        LOG_WARN("Background image has {} channels, expected 3 (RGB)", bg_image_base_.shape()[0]);
                        bg_image_base_ = {};
                        params_.optimization.bg_mode = lfs::core::param::BackgroundMode::SolidColor;
                    } else {
                        LOG_INFO("Background image: {} [{}x{}]",
                                 lfs::core::path_to_utf8(params.optimization.bg_image_path),
                                 bg_image_base_.shape()[2], bg_image_base_.shape()[1]);
                    }
                } catch (const std::exception& e) {
                    LOG_WARN("Failed to load background image: {}", e.what());
                    params_.optimization.bg_mode = lfs::core::param::BackgroundMode::SolidColor;
                }
            }

            // Create progress bar based on headless flag
            if (params.optimization.headless) {
                progress_ = std::make_unique<TrainingProgress>(
                    params_.optimization.iterations, // This now includes sparsity steps if enabled
                    /*update_frequency=*/100);
                LOG_DEBUG("Progress bar initialized for {} total iterations", params_.optimization.iterations);
            }

            // Initialize the evaluator - it handles all metrics internally
            evaluator_ = std::make_unique<lfs::training::MetricsEvaluator>(params_);
            LOG_DEBUG("Metrics evaluator initialized");

            // Resume from checkpoint if provided
            if (params_.resume_checkpoint.has_value()) {
                auto resume_result = load_checkpoint(*params_.resume_checkpoint);
                if (!resume_result) {
                    return std::unexpected(std::format("Failed to resume from checkpoint: {}", resume_result.error()));
                }
                LOG_INFO("Resumed training from checkpoint at iteration {}", *resume_result);

                // Reload bg_image if checkpoint restored different settings
                if (params_.optimization.bg_mode == lfs::core::param::BackgroundMode::Image &&
                    !params_.optimization.bg_image_path.empty() &&
                    std::filesystem::exists(params_.optimization.bg_image_path) &&
                    !bg_image_base_.is_valid()) {
                    try {
                        auto& loader = lfs::io::CacheLoader::getInstance();
                        lfs::io::LoadParams load_params{.resize_factor = 1, .max_width = 0, .cuda_stream = nullptr};
                        bg_image_base_ = loader.load_cached_image(params_.optimization.bg_image_path, load_params);
                        if (bg_image_base_.device() != lfs::core::Device::CUDA) {
                            bg_image_base_ = bg_image_base_.to(lfs::core::Device::CUDA);
                        }
                        if (bg_image_base_.shape()[0] != 3) {
                            LOG_WARN("Background image has {} channels, expected 3", bg_image_base_.shape()[0]);
                            bg_image_base_ = {};
                            params_.optimization.bg_mode = lfs::core::param::BackgroundMode::SolidColor;
                        } else {
                            LOG_INFO("Background image from checkpoint: {} [{}x{}]",
                                     lfs::core::path_to_utf8(params_.optimization.bg_image_path),
                                     bg_image_base_.shape()[2], bg_image_base_.shape()[1]);
                        }
                    } catch (const std::exception& e) {
                        LOG_WARN("Failed to load background image from checkpoint: {}", e.what());
                        params_.optimization.bg_mode = lfs::core::param::BackgroundMode::SolidColor;
                    }
                }
            }
            if (auto result = initialize_ngs(); !result) {
                return std::unexpected(result.error());
            }

            // Print configuration
            LOG_INFO("Visualization: {}", params.optimization.headless ? "disabled" : "enabled");
            LOG_INFO("Strategy: {}", params.optimization.strategy);
            if (params.optimization.mask_mode != lfs::core::param::MaskMode::None) {
                static constexpr const char* MASK_MODE_NAMES[] = {"none", "segment", "ignore", "alpha_consistent", "hardmatting", "softmatting"};
                LOG_INFO("Mask mode: {}", MASK_MODE_NAMES[static_cast<int>(params.optimization.mask_mode)]);
            }
            if (current_iteration_ > 0) {
                LOG_INFO("Starting from iteration: {}", current_iteration_.load());
            }

            initialized_ = true;
            LOG_INFO("Trainer initialization complete");
            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to initialize trainer: {}", e.what()));
        }
    }

    Trainer::~Trainer() {
        shutdown();
    }

    void Trainer::shutdown() {
        if (shutdown_complete_.exchange(true)) {
            return;
        }

        LOG_DEBUG("Trainer shutdown");
        stop_requested_ = true;

        lfs::core::image_io::BatchImageSaver::instance().wait_all();

        if (callback_stream_) {
            cudaStreamSynchronize(callback_stream_);
            cudaStreamDestroy(callback_stream_);
            callback_stream_ = nullptr;
        }
        callback_busy_ = false;

        cudaDeviceSynchronize();

        strategy_.reset();
        bilateral_grid_.reset();
        sparsity_optimizer_.reset();
        evaluator_.reset();
        progress_.reset();
        train_dataset_.reset();
        val_dataset_.reset();

        // Release GPU memory pools back to system
        lfs::core::CudaMemoryPool::instance().trim_cached_memory();
        lfs::core::GlobalArenaManager::instance().get_arena().emergency_cleanup();
        cudaDeviceSynchronize();
        LOG_DEBUG("GPU memory released");

        initialized_ = false;
        is_running_ = false;
        training_complete_ = false;
    }

    void Trainer::setParams(const lfs::core::param::TrainingParameters& params) {
        // Check if background image path changed and needs to be (re)loaded
        const bool bg_image_path_changed =
            params.optimization.bg_image_path != params_.optimization.bg_image_path;
        const bool bg_mode_is_image =
            params.optimization.bg_mode == lfs::core::param::BackgroundMode::Image;

        // Update params first
        params_ = params;

        // Load/reload background image if needed
        if (bg_mode_is_image && bg_image_path_changed &&
            !params.optimization.bg_image_path.empty() &&
            std::filesystem::exists(params.optimization.bg_image_path)) {
            try {
                auto& loader = lfs::io::CacheLoader::getInstance();
                lfs::io::LoadParams load_params{
                    .resize_factor = 1,
                    .max_width = 0,
                    .cuda_stream = nullptr};
                bg_image_base_ = loader.load_cached_image(params.optimization.bg_image_path, load_params);
                if (bg_image_base_.device() != lfs::core::Device::CUDA) {
                    bg_image_base_ = bg_image_base_.to(lfs::core::Device::CUDA);
                }
                bg_image_cache_.clear();
                if (bg_image_base_.shape()[0] != 3) {
                    LOG_WARN("Background image has {} channels, expected 3 (RGB)", bg_image_base_.shape()[0]);
                    bg_image_base_ = {};
                    params_.optimization.bg_mode = lfs::core::param::BackgroundMode::SolidColor;
                } else {
                    LOG_INFO("Background image: {} [{}x{}]",
                             lfs::core::path_to_utf8(params.optimization.bg_image_path),
                             bg_image_base_.shape()[2], bg_image_base_.shape()[1]);
                }
            } catch (const std::exception& e) {
                LOG_WARN("Failed to load background image: {}", e.what());
                params_.optimization.bg_mode = lfs::core::param::BackgroundMode::SolidColor;
            }
        }

        if (!bg_mode_is_image && (bg_image_base_.is_valid() || !bg_image_cache_.empty())) {
            bg_image_cache_.clear();
            bg_image_base_ = {};
        }

        // Update background color tensor if changed
        const auto& bg_color = params.optimization.bg_color;
        if (background_.is_valid()) {
            auto bg_cpu = lfs::core::Tensor::empty({3}, lfs::core::Device::CPU, lfs::core::DataType::Float32);
            auto* bg_ptr = bg_cpu.ptr<float>();
            bg_ptr[0] = bg_color[0];
            bg_ptr[1] = bg_color[1];
            bg_ptr[2] = bg_color[2];
            background_ = bg_cpu.to(lfs::core::Device::CUDA);
        }
    }

    void Trainer::handle_control_requests(int iter, std::stop_token stop_token) {
        // Check stop token first
        if (stop_token.stop_requested()) {
            stop_requested_ = true;
            return;
        }

        // Handle pause/resume
        if (pause_requested_.load() && !is_paused_.load()) {
            is_paused_ = true;
            if (progress_) {
                progress_->pause();
            }
            LOG_INFO("Training paused at iteration {}", iter);
            LOG_DEBUG("Click 'Resume Training' to continue.");
        } else if (!pause_requested_.load() && is_paused_.load()) {
            is_paused_ = false;
            if (progress_) {
                progress_->resume(iter, current_loss_.load(), static_cast<int>(strategy_->get_model().size()));
            }
            LOG_INFO("Training resumed at iteration {}", iter);
        }

        if (save_requested_.exchange(false)) {
            LOG_INFO("Saving checkpoint and PLY at iteration {}...", iter);
            save_ply(params_.dataset.output_path, iter, /*join=*/false);
            auto result = save_checkpoint(iter);
            if (result) {
                auto checkpoint_path = params_.dataset.output_path / "checkpoints" /
                                       std::format("checkpoint_{}.resume", iter);
                LOG_INFO("Checkpoint and PLY saved to {}", lfs::core::path_to_utf8(params_.dataset.output_path));
            } else {
                LOG_ERROR("Failed to save checkpoint: {}", result.error());
            }
        }

        // Handle stop request - this permanently stops training
        if (stop_requested_.load()) {
            LOG_INFO("Stopping training permanently at iteration {}...", iter);
            LOG_DEBUG("Saving final model...");
            save_ply(params_.dataset.output_path, iter, /*join=*/true);
            is_running_ = false;
        }
    }

    // Calculate noise intensity based on training progress.
    // Strategy:
    // 1. Grace Period: No noise initially to allow structure to solidify from SfM/Random points.
    // 2. Warmup: Ramp up noise to clear "floaters" and aggressively carve empty space.
    // 3. Decay: Reduce noise towards the end to allow fine-tuning of semi-transparent details (hair, edges).
    inline float inv_weight_piecewise(int step, int max_steps, bool isNoise) {
        if (max_steps <= 0)
            return 0.0f;

        const float phase = std::clamp(
            static_cast<float>(step) / static_cast<float>(max_steps),
            0.0f, 1.0f);

        // Schedule configuration
        constexpr float P_DELAY = 0.15f;      // First 10%: Grace period (Structural initialization)
        constexpr float P_WARMUP_LEN = 0.10f; // Next 10%: Linear ramp-up
        constexpr float P_RAMP_END = P_DELAY + P_WARMUP_LEN;
        constexpr float P_HOLD_END = P_RAMP_END + P_WARMUP_LEN * 2; // Hold peak intensity until
        const float P_DECAY_END = isNoise ? 0.8f : 0.60f; // Decay ends at 80%

        // Intensities
        const float MAX_INTENSITY = isNoise ? 0.85f : 0.6f;
        constexpr float MIN_INTENSITY = 0.01f; // Small floor to prevent late-training artifacts (halos)

        // Phase 1: Grace Period
        if (phase < P_DELAY) {
            return 0.0f; // Silence: Let the model learn the base geometry first.
        }
        // Phase 2: Warmup (Linear Ramp)
        else if (phase < P_RAMP_END) {
            // Normalize 't' from 0.0 to 1.0 within the warmup window
            const float t = (phase - P_DELAY) / P_WARMUP_LEN;
            return MAX_INTENSITY * t;
        }
        // Phase 3: Hold (Peak Cleaning)
        else if (phase < P_HOLD_END) {
            return MAX_INTENSITY; // Aggressive floater removal.
        }
        // Phase 4: Decay (Fine-tuning)
        else if (phase < P_DECAY_END) {
            // Linear decay from MAX to MIN to recover fine details/transparency
            const float t = (phase - P_HOLD_END) / (P_DECAY_END - P_HOLD_END);
            return MAX_INTENSITY + (MIN_INTENSITY - MAX_INTENSITY) * t;
        }
        // Phase 5: Floor (Stability)
        else {
            return MIN_INTENSITY;
        }
    }

    namespace {
        constexpr float TWO_PI = static_cast<float>(M_PI * 2.0);
        constexpr float PHASE_OFFSET_G = TWO_PI / 3.0f;
        constexpr float PHASE_OFFSET_B = TWO_PI * 2.0f / 3.0f;
        constexpr float CLAMP_EPS = 1e-4f;
        constexpr int BG_PERIOD_R = 37;
        constexpr int BG_PERIOD_G = 41;
        constexpr int BG_PERIOD_B = 43;
    } // anonymous namespace

    lfs::core::Tensor& Trainer::background_for_step(int iter) {
        if (!params_.optimization.bg_modulation || params_.optimization.bg_noise) {
            return background_;
        }

        const float w = inv_weight_piecewise(iter, params_.optimization.iterations, false);
        if (w <= 0.0f) {
            return background_;
        }

        // Sine-based RGB with prime periods for color diversity
        const float pr = TWO_PI * static_cast<float>(iter % BG_PERIOD_R) / BG_PERIOD_R;
        const float pg = TWO_PI * static_cast<float>(iter % BG_PERIOD_G) / BG_PERIOD_G;
        const float pb = TWO_PI * static_cast<float>(iter % BG_PERIOD_B) / BG_PERIOD_B;

        const float result[3] = {
            std::clamp(0.5f * (1.0f + std::sin(pr)) * w, CLAMP_EPS, 1.0f - CLAMP_EPS),
            std::clamp(0.5f * (1.0f + std::sin(pg + PHASE_OFFSET_G)) * w, CLAMP_EPS, 1.0f - CLAMP_EPS),
            std::clamp(0.5f * (1.0f + std::sin(pb + PHASE_OFFSET_B)) * w, CLAMP_EPS, 1.0f - CLAMP_EPS)};

        if (bg_mix_buffer_.is_empty()) {
            bg_mix_buffer_ = lfs::core::Tensor::empty({3}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        }

        cudaMemcpyAsync(bg_mix_buffer_.ptr<float>(), result, sizeof(result), cudaMemcpyHostToDevice, nullptr);
        return bg_mix_buffer_;
    }

    
    lfs::core::Tensor Trainer::apply_background_noise(
        const lfs::core::Tensor& rendered, // [C, H, W] (usually C=3)
        const lfs::core::Tensor& alpha,    // [1, H, W]
        int iter) {
        // Compute noise intensity weight based on training progress
        const float w = inv_weight_piecewise(iter, params_.optimization.iterations, true);

        // Early exit: skip blending if noise contribution is negligible
        if (w <= CLAMP_EPS) {
            return rendered; // shallow return is fine
        }

        // Basic shape assumptions (keep lightweight; training path should be consistent)
        const size_t C = rendered.shape()[0];
        const size_t H = rendered.shape()[1];
        const size_t W = rendered.shape()[2];

        const lfs::core::TensorShape noise_shape({C, H, W});
        const lfs::core::TensorShape alpha_shape({1, H, W});
        const auto dev = rendered.device();

        // Persistent buffers to avoid per-iteration VRAM allocations.
        // thread_local keeps it safe if multiple trainers run in parallel on different threads.
        static thread_local lfs::core::Tensor noise_buffer;     // [C, H, W]
        static thread_local lfs::core::Tensor weighted_buffer;  // [C, H, W]
        static thread_local lfs::core::Tensor alpha_inv_buffer; // [1, H, W]

        // Re-allocate only if strictly necessary (shape/device changes)
        bool buffer_reset = false;
        if (noise_buffer.is_empty() || noise_buffer.shape() != noise_shape || noise_buffer.device() != dev) {
            noise_buffer = lfs::core::Tensor::empty(noise_shape, dev, lfs::core::DataType::Float32);
            buffer_reset = true;
        }
        if (weighted_buffer.is_empty() || weighted_buffer.shape() != noise_shape || weighted_buffer.device() != dev) {
            weighted_buffer = lfs::core::Tensor::empty(noise_shape, dev, lfs::core::DataType::Float32);
        }
        if (alpha_inv_buffer.is_empty() || alpha_inv_buffer.shape() != alpha_shape || alpha_inv_buffer.device() != dev) {
            alpha_inv_buffer = lfs::core::Tensor::empty(alpha_shape, dev, lfs::core::DataType::Float32);
        }

        // Constants
        constexpr int NOISE_RESET_INTERVAL = 100;
        constexpr float GOLDEN_RATIO = 0.61803398875f;

        // NOISE GENERATION STRATEGY:
        // 1) Periodic full reset: expensive RNG for statistical independence.
        // 2) Per-step cheap update: add constant offset and wrap to [0,1).
        //
        // IMPORTANT: In this tensor library, "a = a + b" materializes a new tensor.
        // Use in-place scalar ops to avoid allocations and extra kernels.
        if (buffer_reset || (iter % NOISE_RESET_INTERVAL == 0)) {
            // Expensive path
            noise_buffer.uniform_(0.0f, 1.0f);
        } else {
            // Cheap path: noise = fract(noise + phi)
            // In-place scalar add (no allocation)
            noise_buffer.add_(GOLDEN_RATIO);

            // Wrap around 1.0: fract(x) = x - floor(x)
            // NOTE: floor() currently materializes a temporary (1 allocation).
            // Still much cheaper than materializing (a + scalar) AND floor() AND subtraction.
            noise_buffer.sub_(noise_buffer.floor());
        }

        // Prepare alpha_inv = (1 - alpha) in a reusable buffer.
        // Avoid scalar Tensor "{}" (rank-0) which can be fragile depending on broadcast rules.
        alpha_inv_buffer.copy_(alpha); // alpha is expected to be Float32 on the training path
        alpha_inv_buffer.mul_(-1.0f);
        alpha_inv_buffer.add_(1.0f); // alpha_inv_buffer = 1 - alpha

        // weighted_buffer = noise * w (reused buffer, no allocation)
        weighted_buffer.copy_(noise_buffer);
        weighted_buffer.mul_(w);

        // Blend: out = rendered + (1 - alpha) * (noise * w)
        // This multiply will broadcast [1,H,W] over [C,H,W] and materialize ONE output tensor.
        lfs::core::Tensor out = alpha_inv_buffer * weighted_buffer;
        out.add_(rendered); // in-place add, shapes match exactly

        return out;
    }

    lfs::core::Tensor Trainer::get_background_image_for_camera(int width, int height) {
        // Return empty tensor if no background image is loaded
        if (!bg_image_base_.is_valid() || bg_image_base_.is_empty()) {
            return lfs::core::Tensor();
        }

        // Check cache first - key is (height << 32) | width
        const uint64_t cache_key = (static_cast<uint64_t>(height) << 32) | static_cast<uint64_t>(width);
        auto it = bg_image_cache_.find(cache_key);
        if (it != bg_image_cache_.end()) {
            return it->second;
        }

        // Resize background image to match camera dimensions
        const int src_h = static_cast<int>(bg_image_base_.shape()[1]);
        const int src_w = static_cast<int>(bg_image_base_.shape()[2]);
        const int channels = static_cast<int>(bg_image_base_.shape()[0]);

        // If dimensions match, use the original
        if (src_w == width && src_h == height) {
            bg_image_cache_[cache_key] = bg_image_base_;
            return bg_image_base_;
        }

        // Create resized tensor
        auto resized = lfs::core::Tensor::empty(
            {static_cast<size_t>(channels), static_cast<size_t>(height), static_cast<size_t>(width)},
            lfs::core::Device::CUDA,
            lfs::core::DataType::Float32);

        // Use bilinear resize kernel
        kernels::launch_bilinear_resize_chw(
            bg_image_base_.ptr<float>(),
            resized.ptr<float>(),
            channels,
            src_h, src_w,
            height, width,
            nullptr // default stream
        );

        // Cache the resized image
        bg_image_cache_[cache_key] = resized;
        LOG_DEBUG("Background image resized: {}x{} -> {}x{}", src_w, src_h, width, height);

        return resized;
    }

    lfs::core::Tensor Trainer::get_random_background_for_camera(int width, int height, int iteration) {
        const size_t required_size = 3 * static_cast<size_t>(height) * static_cast<size_t>(width);

        if (!random_bg_buffer_.is_valid() || random_bg_buffer_.numel() != required_size) {
            random_bg_buffer_ = lfs::core::Tensor::empty(
                {3, static_cast<size_t>(height), static_cast<size_t>(width)},
                lfs::core::Device::CUDA,
                lfs::core::DataType::Float32);
        }

        kernels::launch_random_background(
            random_bg_buffer_.ptr<float>(),
            height, width,
            static_cast<uint64_t>(iteration),
            nullptr);

        return random_bg_buffer_;
    }

    std::expected<Trainer::StepResult, std::string> Trainer::train_step(
        int iter,
        lfs::core::Camera* cam,
        lfs::core::Tensor gt_image,
        RenderMode render_mode,
        std::stop_token stop_token) {
        try {
            //NGS Noise Guided Splatting transition
            handle_ngs_phase_transition(iter);

            // GUT mode enables Gaussian Unscented Transform for lens distortion handling
            if (params_.optimization.gut) {
                if (cam->camera_model_type() == core::CameraModelType::ORTHO) {
                    return std::unexpected("Training on cameras with ortho model is not supported yet.");
                }
            } else {
                if (cam->radial_distortion().numel() != 0 ||
                    cam->tangential_distortion().numel() != 0) {
                    return std::unexpected("Distorted images detected.  You can use --gut option to train on cameras with distortion.");
                }
                if (cam->camera_model_type() != core::CameraModelType::PINHOLE) {
                    return std::unexpected("You must use --gut option to train on cameras with non-pinhole model.");
                }
            }

            current_iteration_ = iter;

            // Check control requests at the beginning
            handle_control_requests(iter, stop_token);

            // If stop requested, return Stop
            if (stop_requested_.load() || stop_token.stop_requested()) {
                return StepResult::Stop;
            }

            // If paused, wait
            while (is_paused_.load() && !stop_requested_.load() && !stop_token.stop_requested()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                handle_control_requests(iter, stop_token);
            }

            // Check stop again after potential pause
            if (stop_requested_.load() || stop_token.stop_requested()) {
                return StepResult::Stop;
            }

            nvtxRangePush("background_for_step");
            lfs::core::Tensor& bg = background_for_step(iter);
            nvtxRangePop();

            lfs::core::Tensor bg_image;
            if (params_.optimization.bg_mode == lfs::core::param::BackgroundMode::Image) {
                bg_image = get_background_image_for_camera(cam->image_width(), cam->image_height());
            } else if (params_.optimization.bg_mode == lfs::core::param::BackgroundMode::Random) {
                bg_image = get_random_background_for_camera(cam->image_width(), cam->image_height(), iter);
            }

            // Configurable tile-based training to reduce peak memory
            const int full_width = cam->image_width();
            const int full_height = cam->image_height();

            // Read tile mode from parameters (1=1 tile, 2=2 tiles, 4=4 tiles)
            const TileMode tile_mode = static_cast<TileMode>(params_.optimization.tile_mode);

            // Determine tile configuration
            int tile_rows = 1, tile_cols = 1;
            switch (tile_mode) {
            case TileMode::One:
                tile_rows = 1;
                tile_cols = 1;
                break;
            case TileMode::Two:
                tile_rows = 2;
                tile_cols = 1;
                break;
            case TileMode::Four:
                tile_rows = 2;
                tile_cols = 2;
                break;
            }

            const int tile_width = full_width / tile_cols;
            const int tile_height = full_height / tile_rows;
            const int num_tiles = tile_rows * tile_cols;

            core::Tensor loss_tensor_gpu = core::Tensor::zeros({1}, core::Device::CUDA);
            RenderOutput r_output;
            int tiles_processed = 0;

            // Loop over tiles (row-major order)
            for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
                const int tile_row = tile_idx / tile_cols;
                const int tile_col = tile_idx % tile_cols;
                const int tile_x_offset = tile_col * tile_width;
                const int tile_y_offset = tile_row * tile_height;

                nvtxRangePush(std::format("tile_{}x{}", tile_row, tile_col).c_str());

                // Extract GT image tile
                lfs::core::Tensor gt_tile;
                if (num_tiles == 1) {
                    // No tiling - use full image
                    gt_tile = gt_image;
                } else if (gt_image.shape()[0] == 3) {
                    // CHW layout: gt_image is [3, H, W]
                    // Slice both height and width dimensions
                    auto tile_h = gt_image.slice(1, tile_y_offset, tile_y_offset + tile_height);
                    gt_tile = tile_h.slice(2, tile_x_offset, tile_x_offset + tile_width);
                } else {
                    // HWC layout: gt_image is [H, W, 3]
                    auto tile_h = gt_image.slice(0, tile_y_offset, tile_y_offset + tile_height);
                    gt_tile = tile_h.slice(1, tile_x_offset, tile_x_offset + tile_width);
                }

                // Extract background image tile (if using background image)
                lfs::core::Tensor bg_tile;
                if (bg_image.is_valid() && !bg_image.is_empty()) {
                    if (num_tiles == 1) {
                        // No tiling - use full image
                        bg_tile = bg_image;
                    } else {
                        // CHW layout: bg_image is [3, H, W]
                        // Slice both height and width dimensions
                        auto tile_h = bg_image.slice(1, tile_y_offset, tile_y_offset + tile_height);
                        bg_tile = tile_h.slice(2, tile_x_offset, tile_x_offset + tile_width);
                    }
                }

                // Render the tile
                nvtxRangePush("rasterize_forward");

                // Storage for render output (used by both paths)
                RenderOutput output;
                std::optional<FastRasterizeContext> fast_ctx;
                std::optional<GsplatRasterizeContext> gsplat_ctx;

                if (params_.optimization.gut) {
                    const int tw = (num_tiles > 1) ? tile_width : 0;
                    const int th = (num_tiles > 1) ? tile_height : 0;
                    auto rasterize_result = gsplat_rasterize_forward(
                        *cam, strategy_->get_model(), bg,
                        tile_x_offset, tile_y_offset, tw, th,
                        1.0f, false, GsplatRenderMode::RGB, true, bg_tile);

                    if (!rasterize_result) {
                        nvtxRangePop(); // rasterize_forward
                        nvtxRangePop(); // tile
                        return std::unexpected(rasterize_result.error());
                    }

                    output = std::move(rasterize_result->first);
                    gsplat_ctx.emplace(std::move(rasterize_result->second));
                } else {
                    // Standard mode: use fast rasterizer with tiling support
                    auto rasterize_result = fast_rasterize_forward(
                        *cam, strategy_->get_model(), bg,
                        tile_x_offset, tile_y_offset,
                        (num_tiles > 1) ? tile_width : 0, // 0 means full image
                        (num_tiles > 1) ? tile_height : 0,
                        params_.optimization.mip_filter, bg_tile);

                    // Check for OOM error
                    if (!rasterize_result) {
                        const std::string& error = rasterize_result.error();
                        if (error.find("OUT_OF_MEMORY") != std::string::npos) {
                            nvtxRangePop(); // rasterize_forward
                            nvtxRangePop(); // tile

                            // Handle OOM by switching tile mode
                            if (tile_mode == TileMode::Four) {
                                // Already at maximum tiling - can't tile further, return error
                                LOG_ERROR("OUT OF MEMORY at maximum tile mode (2x2). Cannot continue training.");
                                LOG_ERROR("Arena error: {}", error);
                                return std::unexpected(error);
                            } else {
                                // Upgrade to next tile mode
                                TileMode new_mode = (tile_mode == TileMode::One) ? TileMode::Two : TileMode::Four;
                                LOG_WARN("OUT OF MEMORY detected. Switching tile mode from {} to {}",
                                         static_cast<int>(tile_mode), static_cast<int>(new_mode));
                                LOG_WARN("Arena error: {}", error);
                                params_.optimization.tile_mode = static_cast<int>(new_mode);

                                // Retry this step with new tile mode
                                return std::unexpected("OOM_RETRY"); // Signal to retry the step
                            }
                        } else {
                            // Non-OOM error - propagate
                            nvtxRangePop();
                            nvtxRangePop();
                            return std::unexpected(error);
                        }
                    }

                    output = std::move(rasterize_result->first);
                    fast_ctx.emplace(std::move(rasterize_result->second));

                    if (fast_ctx->forward_ctx.n_visible_primitives == 0) {
                        auto& arena = lfs::core::GlobalArenaManager::instance().get_arena();
                        arena.end_frame(fast_ctx->forward_ctx.frame_id);
                        nvtxRangePop();
                        nvtxRangePop();
                        continue;
                    }
                }

                r_output = output; // Save last tile for densification
                nvtxRangePop();

                // Apply bilateral grid if enabled (before loss computation)
                lfs::core::Tensor corrected_image = output.image;
                if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
                    nvtxRangePush("bilateral_grid_forward");
                    corrected_image = bilateral_grid_->apply(output.image, cam->uid());
                    nvtxRangePop();
                }

                // Apply background noise if enabled
                if (params_.optimization.bg_noise) {
                    nvtxRangePush("apply_background_noise");
                    corrected_image = apply_background_noise(corrected_image, output.alpha, iter);
                    nvtxRangePop();
                }

                // Compute photometric loss and gradients for this tile
                nvtxRangePush("compute_photometric_loss");
                lfs::core::Tensor tile_loss;
                lfs::core::Tensor tile_grad;
                lfs::core::Tensor tile_grad_alpha; // Gradient for alpha (from mask penalty)

                const bool use_mask = params_.optimization.mask_mode != lfs::core::param::MaskMode::None && cam->has_mask();
                if (use_mask) {
                    // Use pipelined mask if available, otherwise load from camera (fallback for validation, etc.)
                    lfs::core::Tensor mask;
                    if (pipelined_mask_.is_valid() && pipelined_mask_.numel() > 0) {
                        mask = pipelined_mask_;
                    } else {
                        // Fallback: load mask from camera (cached after first load)
                        mask = cam->load_and_get_mask(
                            params_.dataset.resize_factor,
                            params_.dataset.max_width,
                            params_.optimization.invert_masks,
                            params_.optimization.mask_threshold);
                    }

                    // Extract mask tile if tiling
                    lfs::core::Tensor mask_tile = mask;
                    if (num_tiles > 1 && mask.ndim() == 2) {
                        const lfs::core::Tensor tile_h = mask.slice(0, tile_y_offset, tile_y_offset + tile_height);
                        mask_tile = tile_h.slice(1, tile_x_offset, tile_x_offset + tile_width);
                    }
 
                    // === NUEVA LÓGICA: HYBRID SWITCH ===
                    // Creamos una copia local de los parámetros para este paso
                    lfs::core::param::OptimizationParameters step_params = params_.optimization;

                    const float progress = static_cast<float>(iter) / static_cast<float>(params_.optimization.iterations);

                    // ESTRATEGIA: "Cleaning Phase" (Último 20%)
                    // Si estamos acabando y estamos en modo Matting, cambiamos temporalmente a AlphaConsistent
                    // para limpiar los floaters.
                    if (progress < 0.2f) {
                        if (step_params.mask_mode == lfs::core::param::MaskMode::SoftMatting ||
                            step_params.mask_mode == lfs::core::param::MaskMode::HardMatting) {

                            // Forzamos el modo estricto para limpiar
                            step_params.mask_mode = lfs::core::param::MaskMode::None;

                            // Opcional: Aumentar lambda_dssim ligeramente en esta fase para preservar estructura
                            // step_params.lambda_dssim = 0.2f;
                        }
                    }
                    else if (progress > 0.8f)
                    {
                        step_params.mask_opacity_penalty_weight = 0.0f;
                    }

                    // ===================================
                    

                    // Matting uses precomputed eroded "core" masks to avoid fighting uncertain boundaries.
                    // These cores are computed lazily and cached per Camera, so this is NOT a per-iteration cost.
                    lfs::core::Tensor fg_core;
                    lfs::core::Tensor bg_core;
                    bool use_matting_cores = false;

                    if (step_params.mask_mode == lfs::core::param::MaskMode::SoftMatting) {
                        constexpr int kMattingCoreErodeRadiusPx = 2;

                        fg_core = cam->load_and_get_mask_fg_core(
                            params_.dataset.resize_factor,
                            params_.dataset.max_width,
                            step_params.invert_masks,
                            step_params.mask_threshold,
                            kMattingCoreErodeRadiusPx);

                        bg_core = cam->load_and_get_mask_bg_core(
                            params_.dataset.resize_factor,
                            params_.dataset.max_width,
                            step_params.invert_masks,
                            step_params.mask_threshold,
                            kMattingCoreErodeRadiusPx);

                        if (!fg_core.is_valid() || !bg_core.is_valid()) {
                            nvtxRangePop();
                            nvtxRangePop();
                            return std::unexpected("MaskMode::Matting: failed to load fg/bg core masks.");
                        }
                        use_matting_cores = true;
                    }
                    
                    // Extract core tiles if Matting is enabled (must match mask_tile spatially)
                    lfs::core::Tensor fg_core_tile;
                    lfs::core::Tensor bg_core_tile;
                    const lfs::core::Tensor* fg_core_ptr = nullptr;
                    const lfs::core::Tensor* bg_core_ptr = nullptr;

                    if (use_matting_cores) {
                        fg_core_tile = fg_core;
                        bg_core_tile = bg_core;

                        if (num_tiles > 1 && fg_core.ndim() == 2) {
                            const lfs::core::Tensor tile_h_fg = fg_core.slice(0, tile_y_offset, tile_y_offset + tile_height);
                            fg_core_tile = tile_h_fg.slice(1, tile_x_offset, tile_x_offset + tile_width);

                            const lfs::core::Tensor tile_h_bg = bg_core.slice(0, tile_y_offset, tile_y_offset + tile_height);
                            bg_core_tile = tile_h_bg.slice(1, tile_x_offset, tile_x_offset + tile_width);
                        }

                        fg_core_ptr = &fg_core_tile;
                        bg_core_ptr = &bg_core_tile;
                    }


                    auto result = compute_photometric_loss_with_mask(
                        corrected_image, gt_tile, mask_tile, output.alpha, step_params,
                        fg_core_ptr, bg_core_ptr);
                    if (!result) {
                        nvtxRangePop();
                        nvtxRangePop();
                        return std::unexpected(result.error());
                    }
                    tile_loss = result->loss;
                    tile_grad = result->grad_image;
                    tile_grad_alpha = result->grad_alpha;
                } else {
                    auto result = compute_photometric_loss_with_gradient(
                        corrected_image, gt_tile, params_.optimization);
                    if (!result) {
                        nvtxRangePop();
                        nvtxRangePop();
                        return std::unexpected(result.error());
                    }
                    tile_loss = result->first;
                    tile_grad = result->second;
                }

                // Accumulate tile loss (stay on GPU)
                loss_tensor_gpu = loss_tensor_gpu + tile_loss;
                tiles_processed++;
                nvtxRangePop();

                // Backward through bilateral grid (accumulates gradients, no Adam yet)
                lfs::core::Tensor raster_grad = tile_grad;
                if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
                    nvtxRangePush("bilateral_grid_backward");
                    raster_grad = bilateral_grid_->backward(output.image, tile_grad, cam->uid());
                    nvtxRangePop();
                }

                nvtxRangePush("rasterize_backward");
                if (gsplat_ctx) {
                    // GUT mode: use gsplat backward (needs grad_alpha too)
                    auto grad_alpha = tile_grad_alpha.is_valid()
                                          ? tile_grad_alpha
                                          : lfs::core::Tensor::zeros_like(output.alpha);
                    gsplat_rasterize_backward(*gsplat_ctx, raster_grad, grad_alpha,
                                              strategy_->get_model(), strategy_->get_optimizer());
                } else {
                    // Standard mode: use fast rasterizer backward with optional alpha gradient
                    fast_rasterize_backward(*fast_ctx, raster_grad, strategy_->get_model(),
                                            strategy_->get_optimizer(), tile_grad_alpha);
                }
                nvtxRangePop();

                nvtxRangePop(); // End tile
            }

            if (tiles_processed > 1)
                loss_tensor_gpu = loss_tensor_gpu / static_cast<float>(tiles_processed);

            if (tiles_processed == 0) {
                LOG_DEBUG("Skipping iteration {} - no visible primitives", iter);
                return iter < params_.optimization.iterations && !stop_requested_.load() && !stop_token.stop_requested()
                           ? StepResult::Continue
                           : StepResult::Stop;
            }

            // Regularization losses are computed ONCE on full model (after all tiles)
            // They accumulate gradients on top of the per-tile gradients

            // Scale regularization loss - accumulate on GPU (AFTER rasterizer backward)
            if (params_.optimization.scale_reg > 0.0f) {
                nvtxRangePush("compute_scale_reg_loss");
                auto scale_loss_result = compute_scale_reg_loss(strategy_->get_model(), strategy_->get_optimizer(), params_.optimization);
                if (!scale_loss_result) {
                    return std::unexpected(scale_loss_result.error());
                }
                loss_tensor_gpu = loss_tensor_gpu + *scale_loss_result;
                nvtxRangePop();
            }

            // Opacity regularization loss - accumulate on GPU (AFTER rasterizer backward)
            if (params_.optimization.opacity_reg > 0.0f) {
                nvtxRangePush("compute_opacity_reg_loss");
                auto opacity_loss_result = compute_opacity_reg_loss(strategy_->get_model(), strategy_->get_optimizer(), params_.optimization);
                if (!opacity_loss_result) {
                    return std::unexpected(opacity_loss_result.error());
                }
                loss_tensor_gpu = loss_tensor_gpu + *opacity_loss_result;
                nvtxRangePop();
            }

            // Bilateral grid: TV loss + optimizer step
            if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
                nvtxRangePush("bilateral_grid_tv_and_step");
                const float tv_weight = params_.optimization.tv_loss_weight;

                loss_tensor_gpu = loss_tensor_gpu + bilateral_grid_->tv_loss_gpu() * tv_weight;
                bilateral_grid_->tv_backward(tv_weight);
                bilateral_grid_->optimizer_step();
                bilateral_grid_->zero_grad();
                bilateral_grid_->scheduler_step();

                nvtxRangePop();
            }

            // Sparsity loss - ALL ON GPU, no CPU sync here
            lfs::core::Tensor sparsity_loss_gpu;
            if (sparsity_optimizer_ && sparsity_optimizer_->should_apply_loss(iter)) {
                nvtxRangePush("sparsity_loss");
                auto sparsity_result = compute_sparsity_loss_forward(iter, strategy_->get_model());
                if (!sparsity_result) {
                    nvtxRangePop();
                    return std::unexpected(sparsity_result.error());
                }
                auto& [loss_tensor, ctx] = *sparsity_result;
                sparsity_loss_gpu = std::move(loss_tensor);

                if (ctx.n > 0) {
                    if (auto result = sparsity_optimizer_->compute_loss_backward(
                            ctx, 1.0f, strategy_->get_optimizer().get_grad(ParamType::Opacity));
                        !result) {
                        nvtxRangePop();
                        return std::unexpected(result.error());
                    }
                }
                nvtxRangePop();
            }

            // Sparsification phase logging (once per phase transition)
            if (params_.optimization.enable_sparsity) {
                const int base_iterations = params_.optimization.iterations - params_.optimization.sparsify_steps;
                if (iter == base_iterations + 1) {
                    LOG_INFO("Entering sparsification: {} Gaussians, target prune={}%",
                             strategy_->get_model().size(), params_.optimization.prune_ratio * 100);
                }
            }

            // Sync loss to CPU only at intervals - single sync point
            constexpr int LOSS_SYNC_INTERVAL = 10;
            float loss_value = 0.0f;
            if (iter % LOSS_SYNC_INTERVAL == 0 || iter == 1) {
                // Accumulate on GPU then sync once
                auto total_loss = sparsity_loss_gpu.numel() > 0
                                      ? (loss_tensor_gpu + sparsity_loss_gpu)
                                      : loss_tensor_gpu;
                loss_value = total_loss.item<float>();

                if (std::isnan(loss_value) || std::isinf(loss_value)) {
                    return std::unexpected(std::format("NaN/Inf loss at iteration {}", iter));
                }

                current_loss_ = loss_value;
                if (progress_) {
                    progress_->update(iter, loss_value,
                                      static_cast<int>(strategy_->get_model().size()),
                                      strategy_->is_refining(iter));
                }
                lfs::core::events::state::TrainingProgress{
                    .iteration = iter,
                    .loss = loss_value,
                    .num_gaussians = static_cast<int>(strategy_->get_model().size()),
                    .is_refining = strategy_->is_refining(iter)}
                    .emit();
            }

            {
                DeferredEvents deferred;
                {
                    std::unique_lock<std::shared_mutex> lock(render_mutex_);

                    // Skip topology-changing ops during sparsification OR NGS noise phase
                    const bool in_sparsification = params_.optimization.enable_sparsity &&
                                                   iter > (params_.optimization.iterations - params_.optimization.sparsify_steps);
                    const bool ngs_block_topology = (ngs_current_phase_ == NGSPhase::WithNoise);

                    // Freeze noise gaussians: means/scaling/rotation/SH, keep opacity trainable
                    if (ngs_block_topology && ngs_surface_count_ > 0 && ngs_noise_count_ > 0) {
                        auto& opt = strategy_->get_optimizer();
                        const size_t start = ngs_surface_count_;
                        const size_t count = ngs_noise_count_;
                        opt.zero_grad_range(ParamType::Means, start, count);
                        opt.zero_grad_range(ParamType::Sh0, start, count);
                        opt.zero_grad_range(ParamType::ShN, start, count);
                        opt.zero_grad_range(ParamType::Scaling, start, count);
                        opt.zero_grad_range(ParamType::Rotation, start, count);
                    }
                    if (!in_sparsification && !ngs_block_topology) {
                        strategy_->post_backward(iter, r_output);
                    }
                    strategy_->step(iter);
                }
                
                // During NGS noise phase we keep ordering/size stable (no pruning/densification).
                if (ngs_current_phase_ != NGSPhase::WithNoise) {
                    if (auto result = handle_sparsity_update(iter, strategy_->get_model()); !result) {
                        LOG_ERROR("Sparsity update: {}", result.error());
                    }
                    if (auto result = apply_sparsity_pruning(iter, strategy_->get_model()); !result) {
                        LOG_ERROR("Sparsity pruning: {}", result.error());
                    }
                }

                // Clean evaluation - let the evaluator handle everything
                if (evaluator_->is_enabled() && evaluator_->should_evaluate(iter)) {
                    evaluator_->print_evaluation_header(iter);
                    auto metrics = evaluator_->evaluate(iter,
                                                        strategy_->get_model(),
                                                        val_dataset_,
                                                        background_);
                    LOG_INFO("{}", metrics.to_string());
                }

                // Save checkpoint (not PLY) at specified steps                
                if (!params_.optimization.skip_intermediate) {
                    for (size_t save_step : params_.optimization.save_steps) {
                        if (iter == static_cast<int>(save_step) && iter != params_.optimization.iterations) {
                            auto result = save_checkpoint(iter);
                            if (!result) {
                                LOG_WARN("Failed to save checkpoint at iteration {}: {}", iter, result.error());
                            }
                        }
                    }
                }

                if (!params_.dataset.timelapse_images.empty() && iter % params_.dataset.timelapse_every == 0) {
                    for (const auto& img_name : params_.dataset.timelapse_images) {
                        auto train_cam = train_dataset_->get_camera_by_filename(img_name);
                        auto val_cam = val_dataset_ ? val_dataset_->get_camera_by_filename(img_name) : std::nullopt;
                        if (train_cam.has_value() || val_cam.has_value()) {
                            lfs::core::Camera* cam_to_use = train_cam.has_value() ? train_cam.value() : val_cam.value();

                            // Image size isn't correct until the image has been loaded once
                            // If we use the camera before it's loaded, it will render images at the non-scaled size
                            if ((cam_to_use->camera_height() == cam_to_use->image_height() && params_.dataset.resize_factor != 1) ||
                                cam_to_use->image_height() > params_.dataset.max_width ||
                                cam_to_use->image_width() > params_.dataset.max_width) {
                                cam_to_use->load_image_size(params_.dataset.resize_factor, params_.dataset.max_width);
                            }

                            RenderOutput rendered_timelapse_output;
                            if (params_.optimization.gut) {
                                rendered_timelapse_output = gsplat_rasterize(*cam_to_use, strategy_->get_model(), background_,
                                                                             1.0f, false, GsplatRenderMode::RGB, true);
                            } else {
                                rendered_timelapse_output = fast_rasterize(*cam_to_use, strategy_->get_model(), background_);
                            }

                            // Get folder name to save in by stripping file extension
                            std::string folder_name = lfs::io::strip_extension(img_name);

                            auto output_path = params_.dataset.output_path / "timelapse" / folder_name;
                            std::filesystem::create_directories(output_path);

                            lfs::core::image_io::save_image_async(output_path / std::format("{:06d}.jpg", iter),
                                                                  rendered_timelapse_output.image);
                        } else {
                            LOG_WARN("Timelapse image '{}' not found in dataset.", img_name);
                        }
                    }
                }
            }

            // Return Continue if we should continue training
            if (iter < params_.optimization.iterations && !stop_requested_.load() && !stop_token.stop_requested()) {
                return StepResult::Continue;
            } else {
                return StepResult::Stop;
            }
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Training step failed: {}", e.what()));
        }
    }

    std::expected<void, std::string> Trainer::train(std::stop_token stop_token) {
        // Check if initialized
        if (!initialized_.load()) {
            return std::unexpected("Trainer not initialized. Call initialize() before train()");
        }

        is_running_ = false;
        training_complete_ = false;
        ready_to_start_ = false; // Reset the flag

        ready_to_start_ = true; // Skip GUI wait for now

        is_running_ = true; // Now we can start
        LOG_INFO("Starting training loop");
        // initializing image loader
        auto& cache_loader = lfs::io::CacheLoader::getInstance(params_.dataset.loading_params.use_cpu_memory, params_.dataset.loading_params.use_fs_cache);
        cache_loader.reset_cache();
        // in case we call getInstance multiple times and cache parameters/dataset were changed by user
        cache_loader.update_cache_params(params_.dataset.loading_params.use_cpu_memory,
                                         params_.dataset.loading_params.use_fs_cache,
                                         train_dataset_size_,
                                         params_.dataset.loading_params.min_cpu_free_GB,
                                         params_.dataset.loading_params.min_cpu_free_memory_ratio,
                                         params_.dataset.loading_params.print_cache_status,
                                         params_.dataset.loading_params.print_status_freq_num);

        try {
            // Start from current_iteration_ (allows resume from checkpoint)
            int iter = current_iteration_.load() > 0 ? current_iteration_.load() + 1 : 1;
            const RenderMode render_mode = RenderMode::RGB;

            if (progress_) {
                progress_->update(iter, current_loss_.load(),
                                  static_cast<int>(strategy_->get_model().size()),
                                  strategy_->is_refining(iter));
            }

            // Conservative prefetch to avoid VRAM exhaustion
            lfs::io::PipelinedLoaderConfig pipelined_config;
            pipelined_config.jpeg_batch_size = 8;
            pipelined_config.prefetch_count = 8;
            pipelined_config.output_queue_size = 4;
            pipelined_config.io_threads = 2;

            // Non-JPEG images (PNG, WebP) need CPU decoding - use more threads until cache warms
            constexpr float NON_JPEG_THRESHOLD = 0.1f;
            constexpr size_t MIN_COLD_THREADS = 4;
            constexpr size_t COLD_PREFETCH_COUNT = 16;
            const float non_jpeg_ratio = train_dataset_->get_non_jpeg_ratio();
            if (non_jpeg_ratio > NON_JPEG_THRESHOLD) {
                const size_t cold_threads = std::max(MIN_COLD_THREADS,
                                                     static_cast<size_t>(std::thread::hardware_concurrency() / 2));
                pipelined_config.cold_process_threads = cold_threads;
                pipelined_config.prefetch_count = COLD_PREFETCH_COUNT;
                LOG_INFO("{:.0f}% non-JPEG images, using {} cold threads", non_jpeg_ratio * 100.0f, cold_threads);
            }

            // Configure mask loading if masks are enabled
            PipelinedMaskConfig mask_pipeline_config;
            if (params_.optimization.mask_mode != lfs::core::param::MaskMode::None) {
                mask_pipeline_config.load_masks = true;
                mask_pipeline_config.invert_masks = params_.optimization.invert_masks;
                mask_pipeline_config.mask_threshold = params_.optimization.mask_threshold;
                LOG_INFO("Mask loading enabled in pipeline (invert={}, threshold={})",
                         mask_pipeline_config.invert_masks, mask_pipeline_config.mask_threshold);
            }

            auto train_dataloader = create_infinite_pipelined_dataloader(
                train_dataset_, pipelined_config, mask_pipeline_config);

            LOG_DEBUG("Starting training iterations");
            while (iter <= params_.optimization.iterations) {
                lfs::core::CudaMemoryPool::instance().set_iteration(iter);
                if (stop_token.stop_requested() || stop_requested_.load())
                    break;
                if (callback_busy_.load())
                    cudaStreamSynchronize(callback_stream_);

                lfs::core::Camera* cam = nullptr;
                lfs::core::Tensor gt_image;
                auto example_opt = train_dataloader->next();
                if (!example_opt) {
                    LOG_ERROR("DataLoader returned nullopt unexpectedly");
                    break;
                }
                auto& example = *example_opt;
                cam = example.data.camera;
                gt_image = std::move(example.data.image);

                // Store pipelined mask for use in train_step
                pipelined_mask_ = example.mask.has_value() ? std::move(*example.mask) : lfs::core::Tensor();

                auto step_result = train_step(iter, cam, gt_image, render_mode, stop_token);
                if (!step_result) {
                    // Check if this is an OOM_RETRY signal
                    if (step_result.error() == "OOM_RETRY") {
                        // Aggressive memory cleanup before retry
                        LOG_INFO("Performing aggressive memory cleanup before retry...");

                        // 0. CRITICAL: Synchronize and clear any pending CUDA errors
                        cudaError_t sync_err = cudaDeviceSynchronize();
                        if (sync_err != cudaSuccess) {
                            LOG_WARN("cudaDeviceSynchronize before cleanup returned: {}", cudaGetErrorString(sync_err));
                            // Clear the error so we can continue
                            cudaGetLastError();
                        }

                        // 1. Emergency cleanup of arena (resets offsets, clears inactive frames)
                        lfs::core::GlobalArenaManager::instance().get_arena().emergency_cleanup();

                        // 2. Trim cached memory pool
                        lfs::core::CudaMemoryPool::instance().trim_cached_memory();

                        // 3. Synchronize again after cleanup
                        cudaDeviceSynchronize();

                        // 4. Clear any error state from the OOM
                        cudaGetLastError();

                        // 5. Log memory status
                        size_t free_mem = 0, total_mem = 0;
                        cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
                        if (err == cudaSuccess) {
                            LOG_INFO("CUDA memory after aggressive cleanup: {:.2f} GB free / {:.2f} GB total",
                                     free_mem / (1024.0 * 1024.0 * 1024.0),
                                     total_mem / (1024.0 * 1024.0 * 1024.0));
                        } else {
                            LOG_WARN("cudaMemGetInfo failed: {}", cudaGetErrorString(err));
                            cudaGetLastError(); // Clear this error too
                        }

                        // Retry the same step with upgraded tile mode
                        LOG_INFO("Retrying iteration {} with upgraded tile mode", iter);
                        step_result = train_step(iter, cam, gt_image, render_mode, stop_token);
                        if (!step_result) {
                            // If retry also failed, propagate the error
                            return std::unexpected(step_result.error());
                        }
                    } else {
                        // Regular error - propagate
                        return std::unexpected(step_result.error());
                    }
                }

                if (*step_result == StepResult::Stop) {
                    break;
                }

                // Launch callback for async progress update (except first iteration)
                if (iter > 1 && callback_) {
                    callback_busy_ = true;
                    auto err = cudaLaunchHostFunc(
                        callback_stream_,
                        [](void* self) {
                            auto* trainer = static_cast<Trainer*>(self);
                            if (trainer->callback_) {
                                trainer->callback_();
                            }
                            trainer->callback_busy_ = false;
                        },
                        this);
                    if (err != cudaSuccess) {
                        LOG_WARN("Failed to launch callback: {}", cudaGetErrorString(err));
                        callback_busy_ = false;
                    }
                }

                ++iter;
            }

            // Ensure callback is finished before final save
            if (callback_busy_.load()) {
                cudaStreamSynchronize(callback_stream_);
            }

            // -----------------------------------------------------------------
            // Post-training mask-based pruning
            //
            // For matting modes, remove splats that don't align with masks:
            // - Center-vote: splats whose center is outside mask in most views
            // - Leakage: splats whose footprint extends outside mask boundary
            // -----------------------------------------------------------------
            if (params_.optimization.mask_mode == lfs::core::param::MaskMode::HardMatting ||
                params_.optimization.mask_mode == lfs::core::param::MaskMode::SoftMatting)
            //if (false)
            {

                mask_pruning::CenterVotePruningConfig center_cfg;
                center_cfg.vote_ratio_threshold = 0.80f;
                center_cfg.min_visibility_count = 3;
                center_cfg.invert_masks = params_.optimization.invert_masks;

                mask_pruning::LeakagePruningConfig leak_cfg;
                leak_cfg.enabled = true;
                leak_cfg.leak_keep_threshold = 0.90f;
                leak_cfg.per_view_leak_fraction = 0.50f;
                leak_cfg.min_visibility_count = 1;
                leak_cfg.min_pixel_radius = 2.0f;
                leak_cfg.sample_points = 8;
                leak_cfg.dilate_px = 2;
                leak_cfg.invert_masks = params_.optimization.invert_masks;

                #ifdef NeedsPruningDiagnostics
                    std::filesystem::path diag_dir = params_.dataset.output_path / "pruning_diagnostics";

                    LOG_INFO("===== MASK-BASED PRUNING WITH VISUALIZATION =====");
                    bool viz_success = mask_pruning::visualizer::generate_pruning_diagnostics(
                        *strategy_,
                        *train_dataset_,
                        center_cfg,
                        diag_dir,
                        25 // Max 25 cámaras
                    );

                    if (viz_success) {
                        LOG_INFO("✓ Diagnostic images saved to: {}", diag_dir.string());
                        LOG_INFO("  Generated 6 layers per camera:");
                        LOG_INFO("    01_mask.jpg          - Mask only");
                        LOG_INFO("    02_centers_all.jpg   - All splat centers");
                        LOG_INFO("    03_centers_classified.jpg - Green=inside, Red=outside");
                        LOG_INFO("    04_footprints.jpg    - Splat ellipses");
                        LOG_INFO("    05_problem_splats.jpg - Outside + large radius");
                        LOG_INFO("    06_summary.jpg       - Summary overlay");
                    } else {
                        LOG_WARN("Failed to generate diagnostic visualizations");
                    }
                #endif
                LOG_INFO("Running post-training mask-based pruning...");


                auto pruning_result = mask_pruning::prune_after_training(
                    *strategy_,
                    *train_dataset_,
                    center_cfg,
                    leak_cfg);


                if (!pruning_result) {
                    LOG_WARN("Post-training pruning failed: {}", pruning_result.error());
                } else if (pruning_result->splats_removed > 0) {
                    LOG_INFO("Pruning complete: removed {} splats ({:.1f}%)",
                             pruning_result->splats_removed,
                             pruning_result->removal_ratio() * 100.0f);
                }
            }

            // Final save if not already saved by stop request
            if (!stop_requested_.load() && !stop_token.stop_requested()) {
                auto final_path = params_.dataset.output_path;
                save_ply(final_path, params_.optimization.iterations, /*join=*/true);
            }

            if (progress_) {
                progress_->complete();
            }
            evaluator_->save_report();
            if (progress_) {
                progress_->print_final_summary(static_cast<int>(strategy_->get_model().size()));
            }

            is_running_ = false;
            training_complete_ = true;

            cache_loader.clear_cpu_cache();
            lfs::core::image_io::wait_for_pending_saves();

            LOG_INFO("Training completed successfully");
            return {};
        } catch (const std::exception& e) {
            is_running_ = false;
            cache_loader.clear_cpu_cache();
            lfs::core::image_io::wait_for_pending_saves();

            return std::unexpected(std::format("Training failed: {}", e.what()));
        }
    }

    void Trainer::save_ply(const std::filesystem::path& save_path, const int iter_num, const bool join_threads) {
        const lfs::io::PlySaveOptions ply_options{
            .output_path = save_path / ("splat_" + std::to_string(iter_num) + ".ply"),
            .binary = true,
            .async = !join_threads};

        const auto ply_result = lfs::io::save_ply(strategy_->get_model(), ply_options);
        if (!ply_result) {
            if (ply_result.error().code == lfs::io::ErrorCode::INSUFFICIENT_DISK_SPACE) {
                lfs::core::events::state::DiskSpaceSaveFailed{
                    .iteration = iter_num,
                    .path = ply_options.output_path,
                    .error = ply_result.error().message,
                    .required_bytes = ply_result.error().required_bytes,
                    .available_bytes = ply_result.error().available_bytes,
                    .is_disk_space_error = true,
                    .is_checkpoint = false}
                    .emit();
            }
            LOG_WARN("Failed to save PLY: {}", ply_result.error().message);
            return; // Don't save checkpoint if PLY failed
        }

        // Save checkpoint alongside PLY for training resumption (skip if --skip-intermediate)
        if (!params_.optimization.skip_intermediate) {
            auto ckpt_result = lfs::training::save_checkpoint(
                save_path, iter_num, *strategy_, params_, bilateral_grid_.get());
            if (!ckpt_result) {
                LOG_WARN("Failed to save checkpoint: {}", ckpt_result.error());
            }
        }

        LOG_DEBUG("PLY save initiated: {} (sync={})", lfs::core::path_to_utf8(save_path), join_threads);
    }

    std::expected<void, std::string> Trainer::save_checkpoint(int iteration) {
        if (!strategy_) {
            return std::unexpected("Cannot save checkpoint: no strategy initialized");
        }

        return lfs::training::save_checkpoint(
            params_.dataset.output_path, iteration, *strategy_, params_,
            bilateral_grid_.get());
    }

    void Trainer::save_final_ply_and_checkpoint(const int iteration) {
        save_ply(params_.dataset.output_path, iteration, /*join=*/true);
    }

    std::expected<int, std::string> Trainer::load_checkpoint(const std::filesystem::path& checkpoint_path) {
        if (!strategy_) {
            return std::unexpected("Cannot load checkpoint: no strategy initialized");
        }

        // Create bilateral grid before loading if needed (checkpoint may contain grid state)
        if (params_.optimization.use_bilateral_grid && !bilateral_grid_) {
            if (auto init_result = initialize_bilateral_grid(); !init_result) {
                LOG_WARN("Failed to init bilateral grid for resume: {}", init_result.error());
            }
        }

        auto result = lfs::training::load_checkpoint(
            checkpoint_path, *strategy_, params_, bilateral_grid_.get());
        if (!result) {
            return result;
        }
        current_iteration_ = *result;

        LOG_INFO("Restored training state from checkpoint at iteration {}", *result);
        return result;
    }
        
    
    // ----------------------------------------------------------------------------
    // initialize_ngs() - Call from Trainer::initialize() after strategy init
    // ----------------------------------------------------------------------------

    std::expected<void, std::string> Trainer::initialize_ngs() {
        const auto& opt = params_.optimization;

        if (opt.ngs_noise_path.empty()) {
            ngs_enabled_ = false;
            return {};
        }

        LOG_INFO("Initializing NGS");

        // Create config with just the path
        NGSConfig config;
        config.noise_ply_path = opt.ngs_noise_path;

        // Create phase manager
        ngs_phase_manager_ = std::make_unique<NGSPhaseManager>(config, opt.iterations);

        // Load noise (but don't inject yet)
        ngs_noise_ = std::make_unique<NoiseGaussians>();
        if (auto result = ngs_noise_->load(opt.ngs_noise_path); !result) {
            return std::unexpected(result.error());
        }

        ngs_enabled_ = true;
        ngs_current_phase_ = NGSPhase::StandardTraining;
        ngs_surface_count_ = 0;
        ngs_noise_count_ = 0;

        const int start_iter = ngs_phase_manager_->noise_start();
        const int end_iter = ngs_phase_manager_->noise_end();

        LOG_INFO("NGS ready: {} noise Gaussians, inject at iter {}, remove at iter {}",
                ngs_noise_->size(), start_iter, end_iter);

        return {};
    }

    // ----------------------------------------------------------------------------
    // handle_ngs_phase_transition() - Call at START of train_step()
    // ----------------------------------------------------------------------------

    void Trainer::handle_ngs_phase_transition(int iter) {
        if (!ngs_enabled_ || !ngs_phase_manager_)
            return;

        const NGSPhase new_phase = ngs_phase_manager_->get_phase(iter);

        // Phase transition
        if (new_phase != ngs_current_phase_) {
            const NGSPhase old_phase = ngs_current_phase_;
            ngs_current_phase_ = new_phase;

            if (old_phase == NGSPhase::StandardTraining && new_phase == NGSPhase::WithNoise) {
                LOG_INFO("NGS: Injecting {} noise Gaussians at iteration {}", ngs_noise_ ? ngs_noise_->size() : 0, iter);
                inject_noise_into_model();
            } else if (old_phase == NGSPhase::WithNoise && new_phase == NGSPhase::Cleanup) {
                LOG_INFO("NGS: Removing noise Gaussians at iteration {}", iter);
                remove_noise_from_model();
            }
        }

        // Per-iteration operations during noise phase
        if (ngs_current_phase_ == NGSPhase::WithNoise && ngs_surface_count_ > 0 && ngs_noise_count_ > 0) {
            auto& model = strategy_->get_model();
            auto& optimizer = strategy_->get_optimizer();

            // Randomize noise colors
            ngs_randomize_sh0_range_inplace(model.sh0(), ngs_surface_count_, ngs_noise_count_, static_cast<uint32_t>(iter));

            // Freeze noise opacity UNTIL kNgsOpacityUnfreezePoint (then let it fade)
            const int unfreeze_iter = static_cast<int>(kNgsOpacityUnfreezePoint * params_.optimization.iterations);
            if (iter < unfreeze_iter) {
                optimizer.zero_grad_range(ParamType::Opacity, ngs_surface_count_, ngs_noise_count_);
            }
        }
    }

    // ----------------------------------------------------------------------------
    // inject_noise_into_model() - Add noise to SplatData and expand optimizer
    // ----------------------------------------------------------------------------

    void Trainer::inject_noise_into_model() {
        if (!ngs_noise_) {
            LOG_WARN("NGS: No noise loaded, cannot inject");
            return;
        }

        auto& model = strategy_->get_model();
        auto& optimizer = strategy_->get_optimizer();

        ngs_surface_count_ = model.size();
        ngs_noise_count_ = ngs_noise_->size();
        
        const size_t noise_count = ngs_noise_count_;
        if (noise_count == 0) {
            LOG_WARN("NGS: Noise PLY contains 0 gaussians");
            return;
        }

        LOG_DEBUG("NGS inject: {} surface + {} noise", ngs_surface_count_, noise_count);

        // Concatenate parameter tensors (noise provides means/scaling/rotation/opacity only)
        auto new_means = lfs::core::Tensor::cat({model.means(), ngs_noise_->means()}, 0);
        auto new_scaling = lfs::core::Tensor::cat({model.scaling_raw(), ngs_noise_->scaling_raw()}, 0);
        auto new_rotation = lfs::core::Tensor::cat({model.rotation_raw(), ngs_noise_->rotation_raw()}, 0);
        auto new_opacity = lfs::core::Tensor::cat({model.opacity_raw(), ngs_noise_->opacity_raw()}, 0);
        
        // SH0: allocate zeros for noise, then randomize in-place per-iteration (model-side)
        lfs::core::Tensor new_sh0;
        {
            const auto& shape = model.sh0().shape();
            std::vector<size_t> dims = {noise_count};
            for (size_t i = 1; i < shape.rank(); ++i) dims.push_back(shape[i]);
            lfs::core::TensorShape noise_shape(dims);

            auto noise_sh0 = lfs::core::Tensor::zeros(noise_shape, model.sh0().device(), model.sh0().dtype());
            new_sh0 = lfs::core::Tensor::cat({model.sh0(), noise_sh0}, 0);
        }

        // SHN: noise has no higher-order SH; append zeros if surface uses SHN
        lfs::core::Tensor new_shN;
        if (model.shN().is_valid() && model.shN().numel() > 0) {
            const auto& shape = model.shN().shape();
            std::vector<size_t> dims = {noise_count};
            for (size_t i = 1; i < shape.rank(); ++i) dims.push_back(shape[i]);
            lfs::core::TensorShape noise_shape(dims);

            auto noise_shN = lfs::core::Tensor::zeros(noise_shape, model.shN().device(), model.shN().dtype());             
            new_shN = lfs::core::Tensor::cat({model.shN(), noise_shN}, 0);
        }

        // Update model tensors
        model.set_means_internal(new_means);
        model.set_sh0_internal(new_sh0);
        model.set_scaling_internal(new_scaling);
        model.set_rotation_internal(new_rotation);
        model.set_opacity_internal(new_opacity);
        if (new_shN.is_valid()) {
            model.set_shN_internal(new_shN);
        }

        // Expand optimizer state for new Gaussians
        optimizer.extend_for_new_gaussians(noise_count);

        LOG_INFO("NGS: Model now has {} Gaussians ({} surface + {} noise)",
                 model.size(), ngs_surface_count_, noise_count);
    }

    // ----------------------------------------------------------------------------
    // remove_noise_from_model() - Remove noise from SplatData and shrink optimizer
    // ----------------------------------------------------------------------------

    // Helper: remove [s,e) along dim0 without ever calling slice on an empty range.
    static lfs::core::Tensor remove_range_dim0_no_empty(const lfs::core::Tensor& src, int s, int e, int total) {
        // Preconditions: 0 <= s < e <= total
        const bool keep_prefix = (s > 0);
        const bool keep_suffix = (e < total);

        if (keep_prefix && keep_suffix) {
            // Keep [0,s) + [e,total)
            return lfs::core::Tensor::cat({src.slice(0, 0, s), src.slice(0, e, total)}, 0).contiguous();
        } else if (keep_prefix) {
            // Keep [0,s)
            return src.slice(0, 0, s).contiguous();
        } else if (keep_suffix) {
            // Keep [e,total)
            return src.slice(0, e, total).contiguous();
        }

        // Would remove everything -> shouldn't happen
        LOG_ERROR("NGS: remove_range_dim0 would remove entire tensor (total={})", total);
        return src;
    }

    void Trainer::remove_noise_from_model() {
        auto& model = strategy_->get_model();
        auto& optimizer = strategy_->get_optimizer();

        if (ngs_surface_count_ == 0 || ngs_noise_count_ == 0) {
            LOG_WARN("NGS: Cannot remove noise - nothing injected");
            return;
        }

        const size_t total_sz = model.size();
        const size_t start_sz = ngs_surface_count_;
        const size_t end_sz = std::min(total_sz, start_sz + ngs_noise_count_);

        if (start_sz >= total_sz || end_sz <= start_sz) {
            LOG_WARN("NGS: Cannot remove noise - invalid range (start={}, end={}, total={})",
                     start_sz, end_sz, total_sz);
            return;
        }

        const int t = static_cast<int>(total_sz);
        const int s = static_cast<int>(start_sz);
        const int e = static_cast<int>(end_sz);

        // IMPORTANT: do NOT call slice() with [t,t)
        model.set_means_internal(remove_range_dim0_no_empty(model.means(), s, e, t));
        model.set_sh0_internal(remove_range_dim0_no_empty(model.sh0(), s, e, t));
        model.set_scaling_internal(remove_range_dim0_no_empty(model.scaling_raw(), s, e, t));
        model.set_rotation_internal(remove_range_dim0_no_empty(model.rotation_raw(), s, e, t));
        model.set_opacity_internal(remove_range_dim0_no_empty(model.opacity_raw(), s, e, t));

        if (model.shN().is_valid() && model.shN().numel() > 0) {
            model.set_shN_internal(remove_range_dim0_no_empty(model.shN(), s, e, t));
        }

        // Remove optimizer state for the same range (also must avoid empty slices inside it!)
        optimizer.remove_range(start_sz, end_sz - start_sz);

        LOG_INFO("NGS: Removed {} noise Gaussians, model now has {} Gaussians",
                 (end_sz - start_sz), model.size());

        ngs_noise_.reset();
        ngs_surface_count_ = 0;
        ngs_noise_count_ = 0;
    }


} // namespace lfs::training
