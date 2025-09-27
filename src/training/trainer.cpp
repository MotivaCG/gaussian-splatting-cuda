/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "trainer.hpp"
#include "components/bilateral_grid.hpp"
#include "components/poseopt.hpp"
#include "components/sparsity_optimizer.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "kernels/fused_ssim.cuh"
#include "rasterization/fast_rasterizer.hpp"
#include "rasterization/rasterizer.hpp"
#include "rasterization/projection_fast.hpp"  
#include <ATen/cuda/CUDAEvent.h>
#include <atomic>
#include <chrono>
#include <cuda_runtime.h>
#include <expected>
#include <memory>

namespace gs::training {

    void Trainer::cleanup() {
        LOG_DEBUG("Cleaning up trainer for re-initialization");

        // Stop any ongoing operations
        stop_requested_ = true;

        // Wait for callback to finish if busy
        if (callback_busy_.load()) {
            callback_stream_.synchronize();
            callback_busy_.store(false);
        }

        // Reset all components
        progress_.reset();
        bilateral_grid_.reset();
        bilateral_grid_optimizer_.reset();
        bilateral_grid_scheduler_.reset();
        poseopt_module_.reset();
        poseopt_optimizer_.reset();
        sparsity_optimizer_.reset();
        evaluator_.reset();

        // Clear datasets (will be recreated)
        train_dataset_.reset();
        val_dataset_.reset();

        // Clear camera cache
        m_cam_id_to_cam.clear();

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

        LOG_DEBUG("Trainer cleanup complete");
    }

    std::expected<void, std::string> Trainer::initialize_bilateral_grid() {
        if (!params_.optimization.use_bilateral_grid) {
            return {};
        }

        try {
            bilateral_grid_ = std::make_unique<BilateralGrid>(
                train_dataset_size_,
                params_.optimization.bilateral_grid_X,
                params_.optimization.bilateral_grid_Y,
                params_.optimization.bilateral_grid_W);

            bilateral_grid_optimizer_ = std::make_unique<torch::optim::Adam>(
                std::vector<torch::Tensor>{bilateral_grid_->parameters()},
                torch::optim::AdamOptions(params_.optimization.bilateral_grid_lr)
                    .eps(1e-15));

            // Create scheduler with warmup
            const double gamma = std::pow(0.01, 1.0 / params_.optimization.iterations);
            bilateral_grid_scheduler_ = std::make_unique<WarmupExponentialLR>(
                *bilateral_grid_optimizer_,
                gamma,
                1000, // warmup steps
                0.01, // start at 1% of initial LR
                -1    // all param groups
            );

            LOG_DEBUG("Bilateral grid initialized with size {}x{}x{} and warmup scheduler",
                      params_.optimization.bilateral_grid_X,
                      params_.optimization.bilateral_grid_Y,
                      params_.optimization.bilateral_grid_W);
            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to initialize bilateral grid: {}", e.what()));
        }
    }

    static inline torch::Tensor make_darkness_weight(const torch::Tensor& gt, float boost) {
        torch::NoGradGuard no_grad;

        // gt: [B,H,W,3] (NHWC) o [B,3,H,W] (NCHW)
        constexpr float kR = 0.299f, kG = 0.587f, kB = 0.114f;
        const bool nhwc = (gt.size(-1) == 3);

        torch::Tensor r, g, b; // [B,H,W] en ambos casos
        if (nhwc) {
            r = gt.select(-1, 0);
            g = gt.select(-1, 1);
            b = gt.select(-1, 2);
        } else {
            r = gt.select(1, 0);
            g = gt.select(1, 1);
            b = gt.select(1, 2);
        }

        // brightness in [0,1], darkness = 1 - brightness
        auto darkness = (1.0f - (kR * r + kG * g + kB * b));

        // w = 1 + boost * darkness, normalized by its mean (keeps global scale stable)
        auto w = 1.0f + boost * darkness;
        const float eps = 1e-4f;
        w = w / torch::clamp(w.mean(), eps, 10.0f);

        // Expand for broadcasting across the channel dimension
        return nhwc ? w.unsqueeze(-1) : w.unsqueeze(1);
    }

    std::expected<torch::Tensor, std::string> Trainer::compute_photometric_loss(
        const RenderOutput& render_output,
        const torch::Tensor& gt_image,
        const SplatData& splatData,
        const param::OptimizationParameters& opt_params) {
        try {
            // Ensure images have same dimensions
            torch::Tensor rendered = render_output.image;
            torch::Tensor gt = gt_image;

            // Ensure both tensors are 4D (batch, height, width, channels)
            rendered = rendered.dim() == 3 ? rendered.unsqueeze(0) : rendered;
            gt = gt.dim() == 3 ? gt.unsqueeze(0) : gt;

            TORCH_CHECK(rendered.sizes() == gt.sizes(),
                        "ERROR: size mismatch – rendered ", rendered.sizes(),
                        " vs. ground truth ", gt.sizes());

            // Base loss: L1 + SSIM
            torch::Tensor l1_loss;
            // Weight amplify loss in dark regions
            if (opt_params.darkness_boost <= 0) {
                l1_loss = torch::l1_loss(rendered, gt);
            } else {
                torch::NoGradGuard no_grad;
                auto darkness_weight = make_darkness_weight(gt, opt_params.darkness_boost);
                l1_loss = (torch::l1_loss(rendered, gt, torch::Reduction::None) * darkness_weight).mean();
            }
            auto ssim_loss = 1.f - fused_ssim(rendered, gt, "valid", /*train=*/true);
            torch::Tensor loss = (1.f - opt_params.lambda_dssim) * l1_loss +
                                 opt_params.lambda_dssim * ssim_loss;
            return loss;
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error computing photometric loss: {}", e.what()));
        }
    }
    // Place this new static function inside trainer.cpp

    /**
     * @brief Applies a penalty based on the final rendered alpha channel of the image.
     *
     * This method penalizes pixels inside the mask that are semi-transparent and
     * pixels outside the mask that have any opacity. It's a more direct approach
     * than the splat-based penalty.
     *
     * @param base_loss The pre-computed photometric loss.
     * @param rendered_alpha The final accumulated alpha channel from the rasterizer [H, W].
     * @param weights The float weight map, where values > 0.5 define the ROI.
     * @param w The global weight multiplier for this penalty.
     * @param kIn The weight for penalizing transparency INSIDE the mask.
     * @param kOut The weight for penalizing opacity OUTSIDE the mask.
     * @return torch::Tensor The loss with the pixel-based opacity penalty added.
     */
    static torch::Tensor pixelBasedOpacityPenalty(const torch::Tensor& base_loss,
                                                  const torch::Tensor& rendered_alpha,
                                                  const torch::Tensor& weights,
                                                  const float w) {
        
        const float kOut = 0.333f;
        const float kIn =  0.667f;

        // Convert the float weight map to a clean boolean mask
        auto bool_mask = (weights > 0.5f);

        // 1. Penalty for being transparent INSIDE the mask
        // (1.0 - alpha) is high where pixels are transparent.
        // We multiply by the boolean mask to only consider pixels inside the ROI.
        auto inside_penalty_map = (1.0f - rendered_alpha) * bool_mask;

        // 2. Penalty for being opaque OUTSIDE the mask
        // `alpha` is high where pixels are opaque.
        // We multiply by the inverted mask to only consider pixels outside the ROI.
        auto outside_penalty_map = rendered_alpha * (~bool_mask);

        // 3. Combine and normalize the penalties
        // We sum the penalty over all pixels and normalize by the total number of pixels.
        float num_pixels = static_cast<float>(rendered_alpha.numel());
        auto total_penalty = w * (kIn * inside_penalty_map.sum() + kOut * outside_penalty_map.sum()) / num_pixels;

        // 4. Add the penalty to the base loss
        // For pixel-based losses, adding is often more stable than multiplying.
        //return base_loss + 0.01 * total_penalty;
        
        return (base_loss * (1.0f + total_penalty)) + (1e-2f * total_penalty);
    }

    /**
     * @brief Applies a penalty to guide splat opacity based on an attention mask.
     * @param base_loss The pre-computed photometric loss.
     * @param out The RenderOutput from the rasterizer.
     * @param weights The float weight map, where values > 0.5 define the ROI.
     * @param opacities_alpha The opacity alpha values (0-1) from get_opacity().
     * @param w The global weight multiplier for this penalty.
     */
    static torch::Tensor outsideMaskOpacityPenalty(const torch::Tensor& base_loss,
                                                   const RenderOutput& out,
                                                   const torch::Tensor& weights,
                                                   const torch::Tensor& opacities_alpha,
                                                   float w) {
        if (w == 0.0f) {
            return base_loss;
        }
        if (!weights.defined() || !out.means2d.defined() || !out.radii.defined()) {
            return base_loss;
        }

        torch::Tensor visible_mask = (out.radii > 0.0f);
        if (!visible_mask.any().item<bool>()) {
            return base_loss;
        }
        auto visible_indices = visible_mask.nonzero().squeeze(-1);
        auto xy = out.means2d.index({visible_indices});

        auto alpha = opacities_alpha.index({visible_indices});

        const int W = out.width, H = out.height;
        auto x = torch::round(xy.select(1, 0)).to(torch::kLong).clamp(0, W - 1);
        auto y = torch::round(xy.select(1, 1)).to(torch::kLong).clamp(0, H - 1);
        auto linear_indices = y * W + x;

        auto bool_mask = (weights > 0.5f);
        auto is_in_mask = bool_mask.to(torch::kFloat32).flatten();
        auto is_in = is_in_mask.index({linear_indices});
        auto is_out = 1.0f - is_in;

        const float kOut = 2.0f;
        const float kIn = 0.02f;
        auto outside_penalty = alpha * is_out;
        auto inside_penalty = (1.0f - alpha) * is_in;

        auto combined_penalty = w * (kOut * outside_penalty + kIn * inside_penalty);
        auto mean_penalty = combined_penalty.mean();
        
        return (base_loss * (1.0f + mean_penalty)) + (1e-5f*mean_penalty);
    }

    std::expected<torch::Tensor, std::string> Trainer::compute_photometric_loss(const RenderOutput& render_output,
                                                    const torch::Tensor& gt_image,
                                                    const torch::Tensor& weights,
                                                    const float outOfMaskAlphaPenalty,
                                                    const SplatData& splatData,
                                                    const param::OptimizationParameters& opt_params) {

        if (!weights.defined() || weights.numel() == 0) {
            // fallback to the previous mode
            return compute_photometric_loss(render_output, gt_image, splatData, opt_params);
        }

        try {
            // Ensure images have same dimensions
            torch::Tensor rendered = render_output.image;
            torch::Tensor gt = gt_image;

            // Ensure both tensors are 4D (batch, height, width, channels)
            rendered = rendered.dim() == 3 ? rendered.unsqueeze(0) : rendered;
            gt = gt.dim() == 3 ? gt.unsqueeze(0) : gt;

            const int Height = rendered.size(2);
            const int Width = rendered.size(3);
            TORCH_CHECK(rendered.sizes() == gt.sizes(), "ERROR: size mismatch - rendered ", rendered.sizes(), " vs. ground truth ", gt.sizes());
            TORCH_CHECK((Height == weights.size(1) && Width == weights.size(2)),
                        "ERROR: size mismatch - rendered ", rendered.sizes(), " vs. mask ", weights.sizes());

            torch::Tensor W = weights;

            // Pixel-wise L1 map and weighted L1 loss (+ darkness_boost)
            auto l1_map = torch::abs(rendered - gt); // [B,C,H,W] or [B,H,W,C]

            if (opt_params.darkness_boost > 0.0f) {
                torch::NoGradGuard no_grad;
                auto darkness_weight = make_darkness_weight(gt, opt_params.darkness_boost);
                l1_map = l1_map * darkness_weight;
            }

            const bool nhwc = (rendered.size(-1) == 3);
            l1_map = nhwc ? l1_map.mean(-1) : l1_map.mean(1);
            torch::Tensor W2 = W;
            if (W2.dim() == 3 && W2.size(0) == 1) {
                W2 = W2.squeeze(0);
            }
            auto wSum = W2.sum().clamp_min(1e-6f);
            auto l1_loss = (l1_map * W2).sum() / wSum;

            // 2) pixel-wise SSIM map and weighted SSIM loss
            
            // Compute the SSIM map. Using "valid" padding results in a smaller map.
            auto ssim_map = fused_ssim_map(rendered, gt, "valid", /*train=*/true);

            // Manually crop the weight map `W` to match the `ssim_map` dimensions.
            namespace I = torch::indexing;

            // Get original and target dimensions.
            const int orig_h = W.size(-2);
            const int orig_w = W.size(-1);
            const int target_h = ssim_map.size(-2);
            const int target_w = ssim_map.size(-1);

            // Calculate offsets for a centered crop.
            const int crop_h_start = (orig_h - target_h) / 2;
            const int crop_w_start = (orig_w - target_w) / 2;

            // Apply the crop using tensor slicing.
            torch::Tensor W_cropped = W.index({I::Slice(),
                                                I::Slice(crop_h_start, crop_h_start + target_h),
                                                I::Slice(crop_w_start, crop_w_start + target_w)});
            // Compute the weighted SSIM loss.
            auto ssim_loss_map = 1.0f - ssim_map;

            // Normalize by the sum of the cropped weights for correct loss scaling.
            auto W_cropped_sum = W_cropped.sum().clamp_min(1e-6f);
            auto ssim_loss = (ssim_loss_map * W_cropped).sum() / W_cropped_sum;

            // 3) combined loss
            auto loss = (1.0f - opt_params.lambda_dssim) * l1_loss + opt_params.lambda_dssim * ssim_loss;
                    
            if (outOfMaskAlphaPenalty > 0.0f && opt_params.darkness_boost <= 0.0f) {
                if (!render_output.image.defined() || render_output.image.numel() == 0) {
                    printf("Image failed!\n");
                } else if (!render_output.alpha.defined() || render_output.alpha.numel() == 0) {
                    //printf("Alpha failed!\n");
                }
                else {
                    /* auto opacity = splatData.get_opacity();
                    loss = outsideMaskOpacityPenalty(loss,
                                                    render_output,
                                                    weights,
                                                    opacity,
                                                    outOfMaskAlphaPenalty);*/

                    loss = pixelBasedOpacityPenalty(loss,
                                                    render_output.alpha.squeeze(0), // Squeeze to [H, W] if needed
                                                    weights,
                                                    outOfMaskAlphaPenalty);
                }
            }
        

            return loss;
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error computing photometric loss: {}", e.what()));
        }
    }

    // Gently boost opacity for Gaussians that are inside the mask in at least
    // 'min_visible_views' visible views.
    // Policy: alpha_new = clamp(boost_value * alpha_old, alpha_old, alpha_min).
    //
    // Notes:
    // - Counts ONLY views where the Gaussian is visible (positive projected radii).
    // - Uses explicit index arrays (no boolean mask indexing) to avoid broadcasting traps.
    // - Flattens raw opacity to 1D to ensure shape consistency.
    // - Never decreases opacity; caps at 'alpha_min' (< 1.0 to keep finite logits).
    void Trainer::maybe_alpha_boost(const float boost_value,
                                    const float alpha_min,
                                    const int min_visible_views) {
        torch::NoGradGuard no_grad;

        const float min_support_ratio = 0.0f; // >= 0.0 => at least one inside vote across visible views
        const float alpha_threshold = 0.0f;   // ignore splats behind this level

        // Model & sizes
        SplatData& model = strategy_->get_model();
        const int64_t N = model.get_means().size(0);
        if (N <= 0)
            return;

        // Accumulators: visible-only votes (CUDA int32)
        auto pos = torch::zeros({N}, torch::kInt32).to(torch::kCUDA);
        auto tot = torch::zeros({N}, torch::kInt32).to(torch::kCUDA);

        // Data loader (sequential, batch=1) – iterate the dataloader itself
        const size_t dataset_size = train_dataset_size_;
        if (dataset_size == 0)
            return;

        const int workers = std::max(1, params_.optimization.num_workers);
        auto train_dataloader = create_infinite_dataloader_from_dataset(train_dataset_, workers);
        auto loader = train_dataloader->begin();

        // Model tensors
        auto means3D = model.get_means();      // [N,3] CUDA
        auto scales = model.get_scaling();     // [N,3] CUDA
        auto rotations = model.get_rotation(); // [N,4] CUDA
        auto opacities = model.get_opacity();  // [N] or [N,1] (unused by projection if not needed)
        if (opacities.defined() && opacities.dim() == 2 && opacities.size(-1) == 1)
            opacities = opacities.squeeze(-1);

        // Projection constants
        const float eps2d = 0.3f, near_plane = 0.01f, far_plane = 10000.0f, radius_clip = 0.0f, scaling_mod = 1.0f;

        // Tally visible-only inside votes
        int idx_img = 1;
        size_t skipped_missing = 0, skipped_shape = 0, skipped_size = 0, skipped_proj = 0;

        for (size_t i = 0; i < dataset_size; ++i, ++loader) {
            // Consume batch like in the training loop
            auto& batch = *loader;
            const auto sample = batch[0].data; // {camera, image, attentionMask}
            Camera* cam = sample.camera;
            auto mask_f = sample.attentionMask;

            if (!cam || !mask_f.defined() || mask_f.numel() == 0) {
                if (!cam) {
                    std::cout << "\n[Alpha boost] Warning: null camera; skipping view.\n";
                } else if (skipped_missing < 3) {
                    std::cout << "\n[Alpha boost] Warning: undefined/empty attentionMask; skipping view.\n";
                }
                ++skipped_missing;
                continue;
            }

            // CPU bool mask [H,W]
            auto m3 = (mask_f > 0.5f);
            auto m2 = (m3.dim() == 3 && m3.size(0) == 1) ? m3.squeeze(0) : m3;
            if (m2.dim() != 2) {
                if (skipped_shape < 3)
                    std::cout << "\n[Alpha boost] Warning: mask wrong shape; skipping view.\n";
                ++skipped_shape;
                continue;
            }
            auto mask = m2.to(torch::kCPU).contiguous();

            const int W = (int)cam->image_width();
            const int H = (int)cam->image_height();
            if (mask.size(1) != W || mask.size(0) != H) {
                if (skipped_size < 3)
                    std::cout << "\n[Alpha boost] Warning: mask size != camera size; skipping view.\n";
                ++skipped_size;
                continue;
            }

            // Camera tensors (CUDA) with batch dim if needed
            auto viewmat = cam->world_view_transform().to(torch::kCUDA);
            auto K = cam->K().to(torch::kCUDA);
            if (viewmat.dim() == 2)
                viewmat = viewmat.unsqueeze(0);
            if (K.dim() == 2)
                K = K.unsqueeze(0);

            // Distortion (optional)
            std::optional<torch::Tensor> radial, tangential;
            if (cam->radial_distortion().defined() && cam->radial_distortion().numel() > 0)
                radial = cam->radial_distortion();
            if (cam->tangential_distortion().defined() && cam->tangential_distortion().numel() > 0)
                tangential = cam->tangential_distortion();

            // Projection-only (fast)
            auto [radii_raw, means2d_raw] = ProjectFast(
                means3D, rotations, scales, opacities,
                viewmat, K,
                W, H,
                eps2d, near_plane, far_plane,
                radius_clip, scaling_mod,
                cam->camera_model_type(),
                radial, tangential, /*thin_prism*/ std::nullopt);

            if (!radii_raw.defined() || !means2d_raw.defined()) {
                if (skipped_proj < 3)
                    std::cout << "\n[Alpha boost] Warning: projection returned undefined tensors; skipping view.\n";
                ++skipped_proj;
                continue;
            }

            auto radii = (radii_raw.dim() == 3 && radii_raw.size(0) == 1) ? radii_raw.squeeze(0) : radii_raw;
            auto means2d = (means2d_raw.dim() == 3 && means2d_raw.size(0) == 1) ? means2d_raw.squeeze(0) : means2d_raw;

            // Visible-only splats (positive projected radii) – handle 1D/2D radii
            torch::Tensor visible;
            if (radii.dim() == 2 && radii.size(1) >= 1)
                visible = (radii > 0.0f).all(-1); // [N]
            else if (radii.dim() == 1)
                visible = (radii > 0.0f); // [N]
            else {
                if (skipped_proj < 3)
                    std::cout << "\n[Alpha boost] Warning: invalid radii shape; skipping view.\n";
                ++skipped_proj;
                continue;
            }
            if (!visible.any().item<bool>())
                continue;

            auto vidx = visible.nonzero().squeeze(-1); // [M], CUDA

            // Pixel coords (on CPU) for mask sampling
            auto xy_cuda = means2d.index({vidx}); // [M,2] CUDA
            auto xy = xy_cuda.detach().to(torch::kCPU);
            auto x = torch::round(xy.select(1, 0)).to(torch::kLong).clamp(0, W - 1);
            auto y = torch::round(xy.select(1, 1)).to(torch::kLong).clamp(0, H - 1);
            auto lin = y * W + x; // [M] CPU long

            // Inside votes (CPU read, accumulate on CUDA)
            auto inside_cpu = mask.flatten().index({lin});                   // [M] bool CPU
            auto inside_i32 = inside_cpu.to(torch::kInt32).to(torch::kCUDA); // [M] int32 CUDA

            pos.index_add_(0, vidx, inside_i32);
            tot.index_add_(0, vidx, torch::ones_like(inside_i32, torch::kInt32));
        }

        // Read/prepare opacity logits as a flat 1D tensor
        auto raw = model.opacity_raw(); // logits; can be [N] or [N,1] depending on impl
        if (raw.dim() == 2 && raw.size(-1) == 1)
            raw = raw.squeeze(-1);
        if (raw.dim() != 1)
            raw = raw.view({-1});             // [N]
        auto cur_alpha = torch::sigmoid(raw); // [N], same device as raw

        // Eligibility based on visible-only support
        auto tot_f = tot.to(torch::kFloat32).clamp_min(1.0f);
        auto ratio = pos.to(torch::kFloat32) / tot_f; // [N] float (CUDA)
        auto eligible_mask =
            (tot >= min_visible_views) &
            (ratio >= min_support_ratio) &
            (cur_alpha > alpha_threshold);

        const float eps = 1e-6f;
        const float cap = std::min(alpha_min, 1.0f - eps);

        // Further restrict to those below the cap (avoid needless writes)
        auto below_cap = (cur_alpha < cap);
        auto final_mask = eligible_mask & below_cap;

        auto idx = final_mask.nonzero().squeeze(-1); // [K] (CUDA)
        if (idx.numel() == 0) {
            std::cout << "[Trainer] Alpha boost: no eligible splats.\n";
            return;
        }

        // Compute target alpha = clamp(boost_value * a, a, cap) on the selected indices
        auto a = cur_alpha.index({idx}); // [K]
        auto doubled = a * boost_value;
        auto target = torch::min(doubled, torch::tensor(cap, a.options()));
        target = torch::max(target, a).clamp(eps, cap); // never decrease; keep finite logits

        // Write back logits (use max to be extra safe numerically)
        auto new_logit = torch::logit(target); // [K]
        auto old_logit = raw.index({idx});
        auto write = torch::max(old_logit, new_logit); // monotonicity guard
        raw.index_put_({idx}, write);

        std::cout << "[Trainer] Alpha boost (x" << boost_value
                  << ", cap=" << alpha_min
                  << "): updated " << idx.numel() << " splats.\n";
    }


    std::expected<torch::Tensor, std::string> Trainer::compute_scale_reg_loss(
        const SplatData& splatData,
        const param::OptimizationParameters& opt_params) {
        try {
            if (opt_params.scale_reg > 0.0f) {
                auto scale_l1 = splatData.get_scaling().mean();
                return opt_params.scale_reg * scale_l1;
            }
            return torch::zeros({1}, torch::kFloat32).requires_grad_();
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error computing scale regularization loss: {}", e.what()));
        }
    }

    std::expected<torch::Tensor, std::string> Trainer::compute_opacity_reg_loss(
        const SplatData& splatData,
        const param::OptimizationParameters& opt_params) {
        try {
            if (opt_params.opacity_reg > 0.0f) {
                auto opacity_l1 = splatData.get_opacity().mean();
                return opt_params.opacity_reg * opacity_l1;
            }
            return torch::zeros({1}, torch::kFloat32).requires_grad_();
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error computing opacity regularization loss: {}", e.what()));
        }
    }

    std::expected<torch::Tensor, std::string> Trainer::compute_bilateral_grid_tv_loss(
        const std::unique_ptr<BilateralGrid>& bilateral_grid,
        const param::OptimizationParameters& opt_params) {
        try {
            if (opt_params.use_bilateral_grid) {
                return opt_params.tv_loss_weight * bilateral_grid->tv_loss();
            }
            return torch::zeros({1}, torch::kFloat32).requires_grad_();
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error computing bilateral grid TV loss: {}", e.what()));
        }
    }

    std::expected<torch::Tensor, std::string> Trainer::compute_sparsity_loss(
        int iter,
        const SplatData& splatData) {
        try {
            if (sparsity_optimizer_ && sparsity_optimizer_->should_apply_loss(iter)) {
                // Initialize on first use (lazy initialization)
                if (!sparsity_optimizer_->is_initialized()) {
                    auto init_result = sparsity_optimizer_->initialize(splatData.opacity_raw());
                    if (!init_result) {
                        return std::unexpected(init_result.error());
                    }
                    LOG_INFO("Sparsity optimizer initialized at iteration {}", iter);
                }

                auto loss_result = sparsity_optimizer_->compute_loss(splatData.opacity_raw());
                if (!loss_result) {
                    return std::unexpected(loss_result.error());
                }
                return *loss_result;
            }
            return torch::zeros({1}, torch::kFloat32).to(torch::kCUDA).requires_grad_();
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error computing sparsity loss: {}", e.what()));
        }
    }

    std::expected<void, std::string> Trainer::handle_sparsity_update(
        int iter,
        SplatData& splatData) {
        try {
            if (sparsity_optimizer_ && sparsity_optimizer_->should_update(iter)) {
                LOG_TRACE("Updating sparsity state at iteration {}", iter);
                auto result = sparsity_optimizer_->update_state(splatData.opacity_raw());
                if (!result) {
                    return std::unexpected(result.error());
                }
            }
            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error updating sparsity state: {}", e.what()));
        }
    }

    std::expected<void, std::string> Trainer::apply_sparsity_pruning(
        int iter,
        SplatData& splatData) {
        try {
            if (sparsity_optimizer_ && sparsity_optimizer_->should_prune(iter)) {
                LOG_INFO("Applying sparsity-based pruning at iteration {}", iter);

                auto mask_result = sparsity_optimizer_->get_prune_mask(splatData.opacity_raw());
                if (!mask_result) {
                    return std::unexpected(mask_result.error());
                }

                auto prune_mask = *mask_result;
                int n_prune = prune_mask.sum().item<int>();
                int n_before = splatData.size();

                // Use strategy's remove functionality
                strategy_->remove_gaussians(prune_mask);

                int n_after = splatData.size();
                std::println("Sparsity pruning complete: {} -> {} Gaussians (removed {})",
                             n_before, n_after, n_prune);

                // Clear sparsity optimizer after pruning
                sparsity_optimizer_.reset();
                LOG_DEBUG("Sparsity optimizer cleared after pruning");
            }
            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error applying sparsity pruning: {}", e.what()));
        }
    }

    Trainer::Trainer(std::shared_ptr<CameraDataset> dataset,
                     std::unique_ptr<IStrategy> strategy)
        : base_dataset_(std::move(dataset)),
          strategy_(std::move(strategy)) {
        if (!torch::cuda::is_available()) {
            throw std::runtime_error("CUDA is not available – aborting.");
        }
        LOG_DEBUG("Trainer constructed with {} cameras", base_dataset_->get_cameras().size());
    }

    void Trainer::load_cameras_info() {
        m_cam_id_to_cam.clear();
        // Setup camera cache
        for (const auto& cam : base_dataset_->get_cameras()) {
            m_cam_id_to_cam[cam->uid()] = cam;
        }
    }

    std::expected<void, std::string> Trainer::initialize(const param::TrainingParameters& params) {
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

            // Handle dataset split based on evaluation flag
            if (params.optimization.enable_eval) {
                // Create train/val split
                train_dataset_ = std::make_shared<CameraDataset>(
                    base_dataset_->get_cameras(), params.dataset, CameraDataset::Split::TRAIN);
                val_dataset_ = std::make_shared<CameraDataset>(
                    base_dataset_->get_cameras(), params.dataset, CameraDataset::Split::VAL);

                LOG_INFO("Created train/val split: {} train, {} val images",
                         train_dataset_->size().value(),
                         val_dataset_->size().value());
            } else {
                // Use all images for training
                train_dataset_ = base_dataset_;
                val_dataset_ = nullptr;

                LOG_INFO("Using all {} images for training (no evaluation)",
                         train_dataset_->size().value());
            }

            // chage resize factor (change may comes from gui)
            if (train_dataset_) {
                train_dataset_->set_resize_factor(params.dataset.resize_factor);
            }
            if (val_dataset_) {
                val_dataset_->set_resize_factor(params.dataset.resize_factor);
            }

            train_dataset_size_ = train_dataset_->size().value();

            m_cam_id_to_cam.clear();
            // Setup camera cache
            for (const auto& cam : base_dataset_->get_cameras()) {
                m_cam_id_to_cam[cam->uid()] = cam;
            }
            LOG_DEBUG("Camera cache initialized with {} cameras", m_cam_id_to_cam.size());

            // Re-initialize strategy with new parameters
            strategy_->initialize(params.optimization);
            LOG_DEBUG("Strategy initialized");

            // Initialize bilateral grid if enabled
            if (auto result = initialize_bilateral_grid(); !result) {
                return std::unexpected(result.error());
            }

            // Initialize sparsity optimizer if enabled
            if (params.optimization.enable_sparsity) {
                // Calculate when sparsity should start
                int base_iterations = params.optimization.iterations;
                int sparsity_start = base_iterations; // Start after base training
                int total_iterations = base_iterations + params.optimization.sparsify_steps;

                // Extend the total training iterations
                params_.optimization.iterations = total_iterations;

                ADMMSparsityOptimizer::Config sparsity_config{
                    .sparsify_steps = params.optimization.sparsify_steps,
                    .init_rho = params.optimization.init_rho,
                    .prune_ratio = params.optimization.prune_ratio,
                    .update_every = 50,
                    .start_iteration = sparsity_start // Start after base training completes
                };

                sparsity_optimizer_ = SparsityOptimizerFactory::create("admm", sparsity_config);

                if (sparsity_optimizer_) {
                    // Don't initialize yet - will initialize when we reach start_iteration
                    LOG_INFO("=== Sparsity Optimization Configuration ===");
                    LOG_INFO("Base training iterations: {}", base_iterations);
                    LOG_INFO("Sparsification starts at: iteration {}", sparsity_start);
                    LOG_INFO("Sparsification duration: {} iterations", params.optimization.sparsify_steps);
                    LOG_INFO("Total training iterations: {}", total_iterations);
                    LOG_INFO("Pruning ratio: {}%", params.optimization.prune_ratio * 100);
                    LOG_INFO("ADMM penalty (rho): {}", params.optimization.init_rho);
                }
            }

            background_ = torch::tensor({0.f, 0.f, 0.f},
                                        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

            if (params.optimization.pose_optimization != "none") {
                if (params.optimization.enable_eval) {
                    return std::unexpected("Evaluating with pose optimization is not supported yet. "
                                           "Please disable pose optimization or evaluation.");
                }
                if (params.optimization.gut) {
                    return std::unexpected("The 3DGUT rasterizer doesn't have camera gradients yet. "
                                           "Please disable pose optimization or disable gut.");
                }
                if (params.optimization.pose_optimization == "direct") {
                    poseopt_module_ = std::make_unique<DirectPoseOptimizationModule>(train_dataset_->get_cameras().size());
                    LOG_DEBUG("Direct pose optimization module created");
                } else if (params.optimization.pose_optimization == "mlp") {
                    poseopt_module_ = std::make_unique<MLPPoseOptimizationModule>(train_dataset_->get_cameras().size());
                    LOG_DEBUG("MLP pose optimization module created");
                } else {
                    return std::unexpected("Invalid pose optimization type: " + params.optimization.pose_optimization);
                }
                poseopt_optimizer_ = std::make_unique<torch::optim::Adam>(
                    std::vector<torch::Tensor>{poseopt_module_->parameters()},
                    torch::optim::AdamOptions(1e-5));
            } else {
                poseopt_module_ = std::make_unique<PoseOptimizationModule>();
            }

            // Create progress bar based on headless flag
            if (params.optimization.headless) {
                progress_ = std::make_unique<TrainingProgress>(
                    params_.optimization.iterations, // This now includes sparsity steps if enabled
                    /*update_frequency=*/100);
                LOG_DEBUG("Progress bar initialized for {} total iterations", params_.optimization.iterations);
            }

            // Initialize the evaluator - it handles all metrics internally
            evaluator_ = std::make_unique<MetricsEvaluator>(params_);
            LOG_DEBUG("Metrics evaluator initialized");

            // Print configuration
            LOG_INFO("Render mode: {}", params.optimization.render_mode);
            LOG_INFO("Visualization: {}", params.optimization.headless ? "disabled" : "enabled");
            LOG_INFO("Strategy: {}", params.optimization.strategy);

            initialized_ = true;
            LOG_INFO("Trainer initialization complete");
            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to initialize trainer: {}", e.what()));
        }
    }

    Trainer::~Trainer() {
        // Ensure training is stopped
        stop_requested_ = true;

        // Wait for callback to finish if busy
        if (callback_busy_.load()) {
            callback_stream_.synchronize();
        }
        LOG_DEBUG("Trainer destroyed");
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

        // Handle save request
        if (save_requested_.exchange(false)) {
            LOG_INFO("Saving checkpoint at iteration {}...", iter);
            auto checkpoint_path = params_.dataset.output_path / "checkpoints";
            save_ply(checkpoint_path, iter, /*join=*/true);

            LOG_INFO("Checkpoint saved to {}", checkpoint_path.string());

            // Emit checkpoint saved event
            events::state::CheckpointSaved{
                .iteration = iter,
                .path = checkpoint_path}
                .emit();
        }

        // Handle stop request - this permanently stops training
        if (stop_requested_.load()) {
            LOG_INFO("Stopping training permanently at iteration {}...", iter);
            LOG_DEBUG("Saving final model...");
            save_ply(params_.dataset.output_path, iter, /*join=*/true);
            is_running_ = false;
        }
    }

    inline float inv_weight_piecewise(int step, int max_steps) {
        // Phases by fraction of training
        const float phase = std::max(0.f, std::min(1.f, step / float(std::max(1, max_steps))));

        const float limit_hi = 1.0f / 4.0f;  // start limit
        const float limit_mid = 2.0f / 4.0f; // middle limit
        const float limit_lo = 3.0f / 4.0f;  // final limit

        const float weight_hi = 1.0f;  // start weight
        const float weight_mid = 0.5f; // middle weight
        const float weight_lo = 0.0f;  // final weight

        if (phase < limit_hi) {
            return weight_hi; // hold until bypasses the start limit
        } else if (phase < limit_mid) {
            const float t = (phase - limit_hi) / (limit_mid - limit_hi);
            return weight_hi + (weight_mid - weight_hi) * t; // decay to mid value
        } else {
            const float t = (phase - limit_mid) / (limit_lo - limit_mid);
            return weight_mid + (weight_lo - weight_mid) * t; // decay to final value
        }
    }

    torch::Tensor sine_background_for_step(
        int step, int periodR = 37, int periodG = 41, int periodB = 43, bool grayscale_only = false, float jitter_amp = 0.03f) {
        const float eps = 1e-4f;
        auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        const float two_pi = M_PI * 2.0f;

        // Phase 0..2PI
        const float tR = (periodR > 0) ? float(step % periodR) / float(periodR) : 0.0f;
        const float phaseR = two_pi * tR;

        const float tG = (periodG > 0) ? float(step % periodG) / float(periodG) : 0.0f;
        const float phaseG = two_pi * tG;

        const float tB = (periodB > 0) ? float(step % periodB) / float(periodB) : 0.0f;
        const float phaseB = two_pi * tB;

        torch::Tensor bg;
        if (grayscale_only) {
            // Grayscale: g in [0,1]
            float g = 0.5f * (1.0f + std::sin(phaseG));
            bg = torch::tensor({g, g, g}, opts);
        } else {
            // Phase-shifted RGB: covers the color wheel over the cycle
            float r = 0.5f * (1.0f + std::sin(phaseR + 0.0f * two_pi / 3.0f));
            float g = 0.5f * (1.0f + std::sin(phaseG + 1.0f * two_pi / 3.0f));
            float b = 0.5f * (1.0f + std::sin(phaseB + 2.0f * two_pi / 3.0f));
            bg = torch::tensor({r, g, b}, opts);
        }

        // Small jitter to prevent exact periodic lock-in
        if (jitter_amp > 0.0f) {
            auto jitter = (torch::rand({3}, opts) - 0.5f) * (2.0f * jitter_amp);
            bg = (bg + jitter).clamp(eps, 1.0f - eps);
        } else {
            bg = bg.clamp(eps, 1.0f - eps);
        }
        return bg;
    }

    // Returns a 3-float CUDA tensor where each channel is either min or max.
    // Guarantees at least one channel at max and one at min.
    // Deterministic per 'step' using a tiny LCG-based hash.
    torch::Tensor binary_extreme_background_for_step(int step, float eps = 1e-4f) {
        // Define extremes with tiny epsilon to avoid exact 0/1 logits issues
        const float vmin = eps;
        const float vmax = 1.0f - eps;

        // Simple deterministic PRNG from 'step'
        auto next = [](uint32_t& s) {
            s = s * 1664525u + 1013904223u; // LCG
            return s;
        };
        uint32_t s = static_cast<uint32_t>(step) ^ 0x9E3779B9u;

        // Pick distinct indices for max and min
        int i_max = static_cast<int>(next(s) % 3u);
        int i_min = (i_max + 1 + static_cast<int>(next(s) % 2u)) % 3; // different from i_max

        // Third channel: choose independently min/max
        int i_third = 3 - i_max - i_min; // remaining index
        bool third_is_max = (next(s) & 1u) != 0;

        float rgb[3] = {vmin, vmin, vmin};
        rgb[i_max] = vmax;
        rgb[i_min] = vmin;
        rgb[i_third] = third_is_max ? vmax : vmin;

        auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        return torch::tensor({rgb[0], rgb[1], rgb[2]}, opts);
    }


    // Helper to ensure buf matches base (defined, dtype, device, shape)
    static inline void ensure_like(torch::Tensor& buf, const torch::Tensor& base) {
        bool is_undefined = !buf.defined();
        bool dtype_mismatch = (buf.dtype() != base.dtype());

        bool need = (is_undefined || dtype_mismatch);
        if (!need) {
            bool device_mismatch = (buf.device() != base.device());
            bool shape_mismatch = (buf.sizes().vec() != base.sizes().vec());
            need = (device_mismatch || shape_mismatch);
        }

        if (need)
            buf = torch::empty_like(base);
    }

    torch::Tensor& Trainer::background_for_step(int iter) {
        torch::NoGradGuard no_grad;
        const auto& opt = params_.optimization;

        // Fast path: modulation disabled: return base background_
        if (!opt.bg_modulation) {
            return background_;
        }

        const float w_mix = inv_weight_piecewise(iter, opt.iterations);
        if (w_mix <= 0.0f) {
            return background_;
        }

        // Generate per-iteration sine background
        //auto sine_bg = sine_background_for_step(iter);
        auto sine_bg = binary_extreme_background_for_step(iter);


        // Ensure reusable buffer exists
        ensure_like(bg_mix_buffer_, background_);

        torch::lerp_out(bg_mix_buffer_, /*start=*/background_, /*end=*/sine_bg, /*weight=*/w_mix);
        return bg_mix_buffer_;
    }

    std::expected<Trainer::StepResult, std::string> Trainer::train_step(
        int iter,
        Camera* cam,
        torch::Tensor gt_image,
        torch::Tensor weights,
        RenderMode render_mode,
        bool out_of_mask_penalty,
        std::stop_token stop_token) {
        try {
            if (params_.optimization.gut) {
                if (cam->camera_model_type() == gsplat::CameraModelType::ORTHO) {
                    return std::unexpected("Training on cameras with ortho model is not supported yet.");
                }
            } else {
                if (cam->radial_distortion().numel() != 0 ||
                    cam->tangential_distortion().numel() != 0) {
                    return std::unexpected("Distorted images detected.  You can use --gut option to train on cameras with distortion.");
                }
                if (cam->camera_model_type() != gsplat::CameraModelType::PINHOLE) {
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

            // Add phase transition logging for sparsity
            if (params_.optimization.enable_sparsity) {
                // Calculate base iterations (original iterations before extension)
                int base_iterations = params_.optimization.iterations - params_.optimization.sparsify_steps;

                // Log phase transition
                if (iter == base_iterations + 1) {
                    LOG_INFO("=== Entering Sparsification Phase ===");
                    LOG_INFO("Base training complete at iteration {}", base_iterations);
                    LOG_INFO("Starting ADMM sparsification for {} iterations",
                             params_.optimization.sparsify_steps);
                    LOG_INFO("Current model size: {} Gaussians", strategy_->get_model().size());
                    LOG_INFO("Target pruning: {}% of Gaussians", params_.optimization.prune_ratio * 100);
                }

                // Log when approaching pruning
                if (iter == params_.optimization.iterations - 100) {
                    LOG_INFO("Approaching final pruning in 100 iterations (at iteration {})",
                             params_.optimization.iterations);
                }

                // Log when pruning will occur
                if (iter == params_.optimization.iterations - 1) {
                    LOG_INFO("Final pruning will occur next iteration");
                }
            }

            auto adjusted_cam_pos = poseopt_module_->forward(cam->world_view_transform(), torch::tensor({cam->uid()}));
            auto adjusted_cam = Camera(*cam, adjusted_cam_pos);

            torch::Tensor& bg = background_for_step(iter);

            RenderOutput r_output;
            // Use the render mode from parameters
            if (!params_.optimization.gut) {
                r_output = fast_rasterize(adjusted_cam, strategy_->get_model(), bg);
            } else {
                r_output = rasterize(adjusted_cam, strategy_->get_model(), bg, 1.0f, false, false, render_mode,
                                     nullptr);
            }

            // Apply bilateral grid if enabled
            if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
                r_output.image = bilateral_grid_->apply(r_output.image, cam->uid());
            }

            // Compute loss using the factored-out function
            std::expected<torch::Tensor, std::string> loss_result;
            
            const int total_iters = params_.optimization.iterations;
            const int warmup_end_iter = total_iters * 0.1f;

            const int alpha_boost_iter = total_iters * 0.3f;
            if (weights.defined() && iter == alpha_boost_iter) {
                // maybe_alpha_boost(2.5f /*boost_value*/, 0.90f /*alpha_min*/, 3 /*min_visible_views */);
                // contextual_alpha_boost_once();
            }

            if (!weights.defined() || iter < warmup_end_iter) {
                loss_result = compute_photometric_loss(r_output,
                                                        gt_image,
                                                        strategy_->get_model(),
                                                        params_.optimization);
            } 
            else {
                const int full_penalty_end_iter = total_iters * 0.5f;
                const int decay_end_iter = total_iters * 0.8f;
                float current_penalty_w = 0.0f;
                if (out_of_mask_penalty) {
                    if (iter <= full_penalty_end_iter) {
                        current_penalty_w = 1;
                    } else if (iter < decay_end_iter) {
                        const int decay_start_iter = full_penalty_end_iter;
                        const int decay_duration = decay_end_iter - decay_start_iter;
                        const float decay_progress = static_cast<float>(iter - decay_start_iter) / decay_duration;
                        current_penalty_w = 1.0f - decay_progress;
                    }

                    // Floor at 0.2 so it never vanishes
                    current_penalty_w = std::max(0.2f, current_penalty_w);
                }

                loss_result = compute_photometric_loss( r_output,
                                                        gt_image,
                                                        weights,
                                                        current_penalty_w,
                                                        strategy_->get_model(),
                                                        params_.optimization);
            }
            
            if (!loss_result) {
                return std::unexpected(loss_result.error());
            }

            torch::Tensor loss = *loss_result;
            loss.backward();
            float loss_value = loss.item<float>();

            // Scale regularization loss
            auto scale_loss_result = compute_scale_reg_loss(strategy_->get_model(), params_.optimization);
            if (!scale_loss_result) {
                return std::unexpected(scale_loss_result.error());
            }
            loss = *scale_loss_result;
            loss.backward();
            loss_value += loss.item<float>();

            // Opacity regularization loss
            auto opacity_loss_result = compute_opacity_reg_loss(strategy_->get_model(), params_.optimization);
            if (!opacity_loss_result) {
                return std::unexpected(opacity_loss_result.error());
            }
            loss = *opacity_loss_result;
            loss.backward();
            loss_value += loss.item<float>();

            // Bilateral grid TV loss
            auto tv_loss_result = compute_bilateral_grid_tv_loss(bilateral_grid_, params_.optimization);
            if (!tv_loss_result) {
                return std::unexpected(tv_loss_result.error());
            }
            loss = *tv_loss_result;
            loss.backward();
            loss_value += loss.item<float>();

            // Add sparsity loss
            auto sparsity_loss_result = compute_sparsity_loss(iter, strategy_->get_model());
            if (!sparsity_loss_result) {
                return std::unexpected(sparsity_loss_result.error());
            }
            loss = *sparsity_loss_result;
            loss.backward();
            loss_value += loss.item<float>();

            // Store the loss value immediately
            current_loss_ = loss_value;

            // Update progress synchronously if needed
            if (progress_) {
                progress_->update(iter, loss_value,
                                  static_cast<int>(strategy_->get_model().size()),
                                  strategy_->is_refining(iter));
            }

            // Emit training progress event (throttled to reduce GUI updates)
            if (iter % 10 == 0 || iter == 1) {
                // Only update every 10 iterations
                events::state::TrainingProgress{
                    .iteration = iter,
                    .loss = loss_value,
                    .num_gaussians = static_cast<int>(strategy_->get_model().size()),
                    .is_refining = strategy_->is_refining(iter)}
                    .emit();
            }
            {
                torch::NoGradGuard no_grad;

                DeferredEvents deferred;
                {
                    std::unique_lock<std::shared_mutex> lock(render_mutex_);

                    // Execute strategy post-backward and step
                    // Only call post_backward during base training (not during sparsification)
                    if (params_.optimization.enable_sparsity) {
                        int base_iterations = params_.optimization.iterations - params_.optimization.sparsify_steps;
                        if (iter <= base_iterations) {
                            strategy_->post_backward(iter, r_output);
                        }
                        // During sparsification phase, skip post_backward entirely
                    } else {
                        // No sparsity, always call post_backward
                        strategy_->post_backward(iter, r_output);
                    }

                    strategy_->step(iter);

                    if (params_.optimization.use_bilateral_grid) {
                        bilateral_grid_optimizer_->step();
                        bilateral_grid_optimizer_->zero_grad(true);
                        bilateral_grid_scheduler_->step();
                    }
                    if (params_.optimization.pose_optimization != "none") {
                        poseopt_optimizer_->step();
                        poseopt_optimizer_->zero_grad(true);
                    }

                    // Queue event for emission after lock release
                    deferred.add(events::state::ModelUpdated{
                        .iteration = iter,
                        .num_gaussians = static_cast<size_t>(strategy_->get_model().size())});
                } // Lock released here

                // Events automatically emitted here when deferred destructs

                // Handle sparsity updates
                if (auto result = handle_sparsity_update(iter, strategy_->get_model()); !result) {
                    LOG_ERROR("Sparsity update failed: {}", result.error());
                }

                // Apply sparsity pruning if needed
                if (auto result = apply_sparsity_pruning(iter, strategy_->get_model()); !result) {
                    LOG_ERROR("Sparsity pruning failed: {}", result.error());
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

                // Save model at specified steps
                if (!params_.optimization.skip_intermediate_saving) {
                    for (size_t save_step : params_.optimization.save_steps) {
                        if (iter == static_cast<int>(save_step) && iter != params_.optimization.iterations) {
                            const bool join_threads = (iter == params_.optimization.save_steps.back());
                            auto save_path = params_.dataset.output_path;
                            save_ply(save_path, iter, /*join=*/join_threads);
                            // Emit checkpoint saved event
                            events::state::CheckpointSaved{
                                .iteration = iter,
                                .path = save_path}
                                .emit();
                        }
                    }
                }

                if (!params_.dataset.timelapse_images.empty() && iter % params_.dataset.timelapse_every == 0) {
                    for (const auto& img_name : params_.dataset.timelapse_images) {
                        auto train_cam = train_dataset_->get_camera_by_filename(img_name);
                        auto val_cam = val_dataset_ ? val_dataset_->get_camera_by_filename(img_name) : std::nullopt;
                        if (train_cam.has_value() || val_cam.has_value()) {
                            Camera* cam_to_use = train_cam.has_value() ? train_cam.value() : val_cam.value();

                            // Image size isn't correct until the image has been loaded once
                            // If we use the camera before it's loaded, it will render images at the non-scaled size
                            if (cam_to_use->camera_height() == cam_to_use->image_height() && params_.dataset.resize_factor != 1) {
                                cam_to_use->load_image_size(params_.dataset.resize_factor);
                            }

                            RenderOutput rendered_timelapse_output = fast_rasterize(
                                *cam_to_use, strategy_->get_model(), background_);

                            // Get folder name to save in by stripping file extension
                            std::string folder_name = img_name;
                            auto last_dot = folder_name.find_last_of('.');
                            if (last_dot != std::string::npos) {
                                folder_name = folder_name.substr(0, last_dot);
                            }

                            auto output_path = params_.dataset.output_path / "timelapse" / folder_name;
                            std::filesystem::create_directories(output_path);

                            image_io::save_image_async(output_path / std::format("{:06d}.jpg", iter),
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

        // Event-based ready signaling
        if (!params_.optimization.headless) {
            // Subscribe to start signal (no need to store handle)
            events::internal::TrainingReadyToStart::when([this](const auto&) {
                ready_to_start_ = true;
            });

            // Signal we're ready
            events::internal::TrainerReady{}.emit();

            // Wait for start signal
            LOG_DEBUG("Waiting for start signal from GUI...");
            while (!ready_to_start_.load() && !stop_token.stop_requested()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }

        is_running_ = true; // Now we can start
        LOG_INFO("Starting training loop with {} workers", params_.optimization.num_workers);

        try {
            int iter = 1;
            const int num_workers = params_.optimization.num_workers;
            const RenderMode render_mode = stringToRenderMode(params_.optimization.render_mode);

            if (progress_) {
                progress_->update(iter, current_loss_.load(),
                                  static_cast<int>(strategy_->get_model().size()),
                                  strategy_->is_refining(iter));
            }

            // Use infinite dataloader to avoid epoch restarts
            auto train_dataloader = create_infinite_dataloader_from_dataset(train_dataset_, num_workers);
            auto loader = train_dataloader->begin();

            LOG_DEBUG("Starting training iterations");
            // Single loop without epochs
            while (iter <= params_.optimization.iterations) {
                if (stop_token.stop_requested() || stop_requested_.load()) {
                    break;
                }

                // Wait for previous callback if still running
                if (callback_busy_.load()) {
                    callback_stream_.synchronize();
                }

                auto& batch = *loader;
                auto camera_with_image = batch[0].data;
                Camera* cam = camera_with_image.camera;
                torch::Tensor gt_image = std::move(camera_with_image.image).to(torch::kCUDA, /*non_blocking=*/true);


                std::expected<Trainer::StepResult, std::string> step_result;
                if (!params_.optimization.use_attention_mask || !camera_with_image.attentionMask.defined()) {
                    step_result = train_step(iter, cam, gt_image, torch::Tensor(), render_mode, false, stop_token);
                } else {
                    torch::Tensor attention_image = std::move(camera_with_image.attentionMask).to(torch::kCUDA, /*non_blocking=*/true);
                    bool out_of_mask_penalty = true;
                    step_result = train_step(iter, cam, gt_image, attention_image, render_mode, out_of_mask_penalty, stop_token);
                }

                
                if (!step_result) {
                    return std::unexpected(step_result.error());
                }

                if (*step_result == StepResult::Stop) {
                    break;
                }

                // Launch callback for async progress update (except first iteration)
                if (iter > 1 && callback_) {
                    callback_busy_ = true;
                    auto err = cudaLaunchHostFunc(
                        callback_stream_.stream(),
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
                ++loader;
            }

            // Ensure callback is finished before final save
            if (callback_busy_.load()) {
                callback_stream_.synchronize();
            }

            training_complete:
                if (params_.optimization.use_attention_mask)
                    prune_after_training(0.8, 0.90);
            
            // Final save if not already saved by stop request
            if (!stop_requested_.load() && !stop_token.stop_requested()) {
                auto final_path = params_.dataset.output_path;
                save_ply(final_path, params_.optimization.iterations, /*join=*/true);
                // Emit final checkpoint saved event
                events::state::CheckpointSaved{
                    static_cast<int>(params_.optimization.iterations),
                    final_path}
                    .emit();
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

            LOG_INFO("Training completed successfully");
            return {};
        } catch (const std::exception& e) {
            is_running_ = false;
            return std::unexpected(std::format("Training failed: {}", e.what()));
        }
    }

    std::shared_ptr<const Camera> Trainer::getCamById(int camId) const {
        const auto it = m_cam_id_to_cam.find(camId);
        if (it == m_cam_id_to_cam.end()) {
            LOG_ERROR("getCamById - could not find cam with cam id {}", camId);
            return nullptr;
        }
        return it->second;
    }

    std::vector<std::shared_ptr<const Camera>> Trainer::getCamList() const {
        std::vector<std::shared_ptr<const Camera>> cams;
        cams.reserve(m_cam_id_to_cam.size());
        for (auto& [key, value] : m_cam_id_to_cam) {
            cams.push_back(value);
        }

        return cams;
    }

    /* void Trainer::prune_after_training(float center_keep_threshold, int min_visibility_count) {
        torch::NoGradGuard no_grad;

        // 0) Model tensors
        SplatData& model = strategy_->get_model();
        const int64_t N = model.get_means().size(0);
        if (N <= 0) {
            std::cout << "[Prune center] No gaussians to prune.\n";
            return;
        }

        auto means3D = model.get_means();
        auto scales = model.get_scaling();
        auto rotations = model.get_rotation();
        auto opacities = model.get_opacity();
        if (opacities.defined() && opacities.dim() == 2 && opacities.size(-1) == 1)
            opacities = opacities.squeeze(-1);

        // 1) Vote buffers (CUDA int32)
        auto pos = torch::zeros({N}, torch::kInt32).to(torch::kCUDA); // inside votes
        auto tot = torch::zeros({N}, torch::kInt32).to(torch::kCUDA); // visible counts

        // 2) Iterate once over the dataset with the efficient loader
        const size_t dataset_size = train_dataset_size_;
        if (dataset_size == 0) {
            std::cout << "[Prune center] Empty dataset.\n";
            return;
        }
        const int workers = std::max(1, params_.optimization.num_workers);
        auto loader = create_efficient_infinite_dataloader(train_dataset_, workers);
        auto it = loader->begin();

        // 3) Projection constants
        const float eps2d = 0.3f;
        const float near_plane = 0.01f;
        const float far_plane = 10000.0f;
        const float radius_clip = 0.0f;
        const float scaling_mod = 1.0f;

        int idx_img = 1;
        size_t skipped_missing = 0, skipped_shape = 0, skipped_size = 0, skipped_proj = 0;

        for (size_t i = 0; i < dataset_size; ++i, ++it) {
            std::printf("\r[Prune Center] image %d/%zu", idx_img++, dataset_size);
            std::fflush(stdout);

            const auto sample = *it; // CameraDataset::Sample {camera, image, attentionMask}
            Camera* cam = sample.camera;
            auto mask_f = sample.attentionMask;

            if (!cam) {
                std::cout << "\n[Prune center] Warning: null camera; skipping view.\n";
                continue;
            }
            if (!mask_f.defined() || mask_f.numel() == 0) {
                if (skipped_missing < 3)
                    std::cout << "\n[Prune center] Warning: undefined/empty attentionMask; skipping view.\n";
                ++skipped_missing;
                continue;
            }

            // CPU binary mask [H,W]
            auto m3 = (mask_f > 0.5f);
            auto m2 = (m3.dim() == 3 && m3.size(0) == 1) ? m3.squeeze(0) : m3;
            if (m2.dim() != 2) {
                if (skipped_shape < 3)
                    std::cout << "\n[Prune center] Warning: mask wrong shape; skipping view.\n";
                ++skipped_shape;
                continue;
            }
            auto mask = m2.to(torch::kCPU).contiguous();

            const int W = (int)cam->image_width();
            const int H = (int)cam->image_height();
            if (mask.size(1) != W || mask.size(0) != H) {
                if (skipped_size < 3) {
                    std::cout << "\n[Prune center] Warning: mask size ("
                              << (long long)mask.size(1) << "x" << (long long)mask.size(0)
                              << ") != camera size (" << W << "x" << H << "); skipping view.\n";
                }
                ++skipped_size;
                continue;
            }

            // Camera tensors (CUDA) with batch dim [1,...] if needed
            auto view = cam->world_view_transform().to(torch::kCUDA);
            auto K = cam->K().to(torch::kCUDA);
            if (view.dim() == 2)
                view = view.unsqueeze(0);
            if (K.dim() == 2)
                K = K.unsqueeze(0);

            // Distortion (optional)
            std::optional<torch::Tensor> radial, tangential, thin_prism;
            if (cam->radial_distortion().defined() && cam->radial_distortion().numel() > 0)
                radial = cam->radial_distortion();
            if (cam->tangential_distortion().defined() && cam->tangential_distortion().numel() > 0)
                tangential = cam->tangential_distortion();

            // Fast projection: returns std::pair<Tensor,Tensor>
            auto proj_pair = ProjectFast(
                means3D, rotations, scales, opacities,
                view, K,
                W, H,
                eps2d, near_plane, far_plane,
                radius_clip, scaling_mod,
                cam->camera_model_type(),
                radial, tangential, std::nullopt);
            auto radii = proj_pair.first;
            auto means2d = proj_pair.second;

            if (!radii.defined() || !means2d.defined()) {
                if (skipped_proj < 3)
                    std::cout << "\n[Prune center] Warning: projection returned undefined tensors; skipping view.\n";
                ++skipped_proj;
                continue;
            }

            if (radii.dim() == 3 && radii.size(0) == 1)
                radii = radii.squeeze(0);
            if (means2d.dim() == 3 && means2d.size(0) == 1)
                means2d = means2d.squeeze(0);

            // Visible splats: positive projected radii
            torch::Tensor visible;
            if (radii.dim() == 2 && radii.size(1) >= 1)
                visible = (radii > 0.0f).all(-1);
            else if (radii.dim() == 1)
                visible = (radii > 0.0f);
            else {
                if (skipped_proj < 3)
                    std::cout << "\n[Prune center] Warning: invalid radii shape; skipping view.\n";
                ++skipped_proj;
                continue;
            }

            if (!visible.any().item<bool>())
                continue;
            auto vidx = visible.nonzero().squeeze(-1); // [M], CUDA

            // 2D centers → CPU ints in [0..W-1],[0..H-1]
            auto xy_cuda = means2d.index({vidx}); // [M,2], CUDA
            auto xy = xy_cuda.detach().to(torch::kCPU);
            auto x = torch::round(xy.select(1, 0)).to(torch::kLong).clamp(0, W - 1);
            auto y = torch::round(xy.select(1, 1)).to(torch::kLong).clamp(0, H - 1);
            auto lin = y * W + x;

            // Vote on CPU mask, accumulate on CUDA
            auto inside_cpu = mask.flatten().index({lin}); // bool CPU
            auto inside_i32_cuda = inside_cpu.to(torch::kInt32).to(torch::kCUDA);
            auto ones_i32_cuda = torch::ones_like(inside_i32_cuda, torch::kInt32);

            pos.index_add_(0, vidx, inside_i32_cuda);
            tot.index_add_(0, vidx, ones_i32_cuda);
        }
        std::printf("\n");

        // Graceful exits
        if (tot.sum().item<long long>() == 0) {
            std::cout << "[Prune center] No visibility accumulated (invalid/missing masks?); aborting prune.\n";
            return;
        }
        auto meets_vis = (tot >= min_visibility_count);
        if (!meets_vis.any().item<bool>()) {
            std::cout << "[Prune center] No splats reached min_vis=" << min_visibility_count << "; aborting prune.\n";
            return;
        }

        // Final keep & prune
        auto tot_f = tot.to(torch::kFloat32).clamp_min(1.0f);
        auto ratio = pos.to(torch::kFloat32) / tot_f;
        auto keep_mask = meets_vis & (ratio >= center_keep_threshold);

        const int64_t keep_cnt = keep_mask.sum().item<long long>();
        if (keep_cnt == 0) {
            std::cout << "[Prune center] keep_mask would be empty (thr=" << center_keep_threshold << "); skipping prune.\n";
            return;
        }

        const int removed = (keep_mask == 0).sum().item<int>();
        model.filterByMask(keep_mask);

        std::cout << "[Trainer] prune_after_training : removed " << removed
                  << " / " << N << " splats (thr=" << center_keep_threshold
                  << ", min_vis=" << min_visibility_count << ")\n";
    }*/

    // -----------------------------------------------------------------------------
    // Keep splats that land often inside the center-mask (vote-based).
    // -----------------------------------------------------------------------------
    void Trainer::prune_by_center_vote(float center_keep_threshold, int min_visibility_count) {
        torch::NoGradGuard no_grad;

        // 0) Model tensors
        SplatData& model = strategy_->get_model();
        const int64_t N = model.get_means().size(0);
        if (N <= 0) {
            std::cout << "[Prune center] No gaussians to prune.\n";
            return;
        }

        auto means3D = model.get_means();
        auto scales = model.get_scaling();
        auto rotations = model.get_rotation();
        auto opacities = model.get_opacity();
        if (opacities.defined() && opacities.dim() == 2 && opacities.size(-1) == 1)
            opacities = opacities.squeeze(-1);

        // 1) Vote buffers (CUDA int32)
        auto pos = torch::zeros({N}, torch::kInt32).to(torch::kCUDA); // inside votes
        auto tot = torch::zeros({N}, torch::kInt32).to(torch::kCUDA); // visible counts

        // 2) Iterate once over the dataset with the efficient loader
        const size_t dataset_size = train_dataset_size_;
        if (dataset_size == 0) {
            std::cout << "[Prune center] Empty dataset.\n";
            return;
        }

        const int workers = std::max(1, params_.optimization.num_workers);
        auto train_dataloader = create_infinite_dataloader_from_dataset(train_dataset_, workers);
        auto loader = train_dataloader->begin(); // iterate the dataloader itself

        // 3) Projection constants
        const float eps2d = 0.3f;
        const float near_plane = 0.01f;
        const float far_plane = 10000.0f;
        const float radius_clip = 0.0f;
        const float scaling_mod = 1.0f;

        int idx_img = 1;
        size_t skipped_missing = 0, skipped_shape = 0, skipped_size = 0, skipped_proj = 0;

        for (size_t i = 0; i < dataset_size; ++i, ++loader) {
            std::printf("\r[Prune Center] image %d/%zu", idx_img++, dataset_size);
            std::fflush(stdout);

            // Consume a batch like in the training loop
            auto& batch = *loader;
            const auto camera_with_image = batch[0].data; // {camera, image, attentionMask}
            Camera* cam = camera_with_image.camera;
            auto mask_f = camera_with_image.attentionMask;

            if (!cam) {
                std::cout << "\n[Prune center] Warning: null camera; skipping view.\n";
                continue;
            }
            if (!mask_f.defined() || mask_f.numel() == 0) {
                if (skipped_missing < 3)
                    std::cout << "\n[Prune center] Warning: undefined/empty attentionMask; skipping view.\n";
                ++skipped_missing;
                continue;
            }

            // CPU binary mask [H,W]
            auto m3 = (mask_f > 0.5f);
            auto m2 = (m3.dim() == 3 && m3.size(0) == 1) ? m3.squeeze(0) : m3;
            if (m2.dim() != 2) {
                if (skipped_shape < 3)
                    std::cout << "\n[Prune center] Warning: mask wrong shape; skipping view.\n";
                ++skipped_shape;
                continue;
            }
            auto mask = m2.to(torch::kCPU).contiguous();

            const int W = (int)cam->image_width();
            const int H = (int)cam->image_height();
            if (mask.size(1) != W || mask.size(0) != H) {
                if (skipped_size < 3) {
                    std::cout << "\n[Prune center] Warning: mask size ("
                              << (long long)mask.size(1) << "x" << (long long)mask.size(0)
                              << ") != camera size (" << W << "x" << H << "); skipping view.\n";
                }
                ++skipped_size;
                continue;
            }

            // Camera tensors (CUDA) with batch dim [1,...] if needed
            auto view = cam->world_view_transform().to(torch::kCUDA);
            auto K = cam->K().to(torch::kCUDA);
            if (view.dim() == 2)
                view = view.unsqueeze(0);
            if (K.dim() == 2)
                K = K.unsqueeze(0);

            // Distortion (optional) – ProjectFast handles device/shape
            std::optional<torch::Tensor> radial, tangential;
            if (cam->radial_distortion().defined() && cam->radial_distortion().numel() > 0)
                radial = cam->radial_distortion();
            if (cam->tangential_distortion().defined() && cam->tangential_distortion().numel() > 0)
                tangential = cam->tangential_distortion();

            // Fast projection
            auto [radii, means2d] = ProjectFast(
                means3D, rotations, scales, opacities,
                view, K,
                W, H,
                eps2d, near_plane, far_plane,
                radius_clip, scaling_mod,
                cam->camera_model_type(),
                radial, tangential, /*thin_prism*/ std::nullopt);

            if (!radii.defined() || !means2d.defined()) {
                if (skipped_proj < 3)
                    std::cout << "\n[Prune center] Warning: projection returned undefined tensors; skipping view.\n";
                ++skipped_proj;
                continue;
            }

            if (radii.dim() == 3 && radii.size(0) == 1)
                radii = radii.squeeze(0);
            if (means2d.dim() == 3 && means2d.size(0) == 1)
                means2d = means2d.squeeze(0);

            // Visible splats: positive projected radii
            torch::Tensor visible;
            if (radii.dim() == 2 && radii.size(1) >= 1)
                visible = (radii > 0.0f).all(-1);
            else if (radii.dim() == 1)
                visible = (radii > 0.0f);
            else {
                if (skipped_proj < 3)
                    std::cout << "\n[Prune center] Warning: invalid radii shape; skipping view.\n";
                ++skipped_proj;
                continue;
            }
            if (!visible.any().item<bool>())
                continue;

            auto vidx = visible.nonzero().squeeze(-1); // [M], CUDA

            // 2D centers -> CPU ints in [0..W-1],[0..H-1]
            auto xy_cuda = means2d.index({vidx}); // [M,2], CUDA
            auto xy = xy_cuda.detach().to(torch::kCPU);
            auto x = torch::round(xy.select(1, 0)).to(torch::kLong).clamp(0, W - 1);
            auto y = torch::round(xy.select(1, 1)).to(torch::kLong).clamp(0, H - 1);
            auto lin = y * W + x; // [M] long CPU

            // Vote on CPU mask, accumulate on CUDA
            auto inside_cpu = mask.flatten().index({lin}); // bool CPU
            auto inside_i32_cuda = inside_cpu.to(torch::kInt32).to(torch::kCUDA);
            auto ones_i32_cuda = torch::ones_like(inside_i32_cuda, torch::kInt32);

            pos.index_add_(0, vidx, inside_i32_cuda);
            tot.index_add_(0, vidx, ones_i32_cuda);
        }
        std::printf("\n");

        // Graceful exits
        if (tot.sum().item<long long>() == 0) {
            std::cout << "[Prune center] No visibility accumulated (invalid/missing masks?); aborting prune.\n";
            return;
        }
        auto meets_vis = (tot >= min_visibility_count);
        if (!meets_vis.any().item<bool>()) {
            std::cout << "[Prune center] No splats reached min_vis=" << min_visibility_count << "; aborting prune.\n";
            return;
        }

        // Final keep & prune
        auto tot_f = tot.to(torch::kFloat32).clamp_min(1.0f);
        auto ratio = pos.to(torch::kFloat32) / tot_f;
        auto keep_mask = meets_vis & (ratio >= center_keep_threshold);

        const int64_t keep_cnt = keep_mask.sum().item<long long>();
        if (keep_cnt == 0) {
            std::cout << "[Prune center] keep_mask would be empty (thr=" << center_keep_threshold << "); skipping prune.\n";
            return;
        }

        const int removed = (keep_mask == 0).sum().item<int>();
        model.filterByMask(keep_mask);

        std::cout << "[Trainer] Prune center-vote: removed " << removed
                  << " / " << N << " splats (thr=" << center_keep_threshold
                  << ", min_vis=" << min_visibility_count << ")\n";
    }


    // -----------------------------------------------------------------------------
    // Prune splats whose projected footprint leaks outside the mask too often.
    // Success-rate semantics: keep only if (views_without_leak / candidate_views) >= keep_threshold.
    // -----------------------------------------------------------------------------
    void Trainer::prune_by_mask_leakage(float leak_keep_threshold,
                                        float min_pixel_radius,
                                        float min_center_mask,
                                        int sample_points,
                                        int dilate_px,
                                        float per_view_leak_frac) {
        torch::NoGradGuard no_grad;

        // 0) Model tensors
        SplatData& model = strategy_->get_model();
        const int64_t N = model.get_means().size(0);
        if (N <= 0) {
            std::cout << "[Prune leak] No gaussians to prune.\n";
            return;
        }

        auto means3D = model.get_means();
        auto scales = model.get_scaling();
        auto rotations = model.get_rotation();
        auto opacities = model.get_opacity();
        if (opacities.defined() && opacities.dim() == 2 && opacities.size(-1) == 1)
            opacities = opacities.squeeze(-1);

        // 1) Leak vote buffers (CUDA int32)
        auto leak_votes = torch::zeros({N}, torch::kInt32).to(torch::kCUDA); // # leak-views
        auto vis_counts = torch::zeros({N}, torch::kInt32).to(torch::kCUDA); // # candidate views

        // 2) Iterate once over the dataset (same pattern as training loop)
        const size_t dataset_size = train_dataset_size_;
        if (dataset_size == 0) {
            std::cout << "[Prune leak] Empty dataset.\n";
            return;
        }

        const int workers = std::max(1, params_.optimization.num_workers);
        auto train_dataloader = create_infinite_dataloader_from_dataset(train_dataset_, workers);
        auto loader = train_dataloader->begin();

        // 3) Projection constants
        const float eps2d = 0.3f;
        const float near_plane = 0.01f;
        const float far_plane = 10000.0f;
        const float radius_clip = 0.0f;
        const float scaling_mod = 1.0f;

        // 4/8-direction sampling around ellipse (axis-aligned approx)
        std::vector<std::array<float, 2>> dirs = {{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}};
        if (sample_points >= 8) {
            const float s = 0.70710678f;
            dirs.push_back({s, s});
            dirs.push_back({-s, s});
            dirs.push_back({s, -s});
            dirs.push_back({-s, -s});
        }

        int idx_img = 1;
        size_t skipped_missing = 0, skipped_shape = 0, skipped_size = 0, skipped_proj = 0;

        for (size_t i = 0; i < dataset_size; ++i, ++loader) {
            std::printf("\r[Prune Leakage] image %d/%zu", idx_img++, dataset_size);
            std::fflush(stdout);

            // Consume a batch like in the training loop
            auto& batch = *loader;
            const auto sample = batch[0].data; // CameraDataset::Sample {camera, image, attentionMask}
            Camera* cam = sample.camera;
            auto mask_f = sample.attentionMask;

            if (!cam) {
                std::cout << "\n[Prune leak] Warning: null camera; skipping view.\n";
                continue;
            }
            if (!mask_f.defined() || mask_f.numel() == 0) {
                if (skipped_missing < 3)
                    std::cout << "\n[Prune leak] Warning: undefined/empty attentionMask; skipping view.\n";
                ++skipped_missing;
                continue;
            }

            // CUDA mask [H,W] in {0,1}
            auto mask01_3d = (mask_f > 0.5f).to(torch::kFloat32);
            auto mask01 = (mask01_3d.dim() == 3 && mask01_3d.size(0) == 1) ? mask01_3d.squeeze(0) : mask01_3d;
            if (mask01.dim() != 2) {
                if (skipped_shape < 3)
                    std::cout << "\n[Prune leak] Warning: mask wrong shape; skipping view.\n";
                ++skipped_shape;
                continue;
            }
            mask01 = mask01.contiguous().to(torch::kCUDA);

            const int W = (int)cam->image_width();
            const int H = (int)cam->image_height();
            if (mask01.size(1) != W || mask01.size(0) != H) {
                if (skipped_size < 3)
                    std::cout << "\n[Prune leak] Warning: mask size != camera size; skipping view.\n";
                ++skipped_size;
                continue;
            }

            // Optional dilation (pixel tolerance at boundary)
            if (dilate_px > 0) {
                const int k = 2 * dilate_px + 1;
                auto m4 = mask01.unsqueeze(0).unsqueeze(0); // [1,1,H,W]
                auto opts = torch::nn::functional::MaxPool2dFuncOptions(k).stride(1).padding(dilate_px);
                mask01 = torch::nn::functional::max_pool2d(m4, opts).squeeze(0).squeeze(0); // [H,W]
            }

            // Camera tensors (CUDA) with batch dim [1,...] if needed
            auto view = cam->world_view_transform().to(torch::kCUDA);
            auto K = cam->K().to(torch::kCUDA);
            if (view.dim() == 2)
                view = view.unsqueeze(0);
            if (K.dim() == 2)
                K = K.unsqueeze(0);

            // Distortion (optional) – ProjectFast handles device/shape
            std::optional<torch::Tensor> radial, tangential;
            if (cam->radial_distortion().defined() && cam->radial_distortion().numel() > 0)
                radial = cam->radial_distortion();
            if (cam->tangential_distortion().defined() && cam->tangential_distortion().numel() > 0)
                tangential = cam->tangential_distortion();

            // Fast projection
            auto [radii, means2d] = ProjectFast(
                means3D, rotations, scales, opacities,
                view, K,
                W, H,
                eps2d, near_plane, far_plane,
                radius_clip, scaling_mod,
                cam->camera_model_type(),
                radial, tangential, /*thin_prism*/ std::nullopt);

            if (!radii.defined() || !means2d.defined()) {
                if (skipped_proj < 3)
                    std::cout << "\n[Prune leak] Warning: projection returned undefined tensors; skipping view.\n";
                ++skipped_proj;
                continue;
            }

            if (radii.dim() == 3 && radii.size(0) == 1)
                radii = radii.squeeze(0);
            if (means2d.dim() == 3 && means2d.size(0) == 1)
                means2d = means2d.squeeze(0);

            // Visible splats: positive projected radii (handle 1D/2D radii)
            torch::Tensor visible;
            if (radii.dim() == 2 && radii.size(1) >= 1)
                visible = (radii > 0.0f).all(-1);
            else if (radii.dim() == 1)
                visible = (radii > 0.0f);
            else {
                if (skipped_proj < 3)
                    std::cout << "\n[Prune leak] Warning: invalid radii shape; skipping view.\n";
                ++skipped_proj;
                continue;
            }
            if (!visible.any().item<bool>())
                continue;

            auto vidx = visible.nonzero().squeeze(-1); // [M], CUDA

            // Centers & radii for visible splats
            auto xy = means2d.index({vidx}); // [M,2], CUDA
            auto rx = (radii.dim() == 2 ? radii.index({vidx, 0}).abs()
                                        : radii.index({vidx}).abs()); // [M]
            // If radii is 1D, use the same for x/y to keep a conservative ellipse
            auto ry = (radii.dim() == 2 ? radii.index({vidx, 1}).abs()
                                        : rx.clone());

            // Pixel centers
            auto cx = torch::round(xy.select(1, 0)).to(torch::kLong).clamp(0, W - 1);
            auto cy = torch::round(xy.select(1, 1)).to(torch::kLong).clamp(0, H - 1);

            // Eligibility for leakage test
            auto center_mask = mask01.index({cy, cx});       // [M] float {0,1}
            auto center_in = center_mask >= min_center_mask; // bool
            auto large_fp = (torch::max(rx, ry) >= min_pixel_radius);
            auto keep = center_in & large_fp;
            if (!keep.any().item<bool>())
                continue;

            auto kidx = keep.nonzero().squeeze(-1); // [K]
            auto rxk = rx.index({kidx});
            auto ryk = ry.index({kidx});
            auto cxk = cx.index({kidx});
            auto cyk = cy.index({kidx});

            // Sample P points around an axis-aligned ellipse per kept splat
            const int P = (int)dirs.size();
            auto sx = torch::empty({kidx.size(0), P}, torch::dtype(torch::kLong).device(torch::kCUDA));
            auto sy = torch::empty_like(sx);
            for (int p = 0; p < P; ++p) {
                auto dx = torch::round(rxk * dirs[p][0]).to(torch::kLong);
                auto dy = torch::round(ryk * dirs[p][1]).to(torch::kLong);
                sx.index_put_({torch::indexing::Slice(), p}, (cxk + dx).clamp(0, W - 1));
                sy.index_put_({torch::indexing::Slice(), p}, (cyk + dy).clamp(0, H - 1));
            }

            // Per-splat outside ratio in this view
            auto lin = (sy * W + sx).reshape({-1});                    // [K*P]
            auto mvals = mask01.view({-1}).index({lin}).view({-1, P}); // [K,P]
            auto outside_ratio = (1.0f - mvals).mean(1);               // [K] in [0,1]

            // Count a "leak view" if outside_ratio > per_view_leak_frac
            auto leak_here = (outside_ratio > per_view_leak_frac).to(torch::kInt32); // [K]
            auto ids = vidx.index({kidx});                                           // [K] original IDs

            vis_counts.index_add_(0, ids, torch::ones_like(leak_here));
            leak_votes.index_add_(0, ids, leak_here);
        }
        std::printf("\n");

        // Graceful exits
        if (vis_counts.sum().item<long long>() == 0) {
            std::cout << "[Prune leak] No candidate visibility accumulated (invalid/missing masks?); aborting prune.\n";
            return;
        }

        auto vis_ok = (vis_counts >= 1);
        auto fail_r = leak_votes.to(torch::kFloat32) / vis_counts.to(torch::kFloat32).clamp_min(1.0f);
        auto ok_r = 1.0f - fail_r;
        auto is_prune = vis_ok & (ok_r < leak_keep_threshold);
        auto keep_mask = (~is_prune).to(torch::kBool);

        const int64_t keep_cnt = keep_mask.sum().item<long long>();
        if (keep_cnt == 0) {
            std::cout << "[Prune leak] keep_mask would be empty (keep_thr=" << leak_keep_threshold << "); skipping prune.\n";
            return;
        }

        const int removed = is_prune.sum().item<int>();
        model.filterByMask(keep_mask);

        std::cout << "[Trainer] Prune mask-leakage: removed " << removed
                  << " / " << N << " splats (keep_thr=" << leak_keep_threshold
                  << ", min_radius=" << min_pixel_radius
                  << ", samples=" << (sample_points >= 8 ? 8 : 4)
                  << ", dilate_px=" << dilate_px
                  << ", per_view_leak_frac=" << per_view_leak_frac
                  << ")\n";
    }


    void Trainer::prune_after_training(float vote_ratio_threshold, float leak_keep_threshold) {
        // 1) First, center-vote pruning (your original logic)
        prune_by_center_vote(vote_ratio_threshold);

        // 2) Re-fetch model and then leakage pruning on remaining splats
        prune_by_mask_leakage(leak_keep_threshold);
    }

    void Trainer::save_ply(const std::filesystem::path& save_path, int iter_num, bool join_threads) {
        // Save PLY format - join_threads controls sync vs async
        strategy_->get_model().save_ply(save_path, iter_num, join_threads);

        // Save SOG format if requested - ALWAYS synchronous
        if (params_.optimization.save_sog) {
            strategy_->get_model().save_sog(save_path, iter_num,
                                            params_.optimization.sog_iterations,
                                            true); // Always synchronous
        }

        // Update project with PLY info
        if (lf_project_) {
            const std::string ply_name = "splat_" + std::to_string(iter_num);
            const std::filesystem::path ply_path = save_path / (ply_name + ".ply");
            lf_project_->addPly(gs::management::PlyData(false, ply_path, iter_num, ply_name));
        }

        LOG_DEBUG("PLY save initiated: {} (sync={}), SOG always sync", save_path.string(), join_threads);
    }
} // namespace gs::training