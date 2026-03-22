/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

/**
 * @file mask_penalty.hpp
 * @brief Pixel-based opacity penalty for masked Gaussian splatting training.
 *
 * This module implements opacity regularization based on attention masks,
 * encouraging splats to be opaque inside the mask region and transparent
 * outside. The implementation is optimized for GPU execution with minimal
 * synchronization and uses lazy tensor evaluation where possible.
 *
 * @section usage Usage
 * @code
 * #include "mask_penalty.hpp"
 *
 * // In your loss computation:
 * auto result = mask_penalty::compute_and_apply_full_penalty(
 *     loss, grad_image, alpha, mask, iter, total_iters);
 *
 * if (result.applied()) {
 *     loss = result.modified_loss;
 *     grad_image = result.modified_grad_image;
 *     grad_alpha = result.grad_alpha;
 * }
 * @endcode
 *
 * @section compatibility Rasterizer Compatibility
 * - Fast rasterizer: Supported (uses rendered alpha)
 * - Gsplat rasterizer: Supported (uses rendered alpha)
 *
 * @note The splat-center-based penalty (outsideMaskOpacityPenalty) is not
 *       implemented as it requires means2d which fast_rasterizer does not expose.
 */

#include "core/tensor.hpp"

namespace lfs::training {
namespace mask_penalty {

// =============================================================================
// Configuration
// =============================================================================

/**
 * @brief Configuration parameters for pixel-based opacity penalty.
 *
 * Controls the balance between inside/outside penalties and the temporal
 * schedule that modulates penalty strength during training.
 */
struct PixelPenaltyConfig {
    /// Weight for penalizing transparency inside the mask (want opacity)
    float k_inside = 0.667f;

    /// Weight for penalizing opacity outside the mask (want transparency)
    float k_outside = 0.333f;

    /// Training progress fraction where warmup ends (no penalty before this)
    float warmup_fraction = 0.10f;

    /// Training progress fraction where full penalty ends (decay starts)
    float full_penalty_end = 0.50f;

    /// Training progress fraction where decay ends (minimum weight after)
    float decay_end = 0.80f;

    /// Minimum penalty weight after decay (never fully disabled)
    float min_penalty_weight = 0.20f;
};

/// Default configuration with empirically tuned values
inline constexpr PixelPenaltyConfig DEFAULT_CONFIG{};

// =============================================================================
// Penalty Schedule
// =============================================================================

/**
 * @brief Computes the penalty weight based on training progress.
 *
 * Implements a piecewise linear schedule:
 * - [0%, warmup_fraction]: No penalty (allows geometry to stabilize)
 * - [warmup_fraction, full_penalty_end]: Full penalty weight (1.0)
 * - [full_penalty_end, decay_end]: Linear decay to min_penalty_weight
 * - [decay_end, 100%]: Constant min_penalty_weight
 *
 * @param iter Current training iteration (1-indexed)
 * @param total_iters Total number of training iterations
 * @param config Schedule configuration parameters
 * @return Penalty weight in range [0.0, 1.0]
 */
[[nodiscard]] inline float compute_penalty_weight(
    int iter,
    int total_iters,
    const PixelPenaltyConfig& config = DEFAULT_CONFIG) noexcept
{
    if (total_iters <= 0) {
        return 0.0f;
    }

    const float progress = static_cast<float>(iter) / static_cast<float>(total_iters);

    if (progress < config.warmup_fraction) {
        return 0.0f;
    }
    if (progress < config.full_penalty_end) {
        return 1.0f;
    }
    if (progress < config.decay_end) {
        const float t = (progress - config.full_penalty_end) /
                        (config.decay_end - config.full_penalty_end);
        return 1.0f - (1.0f - config.min_penalty_weight) * t;
    }
    return config.min_penalty_weight;
}

// =============================================================================
// Penalty Result Types
// =============================================================================

/**
 * @brief Result of pixel-based penalty computation.
 */
struct PenaltyResult {
    float penalty_value;            ///< Scalar penalty value
    lfs::core::Tensor grad_alpha;   ///< Gradient w.r.t. alpha [H, W]

    [[nodiscard]] bool is_valid() const { return grad_alpha.is_valid(); }
};

/**
 * @brief Complete result for loss integration.
 */
struct FullPenaltyResult {
    lfs::core::Tensor modified_loss;        ///< Loss with penalty applied
    lfs::core::Tensor modified_grad_image;  ///< Image gradient scaled by penalty
    lfs::core::Tensor grad_alpha;           ///< Alpha gradient to accumulate
    float penalty_value;                    ///< Raw penalty value for logging

    /// Returns true if penalty was applied (weight > 0)
    [[nodiscard]] bool applied() const { return penalty_value > 0.0f; }
};

// =============================================================================
// Core Penalty Computation
// =============================================================================

/**
 * @brief Computes pixel-based opacity penalty and its gradient.
 *
 * Penalizes two conditions:
 * - Transparency inside the mask: (1 - alpha) where mask > 0.5
 * - Opacity outside the mask: alpha where mask <= 0.5
 *
 * The penalty formula is:
 * @f[
 *   P = \frac{w}{N} \left( k_{in} \sum_{fg} (1 - \alpha) + k_{out} \sum_{bg} \alpha \right)
 * @f]
 *
 * Where:
 * - @f$ w @f$ is the penalty weight
 * - @f$ N @f$ is the total number of pixels
 * - @f$ k_{in}, k_{out} @f$ are the inside/outside balance factors
 * - @f$ fg, bg @f$ are the foreground and background regions
 *
 * @param alpha Rendered alpha channel [H, W] or [1, H, W], float32, CUDA
 * @param mask Attention mask [H, W] or [1, H, W], float32, CUDA (>0.5 = foreground)
 * @param weight Global penalty weight multiplier
 * @param config Balance and schedule configuration
 * @return PenaltyResult containing scalar penalty and alpha gradient
 *
 * @note Uses lazy tensor operations to minimize GPU kernel launches.
 *       Only sum_scalar() calls trigger synchronization.
 */
[[nodiscard]] inline PenaltyResult compute_pixel_penalty(
    const lfs::core::Tensor& alpha,
    const lfs::core::Tensor& mask,
    float weight,
    const PixelPenaltyConfig& config = DEFAULT_CONFIG)
{
    using namespace lfs::core;

    // Early exit for zero weight
    if (weight <= 0.0f) {
        return {0.0f, Tensor()};
    }

    // Normalize to 2D [H, W]
    const Tensor alpha_2d = (alpha.ndim() == 3) ? alpha.squeeze(0) : alpha;
    const Tensor mask_2d = (mask.ndim() == 3) ? mask.squeeze(0) : mask;

    // Validate shapes match
    if (alpha_2d.shape() != mask_2d.shape()) {
        return {0.0f, Tensor()};
    }

    const float num_pixels = static_cast<float>(alpha_2d.numel());
    if (num_pixels <= 0.0f) {
        return {0.0f, Tensor()};
    }

    // Create binary masks (lazy evaluation)
    const Tensor fg_mask = mask_2d.gt(0.5f).to(DataType::Float32);
    const Tensor bg_mask = fg_mask.mul(-1.0f).add(1.0f);

    // Compute penalty sums (requires GPU sync)
    // inside_penalty = sum((1 - alpha) * fg_mask)
    // outside_penalty = sum(alpha * bg_mask)
    const float inside_sum = alpha_2d.mul(-1.0f).add(1.0f).mul(fg_mask).sum_scalar();
    const float outside_sum = alpha_2d.mul(bg_mask).sum_scalar();

    const float penalty_value = weight *
        (config.k_inside * inside_sum + config.k_outside * outside_sum) / num_pixels;

    // Compute gradient: d(penalty)/d(alpha) = w/N * (-k_in * fg + k_out * bg)
    const float grad_scale = weight / num_pixels;
    Tensor grad_alpha = bg_mask.mul(config.k_outside * grad_scale)
                               .sub(fg_mask.mul(config.k_inside * grad_scale));

    return {penalty_value, std::move(grad_alpha)};
}

// =============================================================================
// Loss Modification Utilities
// =============================================================================

/**
 * @brief Applies penalty to loss using multiplicative-additive formula.
 *
 * The formula amplifies the base loss proportionally to the penalty:
 * @f[
 *   L_{new} = L_{base} \cdot (1 + P) + 0.01 \cdot P
 * @f]
 *
 * This formulation ensures:
 * - Errors in high-penalty regions are weighted more heavily
 * - A small direct penalty term maintains gradient even when base loss is low
 *
 * @param loss Current loss tensor (scalar)
 * @param penalty_value Penalty value to apply
 * @return Modified loss tensor
 */
[[nodiscard]] inline lfs::core::Tensor apply_penalty_to_loss(
    const lfs::core::Tensor& loss,
    float penalty_value)
{
    if (penalty_value <= 0.0f) {
        return loss;
    }

    const float multiplier = 1.0f + penalty_value;
    const float additive = 0.01f * penalty_value;

    return loss.mul(multiplier).add(additive);
}

/**
 * @brief Scales image gradient by penalty factor for consistency.
 *
 * @param grad Current image gradient tensor
 * @param penalty_value Penalty value used for loss modification
 * @return Scaled gradient tensor
 */
[[nodiscard]] inline lfs::core::Tensor scale_gradient_by_penalty(
    const lfs::core::Tensor& grad,
    float penalty_value)
{
    if (penalty_value <= 0.0f || !grad.is_valid()) {
        return grad;
    }

    return grad.mul(1.0f + penalty_value);
}

// =============================================================================
// High-Level Integration API
// =============================================================================

/**
 * @brief Computes and applies complete pixel-based opacity penalty.
 *
 * This is the primary entry point for integrating pixel penalty into
 * the training loop. It handles:
 * - Schedule-based weight computation
 * - Penalty calculation
 * - Loss and gradient modification
 *
 * @param loss Current photometric loss (scalar tensor)
 * @param grad_image Current image gradient
 * @param alpha Rendered alpha channel
 * @param mask Attention mask
 * @param iter Current training iteration
 * @param total_iters Total training iterations
 * @param user_weight User-specified penalty weight multiplier
 * @param config Penalty configuration
 * @return FullPenaltyResult with modified loss, gradients, and metadata
 *
 * @code
 * // Typical usage in compute_photometric_loss_with_mask:
 * auto penalty = compute_and_apply_full_penalty(
 *     loss, grad, alpha, mask, iter, total_iters,
 *     opt_params.mask_opacity_penalty_weight);
 *
 * if (penalty.applied()) {
 *     loss = penalty.modified_loss;
 *     grad = penalty.modified_grad_image;
 *     if (grad_alpha.is_valid()) {
 *         grad_alpha = grad_alpha.mul(1.0f + penalty.penalty_value)
 *                                .add(penalty.grad_alpha);
 *     } else {
 *         grad_alpha = penalty.grad_alpha;
 *     }
 * }
 * @endcode
 */
[[nodiscard]] inline FullPenaltyResult compute_and_apply_full_penalty(
    const lfs::core::Tensor& loss,
    const lfs::core::Tensor& grad_image,
    const lfs::core::Tensor& alpha,
    const lfs::core::Tensor& mask,
    int iter,
    int total_iters,
    float user_weight = 1.0f,
    const PixelPenaltyConfig& config = DEFAULT_CONFIG)
{
    const float schedule_weight = compute_penalty_weight(iter, total_iters, config);
    const float final_weight = schedule_weight * user_weight;

    // Early exit if no penalty to apply
    if (final_weight <= 0.0f || !alpha.is_valid() || !mask.is_valid()) {
        return {loss, grad_image, lfs::core::Tensor(), 0.0f};
    }

    auto [penalty_value, grad_alpha] = compute_pixel_penalty(alpha, mask, final_weight, config);

    if (penalty_value <= 0.0f) {
        return {loss, grad_image, lfs::core::Tensor(), 0.0f};
    }

    auto modified_loss = apply_penalty_to_loss(loss, penalty_value);
    auto modified_grad = scale_gradient_by_penalty(grad_image, penalty_value);

    return {
        std::move(modified_loss),
        std::move(modified_grad),
        std::move(grad_alpha),
        penalty_value
    };
}

// =============================================================================
// Future Work Notes
// =============================================================================

/**
 * @note Splat-center-based penalty (outsideMaskOpacityPenalty) is not implemented.
 *
 * The original implementation uses projected 2D splat centers (means2d) and
 * radii to penalize individual splats based on their center location relative
 * to the mask. This approach requires:
 * - means2d: Projected 2D positions of splat centers [N, 2]
 * - radii: Projected radii of splats [N] or [N, 2]
 *
 * Compatibility issues:
 * - fast_rasterizer: Does NOT expose means2d/radii in RenderOutput
 * - gsplat_rasterizer: Exposes means2d_ptr and radii_ptr in context
 *
 * Potential solutions for future implementation:
 * 1. Modify fast_rasterizer to expose projection data in RenderOutput
 * 2. Only enable for gsplat mode (when gut=true)
 * 3. Re-compute projection (expensive, duplicates rasterizer work)
 *
 * For now, pixel-based penalty provides sufficient regularization for
 * most matting use cases.
 */

/**
 * @note Post-training pruning functions are not yet implemented.
 *
 * The following pruning methods should be called after training completes
 * when using FocusedSegment mode:
 *
 * - prune_by_center_vote(): Removes splats whose center does not consistently
 *   fall inside the mask across multiple views.
 *
 * - prune_by_mask_leakage(): Removes splats whose projected footprint
 *   frequently extends outside the mask boundary.
 *
 * - prune_after_training(): Orchestrates both pruning methods in sequence.
 *
 * These should be called in Trainer::train() before the final save:
 * @code
 * if (mask_mode == MaskMode::FocusedSegment) {
 *     prune_after_training(vote_ratio_threshold, leak_keep_threshold);
 * }
 * @endcode
 */

} // namespace mask_penalty
} // namespace lfs::training