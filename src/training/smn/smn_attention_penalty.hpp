/* SPDX-FileCopyrightText: 2026 SMN | Víctor M. Feliz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// =============================================================================
// SMN — Attention mask mode: training-time loss and opacity penalty
// =============================================================================
//
// This translation unit owns everything the "attention" mask mode does DURING
// optimization (the prune stage lives in smn_attention_prune.*). It produces a
// masked photometric loss with a soft out-of-mask weight and adds a scheduled
// bidirectional opacity penalty that guides the rendered alpha towards the mask.
//
// It reuses the engine's fused masked L1+SSIM kernels for the photometric term,
// so it stays consistent with the rest of the training loss and its manual
// gradient plumbing (grad_corrected / grad_raw / grad_alpha).
// =============================================================================

#include "core/parameters.hpp"
#include "core/tensor.hpp"
#include "lfs/kernels/ssim.cuh"

#include <expected>
#include <string>

namespace lfs::training {
    class IStrategy;
}

namespace lfs::training::smn {

    // True for the attention mask mode (out-of-mask weighting + scheduled opacity
    // penalty + post-training prune). Kept as a helper so call sites read clearly.
    [[nodiscard]] inline constexpr bool is_attention_mask_mode(
        const lfs::core::param::MaskMode mode) noexcept {
        return mode == lfs::core::param::MaskMode::Attention;
    }

    // Morphological dilation (max filter) of a mask by `radius` pixels: grows the
    // masked region so the border band counts as "inside". Returns a [H,W] Float32
    // tensor. radius <= 0 returns the mask unchanged. Used for the attention
    // priority mask (photometric weight + densification), never for alpha/prune.
    [[nodiscard]] lfs::core::Tensor dilate_mask(const lfs::core::Tensor& mask, int radius);

    // Result of the attention photometric loss. Mirrors Trainer::MaskLossResult
    // field-for-field so the single upstream call site can forward it directly,
    // without this module depending on any Trainer-private type.
    struct AttentionLossResult {
        lfs::core::Tensor loss;           // scalar training loss for this tile
        lfs::core::Tensor grad_corrected; // d(loss)/d(corrected image)
        lfs::core::Tensor grad_raw;       // d(loss)/d(raw image) (decoupled path only)
        lfs::core::Tensor grad_alpha;     // d(loss)/d(rendered alpha) from the penalty
    };

    // Compute the attention-mode photometric loss for one (tiled) view.
    //
    //   corrected     : rendered image after appearance correction [C,H,W]
    //   gt_image      : ground-truth image [C,H,W]
    //   mask          : binarized ROI mask [H,W] or [1,H,W] (bool/uint8/float)
    //   roi_weight    : optional crop-box weight map [H,W] float, or invalid
    //   alpha         : rendered alpha [H,W] or [1,H,W], or invalid
    //   opt_params    : optimization parameters (lambda_dssim, iterations, ...)
    //   raw_rendered  : raw render for decoupled D-SSIM, or invalid to disable
    //   iteration     : current training iteration (drives the penalty schedule)
    //   workspace_arena: upstream's shared arena for the mutually exclusive
    //                    masked fused and masked decoupled loss workspaces
    //
    // The `mask` is assumed already loaded/binarized by the caller (the trainer
    // requests binarize=true for attention mode).
    [[nodiscard]] std::expected<AttentionLossResult, std::string>
    compute_attention_photometric_loss(
        const lfs::core::Tensor& corrected,
        const lfs::core::Tensor& gt_image,
        const lfs::core::Tensor& mask,
        const lfs::core::Tensor& roi_weight,
        const lfs::core::Tensor& alpha,
        const lfs::core::param::OptimizationParameters& opt_params,
        const lfs::core::Tensor& raw_rendered,
        int iteration,
        lfs::training::kernels::LossWorkspaceArena& workspace_arena);

    // Schedule weight w(iter) in [0,1] for the opacity penalty. Exposed for
    // testing and logging; see smn_attention_constants.h for the ramp shape.
    [[nodiscard]] float attention_penalty_schedule_weight(int iteration, int total_iterations);

    // Per-optimizer-step hook for the attention modes. Call once per step from the
    // trainer at a safe point (after strategy step). It is a no-op unless `mode` is an
    // attention mode; it fires the scheduled opacity kicks (SMN_OPACITY_KICKS) as
    // training crosses each entry's fraction. Strategy-agnostic (MRNF and MCMC).
    void attention_on_optimizer_step(
        lfs::training::IStrategy& strategy,
        int iteration,
        int total_iterations,
        lfs::core::param::MaskMode mode);

} // namespace lfs::training::smn
