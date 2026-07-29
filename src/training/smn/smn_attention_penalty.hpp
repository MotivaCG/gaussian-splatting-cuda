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

    // True for either attention variant. Both run the identical training-time
    // loss/penalty; they differ only in whether the post-training prune runs
    // (see run_attention_prune, gated on MaskMode::Attention only).
    [[nodiscard]] inline constexpr bool is_attention_mask_mode(
        const lfs::core::param::MaskMode mode) noexcept {
        return mode == lfs::core::param::MaskMode::Attention ||
               mode == lfs::core::param::MaskMode::AttentionNoPrune;
    }

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
    //   fused_ws      : reused workspace for the masked fused L1+SSIM kernel
    //   decoupled_ws  : reused workspace for the masked decoupled variant
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
        lfs::training::kernels::MaskedFusedL1SSIMWorkspace& fused_ws,
        lfs::training::kernels::MaskedDecoupledFusedL1SSIMWorkspace& decoupled_ws);

    // Schedule weight w(iter) in [0,1] for the opacity penalty. Exposed for
    // testing and logging; see smn_attention_constants.h for the ramp shape.
    [[nodiscard]] float attention_penalty_schedule_weight(int iteration, int total_iterations);

    // Per-optimizer-step hook for the attention modes. Call once per step from the
    // trainer at a safe point (after strategy step). It is a no-op unless `mode` is
    // an attention mode; currently it fires the one-shot opacity kick at
    // SMN_OPACITY_KICK_AT_FRACTION. Strategy-agnostic (MRNF and MCMC).
    void attention_on_optimizer_step(
        lfs::training::IStrategy& strategy,
        int iteration,
        int total_iterations,
        lfs::core::param::MaskMode mode);

} // namespace lfs::training::smn
