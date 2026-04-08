/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "training/trainer.hpp"

#include "core/logger.hpp"
#include "smn/mask_pruning.hpp"

#include <algorithm>

namespace lfs::training {

    namespace {

        // FocusedSegment densification mask dilation radius (in pixels).
        //
        // Controls how much the FG mask is dilated before multiplying it against the
        // densification error map in FocusedSegment mode. Affects ONLY densification
        // placement, not the loss or the image gradient.
        //
        //   0  : disabled - hard mask. Safest against halos, but pixels of fine
        //        structures (hair, eyelashes) that straddle the mask border see
        //        error=0 and are never densified there.
        //   2-3: recommended - covers thin fine structures without reintroducing
        //        large background regions. The final prune (leakage, centervote,
        //        ellipse boundary) removes anything that leaks into the background.
        //   6  : matches the SSIM 11x11 half-window. Maximum recovery of boundary
        //        detail but higher risk of creating splats in the near-background
        //        that the prune then has to clean up.
        //
        // Tune per scene: scenes with fine hair / fur benefit from 3; scenes with
        // clean silhouettes are fine at 0.
        constexpr int FOCUSED_DENSIFY_DILATE_RADIUS = 3;

    } // namespace

    // =========================================================================
    // FocusedSegment helpers
    //
    // All FocusedSegment-specific logic is grouped here in Trainer private
    // methods so that the call sites inside Trainer methods are one-liners.
    // This keeps the hot paths in compute_photometric_loss_with_mask,
    // train_step and the post-training prune section lean and minimizes
    // merge-conflict surface area when syncing with upstream.
    //
    // Any rename or refactor of Trainer internals that is coming from main
    // should NOT touch this block - keep FocusedSegment-specific state and
    // logic isolated. If upstream renames a field on OptimizationParameters
    // (e.g. mask_opacity_penalty_weight_bg), update the references inside
    // this block to match main's new names.
    // =========================================================================

    // Apply the 7-phase FocusedSegment schedule to a local OptimizationParameters
    // copy for the current iteration. Called from both the PPISP and non-PPISP
    // paths in train_step. Mutates step_params in place.
    //
    // Spatial L1 weighting (FG=1.0, BG=kBgTarget) is always active from Phase 2
    // onward. Only the alpha penalty is scheduled.
    //
    //  Progress  | Phase              | Alpha penalty
    // -----------+--------------------+--------------------------------------
    //   0 - 15%  | None               | none
    //  15 - 25%  | BG spatial ramp    | none
    //  25 - 40%  | Spatial active     | none (growth still active)
    //  40 - 50%  | FG ramp 0->full    | FG only (fill holes)
    //  50 - 60%  | FG at full         | FG only (consolidate before BG presses)
    //  60 - 70%  | FG full + BG ramp  | FG + BG (trim silhouette)
    //  70 - 85%  | Decay to 0         | both ramp down to 0
    //  85 - 100% | Free refinement    | none (natural border alpha)
    void Trainer::focused_segment_apply_schedule(
        lfs::core::param::OptimizationParameters& step_params,
        const float progress) {

        if (step_params.mask_mode != lfs::core::param::MaskMode::FocusedSegment) {
            return;
        }

        constexpr float kNoneEnd = 0.15f; // End of warm-up with masking disabled.
        constexpr float kBgRampEnd = 0.25f; // End of BG spatial weighting ramp.
        constexpr float kBgTarget = 0.05f; // Active-phase BG gradient weight (FG focused)
        constexpr float kBgTargetFree = 0.08f; // Free-refinement BG weight - slightly higher
                                               // than kBgTarget to allow gentle BG refinement
                                               // without reintroducing detail. Kept low to help
                                               // fine silhouettes (hair) stay sharp.
        step_params.focused_bg_weight = kBgTarget;

        constexpr float kFgPenaltyStart = 0.40f; // Start ramping FG alpha pressure up.
        constexpr float kFgPenaltyEnd = 0.50f; // FG alpha pressure reaches full strength.

        constexpr float kBothPenaltyStart = 0.60f; // Start ramping BG alpha pressure up.
        constexpr float kBothPenaltyEnd = 0.70f; // FG+BG alpha pressure both fully active.

        constexpr float kPenaltyDecayStart = 0.75f; // Start relaxing alpha pressure toward residual floors.
        constexpr float kPenaltyDecayEnd = 0.85f; // End of decay; residual floors remain until training ends.

        constexpr float kFgPenaltyFloor = 0.15f; // Final FG residual penalty. Kept at 0 to avoid carving holes from mask noise.
        constexpr float kBgPenaltyFloor = 0.15f; // Final BG residual penalty. Keeps gentle cleanup of late halos/spikes outside the mask.

        static_assert(kBgRampEnd < kFgPenaltyStart, "BG spatial ramp must finish before FG penalty starts");
        static_assert(kFgPenaltyEnd <= kBothPenaltyStart, "FG penalty must reach full before BG joins");
        static_assert(kBothPenaltyEnd <= kPenaltyDecayStart, "Both at full before decay starts");

        if (progress < kNoneEnd) {
            step_params.mask_mode = lfs::core::param::MaskMode::None;
            step_params.focused_bg_weight = 1.0f;
        } else if (progress < kBgRampEnd) {
            const float t = (progress - kNoneEnd) / (kBgRampEnd - kNoneEnd);
            step_params.focused_bg_weight = 1.0f - t * (1.0f - kBgTarget);
            step_params.mask_opacity_penalty_weight = 0.0f;
            step_params.mask_opacity_penalty_weight_bg = 0.0f;
        } else if (progress < kFgPenaltyStart) {
            step_params.mask_opacity_penalty_weight = 0.0f;
            step_params.mask_opacity_penalty_weight_bg = 0.0f;
        } else if (progress < kFgPenaltyEnd) {
            const float t = (progress - kFgPenaltyStart) / (kFgPenaltyEnd - kFgPenaltyStart);
            step_params.mask_opacity_penalty_weight *= t;
            step_params.mask_opacity_penalty_weight_bg = 0.0f;
        } else if (progress < kBothPenaltyEnd) {
            const float t_bg = (progress - kBothPenaltyStart) / (kBothPenaltyEnd - kBothPenaltyStart);
            step_params.mask_opacity_penalty_weight_bg *= std::clamp(t_bg, 0.0f, 1.0f);
        } else if (progress < kPenaltyDecayEnd) {
            const float t_decay = std::clamp(
                (progress - kPenaltyDecayStart) / (kPenaltyDecayEnd - kPenaltyDecayStart),
                0.0f,
                1.0f);
            const float fg_factor = 1.0f + (kFgPenaltyFloor - 1.0f) * t_decay;
            const float bg_factor = 1.0f + (kBgPenaltyFloor - 1.0f) * t_decay;
            step_params.mask_opacity_penalty_weight *= fg_factor;
            step_params.mask_opacity_penalty_weight_bg *= bg_factor;
            step_params.focused_bg_weight = kBgTarget * (1.0f - t_decay) + kBgTargetFree * t_decay;
        } else {
            step_params.mask_opacity_penalty_weight *= kFgPenaltyFloor;
            step_params.mask_opacity_penalty_weight_bg *= kBgPenaltyFloor;
            step_params.focused_bg_weight = kBgTargetFree;
        }
    }

    // FocusedSegment photometric loss: full-image L1+SSIM forward (same quality as
    // None mode) with a post-hoc spatial weight map (FG=1.0, BG=focused_bg_weight)
    // applied to grad_corrected, plus an independent grad_alpha pressure pushing FG
    // alpha -> 1 and BG alpha -> 0. The scalar loss is rescaled to reflect the
    // FG-focused weighting so that logged values track FG reconstruction quality.
    //
    // photometric_loss_ is a Trainer member because it holds a persistent
    // workspace that is reused across iterations.
    //
    // raw_rendered is currently unused - FocusedSegment does not split into
    // appearance-corrected vs raw gradients. The parameter exists to match the
    // master signature of compute_photometric_loss_with_mask and to allow a
    // future decoupled FocusedSegment variant without another churn.
    std::expected<Trainer::MaskLossResult, std::string> Trainer::focused_segment_compute_loss(
        const lfs::core::Tensor& corrected,
        const lfs::core::Tensor& raw_rendered,
        const lfs::core::Tensor& gt_image,
        const lfs::core::Tensor& mask_2d,
        const lfs::core::Tensor& alpha,
        const lfs::core::param::OptimizationParameters& opt_params) {

        using namespace lfs::core;

        (void)raw_rendered;

        const float kBgWeight = opt_params.focused_bg_weight; // BG gradient weight relative to FG (1.0)
        constexpr float kDarknessBoost = 2.0f; // extra FG weight on dark pixels; 0 = disabled
        constexpr float kAlphaFgWeight = 1.5f; // grad_alpha pressure to push FG alpha -> 1
        constexpr float kAlphaBgWeight = 1.0f; // grad_alpha pressure to push BG alpha -> 0

        const Tensor bg_mask = Tensor::full(mask_2d.shape(), 1.0f, mask_2d.device()) - mask_2d;

        // Step 1: full-image L1+SSIM forward - identical to None mode.
        // Produces correct grad_image and populates fused_workspace ssim_map for
        // pixel-error-based densification. No gradient approximation here.
        losses::PhotometricLoss::Params params{.lambda_dssim = opt_params.lambda_dssim};
        auto full_result = photometric_loss_.forward(corrected, gt_image, params);
        if (!full_result) {
            return std::unexpected(full_result.error());
        }
        auto [full_loss, ctx] = *full_result;
        Tensor grad = ctx.grad_image;
        Tensor loss = full_loss;

        // Step 2: spatial weight map - FG=1.0, BG=kBgWeight.
        // Darkness bonus applied to FG only (Rec.601 perceptual luminance from GT, not rendered).
        // Using GT keeps the weight map stable across iterations - rendered changes every step.
        // BG stays flat at kBgWeight regardless of darkness, avoiding spurious BG gradient boosts.
        Tensor weight_map;
        if (kDarknessBoost > 0.0f) {
            const bool chw = (gt_image.ndim() == 3 && gt_image.shape()[0] == 3);
            const Tensor r = chw ? gt_image.slice(0, 0, 1).squeeze(0) : gt_image.slice(2, 0, 1).squeeze(2);
            const Tensor g = chw ? gt_image.slice(0, 1, 2).squeeze(0) : gt_image.slice(2, 1, 2).squeeze(2);
            const Tensor b = chw ? gt_image.slice(0, 2, 3).squeeze(0) : gt_image.slice(2, 2, 3).squeeze(2);
            const Tensor brightness = r * 0.299f + g * 0.587f + b * 0.114f;
            const Tensor darkness = Tensor::full(brightness.shape(), 1.0f, brightness.device()) - brightness;
            weight_map =
                mask_2d * (Tensor::full(darkness.shape(), 1.0f, darkness.device()) + darkness * kDarknessBoost) +
                bg_mask * kBgWeight;
        } else {
            weight_map = mask_2d + bg_mask * kBgWeight;
        }

        // Normalize weight_map by its mean so the global gradient magnitude stays
        // comparable to None mode - prevents scale drift when mask size varies across scenes.
        const float weight_mean = weight_map.mean().item<float>();
        weight_map = weight_map * (1.0f / std::max(weight_mean, 1e-4f));

        // Step 3: apply spatial weights to gradient.
        // Multiplying grad post-hoc is exact for L1 (pixel-wise) and a good approximation
        // for SSIM (window-based). In practice this outperforms discarding SSIM gradient entirely.
        const Tensor weight_3d = (corrected.ndim() == 3 && corrected.shape()[0] == 3)
                                     ? weight_map.unsqueeze(0)
                                     : weight_map.unsqueeze(2);
        grad = grad * weight_3d;

        // Scale scalar loss to reflect FG-focused weighting.
        // This ensures the logged loss is representative of FG reconstruction quality,
        // not diluted by the large BG area. fg_pixels is cached to avoid a second GPU sync.
        const float total_pixels = static_cast<float>(mask_2d.numel());
        const float fg_pixels = std::max(mask_2d.sum().item<float>(), 1.0f);
        const float bg_pixels = std::max(total_pixels - fg_pixels, 1.0f);
        loss = loss * (fg_pixels / std::max(total_pixels * weight_mean, 1e-6f));

        // Step 4: alpha pressure via grad_alpha.
        // NOTE: modifying the scalar `loss` does NOT affect Gaussian parameters - only
        // grad_image and grad_alpha propagate through rasterize_backward to the optimizer.
        // Alpha pressure is therefore applied purely through grad_alpha.
        //
        // Each zone (FG/BG) is normalized independently by its pixel count so that pressure
        // is balanced regardless of how much of the image the mask occupies.
        //
        // Sign convention (gradient descent: param -= lr * grad):
        //   FG: negative grad_alpha  -> alpha_raw increases -> rendered alpha approaches 1
        //   BG: positive grad_alpha  -> alpha_raw decreases -> rendered alpha approaches 0
        Tensor grad_alpha;
        const float w_fg = opt_params.mask_opacity_penalty_weight;
        const float w_bg = opt_params.mask_opacity_penalty_weight_bg;
        if (alpha.is_valid() && (w_fg > 0.0f || w_bg > 0.0f)) {
            const Tensor alpha_2d = alpha.ndim() == 3 ? alpha.squeeze(0) : alpha;

            grad_alpha = bg_mask * (w_bg * kAlphaBgWeight / bg_pixels)    // BG: push alpha -> 0
                         - mask_2d * (w_fg * kAlphaFgWeight / fg_pixels); // FG: push alpha -> 1
        }

        return Trainer::MaskLossResult{
            .loss = loss,
            .grad_corrected = grad,
            .grad_raw = {},
            .grad_alpha = grad_alpha};
    }

    // FocusedSegment-only helper for densification error maps.
    //
    // Dilate the mask before multiplying so that splats straddling the mask
    // border still accumulate enough error for densification (clone/split).
    // Without dilation, a splat with 50% of its footprint outside the mask
    // sees its error halved, which suppresses cloning and creates holes at
    // the object boundary.
    void Trainer::focused_segment_apply_densification_mask(
        lfs::core::Tensor& error_map,
        const lfs::core::Tensor& mask_tile) {

        if constexpr (FOCUSED_DENSIFY_DILATE_RADIUS > 0) {
            constexpr int kDensifyDilateRadius = FOCUSED_DENSIFY_DILATE_RADIUS;
            constexpr int kDensifyDilateKernel = 2 * kDensifyDilateRadius + 1;
            auto mask_4d = mask_tile.unsqueeze(0).unsqueeze(0); // [1,1,H,W]
            auto dilated = mask_4d.max_pool2d(kDensifyDilateKernel, 1, kDensifyDilateRadius);
            auto dilated_2d = dilated.squeeze(0).squeeze(0); // [H,W]
            error_map.mul_(dilated_2d).contiguous();
        } else {
            error_map.mul_(mask_tile).contiguous();
        }
    }

    // Post-training mask-based pruning for FocusedSegment mode.
    //
    // Runs the full pruning pipeline (geometric dome -> center vote -> leakage ->
    // cluster/extreme -> ellipse boundary -> isolation) using thresholds tuned for
    // the FocusedSegment use case: relaxed vote/leak thresholds that protect
    // legitimate border splats (feet, arms, fine hair) while still removing
    // halos and floaters.
    //
    // The commented-out call at the end of Trainer::train() is intentional -
    // re-enable when the post-training prune phase is desired.
    void Trainer::focused_segment_run_post_training_prune(
        const lfs::core::param::MaskMode mask_mode,
        const bool invert_masks,
        lfs::training::IStrategy& strategy,
        lfs::training::CameraDataset& train_dataset) {

        if (mask_mode != lfs::core::param::MaskMode::FocusedSegment) {
            return;
        }

        mask_pruning::GeometricDomePruningConfig geomdome_cfg;
        // default values are ok

        mask_pruning::CenterVotePruningConfig center_cfg;
        center_cfg.enabled = true;

        // Conservative threshold - protects legitimate border splats (feet, arms).
        // Only removes Gaussians clearly outside the mask in a large majority of views.
        // In a dome with 104 cameras, a splat visible in 30 views can afford 8 bad-mask
        // views and still pass (73% good > 0.72).
        center_cfg.vote_ratio_threshold = 0.8f;
        // Moderate margin - avoids penalizing splats near frustum edges without
        // being as permissive as the original 0.25 that missed lateral floaters.
        center_cfg.border_safe_margin = 0.33f;
        center_cfg.enable_depth_filtering = true;
        // Dome-oriented rule: require visibility in at least 10% of usable cameras.
        // With 104 usable cameras this means 11+ views, which is much stricter than
        // the old fixed threshold and helps remove floaters seen only in a handful
        // of side views.
        center_cfg.min_visibility_ratio = 0.35f;
        center_cfg.invert_masks = invert_masks;

        mask_pruning::LeakagePruningConfig leak_cfg;
        leak_cfg.enabled = true;
        // Main tool for halos - removes elongated Gaussians whose footprint
        // extends outside the mask. Center vote is too permissive for these.
        leak_cfg.leak_keep_threshold = 0.70f;
        // 2 of 8 sample points outside counts as a leak per view.
        // Catches elongated splats extending above heads without being too strict
        // on border splats that legitimately straddle the mask edge.
        leak_cfg.per_view_leak_fraction = 0.25f;
        // Require decisive good/bad leakage evidence in at least 10% of usable cameras.
        leak_cfg.min_visibility_ratio = 0.10f;
        // Low radius - evaluate small/thin elongated splats that caused halos.
        // Original 2.0f missed these entirely.
        leak_cfg.min_pixel_radius = 1.0f;
        leak_cfg.sample_points = 8;
        // Small dilation - tolerates 3px at mask boundary to protect
        // extremities that straddle the mask edge in some views.
        leak_cfg.dilate_px = 2;
        leak_cfg.invert_masks = invert_masks;

        mask_pruning::IsolationPruningConfig iso_cfg;
        iso_cfg.enabled = true;

        LOG_INFO("Running post-training mask-based pruning...");

        auto pruning_result = mask_pruning::prune_after_training(
            strategy,
            train_dataset,
            geomdome_cfg,
            center_cfg,
            leak_cfg,
            iso_cfg);

        if (!pruning_result) {
            LOG_WARN("Post-training pruning failed: {}", pruning_result.error());
        } else if (pruning_result->splats_removed > 0) {
            LOG_INFO("Pruning complete: removed {} splats ({:.1f}%)",
                     pruning_result->splats_removed,
                     pruning_result->removal_ratio() * 100.0f);
        }
    }

} // namespace lfs::training
