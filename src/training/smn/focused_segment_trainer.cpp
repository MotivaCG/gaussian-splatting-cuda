/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "training/trainer.hpp"

#include "core/logger.hpp"
#include "lfs/kernels/ssim.cuh"
#include "optimizer/adam_optimizer.hpp"
#include "smn/focused_segment_profiling.hpp"
#include "smn/mask_pruning.hpp"

#include <algorithm>
#include <cstdio>
#include <string>

namespace lfs::training {

    namespace {

        // =====================================================================
        // FocusedSegment compile-time feature switches.
        //
        // All flags default to true (current production behavior). Setting one
        // to false compiles out that piece via `if constexpr` or via a zero
        // multiplier on a weight constant. Grouped so that alpha pressure
        // inside the mask (transparency penalty) and opacity pressure outside
        // the mask (opacity penalty) can be toggled independently at BOTH the
        // pixel level (grad_alpha) and the splat level (center penalty),
        // giving 4 fully independent knobs.
        // =====================================================================

        // Time-varying weight schedule. When false the 7-phase ramp is skipped
        // entirely: FocusedSegment is active from iter 0 and weights stay at
        // their CLI-provided values for the whole training.
        constexpr bool FOCUSED_ENABLE_SCHEDULE = true;

        // Pixel-level loss shaping (focused_segment_compute_loss).
        constexpr bool FOCUSED_ENABLE_SPATIAL_WEIGHT_MAP = true; // FG=1, BG=focused_bg_weight on grad_image
        constexpr bool FOCUSED_ENABLE_DARKNESS_BOOST     = true; // GT-luminance darkness boost on FG region
        constexpr bool FOCUSED_ENABLE_GRAD_ALPHA_FG      = true; // push FG alpha->1 (force opacity inside; fills holes)
        constexpr bool FOCUSED_ENABLE_GRAD_ALPHA_BG      = true; // push BG alpha->0 (force transparency outside; removes halos)

        // Densification (focused_segment_apply_densification_mask).
        // FOCUSED_DENSIFY_DILATE_RADIUS below still controls the dilation size.
        constexpr bool FOCUSED_ENABLE_DENSIFY_DILATION = true;

        // Splat-level center penalty (focused_segment_apply_center_penalty).
        constexpr bool FOCUSED_ENABLE_CENTER_PENALTY         = true; // master switch (skips hook entirely)
        constexpr bool FOCUSED_ENABLE_CENTER_PENALTY_FG_FILL = true; // push opacity UP inside mask
        constexpr bool FOCUSED_ENABLE_CENTER_PENALTY_BG      = true; // push opacity DOWN outside mask

        // Post-training mask-based pruning pipeline.
        constexpr bool FOCUSED_ENABLE_POST_TRAINING_PRUNE = true;

        // Per-pass switches for the post-training prune. Each can be disabled
        // independently to A/B test the impact of that specific pass. Only
        // consulted when FOCUSED_ENABLE_POST_TRAINING_PRUNE is true.
        // Disable an individual pass to skip it without commenting code out.
        constexpr bool FOCUSED_ENABLE_PRUNE_GEOMETRIC_DOME   = true;
        constexpr bool FOCUSED_ENABLE_PRUNE_CENTER_VOTE      = true;
        constexpr bool FOCUSED_ENABLE_PRUNE_MASK_LEAKAGE     = true;
        constexpr bool FOCUSED_ENABLE_PRUNE_ISOLATION        = true;
        constexpr bool FOCUSED_ENABLE_PRUNE_ELLIPSE_BOUNDARY = true;
        constexpr bool FOCUSED_ENABLE_PRUNE_CLUSTER_EXTREME  = false; // Off by default (slow + experimental)

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
        //
        // For dome captures of people, 3 provides enough border coverage for
        // concave zones (between legs, armpits) to accumulate densification error
        // and spawn small splats that define the silhouette precisely. The
        // post-training prune removes any excess.
        constexpr int FOCUSED_DENSIFY_DILATE_RADIUS = 3;

        lfs::core::Tensor focused_segment_mask_float(const lfs::core::Tensor& mask) {
            using namespace lfs::core;

            if (!mask.is_valid()) {
                return {};
            }

            if (mask.dtype() == DataType::Bool) {
                return mask.to(DataType::Float32).contiguous();
            }

            const Tensor mask_f = mask.to(DataType::Float32);
            if (mask.dtype() == DataType::UInt8) {
                // Loaders differ by platform/path: some return UInt8 masks as 0/1,
                // others as 0/255. Preserve 0/1 masks and scale image-range values.
                return Tensor::where(mask_f.le(1.0f), mask_f, mask_f * (1.0f / 255.0f))
                    .clamp(0.0f, 1.0f)
                    .contiguous();
            }

            return mask_f.clamp(0.0f, 1.0f).contiguous();
        }

        lfs::core::Tensor focused_segment_image_float01(const lfs::core::Tensor& image) {
            using namespace lfs::core;

            if (!image.is_valid()) {
                return {};
            }

            if (image.dtype() == DataType::UInt8) {
                return (image.to(DataType::Float32) * (1.0f / 255.0f)).contiguous();
            }

            return image.to(DataType::Float32).clamp(0.0f, 1.0f).contiguous();
        }

    } // namespace

    // =========================================================================
    // SMN: FocusedSegment per-camera cache getter.
    //
    // BELONGS TO SMN. Lazily populates the entry for `cam` on first hit:
    //   - mask_bool      : src_mask -> thresholded bool [H, W]
    //   - fg/bg_pixels   : scalars (one-time GPU sync only on first hit)
    //   - darkness_fp16  : 1 - Rec.601 luminance(gt_image), stored FP16
    //                      (only if want_lightness is true and not already cached)
    //
    // On subsequent hits returns the existing entry with no GPU work. This
    // eliminates the per-iter focused_segment_mask_float() call and the
    // .sum().item<float>() sync that otherwise stall the pipeline every step.
    //
    // See smn/focused_segment_cache.hpp for the data layout and rationale.
    // =========================================================================
    const lfs::training::smn::FocusedSegmentCameraCache& Trainer::focused_segment_get_cache(
        const lfs::core::Camera& cam,
        const lfs::core::Tensor& src_mask,
        const lfs::core::Tensor& gt_image,
        bool want_lightness) {

        using namespace lfs::core;
        auto& entry = fs_camera_cache_[cam.uid()];

        if (!entry.has_mask()) {
            const lfs::training::smn::ScopedFsTimer _t(
                fs_timings_, lfs::training::smn::FsTimerSlot::CacheBuildMask);
            const Tensor mask_f = focused_segment_mask_float(src_mask);
            entry.mask_bool = mask_f.gt(0.5f).contiguous();
            const float fg = std::max(mask_f.sum().item<float>(), 1.0f); // one-time sync
            entry.fg_pixels = fg;
            entry.bg_pixels = std::max(static_cast<float>(mask_f.numel()) - fg, 1.0f);
        }

        if (want_lightness && !entry.has_darkness() && gt_image.is_valid() && gt_image.numel() > 0) {
            const lfs::training::smn::ScopedFsTimer _t(
                fs_timings_, lfs::training::smn::FsTimerSlot::CacheBuildDarkness);
            const Tensor gt_norm = focused_segment_image_float01(gt_image);
            const bool chw = (gt_norm.ndim() == 3 && gt_norm.shape()[0] == 3);
            const Tensor r = chw ? gt_norm.slice(0, 0, 1).squeeze(0) : gt_norm.slice(2, 0, 1).squeeze(2);
            const Tensor g = chw ? gt_norm.slice(0, 1, 2).squeeze(0) : gt_norm.slice(2, 1, 2).squeeze(2);
            const Tensor b = chw ? gt_norm.slice(0, 2, 3).squeeze(0) : gt_norm.slice(2, 2, 3).squeeze(2);
            const Tensor brightness = r * 0.299f + g * 0.587f + b * 0.114f;
            const Tensor darkness = Tensor::full(brightness.shape(), 1.0f, brightness.device()) - brightness;

            // Mean of darkness restricted to FG region. Lets the per-iter
            // spatial-weight block compute weight_mean analytically without
            // any GPU sync. One-time cost (single .item<float>()) per camera.
            const Tensor mask_f_for_mean = entry.mask_bool.to(DataType::Float32);
            const float sum_dark_fg = (mask_f_for_mean * darkness).sum().item<float>();
            entry.mean_darkness_fg = sum_dark_fg / entry.fg_pixels;

            entry.darkness_fp16 = darkness.to(DataType::Float16).contiguous();
        }

        return entry;
    }

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

        // NOTE: no ScopedFsTimer here. This function is CPU-only (a few `if`s
        // and scalar arithmetic), and the 2 cudaDeviceSynchronize calls a timer
        // would inject not only inflate the reported time but also force the
        // GPU pipeline to drain twice per iter for nothing. The slot
        // FsTimerSlot::ApplySchedule stays in the enum but is never populated;
        // the printer skips empty slots so the report ignores it.

        if constexpr (!FOCUSED_ENABLE_SCHEDULE) {
            return; // skip the time-varying ramp; params stay at CLI-provided values
        }

        constexpr float kNoneEnd = 0.15f; // End of warm-up with masking disabled.
        constexpr float kBgSpatialRampEnd = 0.25f; // End of BG spatial weighting ramp.
        constexpr float kBgTarget = 0.05f; // Active-phase BG gradient weight (FG focused)
        constexpr float kBgTargetFree = 0.08f; // Free-refinement BG weight - slightly higher
                                               // than kBgTarget to allow gentle BG refinement
                                               // without reintroducing detail. Kept low to help
                                               // fine silhouettes (hair) stay sharp.
        step_params.focused_bg_weight = kBgTarget;

        constexpr float kFgAlphaRampStart = 0.30f; // Start ramping FG alpha pressure up.
        constexpr float kFgAlphaRampEnd = 0.40f; // FG alpha pressure reaches full strength.

        constexpr float kBgAlphaRampStart = 0.50f; // Start ramping BG alpha pressure up.
        constexpr float kBgAlphaRampEnd = 0.60f; // BG alpha pressure reaches full strength; both FG+BG fully active.

        constexpr float kAlphaDecayStart = 0.75f; // Start relaxing alpha pressure toward residual floors.
        constexpr float kAlphaDecayEnd = 0.85f; // End of decay; residual floors remain until training ends.

        constexpr float kFgAlphaFloorValue = 0.0f;  // FG residual alpha penalty after decay.
        constexpr float kBgAlphaFloorValue = 0.20f; // BG residual alpha penalty after decay.
                                                     // Prevents BG opacity from creeping back
                                                     // in concave regions (between legs, armpits)
                                                     // during the free refinement phase.

        static_assert(kBgSpatialRampEnd < kFgAlphaRampStart, "BG spatial ramp must finish before FG alpha penalty starts");
        static_assert(kFgAlphaRampEnd <= kBgAlphaRampStart, "FG alpha penalty must reach full before BG joins");
        static_assert(kBgAlphaRampEnd <= kAlphaDecayStart, "Both at full before decay starts");

        if (progress < kNoneEnd) {
            step_params.mask_mode = lfs::core::param::MaskMode::None;
            step_params.focused_bg_weight = 1.0f;
        } else if (progress < kBgSpatialRampEnd) {
            step_params.mask_opacity_penalty_weight = 0.0f;
            step_params.mask_opacity_penalty_weight_bg = 0.0f;
        } else if (progress < kFgAlphaRampStart) {
            step_params.mask_opacity_penalty_weight = 0.0f;
            step_params.mask_opacity_penalty_weight_bg = 0.0f;
        } else if (progress < kFgAlphaRampEnd) {
            const float t = (progress - kFgAlphaRampStart) / (kFgAlphaRampEnd - kFgAlphaRampStart);
            step_params.mask_opacity_penalty_weight *= t;
            step_params.mask_opacity_penalty_weight_bg = 0.0f;
        } else if (progress < kBgAlphaRampEnd) {
            const float t_bg = (progress - kBgAlphaRampStart) / (kBgAlphaRampEnd - kBgAlphaRampStart);
            step_params.mask_opacity_penalty_weight_bg *= std::clamp(t_bg, 0.0f, 1.0f);
        } else if (progress < kAlphaDecayEnd) {
            const float t_decay = std::clamp(
                (progress - kAlphaDecayStart) / (kAlphaDecayEnd - kAlphaDecayStart),
                0.0f,
                1.0f);
            const float fg_factor = 1.0f + (kFgAlphaFloorValue - 1.0f) * t_decay;
            const float bg_factor = 1.0f + (kBgAlphaFloorValue - 1.0f) * t_decay;
            step_params.mask_opacity_penalty_weight *= fg_factor;
            step_params.mask_opacity_penalty_weight_bg *= bg_factor;
            step_params.focused_bg_weight = kBgTarget * (1.0f - t_decay) + kBgTargetFree * t_decay;
        } else {
            step_params.mask_opacity_penalty_weight *= kFgAlphaFloorValue;
            step_params.mask_opacity_penalty_weight_bg *= kBgAlphaFloorValue;
            step_params.focused_bg_weight = kBgTargetFree;
        }

        // Darkness boost schedule: 0 during warmup, ramp up to Max while BG spatial ramp
        // activates, then decay linearly to Min for the rest of training.
        constexpr float kDarknessBoostMax = 3.0f;
        constexpr float kDarknessBoostMin = 1.0f;
        if (progress < kNoneEnd) {
            step_params.darkness_boost = 0.0f;
        } else if (progress < kBgSpatialRampEnd) {
            const float t = (progress - kNoneEnd) / (kBgSpatialRampEnd - kNoneEnd);
            step_params.darkness_boost = kDarknessBoostMax * t;
        } else {
            const float t = (progress - kBgSpatialRampEnd) / (1.0f - kBgSpatialRampEnd);
            step_params.darkness_boost = kDarknessBoostMax - (kDarknessBoostMax - kDarknessBoostMin) * t;
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
    // When raw_rendered is provided (PPISP / appearance correction active),
    // FocusedSegment uses the same decoupled L1+SSIM path as unmasked training:
    // L1/color gradients flow through corrected, while SSIM/structure gradients
    // flow directly to raw_rendered.
    std::expected<Trainer::MaskLossResult, std::string> Trainer::focused_segment_compute_loss(
        const lfs::core::Camera& cam,
        const lfs::core::Tensor& corrected,
        const lfs::core::Tensor& raw_rendered,
        const lfs::core::Tensor& gt_image,
        const lfs::core::Tensor& mask_2d,
        const lfs::core::Tensor& alpha,
        const lfs::core::param::OptimizationParameters& opt_params) {

        const lfs::training::smn::ScopedFsTimer _t_total(
            fs_timings_, lfs::training::smn::FsTimerSlot::ComputeLossTotal);

        using namespace lfs::core;

        const float kBgWeight = opt_params.focused_bg_weight; // BG gradient weight relative to FG (1.0)
        constexpr float kAlphaFgWeight = 1.5f; // grad_alpha pressure to push FG alpha -> 1
        constexpr float kAlphaBgWeight = 1.0f; // grad_alpha pressure to push BG alpha -> 0

        // SMN per-camera memoization: builds mask_bool, fg/bg pixel counts and,
        // when darkness_boost is in use, the FP16 darkness map on first hit per
        // camera. Replaces both the per-iter focused_segment_mask_float() work
        // and the .sum().item<float>() GPU sync that would otherwise stall the
        // pipeline every iteration. See smn/focused_segment_cache.hpp.
        const bool want_lightness = FOCUSED_ENABLE_DARKNESS_BOOST && opt_params.darkness_boost > 0.0f;
        const auto& fs_cache = focused_segment_get_cache(cam, mask_2d, gt_image, want_lightness);

        const Tensor mask_f = fs_cache.mask_bool.to(DataType::Float32);
        const Tensor bg_mask = Tensor::full(mask_f.shape(), 1.0f, mask_f.device()) - mask_f;

        // Step 1: full-image L1+SSIM forward - identical to None mode, with
        // decoupled appearance gradients when raw_rendered is available.
        const bool use_decoupled = raw_rendered.is_valid() &&
                                   raw_rendered.numel() > 0 &&
                                   opt_params.lambda_dssim > 0.0f;

        Tensor grad;
        Tensor grad_raw;
        Tensor loss;
        {
            const lfs::training::smn::ScopedFsTimer _t_photo(
                fs_timings_, lfs::training::smn::FsTimerSlot::ComputeLossPhoto);
            if (use_decoupled) {
                auto [loss_tensor, ctx] = lfs::training::kernels::decoupled_fused_l1_ssim_forward(
                    corrected, raw_rendered, gt_image, opt_params.lambda_dssim, decoupled_fused_workspace_,
                    /*apply_valid_padding=*/true);
                auto grads = lfs::training::kernels::decoupled_fused_l1_ssim_backward(ctx, decoupled_fused_workspace_);

                grad = grads.grad_corrected;
                grad_raw = grads.grad_raw;
                if (grad.ndim() == 4 && corrected.ndim() == 3) {
                    grad = grad.squeeze(0);
                }
                if (grad_raw.ndim() == 4 && corrected.ndim() == 3) {
                    grad_raw = grad_raw.squeeze(0);
                }
                loss = loss_tensor;
            } else {
                losses::PhotometricLoss::Params params{.lambda_dssim = opt_params.lambda_dssim};
                auto full_result = photometric_loss_.forward(corrected, gt_image, params);
                if (!full_result) {
                    return std::unexpected(full_result.error());
                }
                auto [full_loss, ctx] = *full_result;
                grad = ctx.grad_image;
                loss = full_loss;
            }
        }

        // Pixel counts come from the SMN cache (no GPU sync). Total is just the
        // tensor numel (metadata access, no kernel).
        const float total_pixels = static_cast<float>(mask_f.numel());
        const float fg_pixels = fs_cache.fg_pixels;
        const float bg_pixels = fs_cache.bg_pixels;

        // Step 2+3: spatial weight map - FG=1.0, BG=kBgWeight applied to grad.
        // Darkness bonus applied to FG only (Rec.601 perceptual luminance from
        // GT, not rendered). Using GT keeps the weight map stable across
        // iterations - rendered changes every step. BG stays flat at kBgWeight
        // regardless of darkness, avoiding spurious BG gradient boosts.
        // Multiplying grad post-hoc is exact for L1 (pixel-wise) and a good
        // approximation for SSIM (window-based); in practice this outperforms
        // discarding SSIM gradient entirely.
        if constexpr (FOCUSED_ENABLE_SPATIAL_WEIGHT_MAP) {
            const lfs::training::smn::ScopedFsTimer _t_sw(
                fs_timings_, lfs::training::smn::FsTimerSlot::ComputeLossSpatialWeight);
            Tensor weight_map;
            if (want_lightness && fs_cache.has_darkness()) {
                // Darkness comes from the SMN cache as FP16 (~6 MB/cam at 1296x2304);
                // promote to FP32 once for the per-iter math.
                const Tensor darkness = fs_cache.darkness_fp16.to(DataType::Float32);
                weight_map =
                    mask_f * (Tensor::full(darkness.shape(), 1.0f, darkness.device()) + darkness * opt_params.darkness_boost) +
                    bg_mask * kBgWeight;
            } else {
                weight_map = mask_f + bg_mask * kBgWeight;
            }

            // Normalize weight_map by its mean so the global gradient magnitude
            // stays comparable to None mode - prevents scale drift when mask
            // size varies across scenes.
            //
            // weight_mean computed analytically from cached scalars (no GPU sync).
            // With darkness boost:
            //   sum(weight_map) = fg * (1 + boost*<darkness>_FG) + bg * kBgWeight
            // Without darkness boost:
            //   sum(weight_map) = fg + bg * kBgWeight
            // weight_mean = sum(weight_map) / total_pixels
            const float weight_mean = (want_lightness && fs_cache.has_darkness())
                ? (fs_cache.fg_pixels * (1.0f + opt_params.darkness_boost * fs_cache.mean_darkness_fg)
                   + fs_cache.bg_pixels * kBgWeight) / total_pixels
                : (fs_cache.fg_pixels + fs_cache.bg_pixels * kBgWeight) / total_pixels;
            weight_map = weight_map * (1.0f / std::max(weight_mean, 1e-4f));

            const Tensor weight_3d = (corrected.ndim() == 3 && corrected.shape()[0] == 3)
                                         ? weight_map.unsqueeze(0)
                                         : weight_map.unsqueeze(2);
            grad = grad * weight_3d;
            if (grad_raw.is_valid() && grad_raw.numel() > 0) {
                grad_raw = grad_raw * weight_3d;
            }

            // Rescale scalar loss so logged values track FG reconstruction
            // quality, not diluted by the large BG area.
            loss = loss * (fg_pixels / std::max(total_pixels * weight_mean, 1e-6f));
        }

        // Step 4: pixel-based alpha pressure via grad_alpha. Constant per-pixel
        // magnitude (no scaling by (1-alpha) or alpha) so the natural gradient
        // accumulation across views performs an implicit majority vote: each
        // FG pixel contributes -w_fg*k/fg_pixels to alpha, each BG pixel
        // contributes +w_bg*k/bg_pixels. If a region is labelled FG in more
        // views than BG, the net accumulated gradient drives the covering
        // splats toward opacity, and vice versa, with no per-iteration trick.
        //
        // Earlier formulations scaled by (1-alpha)/alpha for "self-regulation"
        // (zero pressure when already correct, max when wrong). That broke
        // majority vote: at alpha=0.9 a BG vote contributed 0.9 while an FG
        // vote only 0.1, so a few wrong masks could outweigh many correct ones
        // even when alpha was already high. Trusting the gradient accumulation
        // gives the expected behavior: more votes win, period.
        //
        // Per-area normalization (/fg_pixels, /bg_pixels) keeps the total
        // scalar pressure bounded by the user-set weights so neither side
        // overwhelms the photometric loss. The asymmetry it introduces
        // (each FG pixel matters more than each BG pixel because fg_pixels
        // << bg_pixels in dome captures) is intentional: the subject is small,
        // each FG pixel carries more information.
        //
        // NOTE: modifying the scalar `loss` does NOT affect Gaussian parameters
        // - only grad_image and grad_alpha propagate through rasterize_backward
        // to the optimizer. Alpha pressure is therefore applied purely through
        // grad_alpha.
        //
        // Sign convention (gradient descent: param -= lr * grad):
        //   FG: negative grad_alpha  -> alpha_raw increases -> rendered alpha approaches 1
        //   BG: positive grad_alpha  -> alpha_raw decreases -> rendered alpha approaches 0
        Tensor grad_alpha;
        const float w_fg = opt_params.mask_opacity_penalty_weight;
        const float w_bg = opt_params.mask_opacity_penalty_weight_bg;
        const bool apply_fg = FOCUSED_ENABLE_GRAD_ALPHA_FG && w_fg > 0.0f;
        const bool apply_bg = FOCUSED_ENABLE_GRAD_ALPHA_BG && w_bg > 0.0f;
        if (alpha.is_valid() && (apply_fg || apply_bg)) {
            const lfs::training::smn::ScopedFsTimer _t_ga(
                fs_timings_, lfs::training::smn::FsTimerSlot::ComputeLossGradAlpha);
            Tensor fg_term;
            if (apply_fg) {
                fg_term = mask_f * (w_fg * kAlphaFgWeight / fg_pixels);
            }

            Tensor bg_term;
            if (apply_bg) {
                bg_term = bg_mask * (w_bg * kAlphaBgWeight / bg_pixels);
            }

            if (bg_term.is_valid() && fg_term.is_valid()) {
                grad_alpha = bg_term - fg_term;
            } else if (bg_term.is_valid()) {
                grad_alpha = bg_term;
            } else {
                grad_alpha = fg_term * (-1.0f);
            }
        }

        return Trainer::MaskLossResult{
            .loss = loss,
            .grad_corrected = grad,
            .grad_raw = grad_raw,
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

        const lfs::training::smn::ScopedFsTimer _t(
            fs_timings_, lfs::training::smn::FsTimerSlot::ApplyDensifyMask);

        if constexpr (FOCUSED_ENABLE_DENSIFY_DILATION && FOCUSED_DENSIFY_DILATE_RADIUS > 0) {
            constexpr int kDensifyDilateRadius = FOCUSED_DENSIFY_DILATE_RADIUS;
            constexpr int kDensifyDilateKernel = 2 * kDensifyDilateRadius + 1;
            auto mask_4d = focused_segment_mask_float(mask_tile).unsqueeze(0).unsqueeze(0); // [1,1,H,W]
            auto dilated = mask_4d.max_pool2d(kDensifyDilateKernel, 1, kDensifyDilateRadius);
            auto dilated_2d = dilated.squeeze(0).squeeze(0); // [H,W]
            error_map.mul_(dilated_2d).contiguous();
        } else {
            error_map.mul_(focused_segment_mask_float(mask_tile)).contiguous();
        }
    }

    // =========================================================================
    // Per-splat center-based opacity penalty.
    //
    // Projects each Gaussian center to 2D, samples the mask, and directly
    // pushes opacity down for splats whose center lands outside the mask.
    //
    // This complements the pixel-level grad_alpha in focused_segment_compute_loss:
    //  - grad_alpha:     pixel-level, distributed to splats by rasterizer backward.
    //                    Large splats spanning FG/BG get mixed signals.
    //  - center_penalty: per-splat, based only on the center projection.
    //                    A large splat with center outside the mask gets a direct
    //                    unambiguous "reduce opacity" gradient.
    //
    // This targets concave mask zones (between legs, armpits) where large
    // gaussians with centers in the gap create false opacity.
    // =========================================================================
    void Trainer::focused_segment_apply_center_penalty(
        const lfs::core::Tensor& mask,
        const lfs::core::Camera& cam,
        const lfs::core::param::OptimizationParameters& step_params) {

        using namespace lfs::core;

        if (step_params.mask_mode != param::MaskMode::FocusedSegment)
            return;

        const lfs::training::smn::ScopedFsTimer _t(
            fs_timings_, lfs::training::smn::FsTimerSlot::ApplyCenterPenalty);

        const float w_bg = step_params.mask_opacity_penalty_weight_bg;
        if (w_bg <= 0.0f)
            return;

        // If both per-direction switches are off, the resulting gradient
        // would be all zeros; bail before doing any projection work.
        if constexpr (!FOCUSED_ENABLE_CENTER_PENALTY_BG && !FOCUSED_ENABLE_CENTER_PENALTY_FG_FILL)
            return;

        if (!mask.is_valid() || mask.numel() == 0)
            return;

        auto& model = strategy_->get_model();
        auto& optimizer = strategy_->get_optimizer();
        const size_t N = model.size();
        if (N == 0)
            return;

        const int imgW = cam.image_width();
        const int imgH = cam.image_height();
        if (imgW <= 0 || imgH <= 0)
            return;
        const size_t W = static_cast<size_t>(imgW);
        const size_t H = static_cast<size_t>(imgH);

        // ---- 1. Project means3D to camera space ----
        auto means3d = model.get_means(); // [N, 3] CUDA
        auto wvt = cam.world_view_transform(); // [4, 4] or [1, 4, 4] CUDA
        if (wvt.ndim() == 3)
            wvt = wvt.squeeze(0); // ensure [4, 4]

        // Homogeneous coords: [N, 4]
        auto means_h = Tensor::cat(
            {means3d, Tensor::ones({N, size_t(1)}, Device::CUDA)}, 1);

        // Camera space: cam = means_h @ W2V^T -> [N, 4]
        auto cam_space = means_h.mm(wvt.t());

        // Depth: z coordinate
        auto depth = cam_space.slice(1, 2, 3).squeeze(1); // [N]

        // Visible: depth > near_plane
        constexpr float kNearPlane = 0.01f;
        auto visible_mask = depth.gt(kNearPlane); // [N] bool

        // ---- 2. Pinhole projection to 2D pixel coords ----
        const float fx = cam.focal_x();
        const float fy = cam.focal_y();
        const float cx = cam.center_x();
        const float cy = cam.center_y();

        auto depth_safe = depth.clamp_min(kNearPlane); // avoid div-by-zero
        auto x_proj = cam_space.slice(1, 0, 1).squeeze(1) * fx / depth_safe + cx; // [N]
        auto y_proj = cam_space.slice(1, 1, 2).squeeze(1) * fy / depth_safe + cy; // [N]

        const float Wf = static_cast<float>(W);
        const float Hf = static_cast<float>(H);

        // Build a validity mask before indexing. Invalid/NaN projections are
        // replaced with zero so the gather index is always inside [0, H*W).
        auto in_x = x_proj.ge(0.0f).logical_and(x_proj.lt(Wf));
        auto in_y = y_proj.ge(0.0f).logical_and(y_proj.lt(Hf));
        auto valid = visible_mask.logical_and(in_x.logical_and(in_y)); // [N] bool

        auto zero_n = Tensor::zeros({N}, Device::CUDA);
        auto x_safe = Tensor::where(valid, x_proj, zero_n);
        auto y_safe = Tensor::where(valid, y_proj, zero_n);

        auto x_px = x_safe.round().clamp(0.0f, Wf - 1.0f); // [N]
        auto y_px = y_safe.round().clamp(0.0f, Hf - 1.0f); // [N]

        // ---- 3. Sample mask at projected centers ----
        // Pull from the SMN per-camera cache: avoids re-running
        // focused_segment_mask_float() per iter and shares the bool buffer
        // already populated by focused_segment_compute_loss.
        const auto& fs_cache = focused_segment_get_cache(cam, mask, lfs::core::Tensor(), /*want_lightness=*/false);
        auto lin_idx = (y_px * Wf + x_px).to(DataType::Int64); // [N] int64
        auto mask_flat = fs_cache.mask_bool.to(DataType::Float32).reshape({static_cast<int>(H * W)}); // [H*W]
        auto is_inside = mask_flat.gather(0, lin_idx); // [N] float {0..1}

        // ---- 4. Compute opacity gradient ----
        // For splats outside the mask: positive gradient pushes raw opacity down (sigmoid decreases).
        // For splats inside the mask: small negative gradient fills transparency holes.
        constexpr float kOutWeight = FOCUSED_ENABLE_CENTER_PENALTY_BG      ? 1.5f  : 0.0f; // BG center penalty strength
        constexpr float kInWeight  = FOCUSED_ENABLE_CENTER_PENALTY_FG_FILL ? 0.05f : 0.0f; // gentle FG fill (most work done by grad_alpha)

        auto is_outside = (Tensor::full({N}, 1.0f, Device::CUDA) - is_inside);
        auto valid_f = valid.to(DataType::Float32); // [N] float {0, 1}

        // Gradient w.r.t. activated opacity (sigmoid output)
        auto grad_activated = valid_f *
            (is_outside * kOutWeight - is_inside * kInWeight) *
            (w_bg / static_cast<float>(N));  // normalize by splat count

        // Chain rule: d(loss)/d(raw) = d(loss)/d(sigmoid) * sigmoid'(raw)
        //           = grad_activated * sigmoid(raw) * (1 - sigmoid(raw))
        auto opacity_raw = model.opacity_raw(); // [N, 1]
        auto raw_1d = opacity_raw.ndim() == 2 ? opacity_raw.squeeze(1) : opacity_raw; // [N]
        auto sig = raw_1d.sigmoid(); // [N]
        auto grad_raw = grad_activated * sig * (Tensor::full({N}, 1.0f, Device::CUDA) - sig);

        // ---- 5. Accumulate into optimizer gradient buffer ----
        auto& opacity_grad = optimizer.get_grad(ParamType::Opacity); // [N, 1]
        if (opacity_grad.ndim() == 2 && grad_raw.ndim() == 1) {
            opacity_grad.add_(grad_raw.unsqueeze(1));
        } else {
            opacity_grad.add_(grad_raw);
        }
    }

    // =========================================================================
    // Post-backward hook: resolves the mask and dispatches per-splat penalties.
    //
    // This is the single call site from trainer.cpp (one-liner), keeping the
    // upstream file minimal. All mask resolution and penalty logic stays here.
    // =========================================================================
    void Trainer::focused_segment_post_backward(
        const lfs::core::Tensor& pipelined_mask,
        lfs::core::Camera& cam,
        int iter,
        int num_tiles) {

        if (params_.optimization.mask_mode != lfs::core::param::MaskMode::FocusedSegment)
            return;

        const lfs::training::smn::ScopedFsTimer _t(
            fs_timings_, lfs::training::smn::FsTimerSlot::PostBackward);

        if constexpr (!FOCUSED_ENABLE_CENTER_PENALTY)
            return;

        // Center penalty only makes sense with full image (single tile)
        if (num_tiles != 1)
            return;

        // Build step_params with schedule applied for this iteration before
        // touching the mask cache. Most early FocusedSegment phases do not apply
        // the center penalty, so loading a mask there is wasted work.
        lfs::core::param::OptimizationParameters step_params = params_.optimization;
        const float progress = static_cast<float>(iter) /
                               static_cast<float>(params_.optimization.iterations);
        focused_segment_apply_schedule(step_params, progress);

        if (step_params.mask_mode != lfs::core::param::MaskMode::FocusedSegment ||
            step_params.mask_opacity_penalty_weight_bg <= 0.0f) {
            return;
        }

        // MCMC can carry millions of splats around the 50% FocusedSegment phase.
        // The center penalty projects every splat, so running it every iteration
        // can dominate training. Apply it sparsely and scale the gradient to keep
        // the average pressure comparable.
        constexpr int kMcmcCenterPenaltyInterval = 8;
        const bool is_mcmc = lfs::core::param::canonical_strategy_name(params_.optimization.strategy) ==
                             lfs::core::param::kStrategyMCMC;
        if (is_mcmc) {
            if ((iter % kMcmcCenterPenaltyInterval) != 0) {
                return;
            }
            step_params.mask_opacity_penalty_weight_bg *= static_cast<float>(kMcmcCenterPenaltyInterval);
        }

        lfs::core::Tensor full_mask;
        if (pipelined_mask.is_valid() && pipelined_mask.numel() > 0) {
            full_mask = pipelined_mask;
        } else {
            full_mask = cam.load_and_get_mask(
                params_.dataset.resize_factor,
                params_.dataset.max_width,
                params_.optimization.invert_masks,
                params_.optimization.mask_threshold);
        }

        if (!full_mask.is_valid() || full_mask.numel() == 0)
            return;

        focused_segment_apply_center_penalty(full_mask, cam, step_params);
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

        if constexpr (!FOCUSED_ENABLE_POST_TRAINING_PRUNE) {
            return;
        }

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
        leak_cfg.leak_keep_threshold = 0.60f;
        // 1-2 of 8 sample points outside counts as a leak per view.
        // More sensitive to thin elongated spikes that poke outside the silhouette.
        leak_cfg.per_view_leak_fraction = 0.20f;
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

        // Ellipse boundary: removes Gaussians whose 2D ellipse boundary points
        // extend outside an expanded mask. Catches thin spikes that leakage misses
        // because their center is inside the mask.
        mask_pruning::EllipseBoundaryPruningConfig ellipse_cfg;
        ellipse_cfg.enabled = true;
        ellipse_cfg.mask_expansion_fraction = 0.02f; // ~4px at 2K - tighter than default to catch thin spikes
        ellipse_cfg.negative_vote_threshold = 0.10f; // remove if >=10% of evaluating cameras flagged leakage
        ellipse_cfg.min_evaluating_cameras = 3;
        ellipse_cfg.invert_masks = invert_masks;

        // Cluster + extremes: removes isolated 3D clusters and Gaussians whose
        // oriented 3D extremes are far from the main point cloud - exactly what spikes are.
        mask_pruning::ClusterExtremePruningConfig cluster_cfg;
        cluster_cfg.enabled = true;
        // default values are ok for dome scenes

        LOG_INFO("Running post-training mask-based pruning...");

        // SMN: pasadas individuales en vez de prune_after_training() para poder
        // medir el tiempo de cada una por separado en fs_timings_. Si una
        // pasada falla se loguea pero no se aborta el resto, igual que hacia
        // el bundle de upstream.

        auto report = [](const char* name, const std::expected<mask_pruning::PruningResult, std::string>& r) {
            if (!r) {
                LOG_WARN("{} pruning failed: {}", name, r.error());
            } else if (r->splats_removed > 0) {
                LOG_INFO("{}: removed {} splats ({:.1f}%)",
                         name, r->splats_removed, r->removal_ratio() * 100.0f);
            }
        };

        if constexpr (FOCUSED_ENABLE_PRUNE_GEOMETRIC_DOME) {
            const lfs::training::smn::ScopedFsTimer _t(
                fs_timings_, lfs::training::smn::FsTimerSlot::PruneGeometricDome);
            auto r = mask_pruning::prune_by_geometric_dome(strategy, train_dataset, geomdome_cfg);
            report("Geometric dome", r);
        }
        if constexpr (FOCUSED_ENABLE_PRUNE_CENTER_VOTE) {
            const lfs::training::smn::ScopedFsTimer _t(
                fs_timings_, lfs::training::smn::FsTimerSlot::PruneCenterVote);
            auto r = mask_pruning::prune_by_center_vote(strategy, train_dataset, center_cfg);
            report("Center vote", r);
        }
        if constexpr (FOCUSED_ENABLE_PRUNE_MASK_LEAKAGE) {
            const lfs::training::smn::ScopedFsTimer _t(
                fs_timings_, lfs::training::smn::FsTimerSlot::PruneMaskLeakage);
            auto r = mask_pruning::prune_by_mask_leakage(strategy, train_dataset, leak_cfg);
            report("Mask leakage", r);
        }
        if constexpr (FOCUSED_ENABLE_PRUNE_ISOLATION) {
            const lfs::training::smn::ScopedFsTimer _t(
                fs_timings_, lfs::training::smn::FsTimerSlot::PruneIsolation);
            auto r = mask_pruning::prune_by_isolation_distance(strategy, iso_cfg);
            report("Isolation", r);
        }
        if constexpr (FOCUSED_ENABLE_PRUNE_ELLIPSE_BOUNDARY) {
            const lfs::training::smn::ScopedFsTimer _t(
                fs_timings_, lfs::training::smn::FsTimerSlot::PruneEllipseBoundary);
            auto r = mask_pruning::prune_by_ellipse_boundary(strategy, train_dataset, ellipse_cfg);
            report("Ellipse boundary", r);
        }
        if constexpr (FOCUSED_ENABLE_PRUNE_CLUSTER_EXTREME) {
            const lfs::training::smn::ScopedFsTimer _t(
                fs_timings_, lfs::training::smn::FsTimerSlot::PruneClusterExtreme);
            auto r = mask_pruning::prune_by_cluster_and_extremes(strategy, cluster_cfg);
            report("Cluster/extremes", r);
        }
    }

    // =========================================================================
    // SMN: dump cumulative FocusedSegment / prune timings to the log.
    //
    // Called once at the end of Trainer::train(). Builds a single multi-line
    // string and emits it via LOG_INFO so it lands at the same level as the
    // training stats; nothing is printed for slots that were never timed
    // (so non-FocusedSegment training stays clean).
    // =========================================================================
    void Trainer::focused_segment_print_timings() const {
        using lfs::training::smn::FsTimerSlot;
        using lfs::training::smn::fs_timer_slot_name;
        using lfs::training::smn::kFsTimerSlotCount;

        if (!fs_timings_.enabled) {
            return;
        }

        bool any = false;
        for (auto c : fs_timings_.calls) {
            if (c > 0) { any = true; break; }
        }
        if (!any) {
            return;
        }

        std::string out;
        out.reserve(2048);
        out += "\n";
        out += "[SMN] FocusedSegment cumulative timings (GPU-synced)\n";
        out += "  Subtask                                       Calls          Total (s)      Avg/call (s)\n";
        out += "  --------------------------------------------- -------------- -------------- ---------------\n";

        char line[256];
        for (int i = 0; i < kFsTimerSlotCount; ++i) {
            const std::uint64_t calls = fs_timings_.calls[i];
            if (calls == 0) {
                continue;
            }
            const double total_s = fs_timings_.total_ms[i] / 1000.0;
            const double avg_s = total_s / static_cast<double>(calls);
            const std::string name(fs_timer_slot_name(static_cast<FsTimerSlot>(i)));
            std::snprintf(line, sizeof(line),
                          "  %-45s %14llu %14.4f %15.6f\n",
                          name.c_str(),
                          static_cast<unsigned long long>(calls),
                          total_s,
                          avg_s);
            out += line;
        }

        LOG_INFO("{}", out);
    }

} // namespace lfs::training
