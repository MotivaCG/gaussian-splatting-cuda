/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "training/trainer.hpp"

#include "core/logger.hpp"
#include "lfs/kernels/ssim.cuh"
#include "optimizer/adam_optimizer.hpp"
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
        //
        // For dome captures of people, 3 provides enough border coverage for
        // concave zones (between legs, armpits) to accumulate densification error
        // and spawn small splats that define the silhouette precisely. The
        // post-training prune removes any excess.
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
    // raw_rendered is currently unused - FocusedSegment does not split into
    // appearance-corrected vs raw gradients. The parameter exists to match the
    // master signature of compute_photometric_loss_with_mask and to allow a
    // future decoupled FocusedSegment variant without another churn.
    std::expected<Trainer::MaskLossResult, std::string> Trainer::focused_segment_compute_loss(
        const lfs::core::Tensor& corrected,
        const lfs::core::Tensor& raw_rendered,
        const lfs::core::Tensor& gt_image,
        const lfs::core::Tensor& mask_in,
        const lfs::core::Tensor& alpha,
        const lfs::core::param::OptimizationParameters& opt_params) {

        using namespace lfs::core;

        const float kBgWeight = opt_params.focused_bg_weight; // BG gradient weight relative to FG (1.0)
        constexpr float kAlphaFgWeight = 1.5f; // grad_alpha pressure to push FG alpha -> 1
        constexpr float kAlphaBgWeight = 1.0f; // grad_alpha pressure to push BG alpha -> 0

        // Normalize mask to Float32 in [0,1]. Loaders deliver different formats per OS:
        //   - Linux (NvCodec): UInt8 [0,255]
        //   - Windows (cache_image_loader): Float32 [0,1]
        // Without rescaling UInt8, downstream ops break: bg_mask = 1 - mask becomes
        // [-254, 1], weight_map blows up, fg_pixels = sum() is ~255x off, and gradients
        // diverge to NaN. Upstream's mask_as_float lambda only converts dtype and shares
        // this latent bug; we fix it locally to keep the fork minimal.
        Tensor mask_2d;
        if (mask_in.dtype() == DataType::UInt8) {
            mask_2d = mask_in.to(DataType::Float32) * (1.0f / 255.0f);
        } else if (mask_in.dtype() == DataType::Bool) {
            mask_2d = mask_in.to(DataType::Float32);
        } else {
            mask_2d = mask_in;
        }
        mask_2d = mask_2d.clamp(0.0f, 1.0f);

        const Tensor bg_mask = Tensor::full(mask_2d.shape(), 1.0f, mask_2d.device()) - mask_2d;

        // Step 1: photometric loss forward+backward.
        // When raw_rendered is provided (PPISP / bilateral grid active), use the decoupled
        // L1/SSIM path so that L1 grad goes through `corrected` (-> PPISP backward) and
        // SSIM grad goes through `raw` (bypassing PPISP). Without this split, the PPISP
        // backward amplifies the SSIM gradient and destabilises the transition from
        // None mode (decoupled) to FocusedSegment, producing NaN/Inf around iter ~kNoneEnd.
        const bool use_decoupled = raw_rendered.is_valid() &&
                                   raw_rendered.numel() > 0 &&
                                   opt_params.lambda_dssim > 0.0f;
        Tensor grad;
        Tensor grad_raw;
        Tensor loss;
        if (use_decoupled) {
            auto [loss_tensor, ctx] = lfs::training::kernels::decoupled_fused_l1_ssim_forward(
                corrected, raw_rendered, gt_image, opt_params.lambda_dssim, decoupled_fused_workspace_,
                /*apply_valid_padding=*/true);
            auto grads = lfs::training::kernels::decoupled_fused_l1_ssim_backward(ctx, decoupled_fused_workspace_);
            if (corrected.ndim() == 3) {
                grads.grad_corrected = grads.grad_corrected.squeeze(0);
                grads.grad_raw = grads.grad_raw.squeeze(0);
            }
            grad = grads.grad_corrected;
            grad_raw = grads.grad_raw;
            loss = loss_tensor;
        } else {
            losses::PhotometricLoss::Params params{.lambda_dssim = opt_params.lambda_dssim};
            auto full_result = photometric_loss_.forward(corrected, gt_image, params);
            if (!full_result) {
                return std::unexpected(full_result.error());
            }
            auto [full_loss, fp_ctx] = *full_result;
            grad = fp_ctx.grad_image;
            loss = full_loss;
        }

        // Step 2: spatial weight map - FG=1.0, BG=kBgWeight.
        // Darkness bonus applied to FG only (Rec.601 perceptual luminance from GT, not rendered).
        // Using GT keeps the weight map stable across iterations - rendered changes every step.
        // BG stays flat at kBgWeight regardless of darkness, avoiding spurious BG gradient boosts.
        Tensor weight_map;
        if (opt_params.darkness_boost > 0.0f) {
            const bool chw = (gt_image.ndim() == 3 && gt_image.shape()[0] == 3);
            const Tensor r = chw ? gt_image.slice(0, 0, 1).squeeze(0) : gt_image.slice(2, 0, 1).squeeze(2);
            const Tensor g = chw ? gt_image.slice(0, 1, 2).squeeze(0) : gt_image.slice(2, 1, 2).squeeze(2);
            const Tensor b = chw ? gt_image.slice(0, 2, 3).squeeze(0) : gt_image.slice(2, 2, 3).squeeze(2);
            const Tensor brightness = r * 0.299f + g * 0.587f + b * 0.114f;
            const Tensor darkness = Tensor::full(brightness.shape(), 1.0f, brightness.device()) - brightness;
            weight_map =
                mask_2d * (Tensor::full(darkness.shape(), 1.0f, darkness.device()) + darkness * opt_params.darkness_boost) +
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
        // When decoupled, both grad (L1 path on corrected) and grad_raw (SSIM path on raw)
        // get the same spatial weighting so the per-pixel emphasis stays consistent.
        const Tensor weight_3d = (corrected.ndim() == 3 && corrected.shape()[0] == 3)
                                     ? weight_map.unsqueeze(0)
                                     : weight_map.unsqueeze(2);
        grad = grad * weight_3d;
        if (grad_raw.is_valid() && grad_raw.numel() > 0) {
            grad_raw = grad_raw * weight_3d;
        }

        // Scale scalar loss to reflect FG-focused weighting.
        // This ensures the logged loss is representative of FG reconstruction quality,
        // not diluted by the large BG area. fg_pixels is cached to avoid a second GPU sync.
        const float total_pixels = static_cast<float>(mask_2d.numel());
        const float fg_pixels = std::max(mask_2d.sum().item<float>(), 1.0f);
        const float bg_pixels = std::max(total_pixels - fg_pixels, 1.0f);
        loss = loss * (fg_pixels / std::max(total_pixels * weight_mean, 1e-6f));

        // Step 4: pixel-based adaptive alpha pressure via grad_alpha.
        // NOTE: modifying the scalar `loss` does NOT affect Gaussian parameters - only
        // grad_image and grad_alpha propagate through rasterize_backward to the optimizer.
        // Alpha pressure is therefore applied purely through grad_alpha.
        //
        // Pressure is proportional to the current rendered alpha error, so it
        // self-regulates: a BG pixel already at alpha=0 gets no pressure, a BG pixel
        // at alpha=0.8 gets strong pressure. This avoids over-correction on boundary
        // Gaussians that are already behaving correctly and reduces halo artifacts
        // compared to a constant gradient map.
        //
        // Sign convention (gradient descent: param -= lr * grad):
        //   FG: negative grad_alpha  -> alpha_raw increases -> rendered alpha approaches 1
        //   BG: positive grad_alpha  -> alpha_raw decreases -> rendered alpha approaches 0
        Tensor grad_alpha;
        const float w_fg = opt_params.mask_opacity_penalty_weight;
        const float w_bg = opt_params.mask_opacity_penalty_weight_bg;
        if (alpha.is_valid() && (w_fg > 0.0f || w_bg > 0.0f)) {
            const Tensor alpha_2d = alpha.ndim() == 3 ? alpha.squeeze(0) : alpha;
            const Tensor ones = Tensor::full(alpha_2d.shape(), 1.0f, alpha_2d.device());

            grad_alpha = bg_mask * alpha_2d * (w_bg * kAlphaBgWeight / bg_pixels)          // BG: pressure proportional to current alpha
                         - mask_2d * (ones - alpha_2d) * (w_fg * kAlphaFgWeight / fg_pixels); // FG: pressure proportional to (1 - alpha)
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

        const float w_bg = step_params.mask_opacity_penalty_weight_bg;
        if (w_bg <= 0.0f)
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

        // Clamp to image bounds for safe indexing
        auto x_px = x_proj.round().clamp(0.0f, Wf - 1.0f); // [N]
        auto y_px = y_proj.round().clamp(0.0f, Hf - 1.0f); // [N]

        // Also mask out splats that project outside the image
        auto in_bounds = x_proj.ge(0.0f) * x_proj.lt(Wf)
                       * y_proj.ge(0.0f) * y_proj.lt(Hf);
        auto valid = visible_mask * in_bounds; // [N] bool: visible AND in-bounds

        // ---- 3. Sample mask at projected centers ----
        auto lin_idx = (y_px * Wf + x_px).to(DataType::Int64); // [N] int64
        auto mask_flat = mask.reshape({static_cast<int>(H * W)}); // [H*W]
        auto is_inside = mask_flat.gather(0, lin_idx); // [N] float {0..1}

        // ---- 4. Compute opacity gradient ----
        // For splats outside the mask: positive gradient pushes raw opacity down (sigmoid decreases).
        // For splats inside the mask: small negative gradient fills transparency holes.
        constexpr float kOutWeight = 1.5f;  // BG center penalty strength
        constexpr float kInWeight = 0.05f;  // gentle FG fill (most work done by grad_alpha)

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

        // Center penalty only makes sense with full image (single tile)
        if (num_tiles != 1)
            return;

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

        // Normalize mask to Float32 [0,1]. Loader output differs per OS:
        // Linux (NvCodec) returns UInt8 [0,255]; Windows returns Float32 [0,1].
        // Center penalty samples mask values via gather(), which would otherwise
        // produce 255-valued "is_inside" weights and break the gradient sign logic.
        if (full_mask.dtype() == lfs::core::DataType::UInt8) {
            full_mask = full_mask.to(lfs::core::DataType::Float32) * (1.0f / 255.0f);
        } else if (full_mask.dtype() == lfs::core::DataType::Bool) {
            full_mask = full_mask.to(lfs::core::DataType::Float32);
        }
        full_mask = full_mask.clamp(0.0f, 1.0f);

        // Build step_params with schedule applied for this iteration
        lfs::core::param::OptimizationParameters step_params = params_.optimization;
        const float progress = static_cast<float>(iter) /
                               static_cast<float>(params_.optimization.iterations);
        focused_segment_apply_schedule(step_params, progress);

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

        // Ellipse boundary pass - catches spikes missed by center-vote and leakage.
        auto ellipse_result = mask_pruning::prune_by_ellipse_boundary(strategy, train_dataset, ellipse_cfg);
        if (!ellipse_result) {
            LOG_WARN("Ellipse boundary pruning failed: {}", ellipse_result.error());
        } else if (ellipse_result->splats_removed > 0) {
            LOG_INFO("Ellipse pruning: removed {} splats ({:.1f}%)",
                     ellipse_result->splats_removed,
                     ellipse_result->removal_ratio() * 100.0f);
        }

        // Cluster + extremes pass - removes 3D spikes whose oriented extremes
        // are far from the main point cloud.
        /* auto cluster_result = mask_pruning::prune_by_cluster_and_extremes(strategy, cluster_cfg);
        if (!cluster_result) {
            LOG_WARN("Cluster/extremes pruning failed: {}", cluster_result.error());
        } else if (cluster_result->splats_removed > 0) {
            LOG_INFO("Cluster/extremes pruning: removed {} splats ({:.1f}%)",
                     cluster_result->splats_removed,
                     cluster_result->removal_ratio() * 100.0f);
        }*/
    }

} // namespace lfs::training
