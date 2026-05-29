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
        constexpr bool FOCUSED_ENABLE_DENSIFY_DILATION = true; // dilate the mask before applying it to the densification error map

        // Splat-level center penalty (focused_segment_apply_center_penalty).
        constexpr bool FOCUSED_ENABLE_CENTER_PENALTY         = true; // master switch (skips hook entirely)
        constexpr bool FOCUSED_ENABLE_CENTER_PENALTY_FG_FILL = true; // push opacity UP inside mask
        constexpr bool FOCUSED_ENABLE_CENTER_PENALTY_BG      = true; // push opacity DOWN outside mask

        // Post-training mask-based pruning pipeline.
        constexpr bool FOCUSED_ENABLE_POST_TRAINING_PRUNE    = true;

        // Per-pass switches for the post-training prune. Each can be disabled
        // independently to A/B test the impact of that specific pass. Only
        // consulted when FOCUSED_ENABLE_POST_TRAINING_PRUNE is true.
        // Disable an individual pass to skip it without commenting code out.
        constexpr bool FOCUSED_ENABLE_PRUNE_GEOMETRIC_DOME   = true;
        constexpr bool FOCUSED_ENABLE_PRUNE_CENTER_VOTE      = true;
        constexpr bool FOCUSED_ENABLE_PRUNE_MASK_LEAKAGE     = true;
        constexpr bool FOCUSED_ENABLE_PRUNE_ISOLATION        = true;
        constexpr bool FOCUSED_ENABLE_PRUNE_ELLIPSE_BOUNDARY = true; // catches elongated spike splats
        constexpr bool FOCUSED_ENABLE_PRUNE_CLUSTER_EXTREME  = false; // Off by default (slow + experimental)

        // =====================================================================
        // FocusedSegment band radii (pixels). One independent knob per
        // consumer.
        //
        // Kept here (in the .cpp's anonymous namespace) rather than in
        // focused_segment_cache.hpp on purpose: changing any of these is a
        // common tuning operation, and pulling them out of the header turns
        // it from a ~11 min full-project rebuild into a ~1 min single-file
        // recompile + relink.
        //
        // Naming convention:
        //   <consumer>_INSET_PX   -> how far the zone steps INTO the mask
        //   <consumer>_OUTSET_PX  -> how far the zone steps OUT of the mask
        // =====================================================================

        /// How many pixels INSIDE the annotated mask the alpha-UP zone starts.
        /// Built as: mask_eroded = erode(mask, N).
        /// Consumed by:
        ///   - per-pixel grad_alpha FG term (pushes alpha -> 1)
        ///   - center penalty FG fill (bonus opacity for splats with center inside)
        ///
        /// Raising it = alpha-up trusted only deeper inside the silhouette.
        /// The outer ring (last N px before the border) gets no alpha-up
        /// pressure and is left to the photometric loss. Protects the subject
        /// when some masks are buggy near the silhouette ("this border pixel
        /// is BG" votes from bad masks do not destroy opacity on the rest of
        /// the body). Cost of being generous: low.
        constexpr int FOCUSED_ALPHA_UP_INSET_PX = 3;

        /// SIGNED. How far from the annotated mask the photometric FG region
        /// reaches. Built as:
        ///     N > 0  -> mask_photo = dilate(mask,  N)
        ///     N = 0  -> mask_photo = mask
        ///     N < 0  -> mask_photo = erode (mask, |N|)
        /// Consumed by:
        ///   - spatial weight map FG region (FG weight 1.0, BG weight kBgWeight
        ///     applied to its complement)
        ///   - darkness boost area (linked to the FG region of the weight map)
        ///
        ///   N > 0 : photo loss "sees" more border pixels as FG, giving
        ///           correct-color signal to splats covering fine features the
        ///           matting trimmed (hair, fingers, eyelashes). Cost: if the
        ///           matting includes any BG fuzz in that ring, the photo
        ///           loss will obligate the model to reconstruct it (=halos).
        ///
        ///   N = 0 : photo follows the annotated silhouette exactly.
        ///
        ///   N < 0 : photo region SHRINKS by |N| px inside the silhouette.
        ///           The outer ring (last |N| px of the mask interior) gets
        ///           BG weight (0.05) instead of FG weight. Use when the
        ///           matting is over-generous and includes BG fuzz inside its
        ///           annotated boundary: telling the model "do not try hard
        ///           to reconstruct that ring" lets it converge to either
        ///           dark BG or transparent without fighting a high-weight
        ///           photo gradient that wants to keep the fuzz. Pairs well
        ///           with negative ALPHA_DOWN_OUTSET for cleanest silhouette
        ///           on over-generous mattings.
        constexpr int FOCUSED_PHOTO_OUTSET_PX = 0;

        /// SIGNED. How far from the annotated mask the alpha-DOWN zone starts.
        /// Built as:
        ///     N > 0  -> BG_strict = NOT dilate(mask,  N)
        ///     N = 0  -> BG_strict = NOT mask
        ///     N < 0  -> BG_strict = NOT erode (mask, |N|)
        /// Consumed by:
        ///   - per-pixel grad_alpha BG term (pushes alpha -> 0)
        ///   - center penalty BG push (kills opacity for splats with center
        ///     clearly in BG)
        ///
        /// THIS IS THE ANTI-HALO KNOB.
        ///
        ///   N > 0  : alpha-down starts N px OUTSIDE the silhouette. Opens a
        ///            ring just outside the mask where splats are protected -
        ///            this is where bright "halo" splats nucleate. Use only
        ///            when you need to recover fine border features at the
        ///            cost of some halos.
        ///
        ///   N = 0  : alpha-down starts exactly at the silhouette (matches
        ///            the old trainer_oldmode.cpp). Kills any splat with
        ///            center outside the matting cut.
        ///
        ///   N < 0  : alpha-down extends |N| px INSIDE the silhouette. Kills
        ///            "fuzz" / halo splats whose center sits in the outer
        ///            ring of the matting (typical when the matting is over-
        ///            generous and includes a few px of background around
        ///            fine hair). The rendered silhouette contracts to
        ///            roughly (mask - |N|).
        ///
        /// Note: very negative values also DILUTE per-pixel BG pressure
        /// (the global mean_bg_strict grows -> grad_alpha_BG / mean_bg
        /// shrinks). Keep |N| moderate (0 to 3) for the best tradeoff.
        ///
        /// Constraint when negative: |N| must be < FOCUSED_ALPHA_UP_INSET_PX,
        /// otherwise alpha-up and alpha-down overlap in the same ring and
        /// their gradients fight each other.
        constexpr int FOCUSED_ALPHA_DOWN_OUTSET_PX = 0;

        // Sanity: when the alpha-down outset is negative it eats INTO the FG
        // side of the band; if it eats as far as (or past) the alpha-up
        // inset, the two zones overlap and their gradients pull a single
        // ring in opposite directions every iteration. Forbid that at
        // compile time.
        static_assert(
            FOCUSED_ALPHA_DOWN_OUTSET_PX >= 0 ||
                (-FOCUSED_ALPHA_DOWN_OUTSET_PX) < FOCUSED_ALPHA_UP_INSET_PX,
            "FOCUSED_ALPHA_DOWN_OUTSET_PX is negative AND its magnitude reaches "
            "the alpha-up zone. Pick |OUTSET| < FOCUSED_ALPHA_UP_INSET_PX so the "
            "two rings do not overlap.");

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
        auto& entry = fs_camera_cache_.map[cam.uid()];

        if (!entry.has_mask()) {
            const lfs::training::smn::ScopedFsTimer _t(
                fs_timings_, lfs::training::smn::FsTimerSlot::CacheBuildMask);
            const Tensor mask_f = focused_segment_mask_float(src_mask);
            entry.mask_bool = mask_f.gt(0.5f).contiguous();

            // ------------------------------------------------------------------
            // Three independent morphology passes, one per consumer. See
            // focused_segment_cache.hpp for what each constant means.
            //
            //   FOCUSED_ALPHA_UP_INSET_PX    -> mask_eroded_bool        (alpha-up zone)
            //   FOCUSED_PHOTO_OUTSET_PX      -> mask_photo_dilated_bool (photometric FG)
            //   FOCUSED_ALPHA_DOWN_OUTSET_PX -> mask_bg_dilated_bool    (BG_strict outer edge)
            //
            // Any radius = 0 collapses its mask to the original (shared
            // storage, no extra tensor, no GPU work). If two non-zero
            // dilation radii happen to be equal we still compute them
            // separately for code simplicity; the cost is one extra bool
            // tensor (~3 MB/cam at 2K) - negligible vs darkness_fp16 already
            // cached (~6 MB/cam).
            //
            // Dilation: max_pool2d on a {0,1} mask. Erosion: 1 - dilation(1-mask).
            // Both use kernel = 2*R+1, stride=1, padding=R - same idiom already
            // used by focused_segment_apply_densification_mask.
            // ------------------------------------------------------------------
            constexpr int kErodeR    = FOCUSED_ALPHA_UP_INSET_PX;
            constexpr int kPhotoDilR = FOCUSED_PHOTO_OUTSET_PX;
            constexpr int kBgDilR    = FOCUSED_ALPHA_DOWN_OUTSET_PX;

            const Tensor mask_4d = mask_f.unsqueeze(0).unsqueeze(0); // [1,1,H,W]

            // Erosion -> FG_strict (alpha-up safe zone).
            if constexpr (kErodeR > 0) {
                constexpr int kErodeK = 2 * kErodeR + 1;
                const Tensor inv_4d = Tensor::full(mask_4d.shape(), 1.0f, mask_4d.device()) - mask_4d;
                const Tensor inv_dilated_4d = inv_4d.max_pool2d(kErodeK, 1, kErodeR);
                const Tensor eroded_4d = Tensor::full(inv_dilated_4d.shape(), 1.0f, inv_dilated_4d.device()) - inv_dilated_4d;
                entry.mask_eroded_bool = eroded_4d.squeeze(0).squeeze(0).gt(0.5f).contiguous();
            } else {
                entry.mask_eroded_bool = entry.mask_bool; // identity (no allocation)
            }

            // Photometric dilation -> photometric FG region (spatial weight map and darkness boost).
            // Signed: positive dilates (photo extends past mask), zero is the
            // identity, negative erodes (photo region shrinks inside the mask
            // to dampen the photo loss on over-generous matting borders).
            if constexpr (kPhotoDilR > 0) {
                constexpr int kK = 2 * kPhotoDilR + 1;
                const Tensor dilated_4d = mask_4d.max_pool2d(kK, 1, kPhotoDilR);
                entry.mask_photo_dilated_bool = dilated_4d.squeeze(0).squeeze(0).gt(0.5f).contiguous();
            } else if constexpr (kPhotoDilR < 0) {
                constexpr int kAbsR = -kPhotoDilR;
                constexpr int kK = 2 * kAbsR + 1;
                // Erosion via 1 - dilate(1 - mask, |R|). Same idiom used for
                // the alpha-up zone and (when negative) the alpha-down zone.
                const Tensor inv_4d = Tensor::full(mask_4d.shape(), 1.0f, mask_4d.device()) - mask_4d;
                const Tensor inv_dilated_4d = inv_4d.max_pool2d(kK, 1, kAbsR);
                const Tensor eroded_4d = Tensor::full(inv_dilated_4d.shape(), 1.0f, inv_dilated_4d.device()) - inv_dilated_4d;
                entry.mask_photo_dilated_bool = eroded_4d.squeeze(0).squeeze(0).gt(0.5f).contiguous();
            } else {
                entry.mask_photo_dilated_bool = entry.mask_bool;
            }

            // BG dilation -> BG_strict outer edge (alpha-down zone is its complement).
            // Signed: positive dilates, zero leaves the mask as is, negative
            // erodes (alpha-down extends |R| px INSIDE the silhouette to kill
            // halo splats sitting in the outer ring of an over-generous mask).
            if constexpr (kBgDilR > 0) {
                constexpr int kK = 2 * kBgDilR + 1;
                const Tensor dilated_4d = mask_4d.max_pool2d(kK, 1, kBgDilR);
                entry.mask_bg_dilated_bool = dilated_4d.squeeze(0).squeeze(0).gt(0.5f).contiguous();
            } else if constexpr (kBgDilR < 0) {
                constexpr int kAbsR = -kBgDilR;
                constexpr int kK = 2 * kAbsR + 1;
                // Erosion via 1 - dilate(1 - mask, |R|). Same idiom as the
                // alpha-up zone build above.
                const Tensor inv_4d = Tensor::full(mask_4d.shape(), 1.0f, mask_4d.device()) - mask_4d;
                const Tensor inv_dilated_4d = inv_4d.max_pool2d(kK, 1, kAbsR);
                const Tensor eroded_4d = Tensor::full(inv_dilated_4d.shape(), 1.0f, inv_dilated_4d.device()) - inv_dilated_4d;
                entry.mask_bg_dilated_bool = eroded_4d.squeeze(0).squeeze(0).gt(0.5f).contiguous();
            } else {
                // R=0: alpha-down starts exactly at the annotated silhouette.
                entry.mask_bg_dilated_bool = entry.mask_bool;
            }

            // Pixel counts derived from the STORED bool tensors so the
            // dataset-wide stats below and the per-iter normalization in
            // focused_segment_compute_loss are guaranteed to match exactly the
            // masks consumed downstream (single source of truth). One sync per
            // count per camera at first hit; consumed as cached scalars later.
            const float numel_f = static_cast<float>(mask_f.numel());
            const float fg         = std::max(entry.mask_bool.to(DataType::Float32).sum().item<float>(), 1.0f);
            const float fg_eroded  = std::max(entry.mask_eroded_bool.to(DataType::Float32).sum().item<float>(), 1.0f);
            const float fg_photo   = std::max(entry.mask_photo_dilated_bool.to(DataType::Float32).sum().item<float>(), 1.0f);
            const float fg_bg_dil  = std::max(entry.mask_bg_dilated_bool.to(DataType::Float32).sum().item<float>(), 1.0f);
            entry.fg_pixels = fg;
            entry.bg_pixels = std::max(numel_f - fg, 1.0f);
            entry.fg_eroded_pixels = fg_eroded;
            entry.fg_photo_pixels  = fg_photo;
            entry.bg_photo_pixels  = std::max(numel_f - fg_photo, 1.0f);
            entry.bg_strict_pixels = std::max(numel_f - fg_bg_dil, 1.0f);

            // Accumulate dataset-wide running stats. With focused_segment_prewarm_cache
            // (called once at trainer start) these are complete before iter 1.
            // The lazy path also keeps working: if prewarm is skipped, the
            // running mean stabilizes within the first epoch.
            fs_camera_cache_.stats.total_fg_eroded_pixels += static_cast<double>(entry.fg_eroded_pixels);
            fs_camera_cache_.stats.total_bg_strict_pixels += static_cast<double>(entry.bg_strict_pixels);
            fs_camera_cache_.stats.n_cams_with_mask += 1;
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

            // Mean of darkness restricted to the photometric FG region
            // (= mask_photo_dilated). Matches the weight_map FG region used
            // downstream so weight_mean stays exact.
            const Tensor photo_f = entry.mask_photo_dilated_bool.to(DataType::Float32);
            const float sum_dark_photo = (photo_f * darkness).sum().item<float>();
            entry.mean_darkness_photo = sum_dark_photo / entry.fg_photo_pixels;

            entry.darkness_fp16 = darkness.to(DataType::Float16).contiguous();
        }

        return entry;
    }

    // =========================================================================
    // SMN: pre-warm the FocusedSegment per-camera cache for every camera in
    // the training dataset.
    //
    // Without prewarm, the cache is built lazily on first hit per camera. That
    // means the dataset-wide stats (fs_camera_cache_.stats) start empty and
    // mature over the first epoch, so the early grad_alpha pressure is
    // normalized by a partial mean - typically dominated by whichever cameras
    // happen to be visited first. Pre-warming guarantees the mean is exact
    // from iteration 1.
    //
    // Cost: one disk read + one morphology pass per camera. For ~105 cameras
    // at 2K this is ~1-3 seconds at trainer start. Darkness is intentionally
    // left lazy here (would need to load gt images at startup as well,
    // doubling the cost for marginal benefit).
    //
    // No-op when mask_mode != FocusedSegment or when the dataset has no
    // masked cameras.
    // =========================================================================
    void Trainer::focused_segment_prewarm_cache(lfs::training::CameraDataset& dataset) {
        if (params_.optimization.mask_mode != lfs::core::param::MaskMode::FocusedSegment) {
            return;
        }

        const auto& cams = dataset.get_cameras();
        if (cams.empty()) {
            return;
        }

        LOG_INFO("[SMN] FocusedSegment: pre-warming mask cache for {} cameras...", cams.size());

        int n_built = 0;
        for (const auto& cam_ptr : cams) {
            if (!cam_ptr || !cam_ptr->has_mask()) {
                continue;
            }

            auto mask = cam_ptr->load_and_get_mask(
                params_.dataset.resize_factor,
                params_.dataset.max_width,
                params_.optimization.invert_masks,
                params_.optimization.mask_threshold);

            if (!mask.is_valid() || mask.numel() == 0) {
                continue;
            }

            // Build mask_bool + bands + per-camera counts and accumulate global
            // stats. Darkness is left for the per-iter loss path so we do not
            // need to load gt images here.
            (void)focused_segment_get_cache(
                *cam_ptr, mask, lfs::core::Tensor(), /*want_lightness=*/false);
            ++n_built;
        }

        const auto& s = fs_camera_cache_.stats;
        LOG_INFO("[SMN] FocusedSegment: pre-warm complete - {} cameras built, "
                 "mean_fg_eroded={:.0f}px, mean_bg_strict={:.0f}px",
                 n_built, s.mean_fg_eroded(), s.mean_bg_strict());
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
        constexpr float kBgTargetFree = 0.1f; // Free-refinement BG weight - slightly higher
                                               // than kBgTarget to allow gentle BG refinement
                                               // without reintroducing detail. Kept low to help
                                               // fine silhouettes (hair) stay sharp.
        step_params.focused_bg_weight = kBgTarget;

        constexpr float kFgAlphaRampStart = 0.30f; // Start ramping FG alpha pressure up.
        constexpr float kFgAlphaRampEnd = 0.40f;   // FG alpha pressure reaches full strength.

        constexpr float kBgAlphaRampStart = 0.50f; // Start ramping BG alpha pressure up.
        constexpr float kBgAlphaRampEnd = 0.60f;   // BG alpha pressure reaches full strength; both FG+BG fully active.

        constexpr float kAlphaDecayStart = 0.75f;  // Start relaxing alpha pressure toward residual floors.
        constexpr float kAlphaDecayEnd = 0.85f;    // End of decay; residual floors remain until training ends.

        constexpr float kAlphaPenaltyMult = 0.0066;
        constexpr float kFgAlphaFloorValue = 1.0f * kAlphaPenaltyMult; // FG residual alpha penalty after decay.
        constexpr float kBgAlphaFloorValue = 3.0f * kAlphaPenaltyMult; // BG residual alpha penalty after decay.
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

        // SMN per-camera memoization: on first hit builds mask_bool, the eroded
        // and dilated band masks, fg/bg pixel counts (per-band) and, when
        // darkness_boost is in use, the FP16 darkness map. Also accumulates
        // dataset-wide pixel-count stats used by the global grad_alpha
        // normalization below. See smn/focused_segment_cache.hpp.
        const bool want_lightness = FOCUSED_ENABLE_DARKNESS_BOOST && opt_params.darkness_boost > 0.0f;
        const auto& fs_cache = focused_segment_get_cache(cam, mask_2d, gt_image, want_lightness);

        // Three masks pulled from the cache, each with its own radius:
        //   - fg_photo   (mask dilated by N for photometric focus, captures
        //                 fine border features like hair)
        //   - fg_strict  (mask eroded by X for alpha-UP, safe FG zone)
        //   - bg_strict  (complement of mask dilated by Y for alpha-DOWN,
        //                 safe BG zone)
        // The free band between fg_strict and the complement of bg_strict
        // receives no alpha pressure - the rasterizer is free to set alpha
        // however the photometric loss dictates there.
        const Tensor fg_photo   = fs_cache.mask_photo_dilated_bool.to(DataType::Float32); // photometric FG
        const Tensor fg_strict  = fs_cache.mask_eroded_bool.to(DataType::Float32);        // alpha-up zone
        const Tensor fg_bg_dil  = fs_cache.mask_bg_dilated_bool.to(DataType::Float32);    // for BG_strict complement
        const Tensor bg_photo   = Tensor::full(fg_photo.shape(), 1.0f, fg_photo.device()) - fg_photo;   // photometric BG
        const Tensor bg_strict  = Tensor::full(fg_bg_dil.shape(), 1.0f, fg_bg_dil.device()) - fg_bg_dil; // alpha-down zone

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

        // Pixel counts come from the SMN cache (no GPU sync). Total is just
        // the tensor numel (metadata access, no kernel). All counts are built
        // once per camera in focused_segment_get_cache.
        const float total_pixels   = static_cast<float>(fg_photo.numel());
        const float fg_photo_count = fs_cache.fg_photo_pixels; // photometric FG area
        const float bg_photo_count = fs_cache.bg_photo_pixels; // photometric BG area (= total - fg_photo)

        // Step 2+3: spatial weight map - FG=1.0 on the photo-dilated mask,
        // BG=kBgWeight on its complement, applied to grad. The photo-dilated
        // FG is larger than the original mask so the photometric loss covers
        // fine border features (hair, fingers) that the annotation typically
        // cuts too tight.
        // Darkness bonus applied to photo FG only (Rec.601 perceptual
        // luminance from GT, not rendered). Using GT keeps the weight map
        // stable across iterations - rendered changes every step. BG stays
        // flat at kBgWeight regardless of darkness, avoiding spurious BG
        // gradient boosts.
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
                    fg_photo * (Tensor::full(darkness.shape(), 1.0f, darkness.device()) + darkness * opt_params.darkness_boost) +
                    bg_photo * kBgWeight;
            } else {
                weight_map = fg_photo + bg_photo * kBgWeight;
            }

            // Normalize weight_map by its mean so the global gradient magnitude
            // stays comparable to None mode - prevents scale drift when mask
            // size varies across scenes.
            //
            // weight_mean computed analytically from cached scalars (no GPU sync).
            // With darkness boost:
            //   sum(weight_map) = fg_photo * (1 + boost*<darkness>_photo) + bg_photo * kBgWeight
            // Without darkness boost:
            //   sum(weight_map) = fg_photo + bg_photo * kBgWeight
            // (fg_photo + bg_photo == total_pixels by construction)
            const float weight_mean = (want_lightness && fs_cache.has_darkness())
                ? (fg_photo_count * (1.0f + opt_params.darkness_boost * fs_cache.mean_darkness_photo)
                   + bg_photo_count * kBgWeight) / total_pixels
                : (fg_photo_count + bg_photo_count * kBgWeight) / total_pixels;
            weight_map = weight_map * (1.0f / std::max(weight_mean, 1e-4f));

            const Tensor weight_3d = (corrected.ndim() == 3 && corrected.shape()[0] == 3)
                                         ? weight_map.unsqueeze(0)
                                         : weight_map.unsqueeze(2);
            grad = grad * weight_3d;
            if (grad_raw.is_valid() && grad_raw.numel() > 0) {
                grad_raw = grad_raw * weight_3d;
            }

            // Rescale scalar loss so logged values track the photometric FG
            // reconstruction quality, not diluted by the large BG area.
            loss = loss * (fg_photo_count / std::max(total_pixels * weight_mean, 1e-6f));
        }

        // Step 4: pixel-based alpha pressure via grad_alpha, applied only on
        // the SAFE bands of the 3-band scheme.
        //   - FG_strict (eroded by FOCUSED_ALPHA_UP_INSET_PX): push alpha UP
        //   - BG_strict (complement of dilated):            push alpha DOWN
        //   - free band (between them):                     no pressure
        //
        // The free band absorbs almost all mask-annotation errors (which are
        // overwhelmingly border errors), so a single bad mask cannot push a
        // splat in the wrong direction in the strict regions: those regions
        // are unambiguous even in noisy annotations.
        //
        // Normalization (CHANGED from per-camera 1/fg_pixels to dataset-wide
        // 1/mean_fg_eroded): each pixel contributes a constant amount across
        // ALL cameras of the dataset. With per-camera normalization, a bad
        // mask labelling almost everything as FG (tiny per-pixel push, but
        // huge area) would still dominate the BG term of OTHER cameras
        // through its mirror behavior in BG; conversely a mask with tiny FG
        // would have enormous per-pixel push and one such outlier could
        // overwhelm many correct votes. Normalizing by the dataset mean makes
        // each camera contribute pressure proportional to its area, so bad
        // outliers (extreme area on either side) cannot dominate the natural
        // multi-view majority vote.
        //
        // The accumulators live in fs_camera_cache_.stats and update lazily as
        // cameras are visited. During the very first epoch some cameras may
        // not have contributed yet, in which case the running mean is
        // already representative after a few iterations (one camera per iter).
        //
        // Sign convention (gradient descent: param -= lr * grad):
        //   FG_strict: negative grad_alpha -> alpha_raw increases -> alpha -> 1
        //   BG_strict: positive grad_alpha -> alpha_raw decreases -> alpha -> 0
        Tensor grad_alpha;
        const float w_fg = opt_params.mask_opacity_penalty_weight;
        const float w_bg = opt_params.mask_opacity_penalty_weight_bg;
        const bool apply_fg = FOCUSED_ENABLE_GRAD_ALPHA_FG && w_fg > 0.0f;
        const bool apply_bg = FOCUSED_ENABLE_GRAD_ALPHA_BG && w_bg > 0.0f;
        if (alpha.is_valid() && (apply_fg || apply_bg)) {
            const lfs::training::smn::ScopedFsTimer _t_ga(
                fs_timings_, lfs::training::smn::FsTimerSlot::ComputeLossGradAlpha);

            const float mean_fg = std::max(fs_camera_cache_.stats.mean_fg_eroded(), 1.0f);
            const float mean_bg = std::max(fs_camera_cache_.stats.mean_bg_strict(), 1.0f);

            Tensor fg_term;
            if (apply_fg) {
                // Modulate alpha-up by GT darkness when available: full pressure
                // on dark FG pixels (where photo alone cannot disambiguate FG
                // from BG - dark hair, shadowed concave zones), low pressure on
                // bright FG pixels (where photo already provides decisive
                // signal, and forcing opacity would lock SH to wrong colors
                // from per-view contamination - e.g. backlight bleed visible
                // through fine hair).
                //
                // The darkness map is the same one already cached and used by
                // the spatial weight map's darkness boost - zero extra GPU
                // work, zero extra memory.
                //
                // NOTE: total alpha-up magnitude drops by ~mean(darkness)
                // (typically 0.4-0.6). If pelo oscuro pierde proteccion, sube
                // mask_opacity_penalty_weight 2x desde CLI para compensar.
                //
                // Falls back to flat (un-modulated) alpha-up if darkness is not
                // cached (e.g. darkness_boost disabled in the current phase).
                if (fs_cache.has_darkness()) {
                    const Tensor darkness = fs_cache.darkness_fp16.to(DataType::Float32);
                    fg_term = fg_strict * darkness * (w_fg * kAlphaFgWeight / mean_fg);
                } else {
                    fg_term = fg_strict * (w_fg * kAlphaFgWeight / mean_fg);
                }
            }

            Tensor bg_term;
            if (apply_bg) {
                bg_term = bg_strict * (w_bg * kAlphaBgWeight / mean_bg);
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
        // focused_segment_mask_float() per iter and shares the band buffers
        // already populated by focused_segment_compute_loss.
        //
        // 3-band semantics for the center penalty (mirrors the pixel-level
        // grad_alpha policy):
        //   - splat center in FG_strict  (mask_eroded)              -> FG fill
        //   - splat center in BG_strict  (NOT mask_bg_dilated)      -> BG push
        //   - splat center in the free band (between the two)        -> no penalty
        // Only press where we are confident, leave the silhouette to the
        // photometric loss.
        const auto& fs_cache = focused_segment_get_cache(cam, mask, lfs::core::Tensor(), /*want_lightness=*/false);
        auto lin_idx = (y_px * Wf + x_px).to(DataType::Int64); // [N] int64
        auto eroded_flat     = fs_cache.mask_eroded_bool.to(DataType::Float32).reshape({static_cast<int>(H * W)});
        auto bg_dilated_flat = fs_cache.mask_bg_dilated_bool.to(DataType::Float32).reshape({static_cast<int>(H * W)});
        auto is_in_fg_strict  = eroded_flat.gather(0, lin_idx);     // [N] {0,1}
        auto is_in_bg_dilated = bg_dilated_flat.gather(0, lin_idx); // [N] {0,1}
        auto is_in_bg_strict  = Tensor::full({N}, 1.0f, Device::CUDA) - is_in_bg_dilated; // [N] {0,1}

        // ---- 4. Compute opacity gradient ----
        // For splats whose center is in BG_strict: positive gradient pushes
        // raw opacity down. For splats in FG_strict: small negative gradient
        // fills transparency holes. Splats in the free band fall through with
        // zero gradient in both terms.
        constexpr float kOutWeight = FOCUSED_ENABLE_CENTER_PENALTY_BG      ? 1.5f  : 0.0f; // BG center penalty strength
        constexpr float kInWeight  = FOCUSED_ENABLE_CENTER_PENALTY_FG_FILL ? 0.05f : 0.0f; // gentle FG fill (most work done by grad_alpha)

        auto valid_f = valid.to(DataType::Float32); // [N] float {0, 1}

        // Gradient w.r.t. activated opacity (sigmoid output).
        // is_in_bg_strict and is_in_fg_strict are mutually exclusive by
        // construction (eroded subset of dilated), so the free band cleanly
        // contributes 0 to both terms.
        auto grad_activated = valid_f *
            (is_in_bg_strict * kOutWeight - is_in_fg_strict * kInWeight) *
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
        ellipse_cfg.mask_expansion_fraction = 0.005f; // ~10px at 2K - aggressive to catch thin dark edge fringes (was 0.02 = ~40px, too generous)
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
