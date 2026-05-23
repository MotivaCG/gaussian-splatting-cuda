/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// =============================================================================
// SMN / FocusedSegment per-camera tensor cache.
//
// THIS HEADER BELONGS TO THE SMN FORK (training/smn/).
// It is not used by any upstream code path. It lives in this directory so it
// stays grouped with the rest of the SMN-specific FocusedSegment code and so
// upstream merges never have to touch it.
//
// Purpose
// -------
// Holds memoized derivatives of each camera's mask and GT image that do NOT
// change between training iterations:
//   - mask_bool:     thresholded FG mask as bool (1 byte/px).
//   - darkness_fp16: inverse perceptual luminance (Rec.601), FP16 (2 bytes/px).
//   - fg_pixels / bg_pixels: scalar pixel counts that otherwise force a GPU
//                            sync (mask_f.sum().item<float>()) once per iter.
//
// All four entries are lazily populated on first use per camera inside the
// SMN helpers in training/smn/focused_segment_trainer.cpp; rebuilding them
// every iteration was costing ~250-900 us per iter and forcing two GPU syncs.
//
// Memory footprint per camera at 1296x2304 (~3M pixels):
//   - mask_bool:    ~3 MB
//   - darkness_fp16: ~6 MB
//   - total:        ~9 MB/cam     (e.g. 105 cams = ~945 MB)
//
// Owning container
// ----------------
// lfs::training::Trainer::fs_camera_cache_ (declared in trainer.hpp inside
// the FocusedSegment helpers block, clearly marked as SMN-only state).
// =============================================================================

#include "core/tensor.hpp"

#include <unordered_map>

namespace lfs::training::smn {

    // =========================================================================
    // FocusedSegment band radii (pixels). One independent knob per consumer.
    // Naming convention:
    //   <consumer>_INSET_PX   -> how far the zone steps INTO the mask
    //   <consumer>_OUTSET_PX  -> how far the zone steps OUT of the mask
    //
    // Each constant tells you directly what raising it does. The underlying
    // morphology operation (erode / dilate) is an implementation detail
    // documented per constant.
    // =========================================================================

    /// How many pixels INSIDE the annotated mask the alpha-UP zone starts.
    /// Built as: mask_eroded = erode(mask, N).
    /// Consumed by:
    ///   - per-pixel grad_alpha FG term (pushes alpha -> 1)
    ///   - center penalty FG fill (bonus opacity for splats with center inside)
    ///
    /// Raising it = alpha-up trusted only deeper inside the silhouette. The
    /// outer ring (last N px before the border) gets no alpha-up pressure
    /// and is left to the photometric loss. Protects the subject when some
    /// masks are buggy near the silhouette ("this border pixel is BG" votes
    /// from bad masks do not destroy opacity on the rest of the body).
    /// Cost of being generous: low.
    inline constexpr int FOCUSED_ALPHA_UP_INSET_PX = 7;

    /// How many pixels OUTSIDE the annotated mask the photometric attention
    /// extends. Built as: mask_photo = dilate(mask, N).
    /// Consumed by:
    ///   - spatial weight map FG region (FG weight 1.0, BG weight kBgWeight
    ///     applied to its complement)
    ///   - darkness boost area (linked to the FG region of the weight map)
    ///
    /// Raising it = photometric loss "sees" more border pixels as FG, which
    /// gives correct-color signal to splats covering fine features the
    /// matting trimmed (hair, fingers, eyelashes). Cost of being generous:
    /// some BG pixels near the border get weighted at FG level - mild
    /// dilution of background photometric signal.
    inline constexpr int FOCUSED_PHOTO_OUTSET_PX = 3;

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
    ///            this is where bright "halo" splats nucleate. Use only when
    ///            you need to recover fine border features at the cost of
    ///            some halos.
    ///
    ///   N = 0  : alpha-down starts exactly at the silhouette (matches the
    ///            old trainer_oldmode.cpp). Kills any splat with center
    ///            outside the matting cut.
    ///
    ///   N < 0  : alpha-down extends |N| px INSIDE the silhouette. Kills
    ///            "fuzz" / halo splats whose center sits in the outer ring
    ///            of the matting (typical when the matting is over-generous
    ///            and includes a few px of background around fine hair).
    ///            The rendered silhouette contracts to roughly (mask - |N|).
    ///
    /// Constraint when negative: |N| must be < FOCUSED_ALPHA_UP_INSET_PX,
    /// otherwise alpha-up and alpha-down overlap in the same ring and their
    /// gradients fight each other.
    ///
    /// Typical values: 0 for tight masks, -2 to -3 for matting that includes
    /// a few px of background fuzz around the subject. Avoid > 3 (halos).
    inline constexpr int FOCUSED_ALPHA_DOWN_OUTSET_PX = -2;

    // Sanity: when the alpha-down outset is negative it eats INTO the FG side
    // of the band; if it eats as far as (or past) the alpha-up inset, the two
    // zones overlap and their gradients pull a single ring in opposite
    // directions every iteration. Forbid that at compile time.
    static_assert(
        FOCUSED_ALPHA_DOWN_OUTSET_PX >= 0 ||
            (-FOCUSED_ALPHA_DOWN_OUTSET_PX) < FOCUSED_ALPHA_UP_INSET_PX,
        "FOCUSED_ALPHA_DOWN_OUTSET_PX is negative AND its magnitude reaches the "
        "alpha-up zone. Pick |OUTSET| < FOCUSED_ALPHA_UP_INSET_PX so the two "
        "rings do not overlap.");

    /// Per-camera memoized derivatives used by SMN FocusedSegment helpers.
    /// Constant per camera; rebuilt only when invalidated (resize_factor change).
    struct FocusedSegmentCameraCache {
        lfs::core::Tensor mask_bool;              // [H, W] bool - thresholded original mask (>0.5)
        lfs::core::Tensor mask_eroded_bool;       // [H, W] bool - FG_strict      (alpha-up zone)
        lfs::core::Tensor mask_photo_dilated_bool;// [H, W] bool - photometric FG (spatial weight map FG)
        lfs::core::Tensor mask_bg_dilated_bool;   // [H, W] bool - BG_strict outer edge; complement = BG_strict
        lfs::core::Tensor darkness_fp16;          // [H, W] FP16 - 1 - perceptual luminance

        // Per-camera pixel counts.
        float fg_pixels = 0.0f;            // count(mask_bool)                - kept for compatibility
        float bg_pixels = 0.0f;            // numel - fg_pixels
        float fg_eroded_pixels = 0.0f;     // count(mask_eroded_bool)         - FG_strict area
        float fg_photo_pixels = 0.0f;      // count(mask_photo_dilated_bool)  - photometric FG area
        float bg_photo_pixels = 0.0f;      // numel - fg_photo_pixels         - photometric BG area
        float bg_strict_pixels = 0.0f;     // numel - count(mask_bg_dilated_bool) - BG_strict area

        // mean(darkness) restricted to the photometric FG region (= mask_photo_dilated).
        // Used to compute weight_mean analytically in focused_segment_compute_loss,
        // killing the last per-iter GPU sync.
        float mean_darkness_photo = 0.0f;

        bool has_mask() const { return mask_bool.is_valid(); }
        bool has_bands() const {
            return mask_eroded_bool.is_valid()
                && mask_photo_dilated_bool.is_valid()
                && mask_bg_dilated_bool.is_valid();
        }
        bool has_darkness() const { return darkness_fp16.is_valid(); }
    };

    /// Global stats accumulated across all visited cameras. Used to normalize
    /// grad_alpha pressure globally instead of per-camera, so the contribution
    /// of each view is proportional to its area rather than inversely scaled by
    /// it (the per-camera 1/fg_pixels normalization in the old path amplified
    /// the influence of cameras with extreme FG/BG ratios - exactly the bad
    /// masks we want to dilute).
    ///
    /// Maintained lazily: only the cameras whose cache has been built so far
    /// contribute. As more cameras are visited the mean stabilizes; with
    /// ~105 cameras per scene the mean is essentially final after the first
    /// epoch.
    struct FocusedSegmentGlobalStats {
        double total_fg_eroded_pixels = 0.0;
        double total_bg_strict_pixels = 0.0;
        int n_cams_with_mask = 0;

        float mean_fg_eroded() const {
            return (n_cams_with_mask > 0)
                ? static_cast<float>(total_fg_eroded_pixels / static_cast<double>(n_cams_with_mask))
                : 1.0f;
        }
        float mean_bg_strict() const {
            return (n_cams_with_mask > 0)
                ? static_cast<float>(total_bg_strict_pixels / static_cast<double>(n_cams_with_mask))
                : 1.0f;
        }
    };

    /// Wrapper kept under the original type name so trainer.hpp does not change.
    /// Holds both the per-camera cache map and the dataset-wide running stats.
    struct FocusedSegmentCameraCacheMap {
        std::unordered_map<int, FocusedSegmentCameraCache> map;
        FocusedSegmentGlobalStats stats;
    };

} // namespace lfs::training::smn
