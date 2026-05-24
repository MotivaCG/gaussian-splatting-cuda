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
    // FocusedSegment band radii (FOCUSED_ALPHA_UP_INSET_PX,
    // FOCUSED_PHOTO_OUTSET_PX, FOCUSED_ALPHA_DOWN_OUTSET_PX) live in the
    // anonymous namespace at the top of training/smn/focused_segment_trainer.cpp.
    //
    // They are NOT exposed here on purpose: keeping them in the .cpp turns
    // a tweak ("change one number to A/B test") from a full-project rebuild
    // (~11 min, because trainer.hpp -> visualizer -> mcp -> py -> app all
    // recompile when any header in this dependency line changes) into a
    // single-file recompile + relink (~1 min).
    //
    // If you ever need to consume one of those constants from another
    // translation unit, prefer exposing it through a noinline `int get_X()`
    // function defined in focused_segment_trainer.cpp rather than dragging
    // the constexpr back into this header.
    // =========================================================================

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
