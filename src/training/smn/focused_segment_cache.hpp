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

    /// Per-camera memoized derivatives used by SMN FocusedSegment helpers.
    /// Constant per camera; rebuilt only when invalidated (resize_factor change).
    struct FocusedSegmentCameraCache {
        lfs::core::Tensor mask_bool;     // [H, W] bool   - thresholded FG (>0.5)
        lfs::core::Tensor darkness_fp16; // [H, W] FP16   - 1 - perceptual luminance
        float fg_pixels = 0.0f;          // max(sum(mask_bool), 1)
        float bg_pixels = 0.0f;          // max(numel - fg_pixels, 1)

        bool has_mask() const { return mask_bool.is_valid(); }
        bool has_darkness() const { return darkness_fp16.is_valid(); }
    };

    /// Key = Camera::uid() (int). Owned by Trainer.
    using FocusedSegmentCameraCacheMap = std::unordered_map<int, FocusedSegmentCameraCache>;

} // namespace lfs::training::smn
