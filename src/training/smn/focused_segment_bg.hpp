/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// =============================================================================
// SMN / FocusedSegment bg_noise with a CORRECT alpha gradient.
//
// THIS HEADER BELONGS TO THE SMN FORK (training/smn/).
//
// Upstream Trainer::apply_background_noise composes
//     out = rendered + (1 - alpha) * noise * w
// BEFORE PPISP, but the backward never accounts for this node's
// d(out)/d(alpha) = -noise*w: the generic fast_rasterize_backward only
// derives grad_alpha from the RASTERIZER's background. Result: the loss
// punishes the transparency flicker, but alpha never receives the direct
// "become opaque" gradient - colors/positions churn instead.
//
// This pair fixes that for FocusedSegment mode (gated at the trainer.cpp
// call sites by mask_mode + the --bg-noise CLI flag):
//  - focused_segment_bg_noise_apply(): same composite + noise schedule as
//    upstream (uniform per-pixel noise, full regen every 100 iters,
//    golden-ratio value drift in between, grace/ramp/hold/decay weight
//    curve), but it RETAINS this iteration's weighted noise buffer.
//  - focused_segment_bg_noise_alpha_grad(): called in the backward with
//    the post-PPISP/bilateral gradient (exact chain rule - the noise node
//    sits between the rasterizer and PPISP), accumulates
//    grad_alpha += -sum_c(grad[c] * noise[c]*w) into the loss's
//    grad_alpha via the existing fused kernel. Consumes the stored state.
//
// Per-pixel noise is the strongest anti-fake-transparency background:
// spatially smooth splats cannot match white noise with ANY color in any
// frame, and neighboring-pixel decorrelation makes SSIM see it too.
// =============================================================================

#include "core/tensor.hpp"

namespace lfs::training::smn {

    /// Forward: composite per-pixel noise over the uncovered ray fraction and
    /// retain the weighted noise for the backward. Returns `rendered`
    /// untouched (and arms nothing) when the schedule weight is ~0 or shapes
    /// are unexpected.
    lfs::core::Tensor focused_segment_bg_noise_apply(
        const lfs::core::Tensor& rendered, // [3, H, W] raw rasterizer output
        const lfs::core::Tensor& alpha,    // [1, H, W] or [H, W]
        int iter,
        int total_iters);

    /// Backward: accumulate the noise-compose alpha gradient into
    /// grad_alpha_extra ([H, W]; created if invalid). No-op unless the
    /// matching apply() armed state this iteration. Consumes the state.
    void focused_segment_bg_noise_alpha_grad(
        const lfs::core::Tensor& raster_grad, // [3, H, W] grad after PPISP/bilateral backward
        lfs::core::Tensor& grad_alpha_extra);

} // namespace lfs::training::smn
