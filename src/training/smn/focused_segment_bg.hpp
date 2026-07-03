/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// =============================================================================
// SMN / FocusedSegment background-modulation override.
//
// THIS HEADER BELONGS TO THE SMN FORK (training/smn/).
//
// Old-mode extreme binary background colors for bg_modulation, ported from
// trainer_oldmode.cpp binary_extreme_background_for_step(): each iteration
// picks an extreme corner of the RGB cube (at least one channel at max and
// one at min), deterministically from the iteration index via a small LCG.
//
// Rationale: the implicit alpha gradient the rasterizer backward derives from
// the background compose term is grad_alpha = -sum_c(grad_image[c] * bg[c]).
// Extreme, rapidly changing colors maximize that signal, forcing alpha -> 1
// wherever the (masked) photometric loss is alive - upstream's smooth sinusoid
// is a weaker version of the same trick.
//
// Called from Trainer::background_for_step (single-line SMN hook) only when
// mask_mode == FocusedSegment. Implementation lives in
// focused_segment_trainer.cpp next to the FOCUSED_BG_EXTREME_COLORS switch;
// when that switch is false this function leaves rgb untouched and the caller
// keeps its sinusoidal color.
// =============================================================================

namespace lfs::training::smn {

    /// Overwrite rgb[3] with this iteration's extreme binary modulation color,
    /// scaled by the modulation weight w (same inv_weight_piecewise weight the
    /// sinusoidal path uses). No-op when FOCUSED_BG_EXTREME_COLORS is false.
    void focused_segment_extreme_bg_color(int iter, float w, float rgb[3]);

} // namespace lfs::training::smn
