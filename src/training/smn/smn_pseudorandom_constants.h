/* SPDX-FileCopyrightText: 2026 SMN | Víctor M. Feliz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// =============================================================================
// SMN — Pseudorandom background: fixed tuning constants
// =============================================================================
//
// "pseudorandom" is a training background mode (BackgroundMode::Pseudorandom).
// Each iteration it fills the render background with a blocky map of the 8 RGB
// cube corners (RGBCMYKW — the maximally-adversarial colors) plus a per-step
// pixel offset. The rasterizer composites render·alpha + bg·(1-alpha) and the
// loss is against the raw GT, so any residual transparency lets the vivid random
// background bleed through -> the optimizer is forced to raise alpha to full
// opacity EVERYWHERE (mask-free). Corners maximize the penalty; blocks keep SSIM
// clean; the offset gives a fresh, decorrelated pattern each step for free.
// =============================================================================

namespace lfs::training::smn {

    // Side (pixels) of each solid-color block. Larger = lower spatial frequency
    // (gentler on SSIM), smaller = finer. Per-pixel would be block size 1.
    inline constexpr int SMN_PSEUDORANDOM_BLOCK_PX = 64;

} // namespace lfs::training::smn
