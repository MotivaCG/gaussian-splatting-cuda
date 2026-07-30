/* SPDX-FileCopyrightText: 2026 SMN | Víctor M. Feliz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace lfs::training::smn {

    // Fill a [3,H,W] Float32 CUDA buffer with the pseudorandom background: a blocky
    // map of the 8 RGB cube corners (RGBCMYKW), block colors and a whole-grid pixel
    // offset both derived from `seed` (use the iteration) so the pattern is fresh
    // and decorrelated every call. See smn_pseudorandom_constants.h.
    void launch_pseudorandom_background(
        float* out,      // [3, H, W] CUDA, row-major (channel-planar)
        int height,
        int width,
        uint64_t seed,   // iteration
        cudaStream_t stream = nullptr);

} // namespace lfs::training::smn
