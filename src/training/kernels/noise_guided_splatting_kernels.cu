/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <cuda_runtime.h>
#include <stdint.h>

namespace lfs::training {

    // RGBCMY colors in SH0 format: (color - 0.5) / C0
    // where C0 = 0.28209479177387814
    __constant__ float RGBCMY_SH0[6][3] = {
        {1.7724539f, -1.7724539f, -1.7724539f},  // Red:     (1-0.5)/C0, (0-0.5)/C0, (0-0.5)/C0
        {-1.7724539f, 1.7724539f, -1.7724539f},  // Green
        {-1.7724539f, -1.7724539f, 1.7724539f},  // Blue
        {-1.7724539f, 1.7724539f, 1.7724539f},   // Cyan
        {1.7724539f, -1.7724539f, 1.7724539f},   // Magenta
        {1.7724539f, 1.7724539f, -1.7724539f}    // Yellow
    };


    __device__ __forceinline__ uint32_t ngs_wang_hash(uint32_t x) {
        // Thomas Wang 32-bit integer hash
        x = (x ^ 61u) ^ (x >> 16);
        x *= 9u;
        x = x ^ (x >> 4);
        x *= 0x27d4eb2du;
        x = x ^ (x >> 15);
        return x;
    }

    __global__ void ngs_randomize_colors_seed_kernel(
        float* sh0,     // [N, ...] flattened, writes first 3 coeffs
        const int n,
        const int stride_floats, // floats per gaussian in SH0 tensor (>=3)
        const uint32_t seed) {

        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;

        
        const uint32_t h = ngs_wang_hash(seed ^ static_cast<uint32_t>(idx));
        const int c = static_cast<int>(h % 6u);

        float* out = sh0 + idx * stride_floats;
        out[0] = RGBCMY_SH0[c][0];
        out[1] = RGBCMY_SH0[c][1];
        out[2] = RGBCMY_SH0[c][2];
    }

    void launch_ngs_randomize_colors_seed(float* sh0, int n, int stride_floats, uint32_t seed) {
        constexpr int BLOCK = 256;
        ngs_randomize_colors_seed_kernel<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(sh0, n, stride_floats, seed);
    }

} // namespace lfs::training
