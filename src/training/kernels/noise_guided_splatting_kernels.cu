/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <cuda_runtime.h>

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

    __global__ void ngs_randomize_colors_kernel(
        float* sh0,               // [N, 1, 3] output
        const int* color_indices, // [N] random 0-5
        const int n) {

        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;

        const int c = color_indices[idx] % 6;
        sh0[idx * 3 + 0] = RGBCMY_SH0[c][0];
        sh0[idx * 3 + 1] = RGBCMY_SH0[c][1];
        sh0[idx * 3 + 2] = RGBCMY_SH0[c][2];
    }

    void launch_ngs_randomize_colors(float* sh0, const int* color_indices, int n) {
        constexpr int BLOCK = 256;
        ngs_randomize_colors_kernel<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(sh0, color_indices, n);
    }

} // namespace lfs::training
