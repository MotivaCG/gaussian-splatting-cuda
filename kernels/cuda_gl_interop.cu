/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "config.h"

#ifdef CUDA_GL_INTEROP_ENABLED

#include <cstdint>
#include <cuda_runtime.h>

namespace gs {

    // Kernel for converting RGB float to RGBA uint8
    __global__ void convertRGBFloatToRGBAUint8(
        const float* __restrict__ rgb,
        uint8_t* __restrict__ rgba,
        int width, int height) {

        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < width && y < height) {
            int idx = y * width + x;
            int rgb_idx = idx * 3;
            int rgba_idx = idx * 4;

            // Convert float [0,1] to uint8 [0,255] with clamping
            rgba[rgba_idx + 0] = min(255, max(0, __float2int_rn(rgb[rgb_idx + 0] * 255.0f)));
            rgba[rgba_idx + 1] = min(255, max(0, __float2int_rn(rgb[rgb_idx + 1] * 255.0f)));
            rgba[rgba_idx + 2] = min(255, max(0, __float2int_rn(rgb[rgb_idx + 2] * 255.0f)));
            rgba[rgba_idx + 3] = 255; // Alpha
        }
    }

    // Kernel for converting RGBA float to RGBA uint8
    __global__ void convertRGBAFloatToRGBAUint8(
        const float* __restrict__ rgba_in,
        uint8_t* __restrict__ rgba_out,
        int width, int height) {

        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < width && y < height) {
            int idx = (y * width + x) * 4;

            // Convert float [0,1] to uint8 [0,255] with clamping
            rgba_out[idx + 0] = min(255, max(0, __float2int_rn(rgba_in[idx + 0] * 255.0f)));
            rgba_out[idx + 1] = min(255, max(0, __float2int_rn(rgba_in[idx + 1] * 255.0f)));
            rgba_out[idx + 2] = min(255, max(0, __float2int_rn(rgba_in[idx + 2] * 255.0f)));
            rgba_out[idx + 3] = min(255, max(0, __float2int_rn(rgba_in[idx + 3] * 255.0f)));
        }
    }

    // Kernel for flipping image vertically (OpenGL uses bottom-left origin)
    template <typename T>
    __global__ void flipVertical(
        const T* __restrict__ input,
        T* __restrict__ output,
        int width, int height, int channels) {

        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < width && y < height) {
            int flipped_y = height - 1 - y;
            int src_idx = (y * width + x) * channels;
            int dst_idx = (flipped_y * width + x) * channels;

#pragma unroll
            for (int c = 0; c < channels; ++c) {
                output[dst_idx + c] = input[src_idx + c];
            }
        }
    }

    // Kernel for writing interleaved position+color data to VBO
    // positions: [N, 3], colors: [N, 3] -> output: [N, 6] interleaved
    __global__ void writeInterleavedPosColor(
        const float* __restrict__ positions,
        const float* __restrict__ colors,
        float* __restrict__ output,
        int num_points) {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < num_points) {
            int out_idx = idx * 6;
            int in_idx = idx * 3;

            // Write position (3 floats)
            output[out_idx + 0] = positions[in_idx + 0];
            output[out_idx + 1] = positions[in_idx + 1];
            output[out_idx + 2] = positions[in_idx + 2];

            // Write color (3 floats)
            output[out_idx + 3] = colors[in_idx + 0];
            output[out_idx + 4] = colors[in_idx + 1];
            output[out_idx + 5] = colors[in_idx + 2];
        }
    }

    // Host function to launch the kernel
    void launchWriteInterleavedPosColor(
        const float* positions,
        const float* colors,
        float* output,
        int num_points,
        cudaStream_t stream = 0) {

        const int threads = 256;
        const int blocks = (num_points + threads - 1) / threads;

        writeInterleavedPosColor<<<blocks, threads, 0, stream>>>(
            positions, colors, output, num_points);
    }

} // namespace gs

#endif // CUDA_GL_INTEROP_ENABLED