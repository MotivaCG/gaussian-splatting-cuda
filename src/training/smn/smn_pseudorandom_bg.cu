/* SPDX-FileCopyrightText: 2026 SMN | Víctor M. Feliz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "smn_pseudorandom_bg.hpp"

#include "smn_pseudorandom_constants.h"

#include "core/assert.hpp"
#include "core/cuda_error.hpp"
#include "../kernels/kernel_stream.hpp"

#include <algorithm>
#include <cstddef>

namespace lfs::training::smn {
    namespace {

        constexpr int kThreadsPerBlock = 256;
        constexpr size_t kMaxBlocks = 65535;

        // Integer bit-mix (Murmur-style finalizer). Cheap, well-distributed.
        // __host__ too: the per-step grid offset is computed on the host.
        __host__ __device__ __forceinline__ uint32_t hash_u32(uint32_t x) {
            x ^= x >> 16;
            x *= 0x7feb352dU;
            x ^= x >> 15;
            x *= 0x846ca68bU;
            x ^= x >> 16;
            return x;
        }

        __global__ void pseudorandom_bg_kernel(
            float* __restrict__ out,
            const int height,
            const int width,
            const int block_px,
            const uint32_t seed,
            const int off_x,
            const int off_y) {

            const size_t n = static_cast<size_t>(height) * static_cast<size_t>(width);
            for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
                 idx < n;
                 idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
                const int x = static_cast<int>(idx % width);
                const int y = static_cast<int>(idx / width);

                // Block index under the per-step offset.
                const uint32_t bx = static_cast<uint32_t>((x + off_x) / block_px);
                const uint32_t by = static_cast<uint32_t>((y + off_y) / block_px);

                // One of the 8 cube corners for this block. The low 3 bits of the
                // hash map directly to (R,G,B) in {0,1}^3 = the full RGBCMYKW set.
                const uint32_t h = hash_u32(bx * 2654435761u ^ (by * 40503u) ^ seed);
                out[0 * n + idx] = (h & 1u) ? 1.0f : 0.0f; // R
                out[1 * n + idx] = (h & 2u) ? 1.0f : 0.0f; // G
                out[2 * n + idx] = (h & 4u) ? 1.0f : 0.0f; // B
            }
        }

    } // namespace

    void launch_pseudorandom_background(
        float* out,
        const int height,
        const int width,
        const uint64_t seed,
        cudaStream_t stream) {
        LFS_ASSERT_MSG(out != nullptr, "pseudorandom background: output pointer must not be null");
        LFS_ASSERT_MSG(height > 0 && width > 0, "pseudorandom background: dimensions must be positive");

        constexpr int block_px = SMN_PSEUDORANDOM_BLOCK_PX > 0 ? SMN_PSEUDORANDOM_BLOCK_PX : 1;

        // Whole-grid pixel offset for this step, so blocks don't sit at fixed pixels.
        const uint32_t s = static_cast<uint32_t>(seed ^ (seed >> 32));
        const int off_x = static_cast<int>(hash_u32(s * 2u + 1u) % static_cast<uint32_t>(block_px));
        const int off_y = static_cast<int>(hash_u32(s * 2u + 7u) % static_cast<uint32_t>(block_px));

        stream = resolve_stream(stream);
        const size_t n = static_cast<size_t>(height) * static_cast<size_t>(width);
        const int blocks = static_cast<int>(
            std::min((n + kThreadsPerBlock - 1) / kThreadsPerBlock, kMaxBlocks));
        pseudorandom_bg_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
            out, height, width, block_px, s, off_x, off_y);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.smn.pseudorandom_background");
    }

} // namespace lfs::training::smn
