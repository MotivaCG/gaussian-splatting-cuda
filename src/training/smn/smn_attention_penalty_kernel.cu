/* SPDX-FileCopyrightText: 2026 SMN | Víctor M. Feliz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "smn_attention_penalty_kernel.hpp"

#include "core/assert.hpp"
#include "core/cuda_error.hpp"
#include "../kernels/kernel_stream.hpp"

#include <algorithm>
#include <cstddef>

namespace lfs::training::smn {
    namespace {

        constexpr int kThreadsPerBlock = 256; // power of two: required by the reduction
        constexpr size_t kMaxBlocks = 65535;

        // Per-pixel grad_alpha + block-reduced contribution to the penalty loss.
        // `penalty_loss` doubles as the reduction accumulator and must be pre-zeroed.
        __global__ void attention_penalty_kernel(
            const float* __restrict__ alpha,
            const float* __restrict__ mask,
            const float* __restrict__ roi,
            const size_t n,
            const float cin,
            const float cout,
            const float inv_n,
            const float* __restrict__ couple,
            float* __restrict__ grad_alpha,
            float* __restrict__ penalty_loss) {

            const float cf = couple ? couple[0] : 1.0f;

            __shared__ float sdata[kThreadsPerBlock];
            float local = 0.0f;

            for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
                 i < n;
                 i += static_cast<size_t>(blockDim.x) * gridDim.x) {
                const float m = mask[i];
                const float r = roi ? roi[i] : 1.0f;
                const float m_in = m * r;
                const float m_out = (1.0f - m) * r;
                const float a = alpha[i];

                // Loss contribution (cin/cout folded in) and per-pixel gradient.
                local += cin * (1.0f - a) * m_in + cout * a * m_out;
                grad_alpha[i] = (cout * m_out - cin * m_in) * inv_n * cf;
            }

            // Block reduction into sdata[0].
            sdata[threadIdx.x] = local;
            __syncthreads();
            for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
                if (threadIdx.x < s) {
                    sdata[threadIdx.x] += sdata[threadIdx.x + s];
                }
                __syncthreads();
            }
            if (threadIdx.x == 0) {
                atomicAdd(penalty_loss, sdata[0]);
            }
        }

        // Scale the reduced sum by 1/N and the coupling factor to yield the loss.
        __global__ void attention_penalty_finalize_kernel(
            const float inv_n,
            const float* __restrict__ couple,
            float* __restrict__ penalty_loss) {
            const float cf = couple ? couple[0] : 1.0f;
            penalty_loss[0] = penalty_loss[0] * inv_n * cf;
        }

    } // namespace

    void launch_attention_opacity_penalty(
        const float* alpha,
        const float* mask,
        const float* roi,
        const int height,
        const int width,
        const float cin,
        const float cout,
        const float* couple,
        float* grad_alpha,
        float* penalty_loss,
        cudaStream_t stream) {
        LFS_ASSERT_MSG(alpha != nullptr && mask != nullptr, "attention penalty: alpha/mask must not be null");
        LFS_ASSERT_MSG(grad_alpha != nullptr && penalty_loss != nullptr,
                       "attention penalty: outputs must not be null");
        LFS_ASSERT_MSG(height > 0 && width > 0, "attention penalty: dimensions must be positive");

        stream = resolve_stream(stream);
        const size_t n = static_cast<size_t>(height) * static_cast<size_t>(width);
        const float inv_n = 1.0f / static_cast<float>(n);

        // penalty_loss is the atomic accumulator; start it at zero.
        LFS_CUDA_CHECK(cudaMemsetAsync(penalty_loss, 0, sizeof(float), stream));

        const int blocks = static_cast<int>(
            std::min((n + kThreadsPerBlock - 1) / kThreadsPerBlock, kMaxBlocks));
        attention_penalty_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
            alpha, mask, roi, n, cin, cout, inv_n, couple, grad_alpha, penalty_loss);
        attention_penalty_finalize_kernel<<<1, 1, 0, stream>>>(inv_n, couple, penalty_loss);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.smn.attention_opacity_penalty");
    }

} // namespace lfs::training::smn
