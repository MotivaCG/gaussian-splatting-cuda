/* SPDX-FileCopyrightText: 2026 SMN | Víctor M. Feliz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "smn_attention_project_kernel.hpp"

#include "core/assert.hpp"
#include "core/cuda_error.hpp"
#include "../kernels/kernel_stream.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace lfs::training::smn {
    namespace {

        constexpr int kThreadsPerBlock = 256;
        constexpr size_t kMaxBlocks = 65535;

        // world_to_cam is row-major with element (r,c) at index r*4+c, matching the
        // convention used across the engine (see roi_weight_map.cu). A world point p
        // maps to camera space as cam = R * p + t, with R the top-left 3x3 and
        // t = (w2c[3], w2c[7], w2c[11]).
        __global__ void attention_projection_vote_kernel(
            const float* __restrict__ means,
            const int num_gaussians,
            const float* __restrict__ world_to_cam,
            const float fx, const float fy,
            const float cx, const float cy,
            const int width, const int height,
            const float* __restrict__ mask,
            const float mask_threshold,
            const float near_plane,
            int32_t* __restrict__ inside_votes,
            int32_t* __restrict__ visible_votes) {

            for (int i = blockIdx.x * blockDim.x + threadIdx.x;
                 i < num_gaussians;
                 i += blockDim.x * gridDim.x) {

                const float px = means[i * 3 + 0];
                const float py = means[i * 3 + 1];
                const float pz = means[i * 3 + 2];

                // World -> camera.
                const float cam_x = world_to_cam[0] * px + world_to_cam[1] * py + world_to_cam[2] * pz + world_to_cam[3];
                const float cam_y = world_to_cam[4] * px + world_to_cam[5] * py + world_to_cam[6] * pz + world_to_cam[7];
                const float cam_z = world_to_cam[8] * px + world_to_cam[9] * py + world_to_cam[10] * pz + world_to_cam[11];

                // Behind the near plane => not visible in this view.
                if (cam_z <= near_plane) {
                    continue;
                }

                // Pinhole projection to pixel coordinates.
                const float inv_z = 1.0f / cam_z;
                const int u = static_cast<int>(lroundf(fx * cam_x * inv_z + cx));
                const int v = static_cast<int>(lroundf(fy * cam_y * inv_z + cy));

                // Center must fall inside the frame to count as visible here.
                if (u < 0 || u >= width || v < 0 || v >= height) {
                    continue;
                }

                visible_votes[i] += 1;

                if (mask[static_cast<size_t>(v) * width + u] > mask_threshold) {
                    inside_votes[i] += 1;
                }
            }
        }

    } // namespace

    void launch_attention_projection_vote(
        const float* means,
        const int num_gaussians,
        const float* world_to_cam,
        const float fx, const float fy,
        const float cx, const float cy,
        const int width, const int height,
        const float* mask,
        const float mask_threshold,
        const float near_plane,
        int32_t* inside_votes,
        int32_t* visible_votes,
        cudaStream_t stream) {
        LFS_ASSERT_MSG(means != nullptr, "attention prune: means pointer must not be null");
        LFS_ASSERT_MSG(world_to_cam != nullptr, "attention prune: world_to_cam pointer must not be null");
        LFS_ASSERT_MSG(mask != nullptr, "attention prune: mask pointer must not be null");
        LFS_ASSERT_MSG(inside_votes != nullptr && visible_votes != nullptr,
                       "attention prune: vote buffers must not be null");
        LFS_ASSERT_MSG(width > 0 && height > 0, "attention prune: image dimensions must be positive");
        LFS_ASSERT_MSG(std::isfinite(fx) && fx > 0.0f && std::isfinite(fy) && fy > 0.0f,
                       "attention prune: focal lengths must be positive and finite");

        if (num_gaussians <= 0) {
            return;
        }

        stream = resolve_stream(stream);
        const size_t blocks = std::min(
            (static_cast<size_t>(num_gaussians) + kThreadsPerBlock - 1) / kThreadsPerBlock,
            kMaxBlocks);
        attention_projection_vote_kernel<<<static_cast<int>(blocks), kThreadsPerBlock, 0, stream>>>(
            means,
            num_gaussians,
            world_to_cam,
            fx, fy, cx, cy,
            width, height,
            mask,
            mask_threshold,
            near_plane,
            inside_votes,
            visible_votes);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.smn.attention_projection_vote");
    }

} // namespace lfs::training::smn
