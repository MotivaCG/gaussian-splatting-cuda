/* SPDX-FileCopyrightText: 2026 SMN | Víctor M. Feliz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// =============================================================================
// SMN — Attention prune: per-Gaussian projection + inside/outside vote kernel
// =============================================================================
//
// For a single camera view this kernel projects every Gaussian center into the
// image, and accumulates two per-Gaussian counters:
//
//   visible_votes[i] += 1   if the center lands in front of the camera and
//                           within the image bounds for this view
//   inside_votes[i]  += 1   if, additionally, it lands on a masked ("inside")
//                           pixel
//
// One thread handles one Gaussian for one view, so no atomics are needed: the
// host loops over views and re-launches, letting each thread read-modify-write
// its own slot. Buffers must be zero-initialized before the first view.
//
// Kept deliberately raw-pointer based (like roi_weight_map.cu) so it lives in
// the LibTorch-free kernels library with no lfs::core::Tensor dependency.
// =============================================================================

#include <cstdint>

#include <cuda_runtime.h>

namespace lfs::training::smn {

    void launch_attention_projection_vote(
        const float* means,         // [N,3] world-space Gaussian centers, CUDA, row-major
        int num_gaussians,          // N
        const float* world_to_cam,  // [4,4] world->camera matrix, CUDA, row-major
        float fx, float fy,         // focal lengths (pixels)
        float cx, float cy,         // principal point (pixels)
        int width, int height,      // image dimensions for this view
        const float* mask,          // [H,W] binarized mask, CUDA, row-major (0/1)
        float mask_threshold,       // mask > threshold counts as "inside"
        float near_plane,           // camera-space z must exceed this to be visible
        int32_t* inside_votes,      // [N] accumulator, CUDA (pre-zeroed)
        int32_t* visible_votes,     // [N] accumulator, CUDA (pre-zeroed)
        cudaStream_t stream = nullptr);

} // namespace lfs::training::smn
