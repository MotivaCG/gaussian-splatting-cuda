/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>

namespace gs::training {

    void launch_generate_normal_noise(
        float* noise_buffer,
        size_t num_elements,
        unsigned long long seed,
        cudaStream_t stream);

    void launch_inject_noise_to_gaussians(
        float* raw_opacities,
        float* raw_scales,
        float* raw_quats,
        const float* noise,
        float* means,
        int N,
        float current_lr,
        cudaStream_t stream);

    // Background computation
    void launch_compute_sine_background(
        float* output,
        int step,
        int periodR, int periodG, int periodB,
        float jitter_amp,
        cudaStream_t stream);

    void launch_background_blend(
        float* output,
        const float* base_bg,
        const float* sine_bg,
        float mix_weight,
        cudaStream_t stream);

    // Regularization
    void launch_add_scale_regularization(
        float* grad_scales,
        const float* scales_raw,
        float reg_weight,
        int n,
        cudaStream_t stream);

    void launch_add_opacity_regularization(
        float* grad_opacity,
        const float* opacity_raw,
        float reg_weight,
        int n,
        cudaStream_t stream);

    // Basic tensor operations
    void launch_zero_tensor(
        float* data,
        int n,
        cudaStream_t stream);

    void launch_fill_tensor(
        float* data,
        float value,
        int n,
        cudaStream_t stream);

    // Loss computation
    void launch_compute_l1_loss_forward(
        const float* pred,
        const float* gt,
        float* diff,
        float* abs_diff,
        float* l1_grad,
        int n,
        cudaStream_t stream);

    void launch_compute_mean(
        const float* data,
        float* result,
        int n,
        cudaStream_t stream);

    void launch_combine_gradients(
        float* output,
        const float* grad1,
        float weight1,
        const float* grad2,
        float weight2,
        int n,
        cudaStream_t stream);

    // Memory operations
    void launch_copy(
        float* dst,
        const float* src,
        int n,
        cudaStream_t stream);

    // Image blending
    void launch_blend_image_with_background(
        float* image,
        const float* alpha,
        const float* bg_color,
        int width,
        int height,
        cudaStream_t stream);

    // Transform kernels
    void launch_transform_positions(
        float* positions,
        const float* transform_matrix,
        int n,
        cudaStream_t stream);

    void launch_transform_quaternions(
        float* quaternions,
        const float* rot_quat,
        int n,
        cudaStream_t stream);

    void launch_add_scalar_to_tensor(
        float* data,
        float scalar,
        int n,
        cudaStream_t stream);

    void launch_compute_mean_3d(
        const float* positions,
        float* mean,
        int n,
        cudaStream_t stream);

    void launch_compute_distances_from_center(
        const float* positions,
        const float* center,
        float* distances,
        int n,
        cudaStream_t stream);

} // namespace gs::training