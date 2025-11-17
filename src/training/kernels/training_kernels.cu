/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <algorithm>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace gs::training {

    __global__ void generate_normal_noise_kernel(
        float* noise,
        int n_elements,
        unsigned long long seed) {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_elements)
            return;

        curandState state;
        curand_init(seed, idx, 0, &state);
        noise[idx] = curand_normal(&state);
    }

    void launch_generate_normal_noise(
        float* noise_buffer,
        size_t num_elements,
        unsigned long long seed,
        cudaStream_t stream) {

        int block_size = 256;
        int grid_size = (num_elements + block_size - 1) / block_size;

        generate_normal_noise_kernel<<<grid_size, block_size, 0, stream>>>(
            noise_buffer, num_elements, seed);
    }

    __device__ inline void raw_quat_to_rotmat(
        const float* raw_quat,
        float* R) {

        float w = raw_quat[0], x = raw_quat[1], y = raw_quat[2], z = raw_quat[3];

        float inv_norm = rsqrtf(x * x + y * y + z * z + w * w);
        inv_norm = fminf(inv_norm, 1e+12f);
        x *= inv_norm;
        y *= inv_norm;
        z *= inv_norm;
        w *= inv_norm;

        float x2 = x * x, y2 = y * y, z2 = z * z;
        float xy = x * y, xz = x * z, yz = y * z;
        float wx = w * x, wy = w * y, wz = w * z;

        R[0] = 1.f - 2.f * (y2 + z2);
        R[1] = 2.f * (xy - wz);
        R[2] = 2.f * (xz + wy);

        R[3] = 2.f * (xy + wz);
        R[4] = 1.f - 2.f * (x2 + z2);
        R[5] = 2.f * (yz - wx);

        R[6] = 2.f * (xz - wy);
        R[7] = 2.f * (yz + wx);
        R[8] = 1.f - 2.f * (x2 + y2);
    }

    __global__ void inject_noise_kernel(
        const float* raw_opacities,
        const float* raw_scales,
        const float* raw_quats,
        const float* noise,
        float* means,
        int N,
        float current_lr) {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= N)
            return;

        int idx_3d = 3 * idx;
        int idx_4d = 4 * idx;

        float scale_x = raw_scales[idx_3d + 0];
        float scale_y = raw_scales[idx_3d + 1];
        float scale_z = raw_scales[idx_3d + 2];

        float s2_x = expf(2.f * scale_x);
        float s2_y = expf(2.f * scale_y);
        float s2_z = expf(2.f * scale_z);

        float R[9];
        raw_quat_to_rotmat(raw_quats + idx_4d, R);

        float S2R[9];
        S2R[0] = s2_x * R[0];
        S2R[1] = s2_x * R[3];
        S2R[2] = s2_x * R[6];
        S2R[3] = s2_y * R[1];
        S2R[4] = s2_y * R[4];
        S2R[5] = s2_y * R[7];
        S2R[6] = s2_z * R[2];
        S2R[7] = s2_z * R[5];
        S2R[8] = s2_z * R[8];

        float cov[9];
        cov[0] = R[0] * S2R[0] + R[1] * S2R[3] + R[2] * S2R[6];
        cov[1] = R[0] * S2R[1] + R[1] * S2R[4] + R[2] * S2R[7];
        cov[2] = R[0] * S2R[2] + R[1] * S2R[5] + R[2] * S2R[8];

        cov[3] = R[3] * S2R[0] + R[4] * S2R[3] + R[5] * S2R[6];
        cov[4] = R[3] * S2R[1] + R[4] * S2R[4] + R[5] * S2R[7];
        cov[5] = R[3] * S2R[2] + R[4] * S2R[5] + R[5] * S2R[8];

        cov[6] = R[6] * S2R[0] + R[7] * S2R[3] + R[8] * S2R[6];
        cov[7] = R[6] * S2R[1] + R[7] * S2R[4] + R[8] * S2R[7];
        cov[8] = R[6] * S2R[2] + R[7] * S2R[5] + R[8] * S2R[8];

        float noise_x = noise[idx_3d + 0];
        float noise_y = noise[idx_3d + 1];
        float noise_z = noise[idx_3d + 2];

        float transformed_x = cov[0] * noise_x + cov[1] * noise_y + cov[2] * noise_z;
        float transformed_y = cov[3] * noise_x + cov[4] * noise_y + cov[5] * noise_z;
        float transformed_z = cov[6] * noise_x + cov[7] * noise_y + cov[8] * noise_z;

        float opacity = 1.f / (1.f + expf(-raw_opacities[idx]));
        float op_sigmoid = 1.f / (1.f + expf(100.f * opacity - 0.5f));
        float noise_factor = current_lr * op_sigmoid;

        means[idx_3d + 0] += noise_factor * transformed_x;
        means[idx_3d + 1] += noise_factor * transformed_y;
        means[idx_3d + 2] += noise_factor * transformed_z;
    }

    void launch_inject_noise_to_gaussians(
        float* raw_opacities,
        float* raw_scales,
        float* raw_quats,
        const float* noise,
        float* means,
        int N,
        float current_lr,
        cudaStream_t stream) {

        int block_size = 256;
        int grid_size = (N + block_size - 1) / block_size;

        inject_noise_kernel<<<grid_size, block_size, 0, stream>>>(
            raw_opacities, raw_scales, raw_quats, noise, means, N, current_lr);
    }

    __global__ void background_blend_kernel(
        float* output,
        const float* base_bg,
        const float* sine_bg,
        float mix_weight) {
        // Properly blend base background with sine background
        output[0] = base_bg[0] * (1.0f - mix_weight) + sine_bg[0] * mix_weight;
        output[1] = base_bg[1] * (1.0f - mix_weight) + sine_bg[1] * mix_weight;
        output[2] = base_bg[2] * (1.0f - mix_weight) + sine_bg[2] * mix_weight;
    }

    __global__ void add_scale_regularization_kernel(
        float* grad_scales,
        const float* scales_raw,
        float reg_weight,
        int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            // exp(scale_raw) * reg_weight / n
            float scale = expf(scales_raw[idx]);
            grad_scales[idx] += scale * reg_weight / float(n);
        }
    }

    __global__ void add_opacity_regularization_kernel(
        float* grad_opacity,
        const float* opacity_raw,
        float reg_weight,
        int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            // sigmoid(opacity_raw) * (1 - sigmoid(opacity_raw)) * reg_weight / n
            float sig = 1.0f / (1.0f + expf(-opacity_raw[idx]));
            float sigmoid_grad = sig * (1.0f - sig);
            grad_opacity[idx] += sigmoid_grad * reg_weight / float(n);
        }
    }

    __global__ void zero_tensor_kernel(float* data, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] = 0.0f;
        }
    }

    __global__ void fill_tensor_kernel(float* data, float value, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] = value;
        }
    }

    // L1 loss kernels
    __global__ void compute_l1_loss_forward_kernel(
        const float* pred,
        const float* gt,
        float* diff,
        float* abs_diff,
        float* l1_grad,
        int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            float d = pred[idx] - gt[idx];
            diff[idx] = d;
            abs_diff[idx] = fabsf(d);
            l1_grad[idx] = (d > 0.0f) ? (1.0f / float(n)) : (-1.0f / float(n));
        }
    }

    // Kernel to compute mean of array (sum then divide)
    __global__ void compute_mean_kernel(
        const float* data,
        float* result,
        int n) {
        // Simple reduction - for production use CUB
        extern __shared__ float sdata[];

        int tid = threadIdx.x;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        float val = (idx < n) ? data[idx] : 0.0f;
        sdata[tid] = val;
        __syncthreads();

        // Reduction in shared memory
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            atomicAdd(result, sdata[0] / float(n));
        }
    }

    // Combine gradients with weights
    __global__ void combine_gradients_kernel(
        float* output,
        const float* grad1,
        float weight1,
        const float* grad2,
        float weight2,
        int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            output[idx] = grad1[idx] * weight1 + grad2[idx] * weight2;
        }
    }

    // Copy kernel (device to device)
    __global__ void copy_kernel(
        float* dst,
        const float* src,
        int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            dst[idx] = src[idx];
        }
    }

    __global__ void compute_sine_background_kernel(
        float* output,
        int step,
        int periodR, int periodG, int periodB,
        float jitter_seed) {

        const float two_pi = 6.28318530718f;
        const float eps = 1e-4f;

        // Phase calculation
        float tR = (periodR > 0) ? float(step % periodR) / float(periodR) : 0.0f;
        float tG = (periodG > 0) ? float(step % periodG) / float(periodG) : 0.0f;
        float tB = (periodB > 0) ? float(step % periodB) / float(periodB) : 0.0f;

        float phaseR = two_pi * tR;
        float phaseG = two_pi * tG;
        float phaseB = two_pi * tB;

        // Compute sine values with phase shift
        output[0] = 0.5f * (1.0f + sinf(phaseR + 0.0f * two_pi / 3.0f));
        output[1] = 0.5f * (1.0f + sinf(phaseG + 1.0f * two_pi / 3.0f));
        output[2] = 0.5f * (1.0f + sinf(phaseB + 2.0f * two_pi / 3.0f));

        // Simple pseudo-random jitter (using step as seed)
        if (jitter_seed > 0.0f) {
            // Simple hash function for pseudo-random
            unsigned int hash = step * 1664525u + 1013904223u;
            float rand_r = (hash & 0xFFFF) / 65535.0f - 0.5f;
            hash = hash * 1664525u + 1013904223u;
            float rand_g = (hash & 0xFFFF) / 65535.0f - 0.5f;
            hash = hash * 1664525u + 1013904223u;
            float rand_b = (hash & 0xFFFF) / 65535.0f - 0.5f;

            output[0] = fmaxf(eps, fminf(1.0f - eps, output[0] + rand_r * 2.0f * jitter_seed));
            output[1] = fmaxf(eps, fminf(1.0f - eps, output[1] + rand_g * 2.0f * jitter_seed));
            output[2] = fmaxf(eps, fminf(1.0f - eps, output[2] + rand_b * 2.0f * jitter_seed));
        }
    }

    __global__ void blend_image_with_background_kernel(
        float* image,
        const float* alpha,
        const float* bg_color,
        int width,
        int height) {

        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height)
            return;

        int pixel_idx = y * width + x;
        float a = alpha[pixel_idx]; // Alpha is in HW format
        float one_minus_a = 1.0f - a;

        // Image is in CHW format, so each channel is contiguous
        // Channel 0 (R): [0, width*height)
        // Channel 1 (G): [width*height, 2*width*height)
        // Channel 2 (B): [2*width*height, 3*width*height)
        for (int c = 0; c < 3; ++c) {
            int idx = c * width * height + pixel_idx;
            // IMPORTANT: The formula should be: final = rendered + (1-alpha) * background
            // The rendered image already contains the foreground contribution
            image[idx] = image[idx] + one_minus_a * bg_color[c];
        }
    }

    // Transform kernels for SplatData
    __global__ void transform_positions_kernel(
        float* positions,
        const float* transform_matrix, // 4x4 matrix in row-major
        int n) {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            // Load position
            float x = positions[idx * 3 + 0];
            float y = positions[idx * 3 + 1];
            float z = positions[idx * 3 + 2];

            // Apply transform (assuming row-major matrix)
            float new_x = transform_matrix[0] * x + transform_matrix[1] * y + transform_matrix[2] * z + transform_matrix[3];
            float new_y = transform_matrix[4] * x + transform_matrix[5] * y + transform_matrix[6] * z + transform_matrix[7];
            float new_z = transform_matrix[8] * x + transform_matrix[9] * y + transform_matrix[10] * z + transform_matrix[11];

            // Store transformed position
            positions[idx * 3 + 0] = new_x;
            positions[idx * 3 + 1] = new_y;
            positions[idx * 3 + 2] = new_z;
        }
    }

    __global__ void transform_quaternions_kernel(
        float* quaternions,    // [N, 4] in [w, x, y, z] format
        const float* rot_quat, // Single quaternion to multiply with
        int n) {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            // Load quaternion
            float w2 = quaternions[idx * 4 + 0];
            float x2 = quaternions[idx * 4 + 1];
            float y2 = quaternions[idx * 4 + 2];
            float z2 = quaternions[idx * 4 + 3];

            // Load rotation quaternion
            float w1 = rot_quat[0];
            float x1 = rot_quat[1];
            float y1 = rot_quat[2];
            float z1 = rot_quat[3];

            // Quaternion multiplication: q_new = q_rot * q_original
            float new_w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
            float new_x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
            float new_y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2;
            float new_z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;

            // Store result
            quaternions[idx * 4 + 0] = new_w;
            quaternions[idx * 4 + 1] = new_x;
            quaternions[idx * 4 + 2] = new_y;
            quaternions[idx * 4 + 3] = new_z;
        }
    }

    __global__ void add_scalar_to_tensor_kernel(
        float* data,
        float scalar,
        int n) {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] += scalar;
        }
    }

    __global__ void compute_mean_3d_kernel(
        const float* positions,
        float* mean,
        int n) {

        // Simple version - for production use CUB reduction
        extern __shared__ float sdata[];

        int tid = threadIdx.x;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // Each thread accumulates its portion
        float sum_x = 0.0f, sum_y = 0.0f, sum_z = 0.0f;

        for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
            sum_x += positions[i * 3 + 0];
            sum_y += positions[i * 3 + 1];
            sum_z += positions[i * 3 + 2];
        }

        // Store in shared memory
        sdata[tid * 3 + 0] = sum_x;
        sdata[tid * 3 + 1] = sum_y;
        sdata[tid * 3 + 2] = sum_z;
        __syncthreads();

        // Reduction in shared memory
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid * 3 + 0] += sdata[(tid + s) * 3 + 0];
                sdata[tid * 3 + 1] += sdata[(tid + s) * 3 + 1];
                sdata[tid * 3 + 2] += sdata[(tid + s) * 3 + 2];
            }
            __syncthreads();
        }

        // Write result
        if (tid == 0) {
            atomicAdd(&mean[0], sdata[0] / float(n));
            atomicAdd(&mean[1], sdata[1] / float(n));
            atomicAdd(&mean[2], sdata[2] / float(n));
        }
    }

    __global__ void compute_distances_from_center_kernel(
        const float* positions,
        const float* center,
        float* distances,
        int n) {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            float dx = positions[idx * 3 + 0] - center[0];
            float dy = positions[idx * 3 + 1] - center[1];
            float dz = positions[idx * 3 + 2] - center[2];

            distances[idx] = sqrtf(dx * dx + dy * dy + dz * dz);
        }
    }

    // Wrapper functions to call from C++
    void launch_compute_sine_background(
        float* output,
        int step,
        int periodR, int periodG, int periodB,
        float jitter_amp,
        cudaStream_t stream) {
        compute_sine_background_kernel<<<1, 1, 0, stream>>>(
            output, step, periodR, periodG, periodB, jitter_amp);
    }

    void launch_background_blend(
        float* output,
        const float* base_bg,
        const float* sine_bg,
        float mix_weight,
        cudaStream_t stream) {
        background_blend_kernel<<<1, 1, 0, stream>>>(output, base_bg, sine_bg, mix_weight);
    }

    void launch_add_scale_regularization(
        float* grad_scales,
        const float* scales_raw,
        float reg_weight,
        int n,
        cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        add_scale_regularization_kernel<<<grid_size, block_size, 0, stream>>>(
            grad_scales, scales_raw, reg_weight, n);
    }

    void launch_add_opacity_regularization(
        float* grad_opacity,
        const float* opacity_raw,
        float reg_weight,
        int n,
        cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        add_opacity_regularization_kernel<<<grid_size, block_size, 0, stream>>>(
            grad_opacity, opacity_raw, reg_weight, n);
    }

    void launch_zero_tensor(float* data, int n, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        zero_tensor_kernel<<<grid_size, block_size, 0, stream>>>(data, n);
    }

    void launch_fill_tensor(float* data, float value, int n, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        fill_tensor_kernel<<<grid_size, block_size, 0, stream>>>(data, value, n);
    }

    void launch_compute_l1_loss_forward(
        const float* pred,
        const float* gt,
        float* diff,
        float* abs_diff,
        float* l1_grad,
        int n,
        cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        compute_l1_loss_forward_kernel<<<grid_size, block_size, 0, stream>>>(
            pred, gt, diff, abs_diff, l1_grad, n);
    }

    void launch_compute_mean(
        const float* data,
        float* result,
        int n,
        cudaStream_t stream) {
        // Zero the result first
        cudaMemsetAsync(result, 0, sizeof(float), stream);

        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        compute_mean_kernel<<<grid_size, block_size, block_size * sizeof(float), stream>>>(
            data, result, n);
    }

    void launch_combine_gradients(
        float* output,
        const float* grad1,
        float weight1,
        const float* grad2,
        float weight2,
        int n,
        cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        combine_gradients_kernel<<<grid_size, block_size, 0, stream>>>(
            output, grad1, weight1, grad2, weight2, n);
    }

    void launch_copy(
        float* dst,
        const float* src,
        int n,
        cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        copy_kernel<<<grid_size, block_size, 0, stream>>>(dst, src, n);
    }

    void launch_blend_image_with_background(
        float* image,
        const float* alpha,
        const float* bg_color,
        int width,
        int height,
        cudaStream_t stream) {

        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        blend_image_with_background_kernel<<<grid, block, 0, stream>>>(
            image, alpha, bg_color, width, height);
    }

    void launch_transform_positions(
        float* positions,
        const float* transform_matrix,
        int n,
        cudaStream_t stream) {

        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        transform_positions_kernel<<<grid_size, block_size, 0, stream>>>(
            positions, transform_matrix, n);
    }

    void launch_transform_quaternions(
        float* quaternions,
        const float* rot_quat,
        int n,
        cudaStream_t stream) {

        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        transform_quaternions_kernel<<<grid_size, block_size, 0, stream>>>(
            quaternions, rot_quat, n);
    }

    void launch_add_scalar_to_tensor(
        float* data,
        float scalar,
        int n,
        cudaStream_t stream) {

        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        add_scalar_to_tensor_kernel<<<grid_size, block_size, 0, stream>>>(
            data, scalar, n);
    }

    void launch_compute_mean_3d(
        const float* positions,
        float* mean,
        int n,
        cudaStream_t stream) {

        // Zero the mean first
        cudaMemsetAsync(mean, 0, 3 * sizeof(float), stream);

        int block_size = 256;
        int grid_size = std::min(1024, (n + block_size - 1) / block_size);
        compute_mean_3d_kernel<<<grid_size, block_size, block_size * 3 * sizeof(float), stream>>>(
            positions, mean, n);
    }

    void launch_compute_distances_from_center(
        const float* positions,
        const float* center,
        float* distances,
        int n,
        cudaStream_t stream) {

        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        compute_distances_from_center_kernel<<<grid_size, block_size, 0, stream>>>(
            positions, center, distances, n);
    }

} // namespace gs::training