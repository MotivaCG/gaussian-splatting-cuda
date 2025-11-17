/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core_new/logger.hpp"
#include "kernels/morton_encoding_new.cuh"
#include <cstdint>
#include <cuda_runtime.h>
#include <limits>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>

namespace lfs::core {

    __device__ __forceinline__ uint64_t splitBy3(uint32_t a) {
        uint64_t x = a & 0x1fffff;
        x = (x | x << 32) & 0x1f00000000ffff;
        x = (x | x << 16) & 0x1f0000ff0000ff;
        x = (x | x << 8) & 0x100f00f00f00f00f;
        x = (x | x << 4) & 0x10c30c30c30c30c3;
        x = (x | x << 2) & 0x1249249249249249;
        return x;
    }

    // Combined kernel that computes Morton codes given bbox parameters
    __global__ void morton_encode_kernel(
        const float* __restrict__ positions,
        int64_t* __restrict__ morton_codes,
        const int n_positions,
        float min_x, float min_y, float min_z,
        float cube_size) {

        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_positions)
            return;

        // Load position (x, y, z)
        const float x = positions[idx * 3 + 0];
        const float y = positions[idx * 3 + 1];
        const float z = positions[idx * 3 + 2];

        // Normalize to [0, 1] range
        const double size = double(cube_size);
        const double normalized_x = double(x - min_x) / size;
        const double normalized_y = double(y - min_y) / size;
        const double normalized_z = double(z - min_z) / size;

        // Scale to 21-bit integers (2^21 - 1 = 2097151)
        constexpr double factor = 2097151.0;
        const uint32_t ix = static_cast<uint32_t>(normalized_x * factor);
        const uint32_t iy = static_cast<uint32_t>(normalized_y * factor);
        const uint32_t iz = static_cast<uint32_t>(normalized_z * factor);

        // Compute Morton code by interleaving bits
        const uint64_t morton_code = splitBy3(ix) | (splitBy3(iy) << 1) | (splitBy3(iz) << 2);

        // Convert to signed int64 by adding int64_min for compatibility
        constexpr int64_t int64_min = std::numeric_limits<int64_t>::min();
        morton_codes[idx] = static_cast<int64_t>(morton_code) + int64_min;
    }

    // Struct for computing min/max in a single pass using thrust
    struct float3_minmax {
        float3 min_val;
        float3 max_val;

        __host__ __device__
        float3_minmax() : min_val{INFINITY, INFINITY, INFINITY},
                          max_val{-INFINITY, -INFINITY, -INFINITY} {}

        __host__ __device__
        float3_minmax(float3 min_v, float3 max_v) : min_val(min_v),
                                                    max_val(max_v) {}
    };

    struct minmax_op {
        __host__ __device__
            float3_minmax
            operator()(const float3_minmax& a, const float3_minmax& b) const {
            float3_minmax result;
            result.min_val.x = fminf(a.min_val.x, b.min_val.x);
            result.min_val.y = fminf(a.min_val.y, b.min_val.y);
            result.min_val.z = fminf(a.min_val.z, b.min_val.z);
            result.max_val.x = fmaxf(a.max_val.x, b.max_val.x);
            result.max_val.y = fmaxf(a.max_val.y, b.max_val.y);
            result.max_val.z = fmaxf(a.max_val.z, b.max_val.z);
            return result;
        }
    };

    struct position_to_minmax {
        const float* positions;

        __host__ __device__
        position_to_minmax(const float* pos) : positions(pos) {}

        __host__ __device__
            float3_minmax
            operator()(int idx) const {
            float3 pos;
            pos.x = positions[idx * 3 + 0];
            pos.y = positions[idx * 3 + 1];
            pos.z = positions[idx * 3 + 2];
            return float3_minmax(pos, pos);
        }
    };

    Tensor morton_encode_new(const Tensor& positions) {
        // Validate input
        if (!positions.is_valid()) {
            LOG_ERROR("morton_encode_new: Invalid input tensor");
            return Tensor();
        }

        if (positions.ndim() != 2 || positions.size(1) != 3) {
            LOG_ERROR("morton_encode_new: Positions must have shape [N, 3], got {}",
                      positions.shape().str());
            return Tensor();
        }

        if (positions.dtype() != DataType::Float32) {
            LOG_ERROR("morton_encode_new: Positions must be Float32");
            return Tensor();
        }

        if (positions.device() != Device::CUDA) {
            LOG_ERROR("morton_encode_new: Positions must be on CUDA");
            return Tensor();
        }

        const int n_positions = static_cast<int>(positions.size(0));

        // Compute bounding box in a single pass using thrust
        thrust::counting_iterator<int> first(0);
        thrust::counting_iterator<int> last(n_positions);

        position_to_minmax transform_op(positions.ptr<float>());
        float3_minmax init;

        float3_minmax bbox = thrust::transform_reduce(
            first, last,
            transform_op,
            init,
            minmax_op());

        // Compute cube size (maximum range across all dimensions)
        float range_x = bbox.max_val.x - bbox.min_val.x;
        float range_y = bbox.max_val.y - bbox.min_val.y;
        float range_z = bbox.max_val.z - bbox.min_val.z;
        float cube_size = fmaxf(fmaxf(range_x, range_y), range_z);

        // Add small epsilon to avoid division by zero
        cube_size = fmaxf(cube_size, 1e-7f);

        // Allocate output tensor for Morton codes
        auto morton_codes = Tensor::empty({static_cast<size_t>(n_positions)},
                                          Device::CUDA,
                                          DataType::Int64);

        // Launch kernel
        constexpr int block_size = 256;
        const int grid_size = (n_positions + block_size - 1) / block_size;

        morton_encode_kernel<<<grid_size, block_size>>>(
            positions.ptr<float>(),
            morton_codes.ptr<int64_t>(),
            n_positions,
            bbox.min_val.x,
            bbox.min_val.y,
            bbox.min_val.z,
            cube_size);

        // Check for CUDA errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            LOG_ERROR("CUDA error in morton_encode_kernel: {}", cudaGetErrorString(err));
            return Tensor();
        }

        // Synchronize to ensure kernel completion
        cudaDeviceSynchronize();

        return morton_codes;
    }

    Tensor morton_sort_indices_new(const Tensor& morton_codes) {
        // Validate input
        if (!morton_codes.is_valid()) {
            LOG_ERROR("morton_sort_indices_new: Invalid input tensor");
            return Tensor();
        }

        if (morton_codes.ndim() != 1) {
            LOG_ERROR("morton_sort_indices_new: Morton codes must be 1D tensor");
            return Tensor();
        }

        if (morton_codes.dtype() != DataType::Int64) {
            LOG_ERROR("morton_sort_indices_new: Morton codes must be Int64");
            return Tensor();
        }

        if (morton_codes.device() != Device::CUDA) {
            LOG_ERROR("morton_sort_indices_new: Morton codes must be on CUDA");
            return Tensor();
        }

        const size_t n = morton_codes.numel();

        // Create indices tensor [0, 1, 2, ..., n-1]
        auto indices = Tensor::empty({n}, Device::CUDA, DataType::Int64);

        // Use thrust to initialize sequence
        thrust::device_ptr<int64_t> indices_ptr(indices.ptr<int64_t>());
        thrust::sequence(indices_ptr, indices_ptr + n, 0LL);

        // Create a copy of morton codes for sorting (we'll sort by keys)
        auto morton_copy = morton_codes.clone();

        // Get device pointers for thrust
        thrust::device_ptr<int64_t> keys_ptr(morton_copy.ptr<int64_t>());
        thrust::device_ptr<int64_t> values_ptr(indices.ptr<int64_t>());

        // Sort indices by morton codes (ascending order)
        thrust::sort_by_key(keys_ptr, keys_ptr + n, values_ptr);

        // Check for CUDA errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            LOG_ERROR("CUDA error in morton_sort_indices_new: {}", cudaGetErrorString(err));
            return Tensor();
        }

        cudaDeviceSynchronize();

        return indices;
    }

} // namespace lfs::core
