/* SPDX-FileCopyrightText: 2026 SMN | Víctor M. Feliz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "smn_attention_prune.hpp"

#include "smn_attention_constants.h"
#include "smn_attention_project_kernel.hpp"

#include "core/logger.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"

#include <cstdint>

namespace lfs::training::smn {

    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::Tensor;

    std::expected<void, std::string> run_attention_prune(
        lfs::training::IStrategy& strategy,
        const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras,
        const AttentionPruneConfig& config) {

        if constexpr (!SMN_ATTENTION_PRUNE_ENABLED) {
            return {}; // Compile-time disabled: nothing to do.
        }

        try {
            lfs::core::SplatData& model = strategy.get_model();
            const int64_t num_gaussians = static_cast<int64_t>(model.size());
            if (num_gaussians <= 0) {
                LOG_INFO("[SMN attention prune] No Gaussians to prune.");
                return {};
            }

            const cudaStream_t stream = lfs::core::getCurrentCUDAStream();

            // World-space centers [N,3], resident on CUDA for the projection kernel.
            Tensor means = model.get_means().to(Device::CUDA).contiguous();
            means.sync_to_stream(stream);

            // Per-Gaussian vote accumulators, zeroed before the first view.
            Tensor inside_votes = Tensor::zeros(
                {static_cast<size_t>(num_gaussians)}, Device::CUDA, DataType::Int32);
            Tensor visible_votes = Tensor::zeros(
                {static_cast<size_t>(num_gaussians)}, Device::CUDA, DataType::Int32);
            inside_votes.sync_to_stream(stream);
            visible_votes.sync_to_stream(stream);

            size_t views_used = 0;
            size_t views_skipped = 0;
            for (const auto& cam_ptr : cameras) {
                lfs::core::Camera* cam = cam_ptr.get();
                if (cam == nullptr || !cam->has_mask()) {
                    ++views_skipped;
                    continue;
                }

                // Reproduce the training-time mask, binarized to {0,1}.
                Tensor mask = cam->load_and_get_mask(
                    config.resize_factor,
                    config.max_width,
                    config.invert_masks,
                    config.mask_threshold,
                    /*binarize=*/true);
                if (!mask.is_valid() || mask.numel() == 0) {
                    ++views_skipped;
                    continue;
                }

                Tensor mask_2d = mask.ndim() == 3 ? mask.squeeze(0) : mask;
                Tensor mask_f =
                    (mask_2d.dtype() == DataType::UInt8 || mask_2d.dtype() == DataType::Bool)
                        ? mask_2d.gt(0).to(DataType::Float32)
                        : mask_2d;
                mask_f = mask_f.to(Device::CUDA).contiguous();
                mask_f.sync_to_stream(stream);

                const int H = static_cast<int>(mask_f.shape()[0]);
                const int W = static_cast<int>(mask_f.shape()[1]);

                const auto [fx, fy, cx, cy] = cam->get_intrinsics();
                cam->world_view_transform().sync_to_stream(stream);

                launch_attention_projection_vote(
                    means.ptr<float>(),
                    static_cast<int>(num_gaussians),
                    cam->world_view_transform_ptr(),
                    fx, fy, cx, cy,
                    W, H,
                    mask_f.ptr<float>(),
                    SMN_ATTENTION_PRUNE_MASK_THRESHOLD,
                    SMN_ATTENTION_PRUNE_NEAR_PLANE,
                    inside_votes.ptr<int>(),
                    visible_votes.ptr<int>(),
                    stream);
                ++views_used;
            }

            if (views_used == 0) {
                LOG_WARN("[SMN attention prune] No usable masked views ({} skipped); skipping prune.",
                         views_skipped);
                return {};
            }

            // Keep a Gaussian only when it is seen often enough AND lands inside the
            // mask in at least KEEP_THRESHOLD of the views that see it.
            const Tensor visible_f = visible_votes.to(DataType::Float32);
            const float total_visibility = visible_f.sum().item();
            if (total_visibility <= 0.0f) {
                LOG_WARN("[SMN attention prune] No visibility accumulated; skipping prune.");
                return {};
            }

            const Tensor inside_f = inside_votes.to(DataType::Float32);
            const Tensor inside_ratio = inside_f.div(visible_f.clamp_min(1.0f));

            const Tensor meets_visibility =
                visible_votes.ge(SMN_ATTENTION_PRUNE_MIN_VISIBILITY).to(DataType::Float32);
            const Tensor inside_enough =
                inside_ratio.ge(SMN_ATTENTION_PRUNE_KEEP_THRESHOLD).to(DataType::Float32);

            // keep = meets_visibility AND inside_enough ; prune = NOT keep.
            const Tensor keep = meets_visibility.mul(inside_enough);
            const float keep_count = keep.sum().item();
            if (keep_count <= 0.0f) {
                LOG_WARN("[SMN attention prune] Keep set would be empty (threshold={:.2f}); "
                         "skipping prune to avoid deleting the whole model.",
                         SMN_ATTENTION_PRUNE_KEEP_THRESHOLD);
                return {};
            }

            const Tensor prune_mask = keep.lt(0.5f); // bool [N]: true = remove
            const int removed =
                static_cast<int>(prune_mask.to(DataType::Int32).sum().item());
            if (removed <= 0) {
                LOG_INFO("[SMN attention prune] Nothing to remove ({} views used).", views_used);
                return {};
            }

            strategy.remove_gaussians(prune_mask);

            LOG_INFO("[SMN attention prune] Removed {} / {} Gaussians "
                     "(views={}, keep_threshold={:.2f}, min_visibility={}).",
                     removed, num_gaussians, views_used,
                     SMN_ATTENTION_PRUNE_KEEP_THRESHOLD, SMN_ATTENTION_PRUNE_MIN_VISIBILITY);
            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::string("SMN attention prune failed: ") + e.what());
        }
    }

} // namespace lfs::training::smn
