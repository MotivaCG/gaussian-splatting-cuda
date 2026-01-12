/* ScanMeNow File */

/**
 * @file mask_pruning.cpp
 * @brief Implementation of post-training mask-based pruning operations.
 */

#include "core/logger.hpp"
#include "gsplat/Projection.h"
#include "mask_pruning.hpp"

#include <array>
#include <cmath>
#include <cstdio>

namespace lfs::training {
    namespace mask_pruning {

        // =============================================================================
        // Helper: Pad distortion tensor to required size
        // =============================================================================

        static lfs::core::Tensor pad_distortion(const lfs::core::Tensor& t, size_t target_size) {
            using namespace lfs::core;

            if (!t.is_valid() || t.numel() == 0) {
                return Tensor();
            }

            Tensor result = t.to(Device::CUDA).contiguous();
            const size_t current_size = result.shape()[result.ndim() - 1];

            if (current_size >= target_size) {
                return result;
            }

            // Pad with zeros
            const size_t pad_size = target_size - current_size;
            Tensor padding = Tensor::zeros({pad_size}, Device::CUDA);

            return Tensor::cat({result.flatten(), padding}, 0);
        }

        // =============================================================================
        // Helper: normalize mask to [H,W] CUDA contiguous
        // =============================================================================

        static bool normalize_mask_2d(lfs::core::Tensor& mask) {
            using namespace lfs::core;

            if (!mask.is_valid() || mask.numel() == 0) {
                return false;
            }

            if (mask.ndim() == 3 && mask.shape()[0] == 1) {
                mask = mask.squeeze(0);
            }
            if (mask.ndim() != 2) {
                return false;
            }

            if (mask.device() != Device::CUDA) {
                mask = mask.to(Device::CUDA);
            }
            if (!mask.is_contiguous()) {
                mask = mask.contiguous();
            }
            return true;
        }

        // =============================================================================
        // Fast Projection
        // =============================================================================
        // gsplat expects ::CameraModelType (global enum), while our camera returns lfs::core::CameraModelType.
        static_assert(static_cast<int>(lfs::core::CameraModelType::PINHOLE) == ::PINHOLE);
        static_assert(static_cast<int>(lfs::core::CameraModelType::ORTHO) == ::ORTHO);
        static_assert(static_cast<int>(lfs::core::CameraModelType::FISHEYE) == ::FISHEYE);
        static_assert(static_cast<int>(lfs::core::CameraModelType::EQUIRECTANGULAR) == ::EQUIRECTANGULAR);
        static_assert(static_cast<int>(lfs::core::CameraModelType::THIN_PRISM_FISHEYE) == ::THIN_PRISM_FISHEYE);

        static CameraModelType to_gsplat_camera_model(lfs::core::CameraModelType t) {
            switch (t) {
            case lfs::core::CameraModelType::PINHOLE: return ::PINHOLE;
            case lfs::core::CameraModelType::ORTHO: return ::ORTHO;
            case lfs::core::CameraModelType::FISHEYE: return ::FISHEYE;
            case lfs::core::CameraModelType::EQUIRECTANGULAR: return ::EQUIRECTANGULAR;
            case lfs::core::CameraModelType::THIN_PRISM_FISHEYE: return ::THIN_PRISM_FISHEYE;
            default: return ::PINHOLE;
            }
        }

        std::expected<ProjectionResult, std::string> project_splats(
            const lfs::core::Camera& camera,
            const lfs::core::SplatData& splat_data,
            const CenterVotePruningConfig& config) {
            using namespace lfs::core;

            const int64_t N = static_cast<int64_t>(splat_data.size());
            if (N <= 0) {
                return std::unexpected("No splats to project");
            }

            // Get splat data (activated values)
            // IMPORTANT: ensure CUDA + contiguous before passing raw pointers to kernel
            Tensor means3D = splat_data.get_means();     // [N, 3]
            Tensor quats = splat_data.get_rotation();    // [N, 4]
            Tensor scales = splat_data.get_scaling();    // [N, 3]
            Tensor opacities = splat_data.get_opacity(); // [N] or [N,1]

            if (opacities.is_valid() && opacities.ndim() == 2 && opacities.shape()[1] == 1) {
                opacities = opacities.squeeze(1);
            }

            if (means3D.device() != Device::CUDA)
                means3D = means3D.to(Device::CUDA);
            if (quats.device() != Device::CUDA)
                quats = quats.to(Device::CUDA);
            if (scales.device() != Device::CUDA)
                scales = scales.to(Device::CUDA);
            if (opacities.is_valid() && opacities.device() != Device::CUDA)
                opacities = opacities.to(Device::CUDA);

            means3D = means3D.contiguous();
            quats = quats.contiguous();
            scales = scales.contiguous();
            if (opacities.is_valid())
                opacities = opacities.contiguous();

            // Apply scaling modifier
            Tensor scaled_scales = (scales * config.scaling_modifier).contiguous();

            // Camera data (Camera guarantees CUDA + contiguous for w2v and K())
            Tensor viewmat_batched = camera.world_view_transform(); // [1,4,4] CUDA
            Tensor K_batched = camera.K();                          // [1,3,3] CUDA
            const int W = camera.image_width();
            const int H = camera.image_height();

            if (viewmat_batched.ndim() == 2)
                viewmat_batched = viewmat_batched.unsqueeze(0);
            if (K_batched.ndim() == 2)
                K_batched = K_batched.unsqueeze(0);

            viewmat_batched = viewmat_batched.contiguous();
            K_batched = K_batched.contiguous();

            const uint32_t C = 1;
            const uint32_t n_points = static_cast<uint32_t>(N);

            // Distortion (optional)
            Tensor radial = pad_distortion(camera.radial_distortion(), 4);
            Tensor tangential = pad_distortion(camera.tangential_distortion(), 2);

            // Allocate output tensors
            Tensor radii = Tensor::empty({C, static_cast<size_t>(N), 2}, Device::CUDA, DataType::Int32);
            Tensor means2d = Tensor::empty({C, static_cast<size_t>(N), 2}, Device::CUDA, DataType::Float32);
            Tensor depths = Tensor::empty({C, static_cast<size_t>(N)}, Device::CUDA, DataType::Float32);
            Tensor conics = Tensor::empty({C, static_cast<size_t>(N), 3}, Device::CUDA, DataType::Float32);

            gsplat_lfs::launch_projection_ut_3dgs_fused_kernel(
                means3D.ptr<float>(),
                quats.ptr<float>(),
                scaled_scales.ptr<float>(),
                opacities.is_valid() ? opacities.ptr<float>() : nullptr,
                viewmat_batched.ptr<float>(),
                nullptr, // viewmats1
                K_batched.ptr<float>(),
                n_points,
                C,
                static_cast<uint32_t>(W),
                static_cast<uint32_t>(H),
                config.eps2d,
                config.near_plane,
                config.far_plane,
                config.radius_clip,
                to_gsplat_camera_model(camera.camera_model_type()),
                UnscentedTransformParameters{},
                ShutterType::GLOBAL,
                radial.is_valid() ? radial.ptr<float>() : nullptr,
                tangential.is_valid() ? tangential.ptr<float>() : nullptr,
                nullptr, // thin_prism
                radii.ptr<int32_t>(),
                means2d.ptr<float>(),
                depths.ptr<float>(),
                conics.ptr<float>(),
                nullptr, // compensations
                nullptr  // stream
            );

            // Squeeze batch dimension since C=1
            radii = radii.squeeze(0).contiguous();     // [N, 2]
            means2d = means2d.squeeze(0).contiguous(); // [N, 2]
            depths = depths.squeeze(0).contiguous();   // [N]
            conics = conics.squeeze(0).contiguous();   // [N, 3]

            return ProjectionResult{
                .radii = std::move(radii),
                .means2d = std::move(means2d),
                .depths = std::move(depths),
                .conics = std::move(conics)};
        }

        // =============================================================================
        // Center Vote Pruning
        // =============================================================================

        // =============================================================================
        // Center Vote Pruning
        // =============================================================================

        std::expected<PruningResult, std::string> prune_by_center_vote(
            IStrategy& strategy,
            const CameraDataset& dataset,
            const CenterVotePruningConfig& config) {
            using namespace lfs::core;

            lfs::core::SplatData& model = strategy.get_model();
            const int64_t N = static_cast<int64_t>(model.size());

            if (N <= 0) {
                LOG_WARN("[prune_by_center_vote] No gaussians to prune");
                return PruningResult{0, 0, 0, true, ""};
            }

            LOG_INFO("[prune_by_center_vote] Starting: {} splats, {} cameras (invert_masks={})",
                     N, dataset.size(), config.invert_masks);

            Tensor inside_votes = Tensor::zeros({static_cast<size_t>(N)}, Device::CUDA, DataType::Int32);
            Tensor visible_counts = Tensor::zeros({static_cast<size_t>(N)}, Device::CUDA, DataType::Int32);

            const auto& cameras = dataset.get_cameras();
            const size_t total_cameras = cameras.size();

            size_t processed = 0;
            size_t skipped_no_mask = 0;
            size_t skipped_size_mismatch = 0;
            size_t skipped_proj_error = 0;

            for (size_t cam_idx = 0; cam_idx < total_cameras; ++cam_idx) {
                std::printf("\r[prune_by_center_vote] Processing camera %zu/%zu", cam_idx + 1, total_cameras);
                std::fflush(stdout);

                auto& cam = cameras[cam_idx];
                if (!cam) {
                    continue;
                }

                if (!cam->has_mask()) {
                    ++skipped_no_mask;
                    continue;
                }

                // MAINTAINER NOTE: Ensure config.invert_masks is passed correctly.
                // Previously hardcoded to 'false', which caused incorrect pruning
                // when the dataset required inverted masks (0=object, 1=background).
                Tensor mask = cam->load_and_get_mask(
                    dataset.get_resize_factor(),
                    dataset.get_max_width(),
                    config.invert_masks,
                    0.5f);

                if (!normalize_mask_2d(mask)) {
                    ++skipped_no_mask;
                    continue;
                }

                const int W = cam->image_width();
                const int H = cam->image_height();

                if (static_cast<int>(mask.shape()[0]) != H || static_cast<int>(mask.shape()[1]) != W) {
                    if (skipped_size_mismatch < 3) {
                        LOG_WARN("[prune_by_center_vote] Mask size mismatch: mask [{}x{}] vs camera [{}x{}]",
                                 mask.shape()[1], mask.shape()[0], W, H);
                    }
                    ++skipped_size_mismatch;
                    continue;
                }

                auto proj_result = project_splats(*cam, model, config);
                if (!proj_result) {
                    if (skipped_proj_error < 3) {
                        LOG_WARN("[prune_by_center_vote] Projection failed: {}", proj_result.error());
                    }
                    ++skipped_proj_error;
                    continue;
                }

                const Tensor& radii = proj_result->radii;     // [N, 2] int32
                const Tensor& means2d = proj_result->means2d; // [N, 2] float32

                // Calculate visibility based on projected radii
                const Tensor radii_float = radii.to(DataType::Float32);
                const Tensor radii_x = radii_float.slice(1, 0, 1).squeeze(1);
                const Tensor radii_y = radii_float.slice(1, 1, 2).squeeze(1);
                const Tensor visible = radii_x.gt(0.0f).logical_and(radii_y.gt(0.0f));

                const Tensor visible_indices = visible.nonzero();
                if (!visible_indices.is_valid() || visible_indices.numel() == 0) {
                    continue;
                }

                const Tensor vidx = visible_indices.squeeze(1);
                const int64_t M = static_cast<int64_t>(vidx.shape()[0]);

                const Tensor xy_visible = means2d.index_select(0, vidx);

                const Tensor x_float = xy_visible.slice(1, 0, 1).squeeze(1);
                const Tensor y_float = xy_visible.slice(1, 1, 2).squeeze(1);

                const Tensor x_clamped = x_float.round().clamp(0.0f, static_cast<float>(W - 1));
                const Tensor y_clamped = y_float.round().clamp(0.0f, static_cast<float>(H - 1));

                const Tensor x_int = x_clamped.to(DataType::Int64);
                const Tensor y_int = y_clamped.to(DataType::Int64);
                const Tensor linear_idx = y_int * static_cast<int64_t>(W) + x_int;

                const Tensor mask_flat = mask.flatten().contiguous();
                const Tensor mask_values = mask_flat.index_select(0, linear_idx);

                const Tensor inside = mask_values.gt(0.5f);

                const Tensor inside_int = inside.to(DataType::Int32);
                const Tensor ones_int = Tensor::ones({static_cast<size_t>(M)}, Device::CUDA, DataType::Int32);

                inside_votes.index_add_(0, vidx, inside_int);
                visible_counts.index_add_(0, vidx, ones_int);

                ++processed;
            }

            std::printf("\n");
            LOG_INFO("[prune_by_center_vote] Processed {} cameras, skipped: {} no mask, {} size mismatch, {} proj error",
                     processed, skipped_no_mask, skipped_size_mismatch, skipped_proj_error);

            // Convert to float first to avoid dtype mismatch in sum()
            const float total_visible = visible_counts.to(DataType::Float32).sum().item<float>();
            if (total_visible == 0.0f) {
                LOG_WARN("[prune_by_center_vote] No visibility accumulated; aborting prune");
                return PruningResult{static_cast<int>(N), static_cast<int>(N), 0, true, ""};
            }

            // -------------------------------------------------------------------------
            // Conservative Pruning Logic (matching original implementation)
            // -------------------------------------------------------------------------
            // A splat is removed ONLY if BOTH conditions are met:
            // 1. It was seen in at least min_visibility_count views
            // 2. AND its inside/total ratio is below threshold (mostly outside mask)
            //
            // Splats not seen enough are KEPT (conservative approach).
            // This prevents removing valid splats that happen to be occluded.
            // -------------------------------------------------------------------------

            // 1. Check visibility requirement
            const Tensor meets_min_vis = visible_counts.to(DataType::Float32).ge(static_cast<float>(config.min_visibility_count));

            // 2. Calculate inside/total ratio
            const Tensor counts_float = visible_counts.to(DataType::Float32).clamp_min(1.0f);
            const Tensor votes_float = inside_votes.to(DataType::Float32);
            const Tensor vote_ratio = votes_float / counts_float;

           // 3. Identify splats to remove
            // Conservative: Remove ONLY if seen enough AND has bad ratio
            // This matches the old working logic: keep_mask = meets_vis & (ratio >= threshold)
            const Tensor ratio_bad = vote_ratio.lt(config.vote_ratio_threshold);

            // Remove ONLY splats that were seen enough AND have bad ratio
            // Splats not seen enough are KEPT (conservative approach)
            const Tensor remove_mask = meets_min_vis.logical_and(ratio_bad);

            // Convert to float first for summation
            const int removed = static_cast<int>(remove_mask.to(DataType::Float32).sum().item<float>());

            if (removed > 0) {
                strategy.remove_gaussians(remove_mask);
            }

            const int after = static_cast<int>(strategy.get_model().size());

            LOG_INFO("[prune_by_center_vote] Removed {} / {} splats (threshold={}, min_vis={})",
                     removed, N, config.vote_ratio_threshold, config.min_visibility_count);

            return PruningResult{
                .splats_before = static_cast<int>(N),
                .splats_after = after,
                .splats_removed = removed,
                .success = true,
                .error = ""};
        }

        // =============================================================================
        // Leakage Pruning
        // =============================================================================

        std::expected<PruningResult, std::string> prune_by_mask_leakage(
            IStrategy& strategy,
            const CameraDataset& dataset,
            const LeakagePruningConfig& config) {
            using namespace lfs::core;

            lfs::core::SplatData& model = strategy.get_model();
            const int64_t N = static_cast<int64_t>(model.size());

            if (N <= 0) {
                LOG_WARN("[prune_by_mask_leakage] No gaussians to prune");
                return PruningResult{0, 0, 0, true, ""};
            }
            if (!config.enabled) {
                LOG_INFO("[prune_by_mask_leakage] Disabled; skipping");
                return PruningResult{static_cast<int>(N), static_cast<int>(N), 0, true, ""};
            }

            int sample_points = config.sample_points;
            if (sample_points != 4 && sample_points != 8) {
                LOG_WARN("[prune_by_mask_leakage] sample_points={} not supported; forcing to 8", sample_points);
                sample_points = 8;
            }

            LOG_INFO("[prune_by_mask_leakage] Starting: {} splats, {} cameras (invert_masks={})",
                     N, dataset.size(), config.invert_masks);

            Tensor leak_votes = Tensor::zeros({static_cast<size_t>(N)}, Device::CUDA, DataType::Int32);
            Tensor eval_counts = Tensor::zeros({static_cast<size_t>(N)}, Device::CUDA, DataType::Int32);

            constexpr float DIAG = 0.70710678f;
            const std::array<std::pair<float, float>, 8> dirs8 = {{
                {+1.f, 0.f},
                {-1.f, 0.f},
                {0.f, +1.f},
                {0.f, -1.f},
                {+DIAG, +DIAG},
                {+DIAG, -DIAG},
                {-DIAG, +DIAG},
                {-DIAG, -DIAG},
            }};
            const std::array<std::pair<float, float>, 4> dirs4 = {{
                {+1.f, 0.f},
                {-1.f, 0.f},
                {0.f, +1.f},
                {0.f, -1.f},
            }};

            // Reuse projection parameters
            CenterVotePruningConfig proj_cfg;
            proj_cfg.eps2d = config.eps2d;
            proj_cfg.near_plane = config.near_plane;
            proj_cfg.far_plane = config.far_plane;
            proj_cfg.radius_clip = config.radius_clip;
            proj_cfg.scaling_modifier = config.scaling_modifier;

            const auto& cameras = dataset.get_cameras();
            const size_t total_cameras = cameras.size();

            size_t processed = 0;
            size_t skipped_no_mask = 0;
            size_t skipped_size_mismatch = 0;
            size_t skipped_proj_error = 0;

            for (size_t cam_idx = 0; cam_idx < total_cameras; ++cam_idx) {
                std::printf("\r[prune_by_mask_leakage] Processing camera %zu/%zu", cam_idx + 1, total_cameras);
                std::fflush(stdout);

                auto& cam = cameras[cam_idx];
                if (!cam)
                    continue;

                if (!cam->has_mask()) {
                    ++skipped_no_mask;
                    continue;
                }

                Tensor mask_eval;
                // MAINTAINER NOTE: Ensure config.invert_masks is passed to mask loading.
                // Critical fix: previous version hardcoded 'false'.
                if (config.dilate_px > 0) {
                    // Tolerant mask = 1 - bg_core (bg_core is eroded background)
                    Tensor bg_core = cam->load_and_get_mask_bg_core(
                        dataset.get_resize_factor(),
                        dataset.get_max_width(),
                        config.invert_masks,
                        config.mask_inside_threshold,
                        config.dilate_px);

                    if (!normalize_mask_2d(bg_core)) {
                        ++skipped_no_mask;
                        continue;
                    }
                    const Tensor ones = Tensor::full(bg_core.shape(), 1.0f, bg_core.device(), DataType::Float32);
                    mask_eval = (ones - bg_core).contiguous();
                } else {
                    // Standard binarized mask
                    Tensor mask = cam->load_and_get_mask(
                        dataset.get_resize_factor(),
                        dataset.get_max_width(),
                        config.invert_masks,
                        config.mask_inside_threshold);

                    if (!normalize_mask_2d(mask)) {
                        ++skipped_no_mask;
                        continue;
                    }
                    // Strict binarize
                    const Tensor ones = Tensor::full(mask.shape(), 1.0f, mask.device(), DataType::Float32);
                    const Tensor zeros = Tensor::full(mask.shape(), 0.0f, mask.device(), DataType::Float32);
                    mask_eval = ones.where(mask.ge(config.mask_inside_threshold), zeros).contiguous();
                }

                const int W = cam->image_width();
                const int H = cam->image_height();

                if (static_cast<int>(mask_eval.shape()[0]) != H || static_cast<int>(mask_eval.shape()[1]) != W) {
                    if (skipped_size_mismatch < 3) {
                        LOG_WARN("[prune_by_mask_leakage] Mask size mismatch: mask [{}x{}] vs camera [{}x{}]",
                                 mask_eval.shape()[1], mask_eval.shape()[0], W, H);
                    }
                    ++skipped_size_mismatch;
                    continue;
                }

                auto proj_result = project_splats(*cam, model, proj_cfg);
                if (!proj_result) {
                    if (skipped_proj_error < 3) {
                        LOG_WARN("[prune_by_mask_leakage] Projection failed: {}", proj_result.error());
                    }
                    ++skipped_proj_error;
                    continue;
                }

                const Tensor& radii = proj_result->radii;     // [N,2] int32
                const Tensor& means2d = proj_result->means2d; // [N,2] float32

                const Tensor radii_f = radii.to(DataType::Float32);
                const Tensor rx_all = radii_f.slice(1, 0, 1).squeeze(1);
                const Tensor ry_all = radii_f.slice(1, 1, 2).squeeze(1);
                const Tensor visible = rx_all.gt(0.0f).logical_and(ry_all.gt(0.0f));

                const Tensor visible_indices = visible.nonzero();
                if (!visible_indices.is_valid() || visible_indices.numel() == 0) {
                    continue;
                }

                const Tensor vidx = visible_indices.squeeze(1);
                const int64_t M = static_cast<int64_t>(vidx.shape()[0]);

                const Tensor xy_vis = means2d.index_select(0, vidx); // [M,2]
                const Tensor r_vis = radii_f.index_select(0, vidx);  // [M,2]

                const Tensor x0 = xy_vis.slice(1, 0, 1).squeeze(1);
                const Tensor y0 = xy_vis.slice(1, 1, 2).squeeze(1);
                const Tensor rx = r_vis.slice(1, 0, 1).squeeze(1);
                const Tensor ry = r_vis.slice(1, 1, 2).squeeze(1);

                // Center sample to decide candidates
                const Tensor x0i = x0.round().clamp(0.0f, static_cast<float>(W - 1)).to(DataType::Int64);
                const Tensor y0i = y0.round().clamp(0.0f, static_cast<float>(H - 1)).to(DataType::Int64);
                const Tensor lin0 = y0i * static_cast<int64_t>(W) + x0i;

                const Tensor mask_flat = mask_eval.flatten().contiguous();
                const Tensor center_vals = mask_flat.index_select(0, lin0);
                const Tensor center_inside = center_vals.gt(config.mask_inside_threshold);

                // Radius gate to avoid sampling tiny splats
                const Tensor max_r = rx.where(rx.gt(ry), ry);
                const Tensor big_enough = max_r.ge(config.min_pixel_radius);

                const Tensor candidate = center_inside.logical_and(big_enough);
                const Tensor c_nonzero = candidate.nonzero();
                if (!c_nonzero.is_valid() || c_nonzero.numel() == 0) {
                    continue;
                }

                const Tensor cidx = c_nonzero.squeeze(1); // indices into [0..M)
                const int64_t Kc = static_cast<int64_t>(cidx.shape()[0]);

                const Tensor splat_ids = vidx.index_select(0, cidx); // [K]
                const Tensor cx = x0.index_select(0, cidx);
                const Tensor cy = y0.index_select(0, cidx);
                const Tensor crx = rx.index_select(0, cidx);
                const Tensor cry = ry.index_select(0, cidx);

                Tensor outside_count = Tensor::zeros({static_cast<size_t>(Kc)}, Device::CUDA, DataType::Int32);

                const int ndirs = sample_points;
                for (int d = 0; d < ndirs; ++d) {
                    const float dx = (ndirs == 4) ? dirs4[d].first : dirs8[d].first;
                    const float dy = (ndirs == 4) ? dirs4[d].second : dirs8[d].second;

                    const Tensor xs = (cx + crx * dx).round().clamp(0.0f, static_cast<float>(W - 1));
                    const Tensor ys = (cy + cry * dy).round().clamp(0.0f, static_cast<float>(H - 1));

                    const Tensor xi = xs.to(DataType::Int64);
                    const Tensor yi = ys.to(DataType::Int64);
                    const Tensor lin = yi * static_cast<int64_t>(W) + xi;

                    const Tensor mv = mask_flat.index_select(0, lin);
                    const Tensor inside = mv.gt(config.mask_inside_threshold);
                    const Tensor outside = inside.logical_not();
                    outside_count = outside_count + outside.to(DataType::Int32);
                }

                const Tensor outside_ratio = outside_count.to(DataType::Float32) / static_cast<float>(ndirs);
                const Tensor leak_here = outside_ratio.gt(config.per_view_leak_fraction);

                const Tensor leak_i32 = leak_here.to(DataType::Int32);
                const Tensor ones_i32 = Tensor::ones({static_cast<size_t>(Kc)}, Device::CUDA, DataType::Int32);

                leak_votes.index_add_(0, splat_ids, leak_i32);
                eval_counts.index_add_(0, splat_ids, ones_i32);

                ++processed;
            }

            std::printf("\n");
            LOG_INFO("[prune_by_mask_leakage] Processed {} cameras, skipped: {} no mask, {} size mismatch, {} proj error",
                     processed, skipped_no_mask, skipped_size_mismatch, skipped_proj_error);

            // Convert to float first
            const float total_eval = eval_counts.to(DataType::Float32).sum().item<float>();
            if (total_eval == 0.0f) {
                LOG_WARN("[prune_by_mask_leakage] No evaluated splats (no candidates); skipping leakage prune");
                return PruningResult{static_cast<int>(N), static_cast<int>(N), 0, true, ""};
            }

            const Tensor meets_min_vis = eval_counts.to(DataType::Float32).ge(static_cast<float>(config.min_visibility_count));
            if (!meets_min_vis.any().item<bool>()) {
                LOG_WARN("[prune_by_mask_leakage] No splats met min_visibility={}; skipping leakage prune",
                         config.min_visibility_count);
                return PruningResult{static_cast<int>(N), static_cast<int>(N), 0, true, ""};
            }

            const Tensor counts_f = eval_counts.to(DataType::Float32).clamp_min(1.0f);
            const Tensor leak_f = leak_votes.to(DataType::Float32);
            const Tensor leak_ratio = leak_f / counts_f;

            const Tensor ones = Tensor::full(leak_ratio.shape(), 1.0f, leak_ratio.device(), DataType::Float32);
            const Tensor ok_ratio = ones - leak_ratio;

            const Tensor ok_enough = ok_ratio.ge(config.leak_keep_threshold);
            const Tensor keep_mask = meets_min_vis.logical_and(ok_enough);

            // Leakage pass remains conservative: only remove splats that were evaluated
            // enough times AND failed the test. Splats not evaluated here are left
            // for the center_vote pass to handle.
            const Tensor remove_mask = meets_min_vis.logical_and(keep_mask.logical_not());

            // Convert to float first
            const int removed = static_cast<int>(remove_mask.to(DataType::Float32).sum().item<float>());

            if (removed > 0) {
                strategy.remove_gaussians(remove_mask);
            }

            const int after = static_cast<int>(strategy.get_model().size());

            LOG_INFO("[prune_by_mask_leakage] Removed {} / {} splats (keep_thr={}, per_view_leak_frac={}, min_vis={})",
                     removed, N, config.leak_keep_threshold, config.per_view_leak_fraction, config.min_visibility_count);

            return PruningResult{
                .splats_before = static_cast<int>(N),
                .splats_after = after,
                .splats_removed = removed,
                .success = true,
                .error = ""};
        }

        // =============================================================================
        // Main Entry Point
        // =============================================================================

        std::expected<PruningResult, std::string> prune_after_training(
            IStrategy& strategy,
            const CameraDataset& dataset,
            const CenterVotePruningConfig& center_config,
            const LeakagePruningConfig& leakage_config) {
            LOG_INFO("=== Post-training mask-based pruning ===");

            const int before_all = static_cast<int>(strategy.get_model().size());

            // 1) Center-vote pass
            auto center_res = prune_by_center_vote(strategy, dataset, center_config);
            if (!center_res) {
                LOG_ERROR("[prune_after_training] Center-vote failed: {}", center_res.error());
                return center_res;
            }

            // 2) Leakage pass (optional)
            if (false && leakage_config.enabled) {
                auto leak_res = prune_by_mask_leakage(strategy, dataset, leakage_config);
                if (!leak_res) {
                    LOG_ERROR("[prune_after_training] Leakage failed: {}", leak_res.error());
                    return leak_res;
                }

                const int after_all = static_cast<int>(strategy.get_model().size());
                const int removed_all = before_all - after_all;

                LOG_INFO("=== Pruning complete: {} -> {} splats ({:.1f}% removed) ===",
                         before_all, after_all,
                         before_all > 0 ? (100.0f * removed_all / before_all) : 0.0f);

                PruningResult combined;
                combined.splats_before = before_all;
                combined.splats_after = after_all;
                combined.splats_removed = removed_all;
                combined.success = true;
                return combined;
            }

            // Only center-vote
            const int after_all = static_cast<int>(strategy.get_model().size());
            const int removed_all = before_all - after_all;

            LOG_INFO("=== Pruning complete: {} -> {} splats ({:.1f}% removed) ===",
                     before_all, after_all,
                     before_all > 0 ? (100.0f * removed_all / before_all) : 0.0f);

            PruningResult combined;
            combined.splats_before = before_all;
            combined.splats_after = after_all;
            combined.splats_removed = removed_all;
            combined.success = true;
            return combined;
        }

    } // namespace mask_pruning
} // namespace lfs::training