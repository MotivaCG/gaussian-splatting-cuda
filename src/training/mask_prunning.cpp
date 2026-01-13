/* ScanMeNow file */

#include "mask_pruning.hpp"

#include "Projection.h"          // ::gsplat_fwd::launch_projection_ut_3dgs_fused_kernel
#include "Common.h"              // CameraModelType, UnscentedTransformParameters, ShutterType
#include "core/logger.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace lfs::training::mask_pruning {

namespace {



// -----------------------------------------------------------------------------
// Dataset sizing helpers
// -----------------------------------------------------------------------------

struct DatasetSizing {
    int resize_factor = 1;
    int max_width = 0; // 0 means "no limit" (keeps original resolution)
};

static std::expected<DatasetSizing, std::string> get_dataset_sizing_or_error(const CameraDataset& dataset) {
    DatasetSizing out;
    out.resize_factor = dataset.get_resize_factor();
    out.max_width = dataset.get_max_width();

    if (out.resize_factor <= 0) {
        return std::unexpected("CameraDataset::get_resize_factor() is invalid (<= 0).");
    }
    if (out.max_width < 0) {
        return std::unexpected("CameraDataset::get_max_width() is invalid (< 0).");
    }
    return out;
}

static inline int round_to_int(float x) {
    // std::lround returns long; keep explicit cast.
    return static_cast<int>(std::lround(static_cast<double>(x)));
}

static inline bool is_visible_radii(int rx, int ry) {
    // Treat a splat as visible if it has a positive footprint in either axis.
    return (rx > 0) || (ry > 0);
}

} // namespace

// -----------------------------------------------------------------------------
// Projection
// -----------------------------------------------------------------------------

std::expected<ProjectionResult, std::string> project_splats(
    const lfs::core::Camera& camera,
    const lfs::core::SplatData& splat_data,
    const CenterVotePruningConfig& config) {

    using namespace lfs::core;

    const uint32_t image_width = static_cast<uint32_t>(camera.image_width());
    const uint32_t image_height = static_cast<uint32_t>(camera.image_height());

    // Activated Gaussian parameters
    Tensor means = splat_data.get_means().contiguous();          // [N,3]
    Tensor quats = splat_data.get_rotation().contiguous();       // [N,4]
    Tensor scales = splat_data.get_scaling().contiguous();       // [N,3]
    Tensor opacities = splat_data.get_opacity().contiguous();    // [N] or [N,1]

    if (opacities.ndim() == 2 && opacities.shape()[1] == 1) {
        opacities = opacities.squeeze(-1);
    }

    if (config.scaling_modifier != 1.0f) {
        scales = (scales * config.scaling_modifier).contiguous();
    }

    const uint32_t N = static_cast<uint32_t>(means.shape()[0]);
    if (N == 0) {
        return ProjectionResult{};
    }

    const uint32_t C = 1;

    // Camera model conversion (same approach as gsplat_rasterizer.cpp)
    const ::CameraModelType camera_model = static_cast<::CameraModelType>(
        static_cast<int>(camera.camera_model_type()));

    // K must be CUDA for the projection kernel
    Tensor K_tensor = camera.K().contiguous();
    if (K_tensor.device() != Device::CUDA) {
        K_tensor = K_tensor.to(Device::CUDA).contiguous();
    }
    const float* K_ptr = K_tensor.ptr<float>();

    // View matrix pointer is expected to be usable by CUDA kernels (same as rasterizer path)
    const float* viewmat_ptr = camera.world_view_transform_ptr();
    if (!viewmat_ptr) {
        return std::unexpected("Camera::world_view_transform_ptr() returned null");
    }

    // Distortion coefficients (copy to CUDA if provided)
    const Tensor radial_dist = camera.radial_distortion();
    const Tensor tangential_dist = camera.tangential_distortion();
    Tensor radial_cuda, tangential_cuda, thin_prism_cuda;
    const float* radial_ptr = nullptr;
    const float* tangential_ptr = nullptr;
    const float* thin_prism_ptr = nullptr;

    auto to_cuda = [](const Tensor& t) {
        return t.to(Device::CUDA).contiguous();
    };

    switch (camera_model) {
    case ::CameraModelType::THIN_PRISM_FISHEYE:
        if (radial_dist.is_valid() && radial_dist.numel() == 4) {
            radial_cuda = to_cuda(radial_dist);
            radial_ptr = radial_cuda.ptr<float>();
        }
        if (tangential_dist.is_valid() && tangential_dist.numel() == 4) {
            thin_prism_cuda = to_cuda(tangential_dist);
            thin_prism_ptr = thin_prism_cuda.ptr<float>();
        }
        break;
    case ::CameraModelType::FISHEYE:
        if (radial_dist.is_valid() && radial_dist.numel() >= 4) {
            radial_cuda = to_cuda(radial_dist.numel() == 4 ? radial_dist : radial_dist.slice(0, 0, 4));
            radial_ptr = radial_cuda.ptr<float>();
        }
        break;
    case ::CameraModelType::PINHOLE:
        if (radial_dist.is_valid() && radial_dist.numel() > 0) {
            radial_cuda = to_cuda(radial_dist.numel() == 6 ? radial_dist
                                                          : radial_dist.slice(0, 0, std::min(radial_dist.numel(), size_t(6))));
            radial_ptr = radial_cuda.ptr<float>();
        }
        if (tangential_dist.is_valid() && tangential_dist.numel() >= 2) {
            tangential_cuda = to_cuda(tangential_dist.numel() == 2 ? tangential_dist
                                                                  : tangential_dist.slice(0, 0, 2));
            tangential_ptr = tangential_cuda.ptr<float>();
        }
        break;
    default:
        break;
    }

    // Outputs: allocate as [1, N, ...] and squeeze(0) for return.
    Tensor radii = Tensor::empty({static_cast<size_t>(C), static_cast<size_t>(N), 2UL}, Device::CUDA, DataType::Int32);
    Tensor means2d = Tensor::empty({static_cast<size_t>(C), static_cast<size_t>(N), 2UL}, Device::CUDA, DataType::Float32);
    Tensor depths = Tensor::empty({static_cast<size_t>(C), static_cast<size_t>(N)}, Device::CUDA, DataType::Float32);
    Tensor conics = Tensor::empty({static_cast<size_t>(C), static_cast<size_t>(N), 3UL}, Device::CUDA, DataType::Float32);

    // UT parameters: default constructed (same as rasterizer).
    UnscentedTransformParameters ut_params;

    // Note: opacities are optional; passing them is fine.
    ::gsplat_lfs::launch_projection_ut_3dgs_fused_kernel(
        means.ptr<float>(),
        quats.ptr<float>(),
        scales.ptr<float>(),
        opacities.is_valid() ? opacities.ptr<float>() : nullptr,
        viewmat_ptr,
        nullptr, // viewmats1 (rolling shutter not used here)
        K_ptr,
        N,
        C,
        image_width,
        image_height,
        config.eps2d,
        config.near_plane,
        config.far_plane,
        config.radius_clip,
        camera_model,
        ut_params,
        ShutterType::GLOBAL,
        radial_ptr,
        tangential_ptr,
        thin_prism_ptr,
        radii.ptr<int32_t>(),
        means2d.ptr<float>(),
        depths.ptr<float>(),
        conics.ptr<float>(),
        nullptr, // compensations
        nullptr  // stream
    );

    // Squeeze camera dimension
    ProjectionResult out;
    out.radii = radii.squeeze(0).contiguous();
    out.means2d = means2d.squeeze(0).contiguous();
    out.depths = depths.squeeze(0).contiguous();
    out.conics = conics.squeeze(0).contiguous();
    return out;
}

// -----------------------------------------------------------------------------
// Center-vote pruning
// -----------------------------------------------------------------------------

std::expected<PruningResult, std::string> prune_by_center_vote(
    IStrategy& strategy,
    const CameraDataset& dataset,
    const CenterVotePruningConfig& config) {

    using namespace lfs::core;

    auto& splat = strategy.get_model();
    const int splats_before = static_cast<int>(splat.size());
    if (splats_before == 0) {
        return PruningResult{.splats_before = 0, .splats_after = 0, .splats_removed = 0, .success = true};
    }

    // We need dataset sizing to load masks at the correct resolution.
    auto sizing = get_dataset_sizing_or_error(dataset);
    if (!sizing) {
        return std::unexpected(sizing.error());
    }

    std::vector<int> visible_counts(static_cast<size_t>(splats_before), 0);
    std::vector<int> inside_counts(static_cast<size_t>(splats_before), 0);

    int skipped_no_mask = 0;
    int skipped_size_mismatch = 0;
    int skipped_proj_error = 0;

    const auto& cams = dataset.get_cameras();
    const int n_cams = static_cast<int>(cams.size());

    LOG_INFO("[prune_by_center_vote] Starting: {} splats, {} cameras", splats_before, n_cams);

    for (int ci = 0; ci < n_cams; ++ci) {
        const auto& cam_ptr = cams[static_cast<size_t>(ci)];
        if (!cam_ptr) {
            ++skipped_no_mask;
            continue;
        }
        auto& cam = *cam_ptr;
        if (!cam.has_mask()) {
            ++skipped_no_mask;
            continue;
        }

        // Load mask (cached in Camera). Keep threshold hardcoded to 0.5, like old attention-mask.
        // If you need a different threshold, extend CenterVotePruningConfig.
        Tensor mask = cam.load_and_get_mask(sizing->resize_factor, sizing->max_width, config.invert_masks, 0.5f);
        if (!mask.is_valid() || mask.numel() == 0) {
            ++skipped_no_mask;
            continue;
        }
        if (mask.ndim() == 3 && mask.shape()[0] == 1) {
            mask = mask.squeeze(0);
        }

        const int H = static_cast<int>(cam.image_height());
        const int W = static_cast<int>(cam.image_width());
        if (mask.ndim() != 2 || static_cast<int>(mask.shape()[0]) != H || static_cast<int>(mask.shape()[1]) != W) {
            ++skipped_size_mismatch;
            continue;
        }

        // Project splats
        auto proj = project_splats(cam, splat, config);
        if (!proj) {
            ++skipped_proj_error;
            continue;
        }

        // Bring to CPU for robust indexing
        Tensor radii_cpu = proj->radii.cpu().contiguous();
        Tensor means2d_cpu = proj->means2d.cpu().contiguous();
        Tensor mask_cpu = mask.cpu().contiguous();

        auto r_acc = radii_cpu.accessor<int32_t, 2>();
        auto m_acc = means2d_cpu.accessor<float, 2>();
        auto mask_acc = mask_cpu.accessor<float, 2>();

        const int N = splats_before;
        for (int i = 0; i < N; ++i) {
            const int rx = static_cast<int>(r_acc(i, 0));
            const int ry = static_cast<int>(r_acc(i, 1));
            if (!is_visible_radii(rx, ry)) {
                continue;
            }

            visible_counts[static_cast<size_t>(i)]++;

            const float fx = m_acc(i, 0);
            const float fy = m_acc(i, 1);
            int x = round_to_int(fx);
            int y = round_to_int(fy);
            if (x < 0 || x >= W || y < 0 || y >= H) {
                continue;
            }

            const float v = mask_acc(y, x);
            if (v >= 0.5f) {
                inside_counts[static_cast<size_t>(i)]++;
            }
        }

        if ((ci + 1) % 25 == 0 || (ci + 1) == n_cams) {
            LOG_INFO("[prune_by_center_vote] Processing camera {}/{}", (ci + 1), n_cams);
        }
    }

    LOG_INFO("[prune_by_center_vote] Processed {} cameras, skipped: {} no mask, {} size mismatch, {} proj error",
             n_cams, skipped_no_mask, skipped_size_mismatch, skipped_proj_error);

    // Build remove mask
    Tensor remove_mask_cpu = Tensor::zeros({static_cast<size_t>(splats_before)}, Device::CPU, DataType::Bool);
    auto rm_acc = remove_mask_cpu.accessor<uint8_t, 1>();

    int n_remove = 0;
    for (int i = 0; i < splats_before; ++i) {
        const int tot = visible_counts[static_cast<size_t>(i)];
        if (tot < config.min_visibility_count) {
            rm_acc(i) = 0;
            continue;
        }
        const int pos = inside_counts[static_cast<size_t>(i)];
        const float ratio = (tot > 0) ? (static_cast<float>(pos) / static_cast<float>(tot)) : 0.0f;
        const bool remove = (ratio < config.vote_ratio_threshold);
        rm_acc(i) = remove ? 1 : 0;
        if (remove) {
            ++n_remove;
        }
    }

    if (n_remove == 0) {
        LOG_INFO("[prune_by_center_vote] Removed 0 splats (0.000%)");
        return PruningResult{.splats_before = splats_before,
                             .splats_after = splats_before,
                             .splats_removed = 0,
                             .success = true};
    }

    // Remove on CUDA
    Tensor remove_mask = remove_mask_cpu.to(Device::CUDA).contiguous();

    strategy.remove_gaussians(remove_mask);

    const int splats_after = static_cast<int>(strategy.get_model().size());
    const int removed = splats_before - splats_after;

    const double pct = (splats_before > 0)
                           ? (100.0 * static_cast<double>(removed) / static_cast<double>(splats_before))
                           : 0.0;

    LOG_INFO("[prune_by_center_vote] Requested removal: {}, actually removed: {} ({:.3f}%)",
             n_remove, removed, pct);

    return PruningResult{.splats_before = splats_before,
                         .splats_after = splats_after,
                         .splats_removed = removed,
                         .success = true};
}

// -----------------------------------------------------------------------------
// Leakage pruning
// -----------------------------------------------------------------------------

std::expected<PruningResult, std::string> prune_by_mask_leakage(
    IStrategy& strategy,
    const CameraDataset& dataset,
    const LeakagePruningConfig& config) {

    using namespace lfs::core;

    if (!config.enabled) {
        const int n = static_cast<int>(strategy.get_model().size());
        return PruningResult{.splats_before = n, .splats_after = n, .splats_removed = 0, .success = true};
    }

    auto& splat = strategy.get_model();
    const int splats_before = static_cast<int>(splat.size());
    if (splats_before == 0) {
        return PruningResult{.splats_before = 0, .splats_after = 0, .splats_removed = 0, .success = true};
    }

    auto sizing = get_dataset_sizing_or_error(dataset);
    if (!sizing) {
        return std::unexpected(sizing.error());
    }

    std::vector<int> eval_counts(static_cast<size_t>(splats_before), 0);
    std::vector<int> leak_counts(static_cast<size_t>(splats_before), 0);

    int skipped_no_mask = 0;
    int skipped_size_mismatch = 0;
    int skipped_proj_error = 0;

    const auto& cams = dataset.get_cameras();
    const int n_cams = static_cast<int>(cams.size());

    LOG_INFO("[prune_by_mask_leakage] Starting: {} splats, {} cameras", splats_before, n_cams);

    // Direction samples
    struct Dir {
        float dx;
        float dy;
    };
    std::vector<Dir> dirs;
    dirs.reserve(8);
    dirs.push_back({1.0f, 0.0f});
    dirs.push_back({-1.0f, 0.0f});
    dirs.push_back({0.0f, 1.0f});
    dirs.push_back({0.0f, -1.0f});
    if (config.sample_points >= 8) {
        constexpr float k = 0.70710678f; // 1/sqrt(2)
        dirs.push_back({k, k});
        dirs.push_back({-k, k});
        dirs.push_back({k, -k});
        dirs.push_back({-k, -k});
    }

    for (int ci = 0; ci < n_cams; ++ci) {
        const auto& cam_ptr = cams[static_cast<size_t>(ci)];
        if (!cam_ptr) {
            ++skipped_no_mask;
            continue;
        }
        auto& cam = *cam_ptr;
        if (!cam.has_mask()) {
            ++skipped_no_mask;
            continue;
        }

        // Base mask
        Tensor mask = cam.load_and_get_mask(sizing->resize_factor, sizing->max_width, config.invert_masks, config.mask_inside_threshold);
        if (!mask.is_valid() || mask.numel() == 0) {
            ++skipped_no_mask;
            continue;
        }
        if (mask.ndim() == 3 && mask.shape()[0] == 1) {
            mask = mask.squeeze(0);
        }

        // Optional tolerant mask: use (1 - bg_core) if requested.
        // bg_core is "confident background"; its complement tolerates boundary uncertainty.
        Tensor mask_for_leak = mask;
        if (config.dilate_px > 0) {
            Tensor bg_core = cam.load_and_get_mask_bg_core(
                sizing->resize_factor, sizing->max_width,
                config.invert_masks,
                config.mask_inside_threshold,
                config.dilate_px);
            if (bg_core.is_valid() && bg_core.numel() == mask.numel()) {
                if (bg_core.ndim() == 3 && bg_core.shape()[0] == 1) {
                    bg_core = bg_core.squeeze(0);
                }
                // mask_for_leak = 1 - bg_core
                mask_for_leak = Tensor::full(bg_core.shape(), 1.0f, bg_core.device()) - bg_core;
            }
        }

        const int H = static_cast<int>(cam.image_height());
        const int W = static_cast<int>(cam.image_width());
        if (mask_for_leak.ndim() != 2 || static_cast<int>(mask_for_leak.shape()[0]) != H || static_cast<int>(mask_for_leak.shape()[1]) != W) {
            ++skipped_size_mismatch;
            continue;
        }

        // Project
        CenterVotePruningConfig proj_cfg;
        proj_cfg.eps2d = config.eps2d;
        proj_cfg.near_plane = config.near_plane;
        proj_cfg.far_plane = config.far_plane;
        proj_cfg.radius_clip = config.radius_clip;
        proj_cfg.scaling_modifier = config.scaling_modifier;
        proj_cfg.invert_masks = config.invert_masks;

        auto proj = project_splats(cam, splat, proj_cfg);
        if (!proj) {
            ++skipped_proj_error;
            continue;
        }

        Tensor radii_cpu = proj->radii.cpu().contiguous();
        Tensor means2d_cpu = proj->means2d.cpu().contiguous();
        Tensor mask_cpu = mask_for_leak.cpu().contiguous();

        auto r_acc = radii_cpu.accessor<int32_t, 2>();
        auto m_acc = means2d_cpu.accessor<float, 2>();
        auto mask_acc = mask_cpu.accessor<float, 2>();

        const int N = splats_before;
        for (int i = 0; i < N; ++i) {
            const int rx = static_cast<int>(r_acc(i, 0));
            const int ry = static_cast<int>(r_acc(i, 1));
            if (!is_visible_radii(rx, ry)) {
                continue;
            }
            const float frx = static_cast<float>(std::abs(rx));
            const float fry = static_cast<float>(std::abs(ry));
            if (std::max(frx, fry) < config.min_pixel_radius) {
                continue;
            }

            const float fx = m_acc(i, 0);
            const float fy = m_acc(i, 1);
            const int cx = round_to_int(fx);
            const int cy = round_to_int(fy);
            if (cx < 0 || cx >= W || cy < 0 || cy >= H) {
                continue;
            }

            // Evaluate only if center is inside (or near-inside) mask.
            if (mask_acc(cy, cx) < config.mask_inside_threshold) {
                continue;
            }

            eval_counts[static_cast<size_t>(i)]++;

            int outside = 0;
            const int n_samples = static_cast<int>(dirs.size());
            for (const auto& d : dirs) {
                const float sx = static_cast<float>(cx) + d.dx * frx;
                const float sy = static_cast<float>(cy) + d.dy * fry;
                const int ix = round_to_int(sx);
                const int iy = round_to_int(sy);
                if (ix < 0 || ix >= W || iy < 0 || iy >= H) {
                    outside++;
                    continue;
                }
                if (mask_acc(iy, ix) < config.mask_inside_threshold) {
                    outside++;
                }
            }

            const float outside_ratio = (n_samples > 0) ? (static_cast<float>(outside) / static_cast<float>(n_samples)) : 0.0f;
            if (outside_ratio > config.per_view_leak_fraction) {
                leak_counts[static_cast<size_t>(i)]++;
            }
        }

        if ((ci + 1) % 25 == 0 || (ci + 1) == n_cams) {
            LOG_INFO("[prune_by_mask_leakage] Processing camera {}/{}", (ci + 1), n_cams);
        }
    }

    LOG_INFO("[prune_by_mask_leakage] Processed {} cameras, skipped: {} no mask, {} size mismatch, {} proj error",
             n_cams, skipped_no_mask, skipped_size_mismatch, skipped_proj_error);

    // Build remove mask
    Tensor remove_mask_cpu = Tensor::zeros({static_cast<size_t>(splats_before)}, Device::CPU, DataType::Bool);
    auto rm_acc = remove_mask_cpu.accessor<uint8_t, 1>();
    int n_remove = 0;

    for (int i = 0; i < splats_before; ++i) {
        const int tot = eval_counts[static_cast<size_t>(i)];
        if (tot < config.min_visibility_count) {
            rm_acc(i) = 0;
            continue;
        }
        const int leaks = leak_counts[static_cast<size_t>(i)];
        const float leak_ratio = (tot > 0) ? (static_cast<float>(leaks) / static_cast<float>(tot)) : 0.0f;
        const float keep_ratio = 1.0f - leak_ratio;
        const bool remove = (keep_ratio < config.leak_keep_threshold);
        rm_acc(i) = remove ? 1 : 0;
        if (remove) {
            ++n_remove;
        }
    }

    if (n_remove == 0) {
        LOG_INFO("[prune_by_mask_leakage] Removed 0 splats (0.000%)");
        return PruningResult{.splats_before = splats_before,
                             .splats_after = splats_before,
                             .splats_removed = 0,
                             .success = true};
    }

    Tensor remove_mask = remove_mask_cpu.to(Device::CUDA).contiguous();
    strategy.remove_gaussians(remove_mask);

    const int splats_after = static_cast<int>(strategy.get_model().size());
    const int removed = splats_before - splats_after;

    const double pct = (splats_before > 0)
                           ? (100.0 * static_cast<double>(removed) / static_cast<double>(splats_before))
                           : 0.0;

    LOG_INFO("[prune_by_mask_leakage] Requested removal: {}, actually removed: {} ({:.3f}%)",
             n_remove, removed, pct);

    return PruningResult{.splats_before = splats_before,
                         .splats_after = splats_after,
                         .splats_removed = removed,
                         .success = true};
}

// -----------------------------------------------------------------------------
// Orchestrator
// -----------------------------------------------------------------------------

std::expected<PruningResult, std::string> prune_after_training(
    IStrategy& strategy,
    const CameraDataset& dataset,
    const CenterVotePruningConfig& center_config,
    const LeakagePruningConfig& leakage_config) {

    const int before = static_cast<int>(strategy.get_model().size());
    if (before == 0) {
        return PruningResult{.splats_before = 0, .splats_after = 0, .splats_removed = 0, .success = true};
    }

    auto center = prune_by_center_vote(strategy, dataset, center_config);
    if (!center) {
        return std::unexpected(center.error());
    }

    auto leak = prune_by_mask_leakage(strategy, dataset, leakage_config);
    if (!leak) {
        return std::unexpected(leak.error());
    }

    const int after = static_cast<int>(strategy.get_model().size());
    const int removed = before - after;

    PruningResult out;
    out.splats_before = before;
    out.splats_after = after;
    out.splats_removed = removed;
    out.success = true;
    return out;
}

} // namespace lfs::training::mask_pruning
