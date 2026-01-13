/* ScanMeNow file */

#include "mask_pruning.hpp"

#include "Projection.h" // launch_projection_ut_3dgs_fused_kernel
#include "Common.h"     // CameraModelType, UnscentedTransformParameters, ShutterType
#include "core/logger.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace lfs::training::mask_pruning {

namespace {

// -----------------------------------------------------------------------------
// Dataset sizing helpers
// -----------------------------------------------------------------------------

struct DatasetSizing {
    int resize_factor = 1;
    int max_width = 0; // 0 means "no limit"
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
    return static_cast<int>(std::lround(static_cast<double>(x)));
}

static inline int clamp_int(int v, int lo, int hi) {
    return (v < lo) ? lo : (v > hi ? hi : v);
}

// Old semantics: radii must be positive in both axes to be considered visible.
static inline bool is_visible_radii_strict(int rx, int ry) {
    return (rx > 0) && (ry > 0);
}

// Integral image for fast square-window dilation queries on a binary mask.
struct IntegralImage {
    int H = 0;
    int W = 0;
    std::vector<int> sat; // (H+1)*(W+1), row-major

    void build(const uint8_t* mask01, int h, int w) {
        H = h;
        W = w;
        sat.assign(static_cast<size_t>((H + 1) * (W + 1)), 0);

        const int stride = W + 1;
        for (int y = 0; y < H; ++y) {
            int row_sum = 0;
            const uint8_t* row = mask01 + static_cast<size_t>(y) * static_cast<size_t>(W);
            for (int x = 0; x < W; ++x) {
                row_sum += (row[x] != 0) ? 1 : 0;
                sat[static_cast<size_t>((y + 1) * stride + (x + 1))] =
                    sat[static_cast<size_t>(y * stride + (x + 1))] + row_sum;
            }
        }
    }

    inline int sum_rect(int x0, int y0, int x1, int y1) const {
        // Rect is inclusive bounds: [x0..x1], [y0..y1]
        x0 = clamp_int(x0, 0, W - 1);
        y0 = clamp_int(y0, 0, H - 1);
        x1 = clamp_int(x1, 0, W - 1);
        y1 = clamp_int(y1, 0, H - 1);
        if (x1 < x0 || y1 < y0)
            return 0;

        const int stride = W + 1;
        const int xa = x0;
        const int ya = y0;
        const int xb = x1 + 1;
        const int yb = y1 + 1;

        const int A = sat[static_cast<size_t>(ya * stride + xa)];
        const int B = sat[static_cast<size_t>(ya * stride + xb)];
        const int C = sat[static_cast<size_t>(yb * stride + xa)];
        const int D = sat[static_cast<size_t>(yb * stride + xb)];
        return D - B - C + A;
    }
};

} // namespace

// -----------------------------------------------------------------------------
// Projection (same kernel path as rasterizer)
// -----------------------------------------------------------------------------

std::expected<ProjectionResult, std::string> project_splats(
    const lfs::core::Camera& camera,
    const lfs::core::SplatData& splat_data,
    const CenterVotePruningConfig& config) {

    using namespace lfs::core;

    const uint32_t image_width = static_cast<uint32_t>(camera.image_width());
    const uint32_t image_height = static_cast<uint32_t>(camera.image_height());

    Tensor means = splat_data.get_means().contiguous();       // [N,3]
    Tensor quats = splat_data.get_rotation().contiguous();    // [N,4]
    Tensor scales = splat_data.get_scaling().contiguous();    // [N,3]
    Tensor opacities = splat_data.get_opacity().contiguous(); // [N] or [N,1]

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

    // NOTE: camera.camera_model_type() is lfs::core::CameraModelType; kernels expect ::CameraModelType.
    const ::CameraModelType camera_model =
        static_cast<::CameraModelType>(static_cast<int>(camera.camera_model_type()));

    // K on CUDA
    Tensor K_tensor = camera.K().contiguous();
    if (K_tensor.device() != Device::CUDA) {
        K_tensor = K_tensor.to(Device::CUDA).contiguous();
    }
    const float* K_ptr = K_tensor.ptr<float>();

    // View matrix pointer must be valid for CUDA kernels.
    const float* viewmat_ptr = camera.world_view_transform_ptr();
    if (!viewmat_ptr) {
        return std::unexpected("Camera::world_view_transform_ptr() returned null");
    }

    // Distortion
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
            const size_t n = std::min(radial_dist.numel(), size_t(6));
            radial_cuda = to_cuda(radial_dist.numel() == n ? radial_dist : radial_dist.slice(0, 0, n));
            radial_ptr = radial_cuda.ptr<float>();
        }
        if (tangential_dist.is_valid() && tangential_dist.numel() >= 2) {
            tangential_cuda = to_cuda(tangential_dist.numel() == 2 ? tangential_dist : tangential_dist.slice(0, 0, 2));
            tangential_ptr = tangential_cuda.ptr<float>();
        }
        break;
    default:
        break;
    }

    Tensor radii = Tensor::empty({static_cast<size_t>(C), static_cast<size_t>(N), 2UL}, Device::CUDA, DataType::Int32);
    Tensor means2d = Tensor::empty({static_cast<size_t>(C), static_cast<size_t>(N), 2UL}, Device::CUDA, DataType::Float32);
    Tensor depths = Tensor::empty({static_cast<size_t>(C), static_cast<size_t>(N)}, Device::CUDA, DataType::Float32);
    Tensor conics = Tensor::empty({static_cast<size_t>(C), static_cast<size_t>(N), 3UL}, Device::CUDA, DataType::Float32);

    UnscentedTransformParameters ut_params;

    // User note: in your tree this symbol is in gsplat_lfs (not gsplat_fwd).
    // Keep this exact namespace to match your build.
    ::gsplat_lfs::launch_projection_ut_3dgs_fused_kernel(
        means.ptr<float>(),
        quats.ptr<float>(),
        scales.ptr<float>(),
        opacities.is_valid() ? opacities.ptr<float>() : nullptr,
        viewmat_ptr,
        nullptr,
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
        nullptr,
        nullptr);

    ProjectionResult out;
    out.radii = radii.squeeze(0).contiguous();
    out.means2d = means2d.squeeze(0).contiguous();
    out.depths = depths.squeeze(0).contiguous();
    out.conics = conics.squeeze(0).contiguous();
    return out;
}

// -----------------------------------------------------------------------------
// Center-vote pruning (replicates trainer_old.cpp semantics)
// -----------------------------------------------------------------------------

std::expected<PruningResult, std::string> prune_by_center_vote(
    IStrategy& strategy,
    const CameraDataset& dataset,
    const CenterVotePruningConfig& config) {

    using namespace lfs::core;

    auto& model = strategy.get_model();
    const int N = static_cast<int>(model.size());
    if (N <= 0) {
        return PruningResult{.splats_before = 0, .splats_after = 0, .splats_removed = 0, .success = true};
    }

    auto sizing = get_dataset_sizing_or_error(dataset);
    if (!sizing) {
        return std::unexpected(sizing.error());
    }

    std::vector<int> pos(static_cast<size_t>(N), 0);
    std::vector<int> tot(static_cast<size_t>(N), 0);

    int skipped_no_mask = 0;
    int skipped_size_mismatch = 0;
    int skipped_proj_error = 0;

    const auto& cams = dataset.get_cameras();
    const int n_cams = static_cast<int>(cams.size());

    LOG_INFO("[prune_by_center_vote] Starting: {} splats, {} cameras", N, n_cams);

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

        // Old code: attentionMask > 0.5 -> binary. We keep the same threshold.
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

        // Project
        auto proj = project_splats(cam, model, config);
        if (!proj) {
            ++skipped_proj_error;
            continue;
        }

        // CPU views for robust indexing (same effect as old's CPU mask indexing + CUDA counts).
        Tensor radii_cpu = proj->radii.cpu().contiguous();     // [N,2] int32
        Tensor means2d_cpu = proj->means2d.cpu().contiguous(); // [N,2] float
        Tensor mask_cpu = mask.cpu().contiguous();             // [H,W] float/bool

        auto r_acc = radii_cpu.accessor<int32_t, 2>();
        auto m_acc = means2d_cpu.accessor<float, 2>();
        auto mask_acc = mask_cpu.accessor<float, 2>();

        for (int i = 0; i < N; ++i) {
            const int rx = static_cast<int>(r_acc(i, 0));
            const int ry = static_cast<int>(r_acc(i, 1));
            if (!is_visible_radii_strict(rx, ry)) {
                continue;
            }

            tot[static_cast<size_t>(i)]++;

            int x = round_to_int(m_acc(i, 0));
            int y = round_to_int(m_acc(i, 1));
            x = clamp_int(x, 0, W - 1);
            y = clamp_int(y, 0, H - 1);

            if (mask_acc(y, x) >= 0.5f) {
                pos[static_cast<size_t>(i)]++;
            }
        }

        if ((ci + 1) % 25 == 0 || (ci + 1) == n_cams) {
            LOG_INFO("[prune_by_center_vote] Processing camera {}/{}", (ci + 1), n_cams);
        }
    }

    LOG_INFO("[prune_by_center_vote] Processed {} cameras, skipped: {} no mask, {} size mismatch, {} proj error",
             n_cams, skipped_no_mask, skipped_size_mismatch, skipped_proj_error);

    long long sum_tot = 0;
    for (int i = 0; i < N; ++i) {
        sum_tot += static_cast<long long>(tot[static_cast<size_t>(i)]);
    }
    if (sum_tot == 0) {
        LOG_WARN("[prune_by_center_vote] No visibility accumulated; skipping prune.");
        LOG_INFO("[prune_by_center_vote] Removed 0 splats (0.000%)");
        return PruningResult{.splats_before = N, .splats_after = N, .splats_removed = 0, .success = true};
    }

    bool any_meets = false;
    for (int i = 0; i < N; ++i) {
        if (tot[static_cast<size_t>(i)] >= config.min_visibility_count) {
            any_meets = true;
            break;
        }
    }
    if (!any_meets) {
        LOG_WARN("[prune_by_center_vote] No splats reached min_vis={}; skipping prune.", config.min_visibility_count);
        LOG_INFO("[prune_by_center_vote] Removed 0 splats (0.000%)");
        return PruningResult{.splats_before = N, .splats_after = N, .splats_removed = 0, .success = true};
    }

    // Build keep_mask like trainer_old.cpp:
    // keep = (tot >= min_vis) && (pos/max(tot,1) >= threshold)
    Tensor remove_mask_cpu = Tensor::zeros({static_cast<size_t>(N)}, Device::CPU, DataType::Bool);
    auto rm_acc = remove_mask_cpu.accessor<uint8_t, 1>();

    int keep_cnt = 0;
    int n_remove = 0;

    for (int i = 0; i < N; ++i) {
        const int t = tot[static_cast<size_t>(i)];
        const int p = pos[static_cast<size_t>(i)];
        const float ratio = static_cast<float>(p) / static_cast<float>(std::max(1, t));

        const bool keep = (t >= config.min_visibility_count) && (ratio >= config.vote_ratio_threshold);
        if (keep) {
            ++keep_cnt;
            rm_acc(i) = 0;
        } else {
            ++n_remove;
            rm_acc(i) = 1;
        }
    }

    if (keep_cnt == 0) {
        LOG_WARN("[prune_by_center_vote] keep_mask would be empty (thr={}, min_vis={}); skipping prune.",
                 config.vote_ratio_threshold, config.min_visibility_count);
        LOG_INFO("[prune_by_center_vote] Removed 0 splats (0.000%)");
        return PruningResult{.splats_before = N, .splats_after = N, .splats_removed = 0, .success = true};
    }

    if (n_remove == 0) {
        LOG_INFO("[prune_by_center_vote] Removed 0 splats (0.000%)");
        return PruningResult{.splats_before = N, .splats_after = N, .splats_removed = 0, .success = true};
    }

    Tensor remove_mask = remove_mask_cpu.to(Device::CUDA).contiguous();
    strategy.remove_gaussians(remove_mask);

    const int after = static_cast<int>(strategy.get_model().size());
    const int removed = N - after;

    const double pct = (N > 0) ? (100.0 * static_cast<double>(removed) / static_cast<double>(N)) : 0.0;
    LOG_INFO("[prune_by_center_vote] Requested removal: {}, actually removed: {} ({:.3f}%)", n_remove, removed, pct);

    return PruningResult{.splats_before = N,
                         .splats_after = after,
                         .splats_removed = removed,
                         .success = true};
}

// -----------------------------------------------------------------------------
// Leakage pruning (replicates trainer_old.cpp semantics)
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

    auto& model = strategy.get_model();
    const int N = static_cast<int>(model.size());
    if (N <= 0) {
        return PruningResult{.splats_before = 0, .splats_after = 0, .splats_removed = 0, .success = true};
    }

    auto sizing = get_dataset_sizing_or_error(dataset);
    if (!sizing) {
        return std::unexpected(sizing.error());
    }

    // Sampling directions (4 or 8)
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
        constexpr float s = 0.70710678f;
        dirs.push_back({s, s});
        dirs.push_back({-s, s});
        dirs.push_back({s, -s});
        dirs.push_back({-s, -s});
    }

    std::vector<int> leak_votes(static_cast<size_t>(N), 0); // leak views
    std::vector<int> vis_counts(static_cast<size_t>(N), 0); // candidate views

    int skipped_no_mask = 0;
    int skipped_size_mismatch = 0;
    int skipped_proj_error = 0;

    const auto& cams = dataset.get_cameras();
    const int n_cams = static_cast<int>(cams.size());

    LOG_INFO("[prune_by_mask_leakage] Starting: {} splats, {} cameras", N, n_cams);

    // Buffers reused per camera to reduce allocations.
    std::vector<uint8_t> mask01;
    IntegralImage sat;

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

        // Old code: (attentionMask > 0.5).to(float). Here we mirror it using threshold.
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

        // Prepare CPU binary mask01
        Tensor mask_cpu = mask.cpu().contiguous();
        auto mask_acc = mask_cpu.accessor<float, 2>();

        mask01.assign(static_cast<size_t>(H) * static_cast<size_t>(W), 0);
        for (int y = 0; y < H; ++y) {
            uint8_t* row = mask01.data() + static_cast<size_t>(y) * static_cast<size_t>(W);
            for (int x = 0; x < W; ++x) {
                row[x] = (mask_acc(y, x) >= 0.5f) ? uint8_t(1) : uint8_t(0);
            }
        }

        // Optional dilation tolerance (old uses max_pool2d over mask01).
        // We emulate it via integral-image queries (square window).
        if (config.dilate_px > 0) {
            sat.build(mask01.data(), H, W);
        }

        auto mask_inside = [&](int x, int y) -> bool {
            x = clamp_int(x, 0, W - 1);
            y = clamp_int(y, 0, H - 1);

            if (config.dilate_px <= 0) {
                return mask01[static_cast<size_t>(y) * static_cast<size_t>(W) + static_cast<size_t>(x)] != 0;
            }

            const int r = config.dilate_px;
            const int sum = sat.sum_rect(x - r, y - r, x + r, y + r);
            return sum > 0;
        };

        // Project with leakage projection params
        CenterVotePruningConfig proj_cfg;
        proj_cfg.eps2d = config.eps2d;
        proj_cfg.near_plane = config.near_plane;
        proj_cfg.far_plane = config.far_plane;
        proj_cfg.radius_clip = config.radius_clip;
        proj_cfg.scaling_modifier = config.scaling_modifier;
        proj_cfg.invert_masks = config.invert_masks;

        auto proj = project_splats(cam, model, proj_cfg);
        if (!proj) {
            ++skipped_proj_error;
            continue;
        }

        Tensor radii_cpu = proj->radii.cpu().contiguous();
        Tensor means2d_cpu = proj->means2d.cpu().contiguous();

        auto r_acc = radii_cpu.accessor<int32_t, 2>();
        auto m_acc = means2d_cpu.accessor<float, 2>();

        for (int i = 0; i < N; ++i) {
            const int rx = static_cast<int>(r_acc(i, 0));
            const int ry = static_cast<int>(r_acc(i, 1));
            if (!is_visible_radii_strict(rx, ry)) {
                continue;
            }

            const float frx = static_cast<float>(std::abs(rx));
            const float fry = static_cast<float>(std::abs(ry));
            if (std::max(frx, fry) < config.min_pixel_radius) {
                continue;
            }

            int cx = round_to_int(m_acc(i, 0));
            int cy = round_to_int(m_acc(i, 1));
            cx = clamp_int(cx, 0, W - 1);
            cy = clamp_int(cy, 0, H - 1);

            // Old semantics: only evaluate leakage if center is inside mask.
            if (!mask_inside(cx, cy)) {
                continue;
            }

            vis_counts[static_cast<size_t>(i)]++;

            int outside = 0;
            const int P = static_cast<int>(dirs.size());

            for (const auto& d : dirs) {
                const int dx = round_to_int(frx * d.dx);
                const int dy = round_to_int(fry * d.dy);

                const int sx = clamp_int(cx + dx, 0, W - 1);
                const int sy = clamp_int(cy + dy, 0, H - 1);

                if (!mask_inside(sx, sy)) {
                    outside++;
                }
            }

            const float outside_ratio = (P > 0) ? (static_cast<float>(outside) / static_cast<float>(P)) : 0.0f;
            if (outside_ratio > config.per_view_leak_fraction) {
                leak_votes[static_cast<size_t>(i)]++;
            }
        }

        if ((ci + 1) % 25 == 0 || (ci + 1) == n_cams) {
            LOG_INFO("[prune_by_mask_leakage] Processing camera {}/{}", (ci + 1), n_cams);
        }
    }

    LOG_INFO("[prune_by_mask_leakage] Processed {} cameras, skipped: {} no mask, {} size mismatch, {} proj error",
             n_cams, skipped_no_mask, skipped_size_mismatch, skipped_proj_error);

    long long sum_vis = 0;
    for (int i = 0; i < N; ++i) {
        sum_vis += static_cast<long long>(vis_counts[static_cast<size_t>(i)]);
    }
    if (sum_vis == 0) {
        LOG_WARN("[prune_by_mask_leakage] No candidate visibility accumulated; skipping prune.");
        LOG_INFO("[prune_by_mask_leakage] Removed 0 splats (0.000%)");
        return PruningResult{.splats_before = N, .splats_after = N, .splats_removed = 0, .success = true};
    }

    // Build remove mask (old semantics generalized with config.min_visibility_count)
    Tensor remove_mask_cpu = Tensor::zeros({static_cast<size_t>(N)}, Device::CPU, DataType::Bool);
    auto rm_acc = remove_mask_cpu.accessor<uint8_t, 1>();

    int keep_cnt = 0;
    int n_remove = 0;

    for (int i = 0; i < N; ++i) {
        const int v = vis_counts[static_cast<size_t>(i)];
        if (v < std::max(1, config.min_visibility_count)) {
            rm_acc(i) = 0; // not decided by leakage (kept)
            ++keep_cnt;
            continue;
        }

        const int leaks = leak_votes[static_cast<size_t>(i)];
        const float fail_r = static_cast<float>(leaks) / static_cast<float>(std::max(1, v));
        const float ok_r = 1.0f - fail_r;

        const bool remove = (ok_r < config.leak_keep_threshold);
        if (remove) {
            rm_acc(i) = 1;
            ++n_remove;
        } else {
            rm_acc(i) = 0;
            ++keep_cnt;
        }
    }

    if (keep_cnt == 0) {
        LOG_WARN("[prune_by_mask_leakage] keep_mask would be empty (keep_thr={}); skipping prune.", config.leak_keep_threshold);
        LOG_INFO("[prune_by_mask_leakage] Removed 0 splats (0.000%)");
        return PruningResult{.splats_before = N, .splats_after = N, .splats_removed = 0, .success = true};
    }

    if (n_remove == 0) {
        LOG_INFO("[prune_by_mask_leakage] Removed 0 splats (0.000%)");
        return PruningResult{.splats_before = N, .splats_after = N, .splats_removed = 0, .success = true};
    }

    Tensor remove_mask = remove_mask_cpu.to(Device::CUDA).contiguous();
    strategy.remove_gaussians(remove_mask);

    const int after = static_cast<int>(strategy.get_model().size());
    const int removed = N - after;
    const double pct = (N > 0) ? (100.0 * static_cast<double>(removed) / static_cast<double>(N)) : 0.0;

    LOG_INFO("[prune_by_mask_leakage] Requested removal: {}, actually removed: {} ({:.3f}%)", n_remove, removed, pct);

    return PruningResult{.splats_before = N,
                         .splats_after = after,
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
