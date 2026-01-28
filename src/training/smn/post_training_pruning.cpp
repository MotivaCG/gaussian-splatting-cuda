/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * ScanMeNow file
 *
 * post_training_pruning.cpp
 *     Trainer::prune_by_center_vote
 *     Trainer::prune_by_mask_leakage
 *     Trainer::prune_after_training
 */

#include "trainer.hpp"

#include "core/logger.hpp"
#include "loader/cache_image_loader.hpp"
#include "rasterization/projection_fast.hpp"
#include "external/nanoflann.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <optional>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace gs::training {

namespace {

// ------------------------------------------------------------
// Isolation distance pruning (torch + nanoflann)
// ------------------------------------------------------------

struct IsolationDistanceConfig {
    bool enabled = true;

    // Query KNN count: we query max(k_neighbors, kth_neighbor)+1 to safely exclude self.
    int k_neighbors = 8;

    // 1-based neighbor rank excluding self.
    // 1 = nearest neighbor (what you asked originally).
    // 4 = more robust to mini-islands of 2-3 flyers.
    int kth_neighbor = 4;

    // Remove if d_k(i) > threshold_multiplier * median(d_k)
    float threshold_multiplier = 32.0f;

    // Optional absolute clamp in meters: thr = max(abs_distance_min, threshold_multiplier * median)
    float abs_distance_min = 0.0f;
};

// Point adaptor for nanoflann over a flat float buffer [x0,y0,z0, x1,y1,z1, ...]
struct TorchPointCloud {
    const float* pts = nullptr;
    size_t N = 0;

    inline size_t kdtree_get_point_count() const { return N; }
    inline float kdtree_get_pt(const size_t idx, int dim) const {
        return pts[idx * 3ULL + static_cast<size_t>(dim)];
    }
    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
};

static float median_inplace(std::vector<float>& v) {
    if (v.empty())
        return 0.0f;
    const size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(mid), v.end());
    return v[mid];
}

static void prune_by_isolation_distance(SplatData& model, const IsolationDistanceConfig& cfg) {
    if (!cfg.enabled)
        return;

    torch::NoGradGuard no_grad;

    const torch::Tensor means = model.get_means();
    if (!means.defined() || means.numel() == 0 || means.dim() != 2 || means.size(1) != 3) {
        std::cout << "[Prune isolation] Invalid means tensor (expected [N,3]). Skipping.\n";
        return;
    }

    const int64_t N64 = means.size(0);
    if (N64 <= 0) {
        return;
    }

    const int kth = std::max(1, cfg.kth_neighbor);
    if (N64 <= kth) {
        std::cout << "[Prune isolation] Not enough splats (N=" << (long long)N64
                  << ") for kth_neighbor=" << kth << ". Skipping.\n";
        return;
    }

    // Pull positions to CPU float32 contiguous
    torch::Tensor means_cpu = means.detach().to(torch::kCPU);
    if (means_cpu.scalar_type() != torch::kFloat32) {
        means_cpu = means_cpu.to(torch::kFloat32);
    }
    means_cpu = means_cpu.contiguous();

    const int N = static_cast<int>(N64);
    const float* pts = means_cpu.data_ptr<float>();
    if (!pts) {
        std::cout << "[Prune isolation] means_cpu data_ptr is null. Skipping.\n";
        return;
    }

    // Build KD-tree
    TorchPointCloud cloud;
    cloud.pts = pts;
    cloud.N = static_cast<size_t>(N);

    using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, TorchPointCloud>,
        TorchPointCloud, 3>;

    KDTree tree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    tree.buildIndex();

    // Query count: max(k_neighbors, kth) + 1 (for self), clamp to N
    const int base_k = std::max(std::max(1, cfg.k_neighbors), kth);
    const int kq = std::min(N, base_k + 1);

    if (kq <= 1 || kq <= kth) {
        std::cout << "[Prune isolation] Query k too small (kq=" << kq << ", kth=" << kth << "). Skipping.\n";
        return;
    }

    std::vector<float> dk(static_cast<size_t>(N), 0.0f);

#ifdef _OPENMP
#pragma omp parallel
    {
        std::vector<size_t> indices(static_cast<size_t>(kq));
        std::vector<float> dists_sq(static_cast<size_t>(kq));
        std::vector<float> neigh_dists;
        neigh_dists.reserve(static_cast<size_t>(kq));
        nanoflann::KNNResultSet<float> resultSet(kq);

#pragma omp for schedule(static)
        for (int i = 0; i < N; ++i) {
            resultSet.init(indices.data(), dists_sq.data());
            tree.findNeighbors(resultSet, pts + static_cast<size_t>(i) * 3ULL, nanoflann::SearchParameters());

            neigh_dists.clear();

            // Exclude self robustly (do not assume self is at slot 0)
            for (int j = 0; j < kq; ++j) {
                const size_t idx = indices[static_cast<size_t>(j)];
                if (static_cast<int>(idx) == i)
                    continue;
                const float dsq = dists_sq[static_cast<size_t>(j)];
                neigh_dists.push_back(std::sqrt(std::max(0.0f, dsq)));
            }

            if (static_cast<int>(neigh_dists.size()) < kth) {
                dk[static_cast<size_t>(i)] = std::numeric_limits<float>::infinity();
                continue;
            }

            std::nth_element(
                neigh_dists.begin(),
                neigh_dists.begin() + static_cast<std::ptrdiff_t>(kth - 1),
                neigh_dists.end());

            dk[static_cast<size_t>(i)] = neigh_dists[static_cast<size_t>(kth - 1)];
        }
    }
#else
    std::vector<size_t> indices(static_cast<size_t>(kq));
    std::vector<float> dists_sq(static_cast<size_t>(kq));
    std::vector<float> neigh_dists;
    neigh_dists.reserve(static_cast<size_t>(kq));
    nanoflann::KNNResultSet<float> resultSet(kq);

    for (int i = 0; i < N; ++i) {
        resultSet.init(indices.data(), dists_sq.data());
        tree.findNeighbors(resultSet, pts + static_cast<size_t>(i) * 3ULL, nanoflann::SearchParameters());

        neigh_dists.clear();
        for (int j = 0; j < kq; ++j) {
            const size_t idx = indices[static_cast<size_t>(j)];
            if (static_cast<int>(idx) == i)
                continue;
            const float dsq = dists_sq[static_cast<size_t>(j)];
            neigh_dists.push_back(std::sqrt(std::max(0.0f, dsq)));
        }

        if (static_cast<int>(neigh_dists.size()) < kth) {
            dk[static_cast<size_t>(i)] = std::numeric_limits<float>::infinity();
            continue;
        }

        std::nth_element(
            neigh_dists.begin(),
            neigh_dists.begin() + static_cast<std::ptrdiff_t>(kth - 1),
            neigh_dists.end());

        dk[static_cast<size_t>(i)] = neigh_dists[static_cast<size_t>(kth - 1)];
    }
#endif

    // Global median of d_k
    std::vector<float> pool = dk;
    const float med = median_inplace(pool);
    if (!std::isfinite(med) || med <= 0.0f) {
        std::cout << "[Prune isolation] Invalid median d_k=" << med << ". Skipping.\n";
        return;
    }

    const float base_thr = cfg.threshold_multiplier * med;
    const float abs_thr = (cfg.abs_distance_min > 0.0f) ? cfg.abs_distance_min : 0.0f;
    const float thr = (abs_thr > base_thr) ? abs_thr : base_thr;

    std::cout << "[Prune isolation] d" << kth << " median=" << med
              << " m (N=" << N << "), thr=" << thr
              << " m (x" << cfg.threshold_multiplier
              << "), abs_min=" << abs_thr << " m\n";

    // Keep mask on CPU as uint8 then push to original device as bool
    auto keep_u8 = torch::empty({N}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
    uint8_t* keep_ptr = keep_u8.data_ptr<uint8_t>();

    int64_t keep_cnt = 0;
    for (int i = 0; i < N; ++i) {
        const float d = dk[static_cast<size_t>(i)];
        const bool remove = std::isfinite(d) && (d > thr);
        const uint8_t keep = remove ? 0u : 1u;
        keep_ptr[i] = keep;
        keep_cnt += static_cast<int64_t>(keep);
    }

    if (keep_cnt == N64) {
        std::cout << "[Prune isolation] Removed 0 splats.\n";
        return;
    }
    if (keep_cnt == 0) {
        std::cout << "[Prune isolation] Would remove all splats. Skipping.\n";
        return;
    }

    const int64_t removed = N64 - keep_cnt;

    // Convert to bool mask on model device
    auto keep_mask = keep_u8.to(torch::kBool).to(means.device());
    model.filterByMask(keep_mask);

    const double pct = (N64 > 0) ? (100.0 * static_cast<double>(removed) / static_cast<double>(N64)) : 0.0;
    std::cout << "[Trainer] Prune isolation-distance: removed " << (long long)removed
              << " / " << (long long)N64 << " splats (" << pct
              << "%) (kth=" << kth << ", thr=" << thr << ")\n";
}

} // namespace

// -----------------------------------------------------------------------------
// Keep splats that land often inside the center-mask (vote-based).
// (Moved from trainer.cpp without changes.)
// -----------------------------------------------------------------------------
void Trainer::prune_by_center_vote(float center_keep_threshold, int min_visibility_count) {
    torch::NoGradGuard no_grad;

    // 0) Model tensors
    SplatData& model = strategy_->get_model();
    const int64_t N = model.get_means().size(0);
    if (N <= 0) {
        std::cout << "[Prune center] No gaussians to prune.\n";
        return;
    }

    auto means3D = model.get_means();
    auto scales = model.get_scaling();
    auto rotations = model.get_rotation();
    auto opacities = model.get_opacity();
    if (opacities.defined() && opacities.dim() == 2 && opacities.size(-1) == 1)
        opacities = opacities.squeeze(-1);

    // 1) Vote buffers (CUDA int32)
    auto pos = torch::zeros({N}, torch::kInt32).to(torch::kCUDA); // inside votes
    auto tot = torch::zeros({N}, torch::kInt32).to(torch::kCUDA); // visible counts

    // 2) Iterate once over the dataset with the efficient loader
    const size_t dataset_size = train_dataset_size_;
    if (dataset_size == 0) {
        std::cout << "[Prune center] Empty dataset.\n";
        return;
    }

    const int workers = std::max(1, params_.optimization.num_workers);
    auto train_dataloader = create_infinite_dataloader_from_dataset(train_dataset_, workers);
    auto loader = train_dataloader->begin(); // iterate the dataloader itself

    // 3) Projection constants
    const float eps2d = 0.3f;
    const float near_plane = 0.01f;
    const float far_plane = 10000.0f;
    const float radius_clip = 0.0f;
    const float scaling_mod = 1.0f;

    int idx_img = 1;
    size_t skipped_missing = 0, skipped_shape = 0, skipped_size = 0, skipped_proj = 0;

    for (size_t i = 0; i < dataset_size; ++i, ++loader) {
        std::printf("\r[Prune Center] image %d/%zu", idx_img++, dataset_size);
        std::fflush(stdout);

        // Consume a batch like in the training loop
        auto& batch = *loader;
        const auto camera_with_image = batch[0].data; // {camera, image, attentionMask}
        Camera* cam = camera_with_image.camera;
        auto mask_f = camera_with_image.attentionMask;

        if (!cam) {
            std::cout << "\n[Prune center] Warning: null camera; skipping view.\n";
            continue;
        }
        if (!mask_f.defined() || mask_f.numel() == 0) {
            if (skipped_missing < 3)
                std::cout << "\n[Prune center] Warning: undefined/empty attentionMask; skipping view.\n";
            ++skipped_missing;
            continue;
        }

        // CPU binary mask [H,W]
        auto m3 = (mask_f > 0.5f);
        auto m2 = (m3.dim() == 3 && m3.size(0) == 1) ? m3.squeeze(0) : m3;
        if (m2.dim() != 2) {
            if (skipped_shape < 3)
                std::cout << "\n[Prune center] Warning: mask wrong shape; skipping view.\n";
            ++skipped_shape;
            continue;
        }
        auto mask = m2.to(torch::kCPU).contiguous();

        const int W = (int)cam->image_width();
        const int H = (int)cam->image_height();
        if (mask.size(1) != W || mask.size(0) != H) {
            if (skipped_size < 3) {
                std::cout << "\n[Prune center] Warning: mask size ("
                          << (long long)mask.size(1) << "x" << (long long)mask.size(0)
                          << ") != camera size (" << W << "x" << H << "); skipping view.\n";
            }
            ++skipped_size;
            continue;
        }

        // Camera tensors (CUDA) with batch dim [1,...] if needed
        auto view = cam->world_view_transform().to(torch::kCUDA);
        auto K = cam->K().to(torch::kCUDA);
        if (view.dim() == 2)
            view = view.unsqueeze(0);
        if (K.dim() == 2)
            K = K.unsqueeze(0);

        // Distortion (optional) - ProjectFast handles device/shape
        std::optional<torch::Tensor> radial, tangential;
        if (cam->radial_distortion().defined() && cam->radial_distortion().numel() > 0)
            radial = cam->radial_distortion();
        if (cam->tangential_distortion().defined() && cam->tangential_distortion().numel() > 0)
            tangential = cam->tangential_distortion();

        // Fast projection
        auto [radii, means2d] = ProjectFast(
            means3D, rotations, scales, opacities,
            view, K,
            W, H,
            eps2d, near_plane, far_plane,
            radius_clip, scaling_mod,
            cam->camera_model_type(),
            radial, tangential, /*thin_prism*/ std::nullopt);

        if (!radii.defined() || !means2d.defined()) {
            if (skipped_proj < 3)
                std::cout << "\n[Prune center] Warning: projection returned undefined tensors; skipping view.\n";
            ++skipped_proj;
            continue;
        }

        if (radii.dim() == 3 && radii.size(0) == 1)
            radii = radii.squeeze(0);
        if (means2d.dim() == 3 && means2d.size(0) == 1)
            means2d = means2d.squeeze(0);

        // Visible splats: positive projected radii
        torch::Tensor visible;
        if (radii.dim() == 2 && radii.size(1) >= 1)
            visible = (radii > 0.0f).all(-1);
        else if (radii.dim() == 1)
            visible = (radii > 0.0f);
        else {
            if (skipped_proj < 3)
                std::cout << "\n[Prune center] Warning: invalid radii shape; skipping view.\n";
            ++skipped_proj;
            continue;
        }
        if (!visible.any().item<bool>())
            continue;

        auto vidx = visible.nonzero().squeeze(-1); // [M], CUDA

        // 2D centers -> CPU ints in [0..W-1],[0..H-1]
        auto xy_cuda = means2d.index({vidx}); // [M,2], CUDA
        auto xy = xy_cuda.detach().to(torch::kCPU);
        auto x = torch::round(xy.select(1, 0)).to(torch::kLong).clamp(0, W - 1);
        auto y = torch::round(xy.select(1, 1)).to(torch::kLong).clamp(0, H - 1);
        auto lin = y * W + x; // [M] long CPU

        // Vote on CPU mask, accumulate on CUDA
        auto inside_cpu = mask.flatten().index({lin}); // bool CPU
        auto inside_i32_cuda = inside_cpu.to(torch::kInt32).to(torch::kCUDA);
        auto ones_i32_cuda = torch::ones_like(inside_i32_cuda, torch::kInt32);

        pos.index_add_(0, vidx, inside_i32_cuda);
        tot.index_add_(0, vidx, ones_i32_cuda);
    }
    std::printf("\n");

    // Graceful exits
    if (tot.sum().item<long long>() == 0) {
        std::cout << "[Prune center] No visibility accumulated (invalid/missing masks?); aborting prune.\n";
        return;
    }
    auto meets_vis = (tot >= min_visibility_count);
    if (!meets_vis.any().item<bool>()) {
        std::cout << "[Prune center] No splats reached min_vis=" << min_visibility_count << "; aborting prune.\n";
        return;
    }

    // Final keep & prune
    auto tot_f = tot.to(torch::kFloat32).clamp_min(1.0f);
    auto ratio = pos.to(torch::kFloat32) / tot_f;
    auto keep_mask = meets_vis & (ratio >= center_keep_threshold);

    const int64_t keep_cnt = keep_mask.sum().item<long long>();
    if (keep_cnt == 0) {
        std::cout << "[Prune center] keep_mask would be empty (thr=" << center_keep_threshold << "); skipping prune.\n";
        return;
    }

    const int removed = (keep_mask == 0).sum().item<int>();
    model.filterByMask(keep_mask);

    std::cout << "[Trainer] Prune center-vote: removed " << removed
              << " / " << N << " splats (thr=" << center_keep_threshold
              << ", min_vis=" << min_visibility_count << ")\n";
}

// -----------------------------------------------------------------------------
// Prune splats whose projected footprint leaks outside the mask too often.
// Success-rate semantics: keep only if (views_without_leak / candidate_views) >= keep_threshold.
// (Moved from trainer.cpp without changes.)
// -----------------------------------------------------------------------------
void Trainer::prune_by_mask_leakage(float leak_keep_threshold,
                                   float min_pixel_radius,
                                   float min_center_mask,
                                   int sample_points,
                                   int dilate_px,
                                   float per_view_leak_frac) {
    torch::NoGradGuard no_grad;

    // 0) Model tensors
    SplatData& model = strategy_->get_model();
    const int64_t N = model.get_means().size(0);
    if (N <= 0) {
        std::cout << "[Prune leak] No gaussians to prune.\n";
        return;
    }

    auto means3D = model.get_means();
    auto scales = model.get_scaling();
    auto rotations = model.get_rotation();
    auto opacities = model.get_opacity();
    if (opacities.defined() && opacities.dim() == 2 && opacities.size(-1) == 1)
        opacities = opacities.squeeze(-1);

    // 1) Leak vote buffers (CUDA int32)
    auto leak_votes = torch::zeros({N}, torch::kInt32).to(torch::kCUDA); // # leak-views
    auto vis_counts = torch::zeros({N}, torch::kInt32).to(torch::kCUDA); // # candidate views

    // 2) Iterate once over the dataset (same pattern as training loop)
    const size_t dataset_size = train_dataset_size_;
    if (dataset_size == 0) {
        std::cout << "[Prune leak] Empty dataset.\n";
        return;
    }

    const int workers = std::max(1, params_.optimization.num_workers);
    auto train_dataloader = create_infinite_dataloader_from_dataset(train_dataset_, workers);
    auto loader = train_dataloader->begin();

    // 3) Projection constants
    const float eps2d = 0.3f;
    const float near_plane = 0.01f;
    const float far_plane = 10000.0f;
    const float radius_clip = 0.0f;
    const float scaling_mod = 1.0f;

    // 4/8-direction sampling around ellipse (axis-aligned approx)
    std::vector<std::array<float, 2>> dirs = {{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}};
    if (sample_points >= 8) {
        const float s = 0.70710678f;
        dirs.push_back({s, s});
        dirs.push_back({-s, s});
        dirs.push_back({s, -s});
        dirs.push_back({-s, -s});
    }

    int idx_img = 1;
    size_t skipped_missing = 0, skipped_shape = 0, skipped_size = 0, skipped_proj = 0;

    for (size_t i = 0; i < dataset_size; ++i, ++loader) {
        std::printf("\r[Prune Leakage] image %d/%zu", idx_img++, dataset_size);
        std::fflush(stdout);

        // Consume a batch like in the training loop
        auto& batch = *loader;
        const auto sample = batch[0].data; // CameraDataset::Sample {camera, image, attentionMask}
        Camera* cam = sample.camera;
        auto mask_f = sample.attentionMask;

        if (!cam) {
            std::cout << "\n[Prune leak] Warning: null camera; skipping view.\n";
            continue;
        }
        if (!mask_f.defined() || mask_f.numel() == 0) {
            if (skipped_missing < 3)
                std::cout << "\n[Prune leak] Warning: undefined/empty attentionMask; skipping view.\n";
            ++skipped_missing;
            continue;
        }

        // CUDA mask [H,W] in {0,1}
        auto mask01_3d = (mask_f > 0.5f).to(torch::kFloat32);
        auto mask01 = (mask01_3d.dim() == 3 && mask01_3d.size(0) == 1) ? mask01_3d.squeeze(0) : mask01_3d;
        if (mask01.dim() != 2) {
            if (skipped_shape < 3)
                std::cout << "\n[Prune leak] Warning: mask wrong shape; skipping view.\n";
            ++skipped_shape;
            continue;
        }
        mask01 = mask01.contiguous().to(torch::kCUDA);

        const int W = (int)cam->image_width();
        const int H = (int)cam->image_height();
        if (mask01.size(1) != W || mask01.size(0) != H) {
            if (skipped_size < 3)
                std::cout << "\n[Prune leak] Warning: mask size != camera size; skipping view.\n";
            ++skipped_size;
            continue;
        }

        // Optional dilation (pixel tolerance at boundary)
        if (dilate_px > 0) {
            const int k = 2 * dilate_px + 1;
            auto m4 = mask01.unsqueeze(0).unsqueeze(0); // [1,1,H,W]
            auto opts = torch::nn::functional::MaxPool2dFuncOptions(k).stride(1).padding(dilate_px);
            mask01 = torch::nn::functional::max_pool2d(m4, opts).squeeze(0).squeeze(0); // [H,W]
        }

        // Camera tensors (CUDA) with batch dim [1,...] if needed
        auto view = cam->world_view_transform().to(torch::kCUDA);
        auto K = cam->K().to(torch::kCUDA);
        if (view.dim() == 2)
            view = view.unsqueeze(0);
        if (K.dim() == 2)
            K = K.unsqueeze(0);

        // Distortion (optional) - ProjectFast handles device/shape
        std::optional<torch::Tensor> radial, tangential;
        if (cam->radial_distortion().defined() && cam->radial_distortion().numel() > 0)
            radial = cam->radial_distortion();
        if (cam->tangential_distortion().defined() && cam->tangential_distortion().numel() > 0)
            tangential = cam->tangential_distortion();

        // Fast projection
        auto [radii, means2d] = ProjectFast(
            means3D, rotations, scales, opacities,
            view, K,
            W, H,
            eps2d, near_plane, far_plane,
            radius_clip, scaling_mod,
            cam->camera_model_type(),
            radial, tangential, /*thin_prism*/ std::nullopt);

        if (!radii.defined() || !means2d.defined()) {
            if (skipped_proj < 3)
                std::cout << "\n[Prune leak] Warning: projection returned undefined tensors; skipping view.\n";
            ++skipped_proj;
            continue;
        }

        if (radii.dim() == 3 && radii.size(0) == 1)
            radii = radii.squeeze(0);
        if (means2d.dim() == 3 && means2d.size(0) == 1)
            means2d = means2d.squeeze(0);

        // Visible splats: positive projected radii (handle 1D/2D radii)
        torch::Tensor visible;
        if (radii.dim() == 2 && radii.size(1) >= 1)
            visible = (radii > 0.0f).all(-1);
        else if (radii.dim() == 1)
            visible = (radii > 0.0f);
        else {
            if (skipped_proj < 3)
                std::cout << "\n[Prune leak] Warning: invalid radii shape; skipping view.\n";
            ++skipped_proj;
            continue;
        }
        if (!visible.any().item<bool>())
            continue;

        auto vidx = visible.nonzero().squeeze(-1); // [M], CUDA

        // Centers & radii for visible splats
        auto xy = means2d.index({vidx}); // [M,2], CUDA
        auto rx = (radii.dim() == 2 ? radii.index({vidx, 0}).abs()
                                    : radii.index({vidx}).abs()); // [M]
        // If radii is 1D, use the same for x/y to keep a conservative ellipse
        auto ry = (radii.dim() == 2 ? radii.index({vidx, 1}).abs()
                                    : rx.clone());

        // Pixel centers
        auto cx = torch::round(xy.select(1, 0)).to(torch::kLong).clamp(0, W - 1);
        auto cy = torch::round(xy.select(1, 1)).to(torch::kLong).clamp(0, H - 1);

        // Eligibility for leakage test
        auto center_mask = mask01.index({cy, cx});       // [M] float {0,1}
        auto center_in = center_mask >= min_center_mask; // bool
        auto large_fp = (torch::max(rx, ry) >= min_pixel_radius);
        auto keep = center_in & large_fp;
        if (!keep.any().item<bool>())
            continue;

        auto kidx = keep.nonzero().squeeze(-1); // [K]
        auto rxk = rx.index({kidx});
        auto ryk = ry.index({kidx});
        auto cxk = cx.index({kidx});
        auto cyk = cy.index({kidx});

        // Sample P points around an axis-aligned ellipse per kept splat
        const int P = (int)dirs.size();
        auto sx = torch::empty({kidx.size(0), P}, torch::dtype(torch::kLong).device(torch::kCUDA));
        auto sy = torch::empty_like(sx);
        for (int p = 0; p < P; ++p) {
            auto dx = torch::round(rxk * dirs[p][0]).to(torch::kLong);
            auto dy = torch::round(ryk * dirs[p][1]).to(torch::kLong);
            sx.index_put_({torch::indexing::Slice(), p}, (cxk + dx).clamp(0, W - 1));
            sy.index_put_({torch::indexing::Slice(), p}, (cyk + dy).clamp(0, H - 1));
        }

        // Per-splat outside ratio in this view
        auto lin = (sy * W + sx).reshape({-1});                    // [K*P]
        auto mvals = mask01.view({-1}).index({lin}).view({-1, P}); // [K,P]
        auto outside_ratio = (1.0f - mvals).mean(1);               // [K] in [0,1]

        // Count a "leak view" if outside_ratio > per_view_leak_frac
        auto leak_here = (outside_ratio > per_view_leak_frac).to(torch::kInt32); // [K]
        auto ids = vidx.index({kidx});                                           // [K] original IDs

        vis_counts.index_add_(0, ids, torch::ones_like(leak_here));
        leak_votes.index_add_(0, ids, leak_here);
    }
    std::printf("\n");

    // Graceful exits
    if (vis_counts.sum().item<long long>() == 0) {
        std::cout << "[Prune leak] No candidate visibility accumulated (invalid/missing masks?); aborting prune.\n";
        return;
    }

    auto vis_ok = (vis_counts >= 1);
    auto fail_r = leak_votes.to(torch::kFloat32) / vis_counts.to(torch::kFloat32).clamp_min(1.0f);
    auto ok_r = 1.0f - fail_r;
    auto is_prune = vis_ok & (ok_r < leak_keep_threshold);
    auto keep_mask = (~is_prune).to(torch::kBool);

    const int64_t keep_cnt = keep_mask.sum().item<long long>();
    if (keep_cnt == 0) {
        std::cout << "[Prune leak] keep_mask would be empty (keep_thr=" << leak_keep_threshold << "); skipping prune.\n";
        return;
    }

    const int removed = is_prune.sum().item<int>();
    model.filterByMask(keep_mask);

    std::cout << "[Trainer] Prune mask-leakage: removed " << removed
              << " / " << N << " splats (keep_thr=" << leak_keep_threshold
              << ", min_radius=" << min_pixel_radius
              << ", samples=" << (sample_points >= 8 ? 8 : 4)
              << ", dilate_px=" << dilate_px
              << ", per_view_leak_frac=" << per_view_leak_frac
              << ")\n";
}

// -----------------------------------------------------------------------------
// Post-training pruning entrypoint:
//   1) center-vote
//   2) mask-leakage
//   3) isolation-distance (new)
// -----------------------------------------------------------------------------
void Trainer::prune_after_training(float vote_ratio_threshold, float leak_keep_threshold) {
    // 1) Center-vote pruning
    prune_by_center_vote(vote_ratio_threshold);

    // 2) Mask-leakage pruning (on remaining splats)
    prune_by_mask_leakage(leak_keep_threshold);

    // 3) Isolation-distance pruning
    /* IsolationDistanceConfig iso;
    prune_by_isolation_distance(strategy_->get_model(), iso);*/
    //Not required since SMN tool takes care of it
}

} // namespace gs::training
