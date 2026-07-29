/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// SMN: ported from the previous method (see mask_pruning.hpp).

#include "mask_pruning.hpp"

#include "training/rasterization/gsplat/Common.h" // CameraModelType, UnscentedTransformParameters, ShutterType
#include "training/rasterization/gsplat/Projection.h" // launch_projection_ut_3dgs_fused_kernel
#include "core/logger.hpp"
#include "nanoflann.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif


#ifndef _OPENMP
static inline int omp_get_max_threads() { return 1; }
#endif

// SMN: the current LFS Tensor has no accessor<>; minimal contiguous-CPU shim
// so the ported code keeps its original call sites.
namespace {
    template <typename T, int NDIMS>
    struct TensorAcc;

    template <typename T>
    struct TensorAcc<T, 1> {
        T* p;
        T& operator()(int i) const { return p[static_cast<size_t>(i)]; }
    };

    template <typename T>
    struct TensorAcc<T, 2> {
        T* p;
        size_t cols;
        T& operator()(int i, int j) const {
            return p[static_cast<size_t>(i) * cols + static_cast<size_t>(j)];
        }
    };

    template <typename T, int NDIMS>
    TensorAcc<T, NDIMS> make_acc(lfs::core::Tensor& t) {
        static_assert(NDIMS == 1 || NDIMS == 2);
        if constexpr (NDIMS == 1) {
            return TensorAcc<T, 1>{t.ptr<T>()};
        } else {
            return TensorAcc<T, 2>{t.ptr<T>(), t.shape()[1]};
        }
    }
} // namespace

namespace lfs::training::smn::mask_pruning {

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

        static inline int required_views_from_ratio(int usable_cameras, float min_visibility_ratio) {
            const int usable = std::max(0, usable_cameras);
            const float ratio = std::clamp(min_visibility_ratio, 0.0f, 1.0f);
            if (usable <= 0) {
                return 1;
            }
            return std::max(1, static_cast<int>(std::ceil(ratio * static_cast<float>(usable))));
        }

        // Old semantics: radii must be positive in both axes to be considered visible.
        static inline bool is_visible_radii_strict(int rx, int ry) {
            return (rx > 0) && (ry > 0);
        }

        static lfs::core::Tensor mask_to_float01(const lfs::core::Tensor& mask) {
            using namespace lfs::core;

            if (!mask.is_valid()) {
                return {};
            }

            const Tensor mask_f = mask.to(DataType::Float32);
            const Tensor normalized_binary = mask_f.ge(0.5f).to(DataType::Float32);
            const Tensor image_binary = mask_f.ge(127.5f).to(DataType::Float32);
            return Tensor::where(mask_f.le(1.0f), normalized_binary, image_binary).contiguous();
        }

        static void preload_masks_sequential(
            const std::vector<std::shared_ptr<lfs::core::Camera>>& cams,
            const DatasetSizing& sizing,
            bool invert_masks) {

            for (const auto& cam_ptr : cams) {
                if (!cam_ptr || !cam_ptr->has_mask()) {
                    continue;
                }

                (void)cam_ptr->load_and_get_mask(
                    sizing.resize_factor,
                    sizing.max_width,
                    invert_masks,
                    0.5f);
            }
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

        // prepare_undistortion() replaces K and the image domain with a
        // rectified pinhole camera, while intentionally retaining the source
        // distortion metadata for image loading. Applying those coefficients
        // again here would double-distort projected votes relative to the
        // already rectified masks.
        const bool rectified = camera.is_undistort_prepared();
        // NOTE: camera.camera_model_type() is lfs::core::CameraModelType; kernels expect ::CameraModelType.
        const ::CameraModelType camera_model = rectified
                                                   ? ::CameraModelType::PINHOLE
                                                   : static_cast<::CameraModelType>(
                                                         static_cast<int>(camera.camera_model_type()));

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

        if (!rectified) switch (camera_model) {
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
    // Geometric come pruning
    // -----------------------------------------------------------------------------

    std::expected<PruningResult, std::string> prune_by_geometric_dome(
        IStrategy& strategy,
        const CameraDataset& dataset,
        const GeometricDomePruningConfig& config) {

        using namespace lfs::core;

        if (!config.enabled) {
            const int n = static_cast<int>(strategy.get_model().size());
            return PruningResult{.splats_before = n, .splats_after = n, .splats_removed = 0, .success = true};
        }

        auto& model = strategy.get_model();
        const int N = static_cast<int>(model.size());
        if (N == 0) {
            return PruningResult{.splats_before = 0, .splats_after = 0, .splats_removed = 0, .success = true};
        }

        // Pull Gaussian positions to CPU [N, 3]
        Tensor means_cpu = model.get_means().cpu().contiguous();
        if (!means_cpu.is_valid() || means_cpu.ndim() != 2 ||
            static_cast<int>(means_cpu.shape()[0]) != N ||
            static_cast<int>(means_cpu.shape()[1]) != 3) {
            return std::unexpected("prune_by_geometric_dome: invalid means tensor - expected [N,3] float32");
        }
        const float* pts = means_cpu.ptr<float>();

        std::vector<bool> remove_flags(static_cast<size_t>(N), false);
        int behind_removed = 0;
        int floor_removed = 0;
        int scale_removed = 0;

        // =========================================================================
        // Pass 1: behind-camera filter
        // In an inward-facing dome, any Gaussian behind the optical plane of any
        // camera is outside the dome volume by definition.
        // Cost: O(N * n_cams) dot products, no GPU sync, no mask access.
        // =========================================================================
        if (config.enable_behind_camera) {
            const auto& cameras = dataset.get_cameras();
            const int n_cams = static_cast<int>(cameras.size());

            if (n_cams == 0) {
                LOG_WARN("[prune_by_geometric_dome] enable_behind_camera=true but no cameras in dataset - skipping pass");
            } else {
                for (const auto& cam_ptr : cameras) {
                    if (!cam_ptr)
                        continue;

                    // world_view_transform() always returns [1,4,4] because world_to_view()
                    // always calls unsqueeze(0). The batch dim is cosmetic - data starts at
                    // index 0. Do NOT use (ndim()==3 ? 16 : 0) as that always evaluates to
                    // 16 and reads row 3 of the matrix instead of row 2.
                    Tensor w2c_cpu = cam_ptr->world_view_transform().cpu().contiguous();
                    const float* w2c = w2c_cpu.ptr<float>();

                    // Row 2 of the 3x3 rotation block = camera Z axis in world space = forward.
                    // Flat layout of [1,4,4]: element [0, row, col] = w2c[row*4 + col]
                    const float fx = w2c[8];  // [0, 2, 0]
                    const float fy = w2c[9];  // [0, 2, 1]
                    const float fz = w2c[10]; // [0, 2, 2]

                    // cam_position() is [3] on CUDA - must bring to CPU before reading ptr
                    Tensor cam_pos_cpu = cam_ptr->cam_position().cpu().contiguous();
                    const float* cp = cam_pos_cpu.ptr<float>();
                    const float ox = cp[0], oy = cp[1], oz = cp[2];

                    for (int i = 0; i < N; ++i) {
                        if (remove_flags[static_cast<size_t>(i)])
                            continue;

                        const float dx = pts[i * 3 + 0] - ox;
                        const float dy = pts[i * 3 + 1] - oy;
                        const float dz = pts[i * 3 + 2] - oz;

                        // In COLMAP/OpenCV convention, +Z is forward into the scene.
                        // dot(to_splat, forward) <= behind_tolerance -> splat is behind this camera
                        const float dot = dx * fx + dy * fy + dz * fz;
                        if (dot <= config.behind_tolerance) {
                            remove_flags[static_cast<size_t>(i)] = true;
                            behind_removed++;
                        }
                    }
                }

                LOG_INFO("[prune_by_geometric_dome] Behind-camera pass: {} splats removed ({:.1f}%)",
                         behind_removed,
                         100.0f * static_cast<float>(behind_removed) / static_cast<float>(N));
            }
        }

        // =========================================================================
        // Pass 2: floor pruning
        // Removes Gaussians below a world-space Y threshold. Floor reflections,
        // shadow Gaussians, and noise from the lowest camera ring typically
        // reconstruct slightly below y=0. Tests the Gaussian center only -
        // Gaussians that legitimately cross the floor plane (feet, ankles) are
        // preserved as long as their center is above floor_y.
        // =========================================================================
        if (config.enable_floorandceil_pruning) {
            for (int i = 0; i < N; ++i) {
                if (remove_flags[static_cast<size_t>(i)])
                    continue;

                const float y = pts[i * 3 + 1];
                if (y <= config.floor_y) {
                    remove_flags[static_cast<size_t>(i)] = true;
                    floor_removed++;
                } else if (y > config.ceil_y && config.ceil_y > config.floor_y) {
                    remove_flags[static_cast<size_t>(i)] = true;
                    floor_removed++;
                }
            }

            LOG_INFO("[prune_by_geometric_dome] Floor pass (y <= {:.4f}m): {} splats removed ({:.1f}%)",
                     config.floor_y, floor_removed,
                     100.0f * static_cast<float>(floor_removed) / static_cast<float>(N));
        }

        // Pass 3: max scale filter
        // Removes Gaussians whose effective full size exceeds max_scale_in_meters.
        // get_scaling() already returns world-space axis scales.
        // We estimate full splat size as a 3-sigma diameter on the largest axis.
        if (config.max_scale_in_meters > 0.0f) {
            Tensor scales_cpu = model.get_scaling().cpu().contiguous();
            if (scales_cpu.is_valid() && scales_cpu.ndim() == 2 &&
                static_cast<int>(scales_cpu.shape()[0]) == N &&
                static_cast<int>(scales_cpu.shape()[1]) == 3) {

                const float* sc = scales_cpu.ptr<float>();

                int over_threshold_count = 0;
                float biggest_axis_found = 0.0f;

                const float max_scale = config.max_scale_in_meters / 2.0f;
                for (int i = 0; i < N; ++i) {
                    if (remove_flags[static_cast<size_t>(i)])
                        continue;

                    const float s0 = sc[i * 3 + 0];
                    const float s1 = sc[i * 3 + 1];
                    const float s2 = sc[i * 3 + 2];

                    if (!std::isfinite(s0) || !std::isfinite(s1) || !std::isfinite(s2)) {
                        continue;
                    }

                    const float max_axis_m = std::max({s0, s1, s2});
                    biggest_axis_found = std::max(biggest_axis_found, max_axis_m);

                    if (max_axis_m > max_scale) {
                        remove_flags[static_cast<size_t>(i)] = true;
                        scale_removed++;
                        over_threshold_count++;
                    }
                }

                LOG_INFO("[prune_by_geometric_dome] max_scale debug: threshold = {:.4f} m, biggest axis found = {:.6f} m, over threshold = {}",
                         config.max_scale_in_meters, biggest_axis_found, over_threshold_count);
            } else {
                LOG_WARN("[prune_by_geometric_dome] max_scale_in_meters filter: invalid scaling tensor - skipping");
            }

            LOG_INFO("[prune_by_geometric_dome] Max local-axis scale pass (> {:.4f} m): {} splats removed ({:.1f}%)",
                     config.max_scale_in_meters, scale_removed,
                     100.0f * static_cast<float>(scale_removed) / static_cast<float>(N));
        }

        // =========================================================================
        // Apply removal mask
        // =========================================================================
        const int n_remove = behind_removed + floor_removed + scale_removed;

        LOG_INFO("[prune_by_geometric_dome] Total to remove: {} / {} ({:.1f}%)",
                 n_remove, N,
                 100.0f * static_cast<float>(n_remove) / static_cast<float>(N));

        if (n_remove == 0) {
            return PruningResult{.splats_before = N, .splats_after = N, .splats_removed = 0, .success = true};
        }

        Tensor remove_mask_cpu = Tensor::zeros({static_cast<size_t>(N)},
                                               Device::CPU, DataType::Bool);
        auto rm_acc = make_acc<uint8_t, 1>(remove_mask_cpu);
        for (int i = 0; i < N; ++i) {
            rm_acc(i) = remove_flags[static_cast<size_t>(i)] ? 1 : 0;
        }

        Tensor remove_mask = remove_mask_cpu.to(Device::CUDA).contiguous();
        strategy.remove_gaussians(remove_mask);

        const int after = static_cast<int>(strategy.get_model().size());
        const int removed = N - after;
        const double pct = (N > 0) ? (100.0 * removed / N) : 0.0;

        LOG_INFO("[prune_by_geometric_dome] Requested: {}, actually removed: {} ({:.3f}%)",
                 n_remove, removed, pct);

        return PruningResult{.splats_before = N, .splats_after = after, .splats_removed = removed, .success = true};
    }

    // -----------------------------------------------------------------------------
    // Center-vote pruning (replicates trainer_old.cpp semantics)
    // -----------------------------------------------------------------------------

    std::expected<PruningResult, std::string> prune_by_center_vote(
        IStrategy& strategy,
        const CameraDataset& dataset,
        const CenterVotePruningConfig& config) {

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

        std::vector<int> pos(static_cast<size_t>(N), 0);
        std::vector<int> tot(static_cast<size_t>(N), 0);

        int skipped_no_mask = 0;
        int skipped_size_mismatch = 0;
        int skipped_proj_error = 0;

        const auto& cams = dataset.get_cameras();
        const int n_cams = static_cast<int>(cams.size());

        LOG_INFO("[prune_by_center_vote] Starting: {} splats, {} cameras", N, n_cams);

        const int num_threads = std::min(n_cams, omp_get_max_threads());
        LOG_INFO("[prune_by_center_vote] Using {} OpenMP threads for {} cameras", num_threads, n_cams);

        preload_masks_sequential(cams, *sizing, config.invert_masks);

#pragma omp parallel num_threads(num_threads)
        {
            std::vector<int> local_tot(static_cast<size_t>(N), 0);
            std::vector<int> local_pos(static_cast<size_t>(N), 0);

#pragma omp for schedule(dynamic, 1)
            for (int ci = 0; ci < n_cams; ++ci) {
                const auto& cam_ptr = cams[static_cast<size_t>(ci)];
                if (!cam_ptr) {
#pragma omp atomic
                    ++skipped_no_mask;
                    continue;
                }

                auto& cam = *cam_ptr;
                if (!cam.has_mask()) {
#pragma omp atomic
                    ++skipped_no_mask;
                    continue;
                }

                // Old code: attentionMask > 0.5 -> binary. Keep same threshold.
                Tensor mask = cam.load_and_get_mask(sizing->resize_factor, sizing->max_width, config.invert_masks, 0.5f);
                if (!mask.is_valid() || mask.numel() == 0) {
#pragma omp atomic
                    ++skipped_no_mask;
                    continue;
                }

                if (mask.ndim() == 3 && mask.shape()[0] == 1) {
                    mask = mask.squeeze(0);
                }

                const int H = static_cast<int>(cam.image_height());
                const int W = static_cast<int>(cam.image_width());
                if (mask.ndim() != 2 ||
                    static_cast<int>(mask.shape()[0]) != H ||
                    static_cast<int>(mask.shape()[1]) != W) {
#pragma omp atomic
                    ++skipped_size_mismatch;
                    continue;
                }

                auto proj = project_splats(cam, model, config);
                if (!proj) {
#pragma omp atomic
                    ++skipped_proj_error;
                    continue;
                }

                Tensor radii_cpu = proj->radii.cpu().contiguous();     // [N,2] int32
                Tensor means2d_cpu = proj->means2d.cpu().contiguous(); // [N,2] float
                Tensor mask_cpu = mask_to_float01(mask).cpu().contiguous(); // [H,W] float

                auto r_acc = make_acc<int32_t, 2>(radii_cpu);
                auto m_acc = make_acc<float, 2>(means2d_cpu);
                auto mask_acc = make_acc<float, 2>(mask_cpu);

                // -----------------------------------------------------------------
                // Border-tolerant mask queries:
                //   - strict  : exact binary mask
                //   - relaxed : dilated mask used as an "uncertain border band"
                //
                // Inside strict  -> positive vote
                // Inside relaxed -> ignore
                // Outside relaxed-> negative vote
                // -----------------------------------------------------------------
                constexpr int kMaskBorderTolPx = 2;

                std::vector<uint8_t> mask01(static_cast<size_t>(H) * static_cast<size_t>(W), 0);
                for (int yy = 0; yy < H; ++yy) {
                    uint8_t* row = mask01.data() + static_cast<size_t>(yy) * static_cast<size_t>(W);
                    for (int xx = 0; xx < W; ++xx) {
                        row[xx] = (mask_acc(yy, xx) >= 0.5f) ? uint8_t(1) : uint8_t(0);
                    }
                }

                IntegralImage sat;
                if (kMaskBorderTolPx > 0) {
                    sat.build(mask01.data(), H, W);
                }

                auto mask_inside_strict_xy = [&](int x, int y) -> bool {
                    if ((static_cast<unsigned>(x) >= static_cast<unsigned>(W)) ||
                        (static_cast<unsigned>(y) >= static_cast<unsigned>(H))) {
                        return false;
                    }
                    return mask01[static_cast<size_t>(y) * static_cast<size_t>(W) + static_cast<size_t>(x)] != 0;
                };

                auto mask_inside_relaxed_xy = [&](int x, int y) -> bool {
                    if ((static_cast<unsigned>(x) >= static_cast<unsigned>(W)) ||
                        (static_cast<unsigned>(y) >= static_cast<unsigned>(H))) {
                        return false;
                    }

                    if (kMaskBorderTolPx <= 0) {
                        return mask_inside_strict_xy(x, y);
                    }

                    const int r = kMaskBorderTolPx;
                    const int x0 = std::max(0, x - r);
                    const int y0 = std::max(0, y - r);
                    const int x1 = std::min(W - 1, x + r);
                    const int y1 = std::min(H - 1, y + r);

                    return sat.sum_rect(x0, y0, x1, y1) > 0;
                };

                // Helper for optional depth filtering:
                // only splats whose center is in the STRICT mask are candidates
                // for "inside-depth" statistics / vote removal.
                auto center_inside_mask_strict = [&](int i) -> bool {
                    const float mx = m_acc(i, 0);
                    const float my = m_acc(i, 1);
                    if (!std::isfinite(mx) || !std::isfinite(my)) {
                        return false;
                    }

                    const int x = round_to_int(mx);
                    const int y = round_to_int(my);
                    return mask_inside_strict_xy(x, y);
                };

                const int pad_w = static_cast<int>(W * config.border_safe_margin);
                const int pad_h = static_cast<int>(H * config.border_safe_margin);

                // Main voting loop (old-style: clamp to border, no relaxed band)
                for (int i = 0; i < N; ++i) {
                    const int rx = static_cast<int>(r_acc(i, 0));
                    const int ry = static_cast<int>(r_acc(i, 1));

                    if (!is_visible_radii_strict(rx, ry)) {
                        continue;
                    }

                    // Clamp to frame border: out-of-frame centers evaluate at border pixel
                    const int x = std::clamp(round_to_int(m_acc(i, 0)), 0, W - 1);
                    const int y = std::clamp(round_to_int(m_acc(i, 1)), 0, H - 1);

                    // Every visible splat votes, no relaxed band
                    local_tot[static_cast<size_t>(i)]++;
                    if (mask_inside_strict_xy(x, y)) {
                        local_pos[static_cast<size_t>(i)]++;
                    }
                }

                // ===== DEPTH FILTERING =====
                if (config.enable_depth_filtering) {
                    Tensor depths_cpu = proj->depths.cpu().contiguous(); // [N]
                    auto d_acc = make_acc<float, 1>(depths_cpu);

                    std::vector<std::pair<float, int>> inside;
                    inside.reserve(static_cast<size_t>(N) / 10);

                    for (int i = 0; i < N; ++i) {
                        const int rx = static_cast<int>(r_acc(i, 0));
                        const int ry = static_cast<int>(r_acc(i, 1));
                        if (!is_visible_radii_strict(rx, ry)) {
                            continue;
                        }

                        if (!center_inside_mask_strict(i)) {
                            continue;
                        }

                        const float depth = d_acc(i);
                        if (depth > config.near_plane && depth < config.far_plane) {
                            inside.emplace_back(depth, i);
                        }
                    }

                    if (inside.size() >= static_cast<size_t>(config.min_splats_for_depth_stats)) {
                        std::vector<float> depth_vals;
                        depth_vals.reserve(inside.size());
                        for (const auto& p : inside) {
                            depth_vals.push_back(p.first);
                        }

                        const size_t mid = depth_vals.size() / 2;
                        std::nth_element(depth_vals.begin(), depth_vals.begin() + mid, depth_vals.end());
                        const float median_depth = depth_vals[mid];

                        std::vector<float> abs_devs;
                        abs_devs.reserve(inside.size());
                        for (float d : depth_vals) {
                            abs_devs.push_back(std::abs(d - median_depth));
                        }

                        const size_t mid_dev = abs_devs.size() / 2;
                        std::nth_element(abs_devs.begin(), abs_devs.begin() + mid_dev, abs_devs.end());
                        const float mad = abs_devs[mid_dev];

                        const float robust_std = mad * 1.4826f;

                        if (robust_std > 1e-6f) {
                            const float K = config.depth_filter_sigma_multiplier;
                            const float min_acceptable_depth = median_depth - K * robust_std;
                            const float max_acceptable_depth = median_depth + K * robust_std;

                            if ((ci + 1) % 25 == 0 || ci < 3) {
                                LOG_DEBUG("[prune_by_center_vote] Camera {} depth filter: "
                                          "median={:.2f}m, MAD={:.3f}, std={:.3f}, range=[{:.2f}, {:.2f}]m (K={:.1f})",
                                          ci, median_depth, mad, robust_std,
                                          min_acceptable_depth, max_acceptable_depth, K);
                            }

                            int filtered_by_depth = 0;
                            for (const auto& [depth, idx] : inside) {
                                if (depth < min_acceptable_depth || depth > max_acceptable_depth) {
                                    int& p = local_pos[static_cast<size_t>(idx)];
                                    if (p > 0) {
                                        p--;
                                    }
                                    ++filtered_by_depth;
                                }
                            }

                            if (filtered_by_depth > 0 && ((ci + 1) % 25 == 0 || ci < 3)) {
                                LOG_INFO("[prune_by_center_vote] Camera {} filtered {} splats by depth ({:.1f}% of inside)",
                                         ci, filtered_by_depth, 100.0 * filtered_by_depth / inside.size());
                            }
                        }
                    }
                }
                // ===== END DEPTH FILTERING =====

                if ((ci + 1) % 25 == 0 || (ci + 1) == n_cams) {
                    LOG_INFO("[prune_by_center_vote] Processing camera {}/{}", (ci + 1), n_cams);
                }
            }

#pragma omp critical
            {
                for (int i = 0; i < N; ++i) {
                    tot[static_cast<size_t>(i)] += local_tot[static_cast<size_t>(i)];
                    pos[static_cast<size_t>(i)] += local_pos[static_cast<size_t>(i)];
                }
            }
        }

        const int usable_cameras = std::max(0, n_cams - skipped_no_mask - skipped_size_mismatch - skipped_proj_error);
        const int required_visibility = required_views_from_ratio(usable_cameras, config.min_visibility_ratio);

        LOG_INFO("[prune_by_center_vote] Cameras: {} total, {} usable, skipped: {} no mask, {} size mismatch, {} proj error",
                 n_cams, usable_cameras, skipped_no_mask, skipped_size_mismatch, skipped_proj_error);
        LOG_INFO("[prune_by_center_vote] Visibility requirement: >= {} / {} usable cameras ({:.1f}%)",
                 required_visibility, usable_cameras, 100.0f * config.min_visibility_ratio);

        long long sum_tot = 0;
        int count_zero_vis = 0;
        int count_low_vis = 0;
        int count_adequate_vis = 0;

        for (int i = 0; i < N; ++i) {
            const int t = tot[static_cast<size_t>(i)];
            sum_tot += static_cast<long long>(t);

            if (t == 0) {
                count_zero_vis++;
            } else if (t < required_visibility) {
                count_low_vis++;
            } else {
                count_adequate_vis++;
            }
        }

        LOG_INFO("[prune_by_center_vote] === VISIBILITY DISTRIBUTION ===");
        LOG_INFO("[prune_by_center_vote]   Zero visibility:     {:6d} ({:5.1f}%) <- NEVER SEEN, WILL BE REMOVED",
                 count_zero_vis, 100.0 * count_zero_vis / N);
        LOG_INFO("[prune_by_center_vote]   Low visibility:      {:6d} ({:5.1f}%) <- Seen in <{} / {} usable cameras ({:.1f}%), WILL BE REMOVED",
                 count_low_vis, 100.0 * count_low_vis / N, required_visibility, usable_cameras, 100.0f * config.min_visibility_ratio);
        LOG_INFO("[prune_by_center_vote]   Adequate visibility: {:6d} ({:5.1f}%) <- Evaluated by ratio test",
                 count_adequate_vis, 100.0 * count_adequate_vis / N);

        if (sum_tot == 0) {
            LOG_WARN("[prune_by_center_vote] No visibility accumulated; skipping prune.");
            LOG_INFO("[prune_by_center_vote] Removed 0 splats (0.000%)");
            return PruningResult{.splats_before = N, .splats_after = N, .splats_removed = 0, .success = true};
        }

        bool any_meets = false;
        for (int i = 0; i < N; ++i) {
            if (tot[static_cast<size_t>(i)] >= required_visibility) {
                any_meets = true;
                break;
            }
        }

        if (!any_meets) {
            LOG_WARN("[prune_by_center_vote] No splats reached required visibility {} / {} ({:.1f}%); ALL WOULD BE REMOVED, skipping prune.",
                     required_visibility, usable_cameras, 100.0f * config.min_visibility_ratio);
            LOG_INFO("[prune_by_center_vote] Removed 0 splats (0.000%)");
            return PruningResult{.splats_before = N, .splats_after = N, .splats_removed = 0, .success = true};
        }

        Tensor remove_mask_cpu = Tensor::zeros({static_cast<size_t>(N)}, Device::CPU, DataType::Bool);
        auto rm_acc = make_acc<uint8_t, 1>(remove_mask_cpu);

        int keep_cnt = 0;
        int n_remove = 0;
        int removed_zero_vis = 0;
        int removed_low_vis = 0;
        int removed_bad_ratio = 0;
        int adequate_kept = 0;
        int adequate_removed = 0;

        for (int i = 0; i < N; ++i) {
            const int t = tot[static_cast<size_t>(i)];
            const int p = pos[static_cast<size_t>(i)];
            const float ratio = static_cast<float>(p) / static_cast<float>(std::max(1, t));

            const bool keep = (t >= required_visibility) && (ratio >= config.vote_ratio_threshold);

            if (keep) {
                ++keep_cnt;
                rm_acc(i) = 0;
                if (t >= required_visibility) {
                    adequate_kept++;
                }
            } else {
                ++n_remove;
                rm_acc(i) = 1;

                if (t == 0) {
                    removed_zero_vis++;
                } else if (t < required_visibility) {
                    removed_low_vis++;
                } else {
                    removed_bad_ratio++;
                    adequate_removed++;
                }
            }
        }

        LOG_INFO("[prune_by_center_vote] === REMOVAL BREAKDOWN ===");
        LOG_INFO("[prune_by_center_vote]   Removed (zero_vis):     {:6d} <- Never visible in any masked camera",
                 removed_zero_vis);
        LOG_INFO("[prune_by_center_vote]   Removed (low_vis):      {:6d} <- Visible in <{} / {} usable cameras ({:.1f}%)",
                 removed_low_vis, required_visibility, usable_cameras, 100.0f * config.min_visibility_ratio);
        LOG_INFO("[prune_by_center_vote]   Removed (bad_ratio):    {:6d} <- {:.1f}% of valid votes were outside relaxed mask band (threshold={:.1f}%)",
                 removed_bad_ratio, 100.0 * (1.0 - config.vote_ratio_threshold), 100.0 * config.vote_ratio_threshold);
        LOG_INFO("[prune_by_center_vote]   TOTAL TO REMOVE:        {:6d} ({:5.1f}%)",
                 n_remove, 100.0 * n_remove / N);
        LOG_INFO("[prune_by_center_vote]   Adequate vis kept:      {:6d} / {} adequate splats",
                 adequate_kept, adequate_kept + adequate_removed);

        if (keep_cnt == 0) {
            LOG_WARN("[prune_by_center_vote] keep_mask would be empty (thr={}, min_vis_ratio={:.3f} -> {} / {} usable cameras); skipping prune.",
                     config.vote_ratio_threshold, config.min_visibility_ratio, required_visibility, usable_cameras);
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

        return PruningResult{
            .splats_before = N,
            .splats_after = after,
            .splats_removed = removed,
            .success = true};
    }

    // -----------------------------------------------------------------------------
    // Leakage pruning (replicates trainer_old.cpp semantics)
    // -----------------------------------------------------------------------------

    /* std::expected<PruningResult, std::string> prune_by_mask_leakage(
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

        const int num_threads_leak = std::min(n_cams, omp_get_max_threads());
        LOG_INFO("[prune_by_mask_leakage] Using {} OpenMP threads for {} cameras", num_threads_leak, n_cams);

        #pragma omp parallel num_threads(num_threads_leak)
        {
            // Thread-local counters
            std::vector<int> local_vis_counts(static_cast<size_t>(N), 0);
            std::vector<int> local_leak_votes(static_cast<size_t>(N), 0);

            // Thread-local buffers for mask processing (reused per camera by this thread)
            std::vector<uint8_t> local_mask01;
            IntegralImage local_sat;

            #pragma omp for schedule(dynamic, 1)
            for (int ci = 0; ci < n_cams; ++ci) {
                const auto& cam_ptr = cams[static_cast<size_t>(ci)];
                if (!cam_ptr) {
                    #pragma omp atomic
                    ++skipped_no_mask;
                    continue;
                }

                auto& cam = *cam_ptr;
                if (!cam.has_mask()) {
                    #pragma omp atomic
                    ++skipped_no_mask;
                    continue;
                }

                // Old code: (attentionMask > 0.5).to(float). Mirror it with threshold.
                Tensor mask = cam.load_and_get_mask(sizing->resize_factor, sizing->max_width, config.invert_masks, 0.5f);
                if (!mask.is_valid() || mask.numel() == 0) {
                    #pragma omp atomic
                    ++skipped_no_mask;
                    continue;
                }
                if (mask.ndim() == 3 && mask.shape()[0] == 1) {
                    mask = mask.squeeze(0);
                }

                const int H = static_cast<int>(cam.image_height());
                const int W = static_cast<int>(cam.image_width());
                if (mask.ndim() != 2 ||
                    static_cast<int>(mask.shape()[0]) != H ||
                    static_cast<int>(mask.shape()[1]) != W) {
                    #pragma omp atomic
                    ++skipped_size_mismatch;
                    continue;
                }

                // Prepare CPU binary mask01 - usando buffer local del thread
                Tensor mask_cpu = mask.cpu().contiguous();
                auto mask_acc = make_acc<float, 2>(mask_cpu);

                local_mask01.assign(static_cast<size_t>(H) * static_cast<size_t>(W), 0);
                for (int y = 0; y < H; ++y) {
                    uint8_t* row = local_mask01.data() + static_cast<size_t>(y) * static_cast<size_t>(W);
                    for (int x = 0; x < W; ++x) {
                        row[x] = (mask_acc(y, x) >= 0.5f) ? uint8_t(1) : uint8_t(0);
                    }
                }

                // Optional dilation tolerance (old uses max_pool2d over mask01).
                // We emulate it via integral-image queries - usando SAT local del thread
                if (config.dilate_px > 0) {
                    local_sat.build(local_mask01.data(), H, W);
                }

                // Strict mask query: out-of-frame => false (counts as "outside mask")
                auto mask_inside_strict = [&](int x, int y) -> bool {
                    if ((static_cast<unsigned>(x) >= static_cast<unsigned>(W)) ||
                        (static_cast<unsigned>(y) >= static_cast<unsigned>(H))) {
                        return false;
                    }

                    if (config.dilate_px <= 0) {
                        return local_mask01[static_cast<size_t>(y) * static_cast<size_t>(W) + static_cast<size_t>(x)] != 0;
                    }

                    const int r = config.dilate_px;
                    const int x0 = std::max(0, x - r);
                    const int y0 = std::max(0, y - r);
                    const int x1 = std::min(W - 1, x + r);
                    const int y1 = std::min(H - 1, y + r);

                    const int sum = local_sat.sum_rect(x0, y0, x1, y1);
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
                    #pragma omp atomic
                    ++skipped_proj_error;
                    continue;
                }

                Tensor radii_cpu = proj->radii.cpu().contiguous();
                Tensor means2d_cpu = proj->means2d.cpu().contiguous();

                auto r_acc = make_acc<int32_t, 2>(radii_cpu);
                auto m_acc = make_acc<float, 2>(means2d_cpu);

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

                    const float mx = m_acc(i, 0);
                    const float my = m_acc(i, 1);
                    if (!std::isfinite(mx) || !std::isfinite(my)) {
                        continue;
                    }

                    const int cx = round_to_int(mx);
                    const int cy = round_to_int(my);

                    // Old semantics: only evaluate leakage if center is inside mask.
                    // Important change: out-of-frame center is NOT clamped; it's treated as outside mask.
                    if (!mask_inside_strict(cx, cy)) {
                        continue;
                    }

                    local_vis_counts[static_cast<size_t>(i)]++;

                    int outside = 0;
                    const int P = static_cast<int>(dirs.size());

                    for (const auto& d : dirs) {
                        const int dx = round_to_int(frx * d.dx);
                        const int dy = round_to_int(fry * d.dy);

                        const int sx = cx + dx;
                        const int sy = cy + dy;

                        // Important change: sample points out-of-frame count as outside (no clamp).
                        if (!mask_inside_strict(sx, sy)) {
                            outside++;
                        }
                    }

                    const float outside_ratio = (P > 0) ? (static_cast<float>(outside) / static_cast<float>(P)) : 0.0f;
                    if (outside_ratio > config.per_view_leak_fraction) {
                        local_leak_votes[static_cast<size_t>(i)]++;
                    }
                }

                if ((ci + 1) % 25 == 0 || (ci + 1) == n_cams) {
                    LOG_INFO("[prune_by_mask_leakage] Processing camera {}/{}", (ci + 1), n_cams);
                }
            }

            // Merge thread-local results into global counters
            #pragma omp critical
            {
                for (int i = 0; i < N; ++i) {
                    vis_counts[static_cast<size_t>(i)] += local_vis_counts[static_cast<size_t>(i)];
                    leak_votes[static_cast<size_t>(i)] += local_leak_votes[static_cast<size_t>(i)];
                }
            }
        }

        const int usable_cameras = std::max(0, n_cams - skipped_no_mask - skipped_size_mismatch - skipped_proj_error);
        const int required_decisive_views = required_views_from_ratio(usable_cameras, config.min_visibility_ratio);

        LOG_INFO("[prune_by_mask_leakage] Cameras: {} total, {} usable, skipped: {} no mask, {} size mismatch, {} proj error",
                 n_cams, usable_cameras, skipped_no_mask, skipped_size_mismatch, skipped_proj_error);
        LOG_INFO("[prune_by_mask_leakage] Decisive evidence requirement: >= {} / {} usable cameras ({:.1f}%)",
                 required_decisive_views, usable_cameras, 100.0f * config.min_visibility_ratio);

        // =========================================================================
        // DIAGNOSTIC LOGGING - Analyze evaluation distribution
        // =========================================================================
        long long sum_vis = 0;
        int count_zero_eval = 0;     // Never evaluated (center outside OR radius too small)
        int count_low_eval = 0;      // Evaluated but < min_visibility_count
        int count_adequate_eval = 0; // >= min_visibility_count

        for (int i = 0; i < N; ++i) {
            const int v = vis_counts[static_cast<size_t>(i)];
            sum_vis += static_cast<long long>(v);

            if (v == 0) {
                count_zero_eval++;
            } else if (v < config.min_visibility_count) {
                count_low_eval++;
            } else {
                count_adequate_eval++;
            }
        }

        LOG_INFO("[prune_by_mask_leakage] === EVALUATION DISTRIBUTION ===");
        LOG_INFO("[prune_by_mask_leakage]   Zero evaluation:     {:6d} ({:5.1f}%) <- CENTER OUTSIDE MASK OR RADIUS<{:.0f}px",
                 count_zero_eval, 100.0 * count_zero_eval / N, config.min_pixel_radius);
        LOG_INFO("[prune_by_mask_leakage]   Low evaluation:      {:6d} ({:5.1f}%) <- Evaluated in <{} views",
                 count_low_eval, 100.0 * count_low_eval / N, config.min_visibility_count);
        LOG_INFO("[prune_by_mask_leakage]   Adequate evaluation: {:6d} ({:5.1f}%) <- Evaluated by leakage test",
                 count_adequate_eval, 100.0 * count_adequate_eval / N);

        if (sum_vis == 0) {
            LOG_WARN("[prune_by_mask_leakage] No candidate visibility accumulated; skipping prune.");
            LOG_INFO("[prune_by_mask_leakage] Removed 0 splats (0.000%)");
            return PruningResult{.splats_before = N, .splats_after = N, .splats_removed = 0, .success = true};
        }

        // =========================================================================
        // BUILD REMOVAL MASK with detailed tracking
        // =========================================================================
        Tensor remove_mask_cpu = Tensor::zeros({static_cast<size_t>(N)}, Device::CPU, DataType::Bool);
        auto rm_acc = make_acc<uint8_t, 1>(remove_mask_cpu);

        int keep_cnt = 0;
        int n_remove = 0;

        // Track decision reasons
        int kept_zero_eval = 0; // Never evaluated by leakage (v==0)
        int kept_low_eval = 0;
        int removed_by_leakage = 0;
        int kept_by_leakage = 0;

        for (int i = 0; i < N; ++i) {
            const int v = vis_counts[static_cast<size_t>(i)];

            // Never evaluated = keep (same as trainer_old)
            if (v == 0) {
                rm_acc(i) = 0;
                ++keep_cnt;
                ++kept_zero_eval;
                continue;
            }

            // Not enough evaluated views -> keep
            if (v < config.min_visibility_count) {
                rm_acc(i) = 0;
                ++keep_cnt;
                ++kept_low_eval;
                continue;
            }

            // Enough evaluated views: apply leakage test
            const int leaks = leak_votes[static_cast<size_t>(i)];
            const float fail_r = static_cast<float>(leaks) / static_cast<float>(v);
            const float ok_r = 1.0f - fail_r;

            const bool remove = (ok_r < config.leak_keep_threshold);
            if (remove) {
                rm_acc(i) = 1;
                ++n_remove;
                ++removed_by_leakage;
            } else {
                rm_acc(i) = 0;
                ++keep_cnt;
                ++kept_by_leakage;
            }
        }

        LOG_INFO("[prune_by_mask_leakage] === DECISION BREAKDOWN ===");
        LOG_INFO("[prune_by_mask_leakage]   KEPT (zero_eval):       {:6d} <- Not evaluated by leakage (v==0)",
                 kept_zero_eval);
        LOG_INFO("[prune_by_mask_leakage]   KEPT (low_eval):        {:6d} <- Below min_visibility_count (min={})",
                 kept_low_eval, config.min_visibility_count);
        LOG_INFO("[prune_by_mask_leakage]   Removed (by_leakage):   {:6d} <- Failed leakage test (threshold={:.0f}%)",
                 removed_by_leakage, 100.0 * config.leak_keep_threshold);
        LOG_INFO("[prune_by_mask_leakage]   Kept (passed_leakage):  {:6d} <- Passed leakage test",
                 kept_by_leakage);
        LOG_INFO("[prune_by_mask_leakage]   TOTAL TO REMOVE:        {:6d} ({:5.1f}%)",
                 n_remove, 100.0 * n_remove / N);

        if (kept_zero_eval > 0) {
            LOG_INFO("[prune_by_mask_leakage]   Note: {:d} splats were never evaluated by leakage (v==0). "
                     "This is expected when the splat center is never inside the mask or is too small in all views; "
                     "those splats must be handled by other pruning passes (e.g., center vote).",
                     kept_zero_eval);
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
    }*/

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

        // Strong multiview evidence buckets
        std::vector<int> good_votes(static_cast<size_t>(N), 0);      // strong positive views
        std::vector<int> bad_votes(static_cast<size_t>(N), 0);       // strong negative views
        std::vector<int> decisive_counts(static_cast<size_t>(N), 0); // good + bad
        std::vector<int> uncertain_counts(static_cast<size_t>(N), 0);

        int skipped_no_mask = 0;
        int skipped_size_mismatch = 0;
        int skipped_proj_error = 0;

        const auto& cams = dataset.get_cameras();
        const int n_cams = static_cast<int>(cams.size());

        LOG_INFO("[prune_by_mask_leakage] Starting: {} splats, {} cameras", N, n_cams);

        const int num_threads_leak = std::min(n_cams, omp_get_max_threads());
        LOG_INFO("[prune_by_mask_leakage] Using {} OpenMP threads for {} cameras", num_threads_leak, n_cams);

        preload_masks_sequential(cams, *sizing, config.invert_masks);

#pragma omp parallel num_threads(num_threads_leak)
        {
            std::vector<int> local_good_votes(static_cast<size_t>(N), 0);
            std::vector<int> local_bad_votes(static_cast<size_t>(N), 0);
            std::vector<int> local_decisive_counts(static_cast<size_t>(N), 0);
            std::vector<int> local_uncertain_counts(static_cast<size_t>(N), 0);

            std::vector<uint8_t> local_mask01;
            IntegralImage local_sat;

#pragma omp for schedule(dynamic, 1)
            for (int ci = 0; ci < n_cams; ++ci) {
                const auto& cam_ptr = cams[static_cast<size_t>(ci)];
                if (!cam_ptr) {
#pragma omp atomic
                    ++skipped_no_mask;
                    continue;
                }

                auto& cam = *cam_ptr;
                if (!cam.has_mask()) {
#pragma omp atomic
                    ++skipped_no_mask;
                    continue;
                }

                Tensor mask = cam.load_and_get_mask(sizing->resize_factor, sizing->max_width, config.invert_masks, 0.5f);
                if (!mask.is_valid() || mask.numel() == 0) {
#pragma omp atomic
                    ++skipped_no_mask;
                    continue;
                }
                if (mask.ndim() == 3 && mask.shape()[0] == 1) {
                    mask = mask.squeeze(0);
                }

                const int H = static_cast<int>(cam.image_height());
                const int W = static_cast<int>(cam.image_width());
                if (mask.ndim() != 2 ||
                    static_cast<int>(mask.shape()[0]) != H ||
                    static_cast<int>(mask.shape()[1]) != W) {
#pragma omp atomic
                    ++skipped_size_mismatch;
                    continue;
                }

                Tensor mask_cpu = mask_to_float01(mask).cpu().contiguous();
                auto mask_acc = make_acc<float, 2>(mask_cpu);

                // Exact binary mask
                local_mask01.assign(static_cast<size_t>(H) * static_cast<size_t>(W), 0);
                for (int y = 0; y < H; ++y) {
                    uint8_t* row = local_mask01.data() + static_cast<size_t>(y) * static_cast<size_t>(W);
                    for (int x = 0; x < W; ++x) {
                        row[x] = (mask_acc(y, x) >= 0.5f) ? uint8_t(1) : uint8_t(0);
                    }
                }

                // Relaxed mask = dilated query band, used as "uncertain border"
                if (config.dilate_px > 0) {
                    local_sat.build(local_mask01.data(), H, W);
                }

                auto mask_inside_strict = [&](int x, int y) -> bool {
                    if ((static_cast<unsigned>(x) >= static_cast<unsigned>(W)) ||
                        (static_cast<unsigned>(y) >= static_cast<unsigned>(H))) {
                        return false;
                    }
                    return local_mask01[static_cast<size_t>(y) * static_cast<size_t>(W) + static_cast<size_t>(x)] != 0;
                };

                auto mask_inside_relaxed = [&](int x, int y) -> bool {
                    if ((static_cast<unsigned>(x) >= static_cast<unsigned>(W)) ||
                        (static_cast<unsigned>(y) >= static_cast<unsigned>(H))) {
                        return false;
                    }

                    if (config.dilate_px <= 0) {
                        return mask_inside_strict(x, y);
                    }

                    const int r = config.dilate_px;
                    const int x0 = std::max(0, x - r);
                    const int y0 = std::max(0, y - r);
                    const int x1 = std::min(W - 1, x + r);
                    const int y1 = std::min(H - 1, y + r);

                    return local_sat.sum_rect(x0, y0, x1, y1) > 0;
                };

                CenterVotePruningConfig proj_cfg;
                proj_cfg.eps2d = config.eps2d;
                proj_cfg.near_plane = config.near_plane;
                proj_cfg.far_plane = config.far_plane;
                proj_cfg.radius_clip = config.radius_clip;
                proj_cfg.scaling_modifier = config.scaling_modifier;
                proj_cfg.invert_masks = config.invert_masks;

                auto proj = project_splats(cam, model, proj_cfg);
                if (!proj) {
#pragma omp atomic
                    ++skipped_proj_error;
                    continue;
                }

                Tensor radii_cpu = proj->radii.cpu().contiguous();
                Tensor means2d_cpu = proj->means2d.cpu().contiguous();

                auto r_acc = make_acc<int32_t, 2>(radii_cpu);
                auto m_acc = make_acc<float, 2>(means2d_cpu);

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

                    const float mx = m_acc(i, 0);
                    const float my = m_acc(i, 1);
                    if (!std::isfinite(mx) || !std::isfinite(my)) {
                        continue;
                    }

                    const int cx = round_to_int(mx);
                    const int cy = round_to_int(my);

                    const bool center_strict = mask_inside_strict(cx, cy);

                    // 4 cardinal extreme points
                    const int px = round_to_int(frx);
                    const int py = round_to_int(fry);

                    const int xR = cx + px;
                    const int xL = cx - px;
                    const int yU = cy - py;
                    const int yD = cy + py;

                    const bool cardR_strict = mask_inside_strict(xR, cy);
                    const bool cardL_strict = mask_inside_strict(xL, cy);
                    const bool cardU_strict = mask_inside_strict(cx, yU);
                    const bool cardD_strict = mask_inside_strict(cx, yD);

                    const int strict_cardinals =
                        static_cast<int>(cardR_strict) +
                        static_cast<int>(cardL_strict) +
                        static_cast<int>(cardU_strict) +
                        static_cast<int>(cardD_strict);

                    // -----------------------------------------------------------------
                    // STRONG GOOD fast-path:
                    // center + 4 cardinals all strictly inside -> very strong evidence
                    // -----------------------------------------------------------------
                    if (center_strict && strict_cardinals == 4) {
                        local_good_votes[static_cast<size_t>(i)]++;
                        local_decisive_counts[static_cast<size_t>(i)]++;
                        continue;
                    }

                    // -----------------------------------------------------------------
                    // Candidate gate: no relaxed band, strict mask only
                    // -----------------------------------------------------------------
                    const bool near_subject = center_strict || (strict_cardinals > 0);
                    if (!near_subject) {
                        continue;
                    }

                    // -----------------------------------------------------------------
                    // Footprint inspection: strict mask only, out-of-frame = outside
                    // -----------------------------------------------------------------
                    int strict_inside_samples = 0;
                    int clear_outside_samples = 0;

                    const int P = static_cast<int>(dirs.size());
                    for (const auto& d : dirs) {
                        const int dx = round_to_int(frx * d.dx);
                        const int dy = round_to_int(fry * d.dy);

                        const int sx = cx + dx;
                        const int sy = cy + dy;

                        // Out-of-frame = outside (penalize like old behavior)
                        if ((static_cast<unsigned>(sx) >= static_cast<unsigned>(W)) ||
                            (static_cast<unsigned>(sy) >= static_cast<unsigned>(H))) {
                            clear_outside_samples++;
                            continue;
                        }

                        // Strict mask only, no relaxed band
                        if (mask_inside_strict(sx, sy)) {
                            strict_inside_samples++;
                        } else {
                            clear_outside_samples++;
                        }
                    }

                    const float outside_ratio =
                        static_cast<float>(clear_outside_samples) / static_cast<float>(P);

                    const float inside_ratio =
                        static_cast<float>(strict_inside_samples) / static_cast<float>(P);

                    // -----------------------------------------------------------------
                    // STRONG BAD: no relaxed band, simple leakage check
                    // -----------------------------------------------------------------
                    const bool strong_bad = (outside_ratio >= config.per_view_leak_fraction) &&
                        !(center_strict && inside_ratio >= 0.5f);

                    if (strong_bad) {
                        local_bad_votes[static_cast<size_t>(i)]++;
                        local_decisive_counts[static_cast<size_t>(i)]++;
                    } else {
                        local_uncertain_counts[static_cast<size_t>(i)]++;
                    }
                }

                if ((ci + 1) % 25 == 0 || (ci + 1) == n_cams) {
                    LOG_INFO("[prune_by_mask_leakage] Processing camera {}/{}", (ci + 1), n_cams);
                }
            }

#pragma omp critical
            {
                for (int i = 0; i < N; ++i) {
                    good_votes[static_cast<size_t>(i)] += local_good_votes[static_cast<size_t>(i)];
                    bad_votes[static_cast<size_t>(i)] += local_bad_votes[static_cast<size_t>(i)];
                    decisive_counts[static_cast<size_t>(i)] += local_decisive_counts[static_cast<size_t>(i)];
                    uncertain_counts[static_cast<size_t>(i)] += local_uncertain_counts[static_cast<size_t>(i)];
                }
            }
        }

        const int usable_cameras = std::max(0, n_cams - skipped_no_mask - skipped_size_mismatch - skipped_proj_error);
        const int required_decisive_views = required_views_from_ratio(usable_cameras, config.min_visibility_ratio);

        LOG_INFO("[prune_by_mask_leakage] Cameras: {} total, {} usable, skipped: {} no mask, {} size mismatch, {} proj error",
                 n_cams, usable_cameras, skipped_no_mask, skipped_size_mismatch, skipped_proj_error);
        LOG_INFO("[prune_by_mask_leakage] Decisive evidence requirement: >= {} / {} usable cameras ({:.1f}%)",
                 required_decisive_views, usable_cameras, 100.0f * config.min_visibility_ratio);

        // =========================================================================
        // DIAGNOSTIC LOGGING
        // =========================================================================
        long long sum_decisive = 0;
        int count_zero_decisive = 0;
        int count_low_decisive = 0;
        int count_adequate_decisive = 0;

        for (int i = 0; i < N; ++i) {
            const int d = decisive_counts[static_cast<size_t>(i)];
            sum_decisive += static_cast<long long>(d);

            if (d == 0) {
                count_zero_decisive++;
            } else if (d < required_decisive_views) {
                count_low_decisive++;
            } else {
                count_adequate_decisive++;
            }
        }

        LOG_INFO("[prune_by_mask_leakage] === DECISIVE EVIDENCE DISTRIBUTION ===");
        LOG_INFO("[prune_by_mask_leakage]   Zero decisive:       {:6d} ({:5.1f}%) <- No strong good/bad evidence",
                 count_zero_decisive, 100.0 * count_zero_decisive / N);
        LOG_INFO("[prune_by_mask_leakage]   Low decisive:        {:6d} ({:5.1f}%) <- Strong evidence in <{} / {} usable cameras ({:.1f}%)",
                 count_low_decisive, 100.0 * count_low_decisive / N, required_decisive_views, usable_cameras, 100.0f * config.min_visibility_ratio);
        LOG_INFO("[prune_by_mask_leakage]   Adequate decisive:   {:6d} ({:5.1f}%) <- Enough decisive views",
                 count_adequate_decisive, 100.0 * count_adequate_decisive / N);

        if (sum_decisive == 0) {
            LOG_WARN("[prune_by_mask_leakage] No decisive evidence accumulated; skipping prune.");
            LOG_INFO("[prune_by_mask_leakage] Removed 0 splats (0.000%)");
            return PruningResult{.splats_before = N, .splats_after = N, .splats_removed = 0, .success = true};
        }

        // =========================================================================
        // BUILD REMOVAL MASK
        // =========================================================================
        Tensor remove_mask_cpu = Tensor::zeros({static_cast<size_t>(N)}, Device::CPU, DataType::Bool);
        auto rm_acc = make_acc<uint8_t, 1>(remove_mask_cpu);

        int keep_cnt = 0;
        int n_remove = 0;

        int kept_zero_decisive = 0;
        int kept_low_decisive = 0;
        int kept_low_bad = 0;
        int kept_positive_balance = 0;
        int removed_by_negative_evidence = 0;

        constexpr float kBadVoteWeight = 2.0f;
        const int min_bad_votes = std::max(1, (required_decisive_views + 1) / 2);

        for (int i = 0; i < N; ++i) {
            const int good = good_votes[static_cast<size_t>(i)];
            const int bad = bad_votes[static_cast<size_t>(i)];
            const int dec = decisive_counts[static_cast<size_t>(i)];

            // Never decisively evaluated -> keep
            if (dec == 0) {
                rm_acc(i) = 0;
                ++keep_cnt;
                ++kept_zero_decisive;
                continue;
            }

            if (dec < required_decisive_views) {
                rm_acc(i) = 0;
                ++keep_cnt;
                ++kept_low_decisive;
                continue;
            }

            // Negative evidence needs fewer votes than positive evidence,
            // but still must be multiview, not single-view.
            if (bad < min_bad_votes) {
                rm_acc(i) = 0;
                ++keep_cnt;
                ++kept_low_bad;
                continue;
            }

            // Weighted vote:
            // bad evidence counts more than good evidence.
            const float weighted_keep_ratio =
                static_cast<float>(good) /
                std::max(1.0f, static_cast<float>(good) + kBadVoteWeight * static_cast<float>(bad));

            const bool remove = (weighted_keep_ratio < config.leak_keep_threshold);

            if (remove) {
                rm_acc(i) = 1;
                ++n_remove;
                ++removed_by_negative_evidence;
            } else {
                rm_acc(i) = 0;
                ++keep_cnt;
                ++kept_positive_balance;
            }
        }

        LOG_INFO("[prune_by_mask_leakage] === DECISION BREAKDOWN ===");
        LOG_INFO("[prune_by_mask_leakage]   KEPT (zero_decisive):     {:6d} <- No strong multiview evidence",
                 kept_zero_decisive);
        LOG_INFO("[prune_by_mask_leakage]   KEPT (low_decisive):      {:6d} <- Decisive views < {} / {} usable cameras ({:.1f}%)",
                 kept_low_decisive, required_decisive_views, usable_cameras, 100.0f * config.min_visibility_ratio);
        LOG_INFO("[prune_by_mask_leakage]   KEPT (low_bad_votes):     {:6d} <- Bad votes < min_bad_votes ({})",
                 kept_low_bad, min_bad_votes);
        LOG_INFO("[prune_by_mask_leakage]   KEPT (positive_balance):  {:6d} <- Good evidence compensated bad evidence",
                 kept_positive_balance);
        LOG_INFO("[prune_by_mask_leakage]   Removed (neg_evidence):   {:6d} <- Weighted bad evidence won (bad weight={:.1f}, keep_thr={:.0f}%)",
                 removed_by_negative_evidence, kBadVoteWeight, 100.0 * config.leak_keep_threshold);
        LOG_INFO("[prune_by_mask_leakage]   TOTAL TO REMOVE:          {:6d} ({:5.1f}%)",
                 n_remove, 100.0 * n_remove / N);

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

        return PruningResult{
            .splats_before = N,
            .splats_after = after,
            .splats_removed = removed,
            .success = true};
    }

    // -----------------------------------------------------------------------------
    // Alpha consensus accurate mass-weighted approach
    // -----------------------------------------------------------------------------
    std::expected<PruningResult, std::string> prune_by_alpha_consensus(
        IStrategy& strategy,
        const CameraDataset& dataset,
        const AlphaConsensusPruningConfig& config) {

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

        // Per-splat accumulators: sum of (weighted_inside / weighted_total) per camera
        std::vector<double> consensus_sum(static_cast<size_t>(N), 0.0);
        std::vector<int> consensus_count(static_cast<size_t>(N), 0);

        int skipped_no_mask = 0;
        int skipped_size_mismatch = 0;
        int skipped_proj_error = 0;

        const auto& cams = dataset.get_cameras();
        const int n_cams = static_cast<int>(cams.size());

        LOG_INFO("[prune_by_alpha_consensus] Starting: {} splats, {} cameras", N, n_cams);

        // Build a CenterVotePruningConfig just to call project_splats (reuses projection path)
        CenterVotePruningConfig proj_cfg;
        proj_cfg.eps2d = config.eps2d;
        proj_cfg.near_plane = config.near_plane;
        proj_cfg.far_plane = config.far_plane;
        proj_cfg.radius_clip = config.radius_clip;
        proj_cfg.scaling_modifier = config.scaling_modifier;

        const int num_threads = std::min(n_cams, omp_get_max_threads());
        LOG_INFO("[prune_by_alpha_consensus] Using {} OpenMP threads", num_threads);

        preload_masks_sequential(cams, *sizing, config.invert_masks);

        #pragma omp parallel num_threads(num_threads)
        {
            std::vector<double> local_sum(static_cast<size_t>(N), 0.0);
            std::vector<int> local_count(static_cast<size_t>(N), 0);

            #pragma omp for schedule(dynamic, 1)
            for (int ci = 0; ci < n_cams; ++ci) {
                const auto& cam_ptr = cams[static_cast<size_t>(ci)];
                if (!cam_ptr || !cam_ptr->has_mask()) {
                    #pragma omp atomic
                    skipped_no_mask++;
                    continue;
                }

                auto& cam = *cam_ptr;

                Tensor mask = cam.load_and_get_mask(
                    sizing->resize_factor, sizing->max_width, config.invert_masks, 0.5f);
                if (!mask.is_valid() || mask.numel() == 0) {
                    #pragma omp atomic
                    skipped_no_mask++;
                    continue;
                }
                if (mask.ndim() == 3 && mask.shape()[0] == 1) {
                    mask = mask.squeeze(0);
                }

                const int H = static_cast<int>(cam.image_height());
                const int W = static_cast<int>(cam.image_width());
                if (mask.ndim() != 2 ||
                    static_cast<int>(mask.shape()[0]) != H ||
                    static_cast<int>(mask.shape()[1]) != W) {
                    #pragma omp atomic
                    skipped_size_mismatch++;
                    continue;
                }

                auto proj = project_splats(cam, model, proj_cfg);
                if (!proj) {
                #pragma omp atomic
                    skipped_proj_error++;
                    continue;
                }

                Tensor radii_cpu = proj->radii.cpu().contiguous();     // [N, 2] int32
                Tensor means2d_cpu = proj->means2d.cpu().contiguous(); // [N, 2] float
                Tensor conics_cpu = proj->conics.cpu().contiguous();   // [N, 3] float
                Tensor mask_cpu = mask_to_float01(mask).cpu().contiguous(); // [H, W] float

                auto r_acc = make_acc<int32_t, 2>(radii_cpu);
                auto m_acc = make_acc<float, 2>(means2d_cpu);
                auto con_acc = make_acc<float, 2>(conics_cpu);
                auto mask_acc = make_acc<float, 2>(mask_cpu);

                const int half_grid = config.sample_grid / 2;

                for (int i = 0; i < N; ++i) {
                    const int rx = static_cast<int>(r_acc(i, 0));
                    const int ry = static_cast<int>(r_acc(i, 1));

                    // Skip invisible splats (behind camera or culled)
                    if (!is_visible_radii_strict(rx, ry))
                        continue;

                    const float cx = m_acc(i, 0); // projected center X
                    const float cy = m_acc(i, 1); // projected center Y

                    // Conic coefficients: Gaussian value = exp(-0.5*(a*dx^2 + 2*b*dx*dy + c*dy^2))
                    const float a = con_acc(i, 0);
                    const float b = con_acc(i, 1);
                    const float c = con_acc(i, 2);

                    // Sample NxN grid over the projected ellipse bounding box [rx x ry]
                    double weighted_inside = 0.0;
                    double weighted_total = 0.0;

                    for (int gy = -half_grid; gy <= half_grid; ++gy) {
                        for (int gx = -half_grid; gx <= half_grid; ++gx) {
                            // Map grid coords to pixel offsets within [-rx, rx] x [-ry, ry]
                            const float dx = (config.sample_grid > 1)
                                                 ? static_cast<float>(gx) * static_cast<float>(rx) / static_cast<float>(half_grid)
                                                 : 0.0f;
                            const float dy = (config.sample_grid > 1)
                                                 ? static_cast<float>(gy) * static_cast<float>(ry) / static_cast<float>(half_grid)
                                                 : 0.0f;

                            // 2D Gaussian weight at this sample point
                            const float exponent = 0.5f * (a * dx * dx + 2.0f * b * dx * dy + c * dy * dy);
                            if (exponent > 4.0f)
                                continue;

                            const float weight = std::exp(-exponent);

                            // Pixel coordinates of this sample
                            const int px = round_to_int(cx + dx);
                            const int py = round_to_int(cy + dy);

                            // Out of bounds = outside mask
                            const bool in_bounds = (px >= 0 && px < W && py >= 0 && py < H);
                            const bool inside = in_bounds && (mask_acc(py, px) >= 0.5f);

                            weighted_total += static_cast<double>(weight);
                            weighted_inside += inside ? static_cast<double>(weight) : 0.0;
                        }
                    }

                    if (weighted_total > 1e-9) {
                        local_sum[static_cast<size_t>(i)] += weighted_inside / weighted_total;
                        local_count[static_cast<size_t>(i)] += 1;
                    }
                }

                if ((ci + 1) % 25 == 0 || (ci + 1) == n_cams) {
                    LOG_INFO("[prune_by_alpha_consensus] Processing camera {}/{}", (ci + 1), n_cams);
                }
            }

            #pragma omp critical
            {
                for (int i = 0; i < N; ++i) {
                    consensus_sum[static_cast<size_t>(i)] += local_sum[static_cast<size_t>(i)];
                    consensus_count[static_cast<size_t>(i)] += local_count[static_cast<size_t>(i)];
                }
            }
        }

        LOG_INFO("[prune_by_alpha_consensus] Processed {} cameras, skipped: {} no mask, "
                 "{} size mismatch, {} proj error",
                 n_cams, skipped_no_mask, skipped_size_mismatch, skipped_proj_error);

        // =========================================================================
        // Build removal mask
        // =========================================================================
        Tensor remove_mask_cpu = Tensor::zeros({static_cast<size_t>(N)},
                                               Device::CPU, DataType::Bool);
        auto rm_acc = make_acc<uint8_t, 1>(remove_mask_cpu);

        int n_remove = 0;
        int kept_low_vis = 0;
        int removed_by_cons = 0;
        int kept_by_cons = 0;

        for (int i = 0; i < N; ++i) {
            const int cnt = consensus_count[static_cast<size_t>(i)];
            const double cons_val = (cnt > 0)
                                        ? consensus_sum[static_cast<size_t>(i)] / static_cast<double>(cnt)
                                        : 1.0; // not enough views -> keep

            if (cnt < config.min_visibility_count) {
                rm_acc(i) = 0;
                kept_low_vis++;
                continue;
            }

            if (static_cast<float>(cons_val) < config.consensus_threshold) {
                rm_acc(i) = 1;
                n_remove++;
                removed_by_cons++;
            } else {
                rm_acc(i) = 0;
                kept_by_cons++;
            }
        }

        LOG_INFO("[prune_by_alpha_consensus] === DECISION BREAKDOWN ===");
        LOG_INFO("[prune_by_alpha_consensus]   Kept (low visibility): {:6d} (< {} cameras)",
                 kept_low_vis, config.min_visibility_count);
        LOG_INFO("[prune_by_alpha_consensus]   Removed (failed consensus): {:6d} (ratio < {:.2f})",
                 removed_by_cons, config.consensus_threshold);
        LOG_INFO("[prune_by_alpha_consensus]   Kept (passed consensus):  {:6d}",
                 kept_by_cons);
        LOG_INFO("[prune_by_alpha_consensus]   TOTAL TO REMOVE: {:6d} ({:.1f}%)",
                 n_remove, 100.0f * static_cast<float>(n_remove) / static_cast<float>(N));

        if (n_remove == 0) {
            return PruningResult{.splats_before = N, .splats_after = N, .splats_removed = 0, .success = true};
        }
        if (n_remove == N) {
            LOG_WARN("[prune_by_alpha_consensus] Would remove all splats - skipping.");
            return PruningResult{.splats_before = N, .splats_after = N, .splats_removed = 0, .success = true};
        }

        Tensor remove_mask = remove_mask_cpu.to(Device::CUDA).contiguous();
        strategy.remove_gaussians(remove_mask);

        const int after = static_cast<int>(strategy.get_model().size());
        const int removed = N - after;
        const double pct = (N > 0) ? (100.0 * removed / N) : 0.0;

        LOG_INFO("[prune_by_alpha_consensus] Requested: {}, actually removed: {} ({:.3f}%)",
                 n_remove, removed, pct);

        return PruningResult{.splats_before = N, .splats_after = after, .splats_removed = removed, .success = true};
    } 

    
    std::expected<PruningResult, std::string> prune_by_ellipse_boundary(
        IStrategy& strategy,
        const CameraDataset& dataset,
        const EllipseBoundaryPruningConfig& config) {

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

        // Per-splat vote accumulators
        std::vector<int> negative_votes(static_cast<size_t>(N), 0);
        std::vector<int> evaluating_cameras(static_cast<size_t>(N), 0);

        int skipped_no_mask = 0;
        int skipped_size_mismatch = 0;
        int skipped_proj_error = 0;

        const auto& cams = dataset.get_cameras();
        const int n_cams = static_cast<int>(cams.size());

        LOG_INFO("[prune_by_ellipse_boundary] Starting: {} splats, {} cameras, "
                 "mask_expansion={:.1f}%, negative_threshold={:.0f}%",
                 N, n_cams,
                 100.0f * config.mask_expansion_fraction,
                 100.0f * config.negative_vote_threshold);

        CenterVotePruningConfig proj_cfg;
        proj_cfg.eps2d = config.eps2d;
        proj_cfg.near_plane = config.near_plane;
        proj_cfg.far_plane = config.far_plane;
        proj_cfg.radius_clip = config.radius_clip;
        proj_cfg.scaling_modifier = config.scaling_modifier;

        // 8 sample directions on the ellipse boundary:
        //   4 cardinal:  (+-1, 0), (0, +-1)
        //   4 diagonal:  (+-s, +-s) where s = 1/sqrt(2)
        constexpr float S = 0.70710678f;
        struct Dir {
            float dx;
            float dy;
        };
        constexpr Dir dirs[8] = {
            {1.0f, 0.0f},
            {-1.0f, 0.0f}, // right, left
            {0.0f, 1.0f},
            {0.0f, -1.0f}, // down, up
            {S, S},
            {-S, S}, // diagonal
            {S, -S},
            {-S, -S}};

        const int num_threads = std::min(n_cams, omp_get_max_threads());
        LOG_INFO("[prune_by_ellipse_boundary] Using {} OpenMP threads", num_threads);

        preload_masks_sequential(cams, *sizing, config.invert_masks);

#pragma omp parallel num_threads(num_threads)
        {
            std::vector<int> local_negative(static_cast<size_t>(N), 0);
            std::vector<int> local_evaluating(static_cast<size_t>(N), 0);

            // Thread-local buffers for expanded mask
            std::vector<uint8_t> expanded_mask;

#pragma omp for schedule(dynamic, 1)
            for (int ci = 0; ci < n_cams; ++ci) {
                const auto& cam_ptr = cams[static_cast<size_t>(ci)];
                if (!cam_ptr || !cam_ptr->has_mask()) {
#pragma omp atomic
                    skipped_no_mask++;
                    continue;
                }

                auto& cam = *cam_ptr;

                Tensor mask = cam.load_and_get_mask(
                    sizing->resize_factor, sizing->max_width, config.invert_masks, 0.5f);
                if (!mask.is_valid() || mask.numel() == 0) {
#pragma omp atomic
                    skipped_no_mask++;
                    continue;
                }
                if (mask.ndim() == 3 && mask.shape()[0] == 1) {
                    mask = mask.squeeze(0);
                }

                const int H = static_cast<int>(cam.image_height());
                const int W = static_cast<int>(cam.image_width());
                if (mask.ndim() != 2 ||
                    static_cast<int>(mask.shape()[0]) != H ||
                    static_cast<int>(mask.shape()[1]) != W) {
#pragma omp atomic
                    skipped_size_mismatch++;
                    continue;
                }

                // ---------------------------------------------------------
                // Build expanded mask via dilation.
                // Expansion radius = mask_expansion_fraction * max(W, H)
                // Use integral image for O(1) box queries.
                // ---------------------------------------------------------
                Tensor mask_cpu = mask_to_float01(mask).cpu().contiguous();
                auto mask_acc = make_acc<float, 2>(mask_cpu);

                const int expand_px = static_cast<int>(
                    config.mask_expansion_fraction * static_cast<float>(std::max(W, H)));

                // Build binary mask
                const size_t img_size = static_cast<size_t>(H) * static_cast<size_t>(W);
                expanded_mask.resize(img_size);

                if (expand_px <= 0) {
                    // No expansion: direct binary copy
                    for (int y = 0; y < H; ++y) {
                        uint8_t* row = expanded_mask.data() + static_cast<size_t>(y) * static_cast<size_t>(W);
                        for (int x = 0; x < W; ++x) {
                            row[x] = (mask_acc(y, x) >= 0.5f) ? uint8_t(1) : uint8_t(0);
                        }
                    }
                } else {
                    // Build SAT for dilation
                    // First pass: binary mask
                    std::vector<uint8_t> raw_mask(img_size);
                    for (int y = 0; y < H; ++y) {
                        uint8_t* row = raw_mask.data() + static_cast<size_t>(y) * static_cast<size_t>(W);
                        for (int x = 0; x < W; ++x) {
                            row[x] = (mask_acc(y, x) >= 0.5f) ? uint8_t(1) : uint8_t(0);
                        }
                    }

                    // Build integral image (SAT)
                    const int sat_stride = W + 1;
                    std::vector<int> sat(static_cast<size_t>((H + 1) * sat_stride), 0);

                    for (int y = 0; y < H; ++y) {
                        int row_sum = 0;
                        const uint8_t* row = raw_mask.data() + static_cast<size_t>(y) * static_cast<size_t>(W);
                        for (int x = 0; x < W; ++x) {
                            row_sum += (row[x] != 0) ? 1 : 0;
                            sat[static_cast<size_t>((y + 1) * sat_stride + (x + 1))] =
                                sat[static_cast<size_t>(y * sat_stride + (x + 1))] + row_sum;
                        }
                    }

                    // Dilate: pixel is 1 if any pixel in [x-r, x+r] x [y-r, y+r] is 1
                    auto sat_sum = [&](int x0, int y0, int x1, int y1) -> int {
                        x0 = std::max(0, x0);
                        y0 = std::max(0, y0);
                        x1 = std::min(W - 1, x1);
                        y1 = std::min(H - 1, y1);
                        if (x1 < x0 || y1 < y0)
                            return 0;
                        const int xa = x0, ya = y0;
                        const int xb = x1 + 1, yb = y1 + 1;
                        return sat[static_cast<size_t>(yb * sat_stride + xb)] - sat[static_cast<size_t>(ya * sat_stride + xb)] - sat[static_cast<size_t>(yb * sat_stride + xa)] + sat[static_cast<size_t>(ya * sat_stride + xa)];
                    };

                    for (int y = 0; y < H; ++y) {
                        uint8_t* row = expanded_mask.data() + static_cast<size_t>(y) * static_cast<size_t>(W);
                        for (int x = 0; x < W; ++x) {
                            row[x] = (sat_sum(x - expand_px, y - expand_px,
                                              x + expand_px, y + expand_px) > 0)
                                         ? uint8_t(1)
                                         : uint8_t(0);
                        }
                    }
                }

                // Expanded mask query (in-frame only)
                auto inside_expanded = [&](int x, int y) -> bool {
                    return expanded_mask[static_cast<size_t>(y) * static_cast<size_t>(W) +
                                         static_cast<size_t>(x)] != 0;
                };

                auto in_frame = [&](int x, int y) -> bool {
                    return x >= 0 && x < W && y >= 0 && y < H;
                };

                // ---------------------------------------------------------
                // Project splats
                // ---------------------------------------------------------
                auto proj = project_splats(cam, model, proj_cfg);
                if (!proj) {
#pragma omp atomic
                    skipped_proj_error++;
                    continue;
                }

                Tensor radii_cpu = proj->radii.cpu().contiguous();     // [N, 2] int32
                Tensor means2d_cpu = proj->means2d.cpu().contiguous(); // [N, 2] float

                auto r_acc = make_acc<int32_t, 2>(radii_cpu);
                auto m_acc = make_acc<float, 2>(means2d_cpu);

                // ---------------------------------------------------------
                // Evaluate each splat
                // ---------------------------------------------------------
                for (int i = 0; i < N; ++i) {
                    const int rx = static_cast<int>(r_acc(i, 0));
                    const int ry = static_cast<int>(r_acc(i, 1));

                    // Not visible in this camera
                    if (!is_visible_radii_strict(rx, ry))
                        continue;

                    const float mx = m_acc(i, 0);
                    const float my = m_acc(i, 1);
                    if (!std::isfinite(mx) || !std::isfinite(my))
                        continue;

                    const float frx = static_cast<float>(std::abs(rx));
                    const float fry = static_cast<float>(std::abs(ry));

                    // Splats with tiny projected radius: nothing to evaluate
                    // (the boundary is essentially at the center, which center-vote
                    // already validated).
                    if (frx < 1.0f && fry < 1.0f)
                        continue;

                    // Evaluate 8 boundary points
                    bool any_inframe = false;
                    bool any_outside_expanded = false;

                    for (int d = 0; d < 8; ++d) {
                        const int px = round_to_int(mx + frx * dirs[d].dx);
                        const int py = round_to_int(my + fry * dirs[d].dy);

                        // Point outside image -> skip this point, don't penalize
                        if (!in_frame(px, py))
                            continue;

                        any_inframe = true;

                        if (!inside_expanded(px, py)) {
                            any_outside_expanded = true;
                            break; // one outside point is enough for a negative vote
                        }
                    }

                    // Camera didn't have any boundary point in frame -> can't evaluate
                    if (!any_inframe)
                        continue;

                    // This camera could evaluate the splat
                    local_evaluating[static_cast<size_t>(i)]++;

                    // At least one in-frame boundary point was outside expanded mask
                    if (any_outside_expanded) {
                        local_negative[static_cast<size_t>(i)]++;
                    }
                }

                if ((ci + 1) % 25 == 0 || (ci + 1) == n_cams) {
                    LOG_INFO("[prune_by_ellipse_boundary] Processing camera {}/{}", (ci + 1), n_cams);
                }
            }

            // Merge thread-local into global
            #pragma omp critical
            {
                for (int i = 0; i < N; ++i) {
                    negative_votes[static_cast<size_t>(i)] += local_negative[static_cast<size_t>(i)];
                    evaluating_cameras[static_cast<size_t>(i)] += local_evaluating[static_cast<size_t>(i)];
                }
            }
        }

        LOG_INFO("[prune_by_ellipse_boundary] Processed {} cameras, skipped: {} no mask, "
                 "{} size mismatch, {} proj error",
                 n_cams, skipped_no_mask, skipped_size_mismatch, skipped_proj_error);

        // =====================================================================
        // Build removal mask
        // =====================================================================
        Tensor remove_mask_cpu = Tensor::zeros({static_cast<size_t>(N)},
                                               Device::CPU, DataType::Bool);
        auto rm_acc = make_acc<uint8_t, 1>(remove_mask_cpu);

        int n_remove = 0;
        int kept_low_eval = 0;
        int kept_clean = 0;
        int removed_leaking = 0;

        // Diagnostic: distribution of negative ratios
        int diag_0 = 0;      // exactly 0% negative
        int diag_0_5 = 0;    // (0%, 5%)
        int diag_5_10 = 0;   // [5%, 10%)
        int diag_10_20 = 0;  // [10%, 20%)
        int diag_20_50 = 0;  // [20%, 50%)
        int diag_50_100 = 0; // [50%, 100%]

        for (int i = 0; i < N; ++i) {
            const int ev = evaluating_cameras[static_cast<size_t>(i)];
            const int nv = negative_votes[static_cast<size_t>(i)];

            if (ev < config.min_evaluating_cameras) {
                rm_acc(i) = 0;
                kept_low_eval++;
                continue;
            }

            const float neg_ratio = static_cast<float>(nv) / static_cast<float>(ev);

            // Diagnostic bucketing
            if (nv == 0)
                diag_0++;
            else if (neg_ratio < 0.05f)
                diag_0_5++;
            else if (neg_ratio < 0.10f)
                diag_5_10++;
            else if (neg_ratio < 0.20f)
                diag_10_20++;
            else if (neg_ratio < 0.50f)
                diag_20_50++;
            else
                diag_50_100++;

            if (neg_ratio >= config.negative_vote_threshold) {
                rm_acc(i) = 1;
                n_remove++;
                removed_leaking++;
            } else {
                rm_acc(i) = 0;
                kept_clean++;
            }
        }

        LOG_INFO("[prune_by_ellipse_boundary] === DECISION BREAKDOWN ===");
        LOG_INFO("[prune_by_ellipse_boundary]   Kept (low evaluators):  {:6d} (< {} cameras)",
                 kept_low_eval, config.min_evaluating_cameras);
        LOG_INFO("[prune_by_ellipse_boundary]   Kept (clean boundary):  {:6d}",
                 kept_clean);
        LOG_INFO("[prune_by_ellipse_boundary]   Removed (leaking):      {:6d} (neg_ratio >= {:.0f}%)",
                 removed_leaking, 100.0 * config.negative_vote_threshold);
        LOG_INFO("[prune_by_ellipse_boundary]   TOTAL TO REMOVE:        {:6d} ({:.1f}%)",
                 n_remove, 100.0 * static_cast<double>(n_remove) / std::max(1, N));

        LOG_INFO("[prune_by_ellipse_boundary] === NEGATIVE RATIO DISTRIBUTION ===");
        LOG_INFO("[prune_by_ellipse_boundary]     0%%: {:6d}  |  0-5%%: {:6d}  |  5-10%%: {:6d}",
                 diag_0, diag_0_5, diag_5_10);
        LOG_INFO("[prune_by_ellipse_boundary]   10-20%%: {:6d}  | 20-50%%: {:6d}  | 50-100%%: {:6d}",
                 diag_10_20, diag_20_50, diag_50_100);

        if (n_remove == 0) {
            return PruningResult{.splats_before = N, .splats_after = N, .splats_removed = 0, .success = true};
        }

        const int n_keep = N - n_remove;
        if (n_keep == 0) {
            LOG_WARN("[prune_by_ellipse_boundary] Would remove all splats; skipping.");
            return PruningResult{.splats_before = N, .splats_after = N, .splats_removed = 0, .success = true};
        }

        Tensor remove_mask = remove_mask_cpu.to(Device::CUDA).contiguous();
        strategy.remove_gaussians(remove_mask);

        const int after = static_cast<int>(strategy.get_model().size());
        const int removed = N - after;
        const double pct = (N > 0) ? (100.0 * static_cast<double>(removed) / static_cast<double>(N)) : 0.0;

        LOG_INFO("[prune_by_ellipse_boundary] Requested: {}, actually removed: {} ({:.3f}%)",
                 n_remove, removed, pct);

        return PruningResult{
            .splats_before = N,
            .splats_after = after,
            .splats_removed = removed,
            .success = true};
    }
    
    // -----------------------------------------------------------------------------
    // Isolation pruning (3D nearest-neighbor outliers)
    // -----------------------------------------------------------------------------

    // Minimal nanoflann adaptor over a contiguous float array [N,3]
    struct TensorPointCloud {
        const float* pts = nullptr; // points as [x0,y0,z0, x1,y1,z1, ...]
        size_t N = 0;

        inline size_t kdtree_get_point_count() const { return N; }
        inline float kdtree_get_pt(const size_t idx, int dim) const {
            return pts[idx * 3 + static_cast<size_t>(dim)];
        }
        template <class BBOX>
        bool kdtree_get_bbox(BBOX& /*bb*/) const { return false; }
    };

    static inline float median_inplace(std::vector<float>& v) {
        if (v.empty())
            return 0.0f;
        const size_t mid = v.size() / 2;
        std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(mid), v.end());
        return v[mid];
    }

    std::expected<PruningResult, std::string> prune_by_cluster_and_extremes(
        IStrategy& strategy,
        const ClusterExtremePruningConfig& config) {
        using namespace lfs::core;

        if (!config.enabled) {
            const int n = static_cast<int>(strategy.get_model().size());
            return PruningResult{
                .splats_before = n,
                .splats_after = n,
                .splats_removed = 0,
                .success = true};
        }

        auto& model = strategy.get_model();
        const int N = static_cast<int>(model.size());
        if (N <= 0) {
            return PruningResult{
                .splats_before = 0,
                .splats_after = 0,
                .splats_removed = 0,
                .success = true};
        }

        if (N < 2) {
            LOG_WARN("[prune_by_cluster_and_extremes] Not enough splats (N={}) to compute nearest-neighbor median; skipping.", N);
            return PruningResult{
                .splats_before = N,
                .splats_after = N,
                .splats_removed = 0,
                .success = true};
        }

        Tensor means_cpu = model.get_means().cpu().contiguous();    // [N,3]
        Tensor scales_cpu = model.get_scaling().cpu().contiguous(); // [N,3]
        Tensor rots_cpu = model.get_rotation().cpu().contiguous();  // [N,4] as [w,x,y,z]

        if (!means_cpu.is_valid() || means_cpu.ndim() != 2 ||
            static_cast<int>(means_cpu.shape()[0]) != N ||
            static_cast<int>(means_cpu.shape()[1]) != 3) {
            return std::unexpected("prune_by_cluster_and_extremes: model.get_means() is invalid or not shaped [N,3].");
        }

        if (!scales_cpu.is_valid() || scales_cpu.ndim() != 2 ||
            static_cast<int>(scales_cpu.shape()[0]) != N ||
            static_cast<int>(scales_cpu.shape()[1]) != 3) {
            return std::unexpected("prune_by_cluster_and_extremes: model.get_scaling() is invalid or not shaped [N,3].");
        }

        if (!rots_cpu.is_valid() || rots_cpu.ndim() != 2 ||
            static_cast<int>(rots_cpu.shape()[0]) != N ||
            static_cast<int>(rots_cpu.shape()[1]) != 4) {
            return std::unexpected("prune_by_cluster_and_extremes: model.get_rotation() is invalid or not shaped [N,4].");
        }

        const float* pts = means_cpu.ptr<float>();
        auto means_acc = make_acc<float, 2>(means_cpu);
        auto scales_acc = make_acc<float, 2>(scales_cpu);
        auto rots_acc = make_acc<float, 2>(rots_cpu);

        TensorPointCloud cloud;
        cloud.pts = pts;
        cloud.N = static_cast<size_t>(N);

        using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<float, TensorPointCloud>,
            TensorPointCloud, 3>;

        using RadiusItem = nanoflann::ResultItem<uint32_t, float>;

        KDTree tree_all(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        tree_all.buildIndex();

        // -------------------------------------------------------------------------
        // Step 1: median nearest-neighbor distance between centers
        // -------------------------------------------------------------------------
        std::vector<float> nn_dists(static_cast<size_t>(N), 0.0f);

        {
            const int kq = std::min(N, 4);
            std::vector<size_t> indices(static_cast<size_t>(kq));
            std::vector<float> dists_sq(static_cast<size_t>(kq));
            nanoflann::KNNResultSet<float> resultSet(kq);

            for (int i = 0; i < N; ++i) {
                resultSet.init(indices.data(), dists_sq.data());
                tree_all.findNeighbors(
                    resultSet,
                    pts + static_cast<size_t>(i) * 3ULL,
                    nanoflann::SearchParameters());

                float best = std::numeric_limits<float>::infinity();
                for (int j = 0; j < kq; ++j) {
                    const size_t idx = indices[static_cast<size_t>(j)];
                    if (static_cast<int>(idx) == i) {
                        continue;
                    }
                    best = std::min(best, std::sqrt(std::max(0.0f, dists_sq[static_cast<size_t>(j)])));
                }

                nn_dists[static_cast<size_t>(i)] = best;
            }
        }

        std::vector<float> nn_pool;
        nn_pool.reserve(static_cast<size_t>(N));
        for (float d : nn_dists) {
            if (std::isfinite(d) && d > 0.0f) {
                nn_pool.push_back(d);
            }
        }

        if (nn_pool.empty()) {
            LOG_WARN("[prune_by_cluster_and_extremes] NN distance pool is empty or degenerate; skipping.");
            return PruningResult{
                .splats_before = N,
                .splats_after = N,
                .splats_removed = 0,
                .success = true};
        }

        const float median_nn = median_inplace(nn_pool);
        if (!std::isfinite(median_nn) || median_nn <= 0.0f) {
            LOG_WARN("[prune_by_cluster_and_extremes] Invalid median nearest-neighbor distance = {}; skipping.", median_nn);
            return PruningResult{
                .splats_before = N,
                .splats_after = N,
                .splats_removed = 0,
                .success = true};
        }

        const float eps_cluster = config.cluster_eps_multiplier * median_nn;
        const float eps_extreme = config.extreme_eps_multiplier * median_nn;
        const float eps_cluster_sq = eps_cluster * eps_cluster;

        LOG_INFO("[prune_by_cluster_and_extremes] median_nn={:.6f} m, cluster_eps={:.6f} m (x{:.2f}), extreme_eps={:.6f} m (x{:.2f}), min_cluster_size={}",
                 median_nn,
                 eps_cluster, config.cluster_eps_multiplier,
                 eps_extreme, config.extreme_eps_multiplier,
                 config.min_cluster_size);

        // -------------------------------------------------------------------------
        // Step 2: epsilon-connected components on centers
        // -------------------------------------------------------------------------
        std::vector<uint8_t> visited(static_cast<size_t>(N), 0);
        std::vector<uint8_t> removed_by_cluster(static_cast<size_t>(N), 0);

        int removed_small_clusters = 0;
        int num_clusters = 0;

        std::vector<int> stack;
        stack.reserve(1024);
        std::vector<int> component;
        component.reserve(1024);
        std::vector<RadiusItem> radius_matches;
        radius_matches.reserve(128);

        for (int seed = 0; seed < N; ++seed) {
            if (visited[static_cast<size_t>(seed)]) {
                continue;
            }

            ++num_clusters;
            component.clear();
            stack.clear();

            visited[static_cast<size_t>(seed)] = 1;
            stack.push_back(seed);

            while (!stack.empty()) {
                const int u = stack.back();
                stack.pop_back();
                component.push_back(u);

                radius_matches.clear();
                tree_all.radiusSearch(
                    pts + static_cast<size_t>(u) * 3ULL,
                    eps_cluster_sq,
                    radius_matches,
                    nanoflann::SearchParameters());

                for (const auto& item : radius_matches) {
                    const int v = static_cast<int>(item.first);
                    if (!visited[static_cast<size_t>(v)]) {
                        visited[static_cast<size_t>(v)] = 1;
                        stack.push_back(v);
                    }
                }
            }

            if (static_cast<int>(component.size()) < config.min_cluster_size) {
                removed_small_clusters += static_cast<int>(component.size());
                for (int idx : component) {
                    removed_by_cluster[static_cast<size_t>(idx)] = 1;
                }
            }
        }

        LOG_INFO("[prune_by_cluster_and_extremes] Found {} connected components. Removed {} splats in small clusters (<{}).",
                 num_clusters, removed_small_clusters, config.min_cluster_size);

        int survivors_after_cluster = 0;
        for (int i = 0; i < N; ++i) {
            if (!removed_by_cluster[static_cast<size_t>(i)]) {
                ++survivors_after_cluster;
            }
        }

        if (survivors_after_cluster == 0) {
            LOG_WARN("[prune_by_cluster_and_extremes] Cluster phase would remove all splats; skipping prune.");
            return PruningResult{
                .splats_before = N,
                .splats_after = N,
                .splats_removed = 0,
                .success = true};
        }

        // -------------------------------------------------------------------------
        // Step 3: rebuild KD-tree on survivors
        // -------------------------------------------------------------------------
        std::vector<float> survivor_pts;
        survivor_pts.reserve(static_cast<size_t>(survivors_after_cluster) * 3ULL);

        for (int i = 0; i < N; ++i) {
            if (removed_by_cluster[static_cast<size_t>(i)]) {
                continue;
            }
            survivor_pts.push_back(means_acc(i, 0));
            survivor_pts.push_back(means_acc(i, 1));
            survivor_pts.push_back(means_acc(i, 2));
        }

        TensorPointCloud survivor_cloud;
        survivor_cloud.pts = survivor_pts.data();
        survivor_cloud.N = static_cast<size_t>(survivors_after_cluster);

        KDTree tree_survivors(3, survivor_cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        tree_survivors.buildIndex();

        // -------------------------------------------------------------------------
        // Step 4: extreme-point validation
        // -------------------------------------------------------------------------
        std::vector<uint8_t> removed_by_extremes(static_cast<size_t>(N), 0);

        int removed_extreme = 0;
        int skipped_extreme_all_buried = 0;

        std::vector<size_t> nn_idx(1);
        std::vector<float> nn_dist_sq(1);
        nanoflann::KNNResultSet<float> nn_result(1);

        auto test_extreme = [&](const glm::vec3& extreme_world) -> bool {
            float query_pt[3] = {extreme_world.x, extreme_world.y, extreme_world.z};
            nn_result.init(nn_idx.data(), nn_dist_sq.data());
            tree_survivors.findNeighbors(nn_result, query_pt, nanoflann::SearchParameters());
            const float d = std::sqrt(std::max(0.0f, nn_dist_sq[0]));
            return d > eps_extreme;
        };

        for (int i = 0; i < N; ++i) {
            if (removed_by_cluster[static_cast<size_t>(i)]) {
                continue;
            }

            const float cx = means_acc(i, 0);
            const float cy = means_acc(i, 1);
            const float cz = means_acc(i, 2);

            const float sx = scales_acc(i, 0) * config.extreme_axis_multiplier;
            const float sy = scales_acc(i, 1) * config.extreme_axis_multiplier;
            const float sz = scales_acc(i, 2) * config.extreme_axis_multiplier;

            const float qw = rots_acc(i, 0);
            const float qx = rots_acc(i, 1);
            const float qy = rots_acc(i, 2);
            const float qz = rots_acc(i, 3);

            glm::quat q(qw, qx, qy, qz);
            const float qnorm = glm::length(q);
            if (!std::isfinite(qnorm) || qnorm <= 1e-12f) {
                q = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
            } else {
                q = glm::normalize(q);
            }

            const glm::mat3 R = glm::mat3_cast(q);
            const glm::vec3 c(cx, cy, cz);

            const glm::vec3 ex = R * glm::vec3(sx, 0.0f, 0.0f);
            const glm::vec3 ey = R * glm::vec3(0.0f, sy, 0.0f);
            const glm::vec3 ez = R * glm::vec3(0.0f, 0.0f, sz);

            const glm::vec3 extremes[6] = {
                c + ex, c - ex,
                c + ey, c - ey,
                c + ez, c - ez};

            bool any_checked = false;
            bool remove_this = false;

            for (const glm::vec3& p : extremes) {
                if (p.y < config.burial_y_ignore_threshold) {
                    continue;
                }

                any_checked = true;

                if (test_extreme(p)) {
                    remove_this = true;
                    break;
                }
            }

            if (!any_checked) {
                ++skipped_extreme_all_buried;
                continue;
            }

            if (remove_this) {
                removed_by_extremes[static_cast<size_t>(i)] = 1;
                ++removed_extreme;
            }
        }

        LOG_INFO("[prune_by_cluster_and_extremes] Removed {} splats by extreme validation. {} splats had all 6 extremes below y<{:.3f} and were ignored in this phase.",
                 removed_extreme, skipped_extreme_all_buried, config.burial_y_ignore_threshold);

        // -------------------------------------------------------------------------
        // Final removal mask
        // -------------------------------------------------------------------------
        Tensor remove_mask_cpu = Tensor::zeros({static_cast<size_t>(N)}, Device::CPU, DataType::Bool);
        auto rm_acc = make_acc<uint8_t, 1>(remove_mask_cpu);

        int n_remove = 0;
        int n_keep = 0;

        for (int i = 0; i < N; ++i) {
            const bool remove =
                (removed_by_cluster[static_cast<size_t>(i)] != 0) ||
                (removed_by_extremes[static_cast<size_t>(i)] != 0);

            rm_acc(i) = remove ? 1 : 0;
            n_remove += remove ? 1 : 0;
            n_keep += remove ? 0 : 1;
        }

        LOG_INFO("[prune_by_cluster_and_extremes] Removal breakdown:");
        LOG_INFO("[prune_by_cluster_and_extremes]   Removed by small cluster: {:6d}", removed_small_clusters);
        LOG_INFO("[prune_by_cluster_and_extremes]   Removed by extremes:      {:6d}", removed_extreme);
        LOG_INFO("[prune_by_cluster_and_extremes]   Total to remove:          {:6d} ({:5.1f}%)",
                 n_remove, 100.0 * static_cast<double>(n_remove) / std::max(1, N));

        if (n_remove == 0) {
            LOG_INFO("[prune_by_cluster_and_extremes] Removed 0 splats (0.000%)");
            return PruningResult{
                .splats_before = N,
                .splats_after = N,
                .splats_removed = 0,
                .success = true};
        }

        if (n_keep == 0) {
            LOG_WARN("[prune_by_cluster_and_extremes] Would remove all splats; skipping prune.");
            return PruningResult{
                .splats_before = N,
                .splats_after = N,
                .splats_removed = 0,
                .success = true};
        }

        Tensor remove_mask = remove_mask_cpu.to(Device::CUDA).contiguous();
        strategy.remove_gaussians(remove_mask);

        const int after = static_cast<int>(strategy.get_model().size());
        const int removed = N - after;
        const double pct = (N > 0) ? (100.0 * static_cast<double>(removed) / static_cast<double>(N)) : 0.0;

        LOG_INFO("[prune_by_cluster_and_extremes] Requested removal: {}, actually removed: {} ({:.3f}%)",
                 n_remove, removed, pct);

        return PruningResult{
            .splats_before = N,
            .splats_after = after,
            .splats_removed = removed,
            .success = true};
    }

    std::expected<PruningResult, std::string> prune_by_isolation_distance(
        IStrategy& strategy,
        const IsolationPruningConfig& config) {

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

        // kth neighbor excluding self (1-based)
        const int kth = std::max(1, config.kth_neighbor);
        if (N <= kth) {
            LOG_WARN("[prune_by_isolation_distance] Not enough splats (N={}) for kth_neighbor={}; skipping.", N, kth);
            return PruningResult{.splats_before = N, .splats_after = N, .splats_removed = 0, .success = true};
        }

        // Pull positions to CPU
        Tensor means_cpu = model.get_means().cpu().contiguous(); // [N,3] float32
        if (!means_cpu.is_valid() || means_cpu.numel() == 0 ||
            means_cpu.ndim() != 2 ||
            static_cast<int>(means_cpu.shape()[0]) != N ||
            static_cast<int>(means_cpu.shape()[1]) != 3) {
            return std::unexpected("prune_by_isolation_distance: model.get_means() is invalid or not shaped [N,3].");
        }
        const float* pts = means_cpu.ptr<float>();

        // Build KD-tree
        TensorPointCloud cloud;
        cloud.pts = pts;
        cloud.N = static_cast<size_t>(N);

        using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<float, TensorPointCloud>,
            TensorPointCloud, 3>;

        KDTree tree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        tree.buildIndex();

        // Query count: max(k_neighbors, kth_neighbor) + 1 (self) to be safe.
        const int kq = std::min(N, std::max(std::max(1, config.k_neighbors), kth) + 1);
        if (kq <= kth) {
            LOG_WARN("[prune_by_isolation_distance] Query k too small (kq={}, kth={}); skipping.", kq, kth);
            return PruningResult{.splats_before = N, .splats_after = N, .splats_removed = 0, .success = true};
        }

        std::vector<size_t> indices(static_cast<size_t>(kq));
        std::vector<float> dists_sq(static_cast<size_t>(kq));
        nanoflann::KNNResultSet<float> resultSet(kq);

        // Compute d_k for each splat: distance to kth neighbor EXCLUDING self
        std::vector<float> dk(static_cast<size_t>(N), 0.0f);
        std::vector<float> neigh_dists;
        neigh_dists.reserve(static_cast<size_t>(kq));

        for (int i = 0; i < N; ++i) {
            resultSet.init(indices.data(), dists_sq.data());
            tree.findNeighbors(resultSet, pts + static_cast<size_t>(i) * 3ULL, nanoflann::SearchParameters());

            neigh_dists.clear();

            // Collect neighbor distances excluding self (robust to whether self is returned at slot 0 or not)
            for (int j = 0; j < kq; ++j) {
                const size_t idx = indices[static_cast<size_t>(j)];
                if (static_cast<int>(idx) == i) {
                    continue; // skip self
                }
                const float dsq = dists_sq[static_cast<size_t>(j)];
                neigh_dists.push_back(std::sqrt(std::max(0.0f, dsq)));
            }

            // We requested kq = max(k_neighbors,kth)+1 so neigh_dists should have at least kth entries.
            // Still, guard in case of any unexpected behavior.
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

        // Global median over ALL splats
        std::vector<float> pool = dk; // copy
        const float med = median_inplace(pool);

        if (!std::isfinite(med) || med <= 0.0f) {
            LOG_WARN("[prune_by_isolation_distance] Invalid median d_k={}; skipping.", med);
            return PruningResult{.splats_before = N, .splats_after = N, .splats_removed = 0, .success = true};
        }

        const float base_thr = config.threshold_multiplier * med;
        const float abs_thr = (config.abs_distance_min > 0.0f) ? config.abs_distance_min : 0.0f;
        const float thr = (abs_thr > base_thr) ? abs_thr : base_thr;

        LOG_INFO("[prune_by_isolation_distance] d{} median={:.6f} m (N={}), thr={:.6f} m (x{:.2f}), abs_min={:.6f} m",
                 kth, med, N, thr, config.threshold_multiplier, abs_thr);
        // Build removal mask
        Tensor remove_mask_cpu = Tensor::zeros({static_cast<size_t>(N)}, Device::CPU, DataType::Bool);
        auto rm_acc = make_acc<uint8_t, 1>(remove_mask_cpu);

        int n_remove = 0;
        int n_keep = 0;

        for (int i = 0; i < N; ++i) {
            const float d = dk[static_cast<size_t>(i)];
            const bool remove = std::isfinite(d) && (d > thr);

            rm_acc(i) = remove ? 1 : 0;
            n_remove += remove ? 1 : 0;
            n_keep += remove ? 0 : 1;
        }

        if (n_remove == 0) {
            LOG_INFO("[prune_by_isolation_distance] Removed 0 splats (0.000%)");
            return PruningResult{.splats_before = N, .splats_after = N, .splats_removed = 0, .success = true};
        }
        if (n_keep == 0) {
            LOG_WARN("[prune_by_isolation_distance] Would remove all splats; skipping.");
            return PruningResult{.splats_before = N, .splats_after = N, .splats_removed = 0, .success = true};
        }

        Tensor remove_mask = remove_mask_cpu.to(Device::CUDA).contiguous();
        strategy.remove_gaussians(remove_mask);

        const int after = static_cast<int>(strategy.get_model().size());
        const int removed = N - after;
        const double pct = (N > 0) ? (100.0 * static_cast<double>(removed) / static_cast<double>(N)) : 0.0;

        LOG_INFO("[prune_by_isolation_distance] Removed {} splats ({:.2}%) | threshold={:.4f}m ({}x median)",
                 removed, pct, thr, config.threshold_multiplier);

        return PruningResult{.splats_before = N,
                             .splats_after = after,
                             .splats_removed = removed,
                             .success = true};
    }

    // -----------------------------------------------------------------------------
    // SMN: Statistical Outlier Removal (SOR)
    // -----------------------------------------------------------------------------

    std::expected<PruningResult, std::string> prune_by_sor(
        IStrategy& strategy,
        const SORPruningConfig& config) {
           
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

        const int k = clamp_int(config.k_neighbors, 1, std::min(500, N - 1));
        if (k < 1 || N <= k) {
            LOG_WARN("[prune_by_sor] Not enough splats (N={}) for k_neighbors={}; skipping.", N, config.k_neighbors);
            return PruningResult{.splats_before = N, .splats_after = N, .splats_removed = 0, .success = true};
        }

        Tensor means_cpu = model.get_means().cpu().contiguous(); // [N,3] float32
        if (!means_cpu.is_valid() || means_cpu.numel() == 0 ||
            means_cpu.ndim() != 2 ||
            static_cast<int>(means_cpu.shape()[0]) != N ||
            static_cast<int>(means_cpu.shape()[1]) != 3) {
            return std::unexpected("prune_by_sor: model.get_means() is invalid or not shaped [N,3].");
        }
        const float* pts = means_cpu.ptr<float>();

        TensorPointCloud cloud;
        cloud.pts = pts;
        cloud.N = static_cast<size_t>(N);

        using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<float, TensorPointCloud>,
            TensorPointCloud, 3>;

        KDTree tree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        tree.buildIndex();

        // Query k+1: the query point itself comes back as a neighbor.
        const int kq = std::min(N, k + 1);

        // Connectivity guard neighbor (results are sorted by distance).
        const int guard_kth = (config.guard_kth > 0) ? std::min(config.guard_kth, k) : 0;

        // Mean distance to the k nearest neighbors, per splat, plus the
        // distance to the guard_kth neighbor (local connectivity).
        std::vector<float> mean_dist(static_cast<size_t>(N), 0.0f);
        std::vector<float> guard_dist(static_cast<size_t>(N), 0.0f);

        const int num_threads = omp_get_max_threads();
#pragma omp parallel num_threads(num_threads)
        {
            std::vector<size_t> indices(static_cast<size_t>(kq));
            std::vector<float> dists_sq(static_cast<size_t>(kq));

#pragma omp for schedule(static)
            for (int i = 0; i < N; ++i) {
                nanoflann::KNNResultSet<float> resultSet(static_cast<size_t>(kq));
                resultSet.init(indices.data(), dists_sq.data());
                tree.findNeighbors(resultSet, pts + static_cast<size_t>(i) * 3ULL, nanoflann::SearchParameters());

                double sum = 0.0;
                int used = 0;
                float guard_d = std::numeric_limits<float>::infinity();
                for (int j = 0; j < kq && used < k; ++j) {
                    if (static_cast<int>(indices[static_cast<size_t>(j)]) == i) {
                        continue; // skip self
                    }
                    const float d = std::sqrt(std::max(0.0f, dists_sq[static_cast<size_t>(j)]));
                    sum += d;
                    ++used;
                    if (used == guard_kth) {
                        guard_d = d;
                    }
                }

                mean_dist[static_cast<size_t>(i)] =
                    (used > 0) ? static_cast<float>(sum / used)
                               : std::numeric_limits<float>::infinity();
                guard_dist[static_cast<size_t>(i)] = guard_d;
            }
        }

        // Global statistics of the mean-distance distribution (finite values only).
        double sum = 0.0;
        long long n_finite = 0;
        for (int i = 0; i < N; ++i) {
            const float d = mean_dist[static_cast<size_t>(i)];
            if (std::isfinite(d)) {
                sum += static_cast<double>(d);
                ++n_finite;
            }
        }
        if (n_finite < 2) {
            LOG_WARN("[prune_by_sor] Not enough finite neighbor statistics; skipping.");
            return PruningResult{.splats_before = N, .splats_after = N, .splats_removed = 0, .success = true};
        }
        const double mean = sum / static_cast<double>(n_finite);

        double sq_dev = 0.0;
        for (int i = 0; i < N; ++i) {
            const float d = mean_dist[static_cast<size_t>(i)];
            if (std::isfinite(d)) {
                const double dev = static_cast<double>(d) - mean;
                sq_dev += dev * dev;
            }
        }
        const double stddev = std::sqrt(sq_dev / static_cast<double>(n_finite));
        const float thr = static_cast<float>(mean + static_cast<double>(config.std_ratio) * stddev);

        // Connectivity guard threshold: guard_kth-neighbor distance must exceed
        // this multiple of its global median for the splat to be removable.
        // Protects protruding-but-connected geometry (fingertips, hair tips).
        float guard_thr = 0.0f;
        if (guard_kth > 0) {
            std::vector<float> guard_pool;
            guard_pool.reserve(static_cast<size_t>(N));
            for (int i = 0; i < N; ++i) {
                const float g = guard_dist[static_cast<size_t>(i)];
                if (std::isfinite(g)) {
                    guard_pool.push_back(g);
                }
            }
            const float guard_median = guard_pool.empty() ? 0.0f : median_inplace(guard_pool);
            if (std::isfinite(guard_median) && guard_median > 0.0f) {
                guard_thr = config.guard_multiplier * guard_median;
            }
            LOG_INFO("[prune_by_sor] connectivity guard: d{} median={:.6f} m x{:.1f} -> only removable above {:.6f} m",
                     guard_kth, guard_median, config.guard_multiplier, guard_thr);
        }

        LOG_INFO("[prune_by_sor] k={}, mean d_k={:.6f} m, stddev={:.6f} m, std_ratio={:.2f} -> threshold={:.6f} m",
                 k, mean, stddev, config.std_ratio, thr);

        Tensor remove_mask_cpu = Tensor::zeros({static_cast<size_t>(N)}, Device::CPU, DataType::Bool);
        auto rm_acc = make_acc<uint8_t, 1>(remove_mask_cpu);

        int n_remove = 0;
        int n_keep = 0;
        int guard_saved = 0;
        for (int i = 0; i < N; ++i) {
            const float d = mean_dist[static_cast<size_t>(i)];
            bool remove = std::isfinite(d) && (d > thr);
            if (remove && guard_thr > 0.0f) {
                const float g = guard_dist[static_cast<size_t>(i)];
                if (std::isfinite(g) && g <= guard_thr) {
                    remove = false; // locally connected: fingertip, hair tip, ...
                    ++guard_saved;
                }
            }
            rm_acc(i) = remove ? 1 : 0;
            n_remove += remove ? 1 : 0;
            n_keep += remove ? 0 : 1;
        }

        if (guard_saved > 0) {
            LOG_INFO("[prune_by_sor] connectivity guard kept {} statistically-outlying but connected splats", guard_saved);
        }

        if (n_remove == 0) {
            LOG_INFO("[prune_by_sor] Removed 0 splats (0.000%)");
            return PruningResult{.splats_before = N, .splats_after = N, .splats_removed = 0, .success = true};
        }
        if (n_keep == 0) {
            LOG_WARN("[prune_by_sor] Would remove all splats; skipping.");
            return PruningResult{.splats_before = N, .splats_after = N, .splats_removed = 0, .success = true};
        }

        Tensor remove_mask = remove_mask_cpu.to(Device::CUDA).contiguous();
        strategy.remove_gaussians(remove_mask);

        const int after = static_cast<int>(strategy.get_model().size());
        const int removed = N - after;
        const double pct = (N > 0) ? (100.0 * static_cast<double>(removed) / static_cast<double>(N)) : 0.0;

        LOG_INFO("[prune_by_sor] Removed {} splats ({:.2f}%) | threshold={:.4f} m (mean + {:.2f} sigma)",
                 removed, pct, thr, config.std_ratio);

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
        const GeometricDomePruningConfig& geometric_config,
        const CenterVotePruningConfig& center_config,
        const LeakagePruningConfig& leakage_config,
        const IsolationPruningConfig& isolation_config) {

        const int before = static_cast<int>(strategy.get_model().size());
        if (before == 0) {
            return PruningResult{.splats_before = 0, .splats_after = 0, .splats_removed = 0, .success = true};
        }

        if (geometric_config.enabled) {
            auto geomdome = prune_by_geometric_dome(strategy, dataset, geometric_config);
            if (!geomdome) {
                return std::unexpected(geomdome.error());
            }
        }

        if (center_config.enabled) {
            auto center = prune_by_center_vote(strategy, dataset, center_config);
            if (!center) {
                return std::unexpected(center.error());
            }
        }
        
        if (leakage_config.enabled) {
            auto leak = prune_by_mask_leakage(strategy, dataset, leakage_config);
            if (!leak) {
                return std::unexpected(leak.error());
            }
        }

    
        /* ClusterExtremePruningConfig cluster_config;
        auto cluster = prune_by_cluster_and_extremes(strategy, cluster_config);
        if (!cluster) {
            return std::unexpected(cluster.error());
        }*/
                
        auto bound = prune_by_ellipse_boundary(strategy,dataset);
        if (!bound) {
            return std::unexpected(bound.error());
        }

        if (isolation_config.enabled) {
            auto isolation = prune_by_isolation_distance(strategy, isolation_config);
            if (!isolation) {
                return std::unexpected(isolation.error());
            }
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

} // namespace lfs::training::smn::mask_pruning
