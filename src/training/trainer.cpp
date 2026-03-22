/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "trainer.hpp"
#include "components/bilateral_grid.hpp"
#include "components/ppisp.hpp"
#include "components/ppisp_controller_pool.hpp"
#include "components/ppisp_file.hpp"
#include "components/sparsity_optimizer.hpp"
#include "control/command_api.hpp"
#include "control/control_boundary.hpp"
#include "core/checkpoint_format.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/events.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/scene.hpp"
#include "core/splat_data_transform.hpp"
#include "io/cache_image_loader.hpp"
#include "io/exporter.hpp"
#include "io/filesystem_utils.hpp"
#include "lfs/kernels/ssim.cuh"
#include "losses/losses.hpp"
#include "optimizer/adam_optimizer.hpp"
#include "python/runner.hpp"
#include "rasterization/fast_rasterizer.hpp"
#include "rasterization/gsplat_rasterizer.hpp"
#include "strategies/adc.hpp"
#include "strategies/mcmc.hpp"
#include "strategies/strategy_factory.hpp"
#include "training/kernels/grad_alpha.hpp"

#include <filesystem>
#include <fstream>

#include <atomic>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <expected>
#include <memory>
#include <numeric>
#include <nvtx3/nvToolsExt.h>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "smn/mask_penalty.hpp" //matting modes
#include "smn/mask_pruning.hpp"
#include "smn/mask_pruning_visualizer.hpp"

namespace lfs::training {

    namespace {
        template <typename Fn>
        class ScopeGuard {
        public:
            explicit ScopeGuard(Fn fn)
                : fn_(std::move(fn)) {}

            ScopeGuard(const ScopeGuard&) = delete;
            ScopeGuard& operator=(const ScopeGuard&) = delete;

            ScopeGuard(ScopeGuard&& other) noexcept
                : fn_(std::move(other.fn_)),
                  active_(other.active_) {
                other.active_ = false;
            }

            ScopeGuard& operator=(ScopeGuard&&) = delete;

            ~ScopeGuard() {
                if (active_) {
                    fn_();
                }
            }

            void release() noexcept { active_ = false; }

        private:
            Fn fn_;
            bool active_ = true;
        };

        template <typename Fn>
        ScopeGuard<Fn> makeScopeGuard(Fn fn) {
            return ScopeGuard<Fn>(std::move(fn));
        }

        PPISPRenderOverrides toRenderOverrides(const PPISPViewportOverrides& ov) {
            PPISPRenderOverrides r;
            r.exposure_offset = ov.exposure_offset;
            r.vignette_enabled = ov.vignette_enabled;
            r.vignette_strength = ov.vignette_strength;
            r.wb_temperature = ov.wb_temperature;
            r.wb_tint = ov.wb_tint;
            r.color_red_x = ov.color_red_x;
            r.color_red_y = ov.color_red_y;
            r.color_green_x = ov.color_green_x;
            r.color_green_y = ov.color_green_y;
            r.color_blue_x = ov.color_blue_x;
            r.color_blue_y = ov.color_blue_y;
            r.gamma_multiplier = ov.gamma_multiplier;
            r.gamma_red = ov.gamma_red;
            r.gamma_green = ov.gamma_green;
            r.gamma_blue = ov.gamma_blue;
            r.crf_toe = ov.crf_toe;
            r.crf_shoulder = ov.crf_shoulder;
            return r;
        }
    } // namespace

    // Tile configuration for memory-efficient training
    enum class TileMode {
        One = 1, // 1 tile  - 1x1 - Render full image (no tiling)
        Two = 2, // 2 tiles - 2x1 - Two horizontal tiles
        Four = 4 // 4 tiles - 2x2 - Four tiles in a grid
    };

    void Trainer::cleanup() {
        LOG_DEBUG("Cleaning up trainer for re-initialization");

        // Stop any ongoing operations
        stop_requested_ = true;

        // Sync callback stream to avoid race conditions
        if (callback_stream_) {
            cudaStreamSynchronize(callback_stream_);
        }
        callback_busy_ = false;

        // Reset all components
        progress_.reset();
        bilateral_grid_.reset();
        ppisp_.reset();
        ppisp_controller_pool_.reset();
        sparsity_optimizer_.reset();
        evaluator_.reset();

        // Clear datasets (will be recreated)
        train_dataset_.reset();
        val_dataset_.reset();

        // Reset flags
        pause_requested_ = false;
        save_requested_ = false;
        stop_requested_ = false;
        is_paused_ = false;
        is_running_ = false;
        training_complete_ = false;
        ready_to_start_ = false;
        current_iteration_ = 0;
        current_loss_ = 0.0f;
        train_dataset_size_ = 0;
        total_cameras_count_ = 0;

        // NGS
        ngs_noise_.reset();
        ngs_phase_manager_.reset();
        ngs_enabled_ = false;

        LOG_DEBUG("Trainer cleanup complete");
    }

    std::expected<void, std::string> Trainer::initialize_bilateral_grid() {
        if (!params_.optimization.use_bilateral_grid) {
            return {};
        }

        try {
            BilateralGrid::Config config;
            config.lr = params_.optimization.bilateral_grid_lr;

            // BilateralGrid is indexed with cam->uid() in the training loop. Those UIDs stay
            // in the original camera space even when train/val splits are enabled, so the grid
            // must be sized for the full camera set rather than only the training subset.
            bilateral_grid_ = std::make_unique<BilateralGrid>(
                static_cast<int>(total_cameras_count_),
                params_.optimization.bilateral_grid_X,
                params_.optimization.bilateral_grid_Y,
                params_.optimization.bilateral_grid_W,
                params_.optimization.iterations,
                config);

            LOG_INFO("Bilateral grid initialized: {}x{}x{} for {} camera slots ({} train images)",
                     params_.optimization.bilateral_grid_X,
                     params_.optimization.bilateral_grid_Y,
                     params_.optimization.bilateral_grid_W,
                     total_cameras_count_,
                     train_dataset_size_);

            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to init bilateral grid: {}", e.what()));
        }
    }

    std::expected<void, std::string> Trainer::initialize_ppisp() {
        if (!params_.optimization.use_ppisp) {
            return {};
        }

        try {
            PPISPConfig config;
            config.lr = params_.optimization.ppisp_lr;
            config.warmup_steps = params_.optimization.ppisp_warmup_steps;

            ppisp_ = std::make_unique<PPISP>(params_.optimization.iterations, config);
            for (const auto& cam : train_dataset_->get_cameras()) {
                if (cam) {
                    ppisp_->register_frame(cam->uid(), cam->camera_id());
                }
            }
            ppisp_->finalize();

            LOG_INFO("PPISP initialized: {} cameras (physical), {} frames, lr={:.2e}, warmup={}",
                     ppisp_->num_cameras(), ppisp_->num_frames(), params_.optimization.ppisp_lr, config.warmup_steps);

            if (auto result = apply_ppisp_sidecar_if_configured(); !result) {
                return result;
            }

            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to init PPISP: {}", e.what()));
        }
    }

    std::expected<PPISPFileMetadata, std::string> Trainer::build_ppisp_sidecar_metadata() const {
        if (!ppisp_ || !ppisp_->isFinalized()) {
            return std::unexpected("Cannot build PPISP sidecar metadata before PPISP is initialized");
        }
        if (!train_dataset_) {
            return std::unexpected("Cannot build PPISP sidecar metadata without an active training dataset");
        }

        PPISPFileMetadata metadata;
        metadata.dataset_path_utf8 = lfs::core::path_to_utf8(params_.dataset.data_path);
        metadata.images_folder = params_.dataset.images;
        metadata.camera_ids = ppisp_->ordered_camera_ids();

        for (const auto& cam : train_dataset_->get_cameras()) {
            if (!cam) {
                continue;
            }
            metadata.frame_image_names.push_back(cam->image_name());
            metadata.frame_camera_ids.push_back(cam->camera_id());
        }

        if (static_cast<int>(metadata.frame_image_names.size()) != ppisp_->num_frames() ||
            static_cast<int>(metadata.frame_camera_ids.size()) != ppisp_->num_frames()) {
            return std::unexpected(std::format(
                "PPISP metadata frame mismatch: metadata has {} names / {} camera ids but PPISP has {} frames",
                metadata.frame_image_names.size(),
                metadata.frame_camera_ids.size(),
                ppisp_->num_frames()));
        }
        if (static_cast<int>(metadata.camera_ids.size()) != ppisp_->num_cameras()) {
            return std::unexpected(std::format(
                "PPISP metadata camera mismatch: metadata has {} camera ids but PPISP has {} cameras",
                metadata.camera_ids.size(),
                ppisp_->num_cameras()));
        }

        return metadata;
    }

    std::expected<Trainer::PPISPSidecarMappings, std::string> Trainer::build_ppisp_sidecar_mappings(
        const PPISP& loaded_ppisp,
        const PPISPFileMetadata& metadata,
        const std::filesystem::path& sidecar_path) const {

        if (!ppisp_ || !ppisp_->isFinalized()) {
            return std::unexpected("Cannot apply PPISP sidecar before PPISP initialization is complete");
        }
        if (!train_dataset_) {
            return std::unexpected("Cannot apply PPISP sidecar without an active training dataset");
        }
        if (metadata.empty()) {
            return std::unexpected(std::format(
                "Frozen PPISP sidecar '{}' has no dataset metadata. Older sidecars cannot be verified against the current dataset; resave the source model with sidecar metadata first.",
                lfs::core::path_to_utf8(sidecar_path)));
        }
        if (static_cast<int>(metadata.frame_image_names.size()) != loaded_ppisp.num_frames() ||
            static_cast<int>(metadata.frame_camera_ids.size()) != loaded_ppisp.num_frames()) {
            return std::unexpected(std::format(
                "PPISP sidecar metadata frame count mismatch: metadata has {} names / {} camera ids but sidecar has {} frames",
                metadata.frame_image_names.size(),
                metadata.frame_camera_ids.size(),
                loaded_ppisp.num_frames()));
        }
        if (static_cast<int>(metadata.camera_ids.size()) != loaded_ppisp.num_cameras()) {
            return std::unexpected(std::format(
                "PPISP sidecar metadata camera count mismatch: metadata has {} camera ids but sidecar has {} cameras",
                metadata.camera_ids.size(),
                loaded_ppisp.num_cameras()));
        }

        const auto current_dataset_path = lfs::core::path_to_utf8(params_.dataset.data_path);
        if (!metadata.dataset_path_utf8.empty() && metadata.dataset_path_utf8 != current_dataset_path) {
            LOG_INFO("Frozen PPISP sidecar dataset path differs from current dataset path: '{}' vs '{}'",
                     metadata.dataset_path_utf8, current_dataset_path);
        }
        if (!metadata.images_folder.empty() && metadata.images_folder != params_.dataset.images) {
            LOG_INFO("Frozen PPISP sidecar images folder differs from current training config: '{}' vs '{}'",
                     metadata.images_folder, params_.dataset.images);
        }

        auto make_frame_key = [](std::string_view image_name, int camera_id) {
            return std::format("{}\n{}", image_name, camera_id);
        };

        std::unordered_map<std::string, int> source_frame_index_by_key;
        source_frame_index_by_key.reserve(metadata.frame_image_names.size());
        for (size_t i = 0; i < metadata.frame_image_names.size(); ++i) {
            auto [_, inserted] = source_frame_index_by_key.emplace(
                make_frame_key(metadata.frame_image_names[i], metadata.frame_camera_ids[i]),
                static_cast<int>(i));
            if (!inserted) {
                return std::unexpected(std::format(
                    "PPISP sidecar metadata contains duplicate frame key for image '{}' and camera {}",
                    metadata.frame_image_names[i],
                    metadata.frame_camera_ids[i]));
            }
        }

        PPISPSidecarMappings mappings;
        mappings.frame_mapping.reserve(static_cast<size_t>(ppisp_->num_frames()));
        std::unordered_set<std::string> seen_target_frames;
        seen_target_frames.reserve(static_cast<size_t>(ppisp_->num_frames()));
        for (const auto& cam : train_dataset_->get_cameras()) {
            if (!cam) {
                continue;
            }
            const auto key = make_frame_key(cam->image_name(), cam->camera_id());
            if (!seen_target_frames.insert(key).second) {
                return std::unexpected(std::format(
                    "Current training dataset contains duplicate frame key for image '{}' and camera {}",
                    cam->image_name(),
                    cam->camera_id()));
            }
            const auto it = source_frame_index_by_key.find(key);
            if (it == source_frame_index_by_key.end()) {
                return std::unexpected(std::format(
                    "Frozen PPISP sidecar is missing frame '{}' for camera {}",
                    cam->image_name(),
                    cam->camera_id()));
            }
            mappings.frame_mapping.push_back(it->second);
        }
        if (seen_target_frames.size() != source_frame_index_by_key.size()) {
            return std::unexpected(std::format(
                "Frozen PPISP sidecar dataset mismatch: sidecar has {} frame keys but current training dataset has {}",
                source_frame_index_by_key.size(),
                seen_target_frames.size()));
        }

        std::unordered_map<int, int> source_camera_index_by_id;
        source_camera_index_by_id.reserve(metadata.camera_ids.size());
        for (size_t i = 0; i < metadata.camera_ids.size(); ++i) {
            auto [_, inserted] = source_camera_index_by_id.emplace(metadata.camera_ids[i], static_cast<int>(i));
            if (!inserted) {
                return std::unexpected(std::format(
                    "PPISP sidecar metadata contains duplicate camera id {}",
                    metadata.camera_ids[i]));
            }
        }

        const auto target_camera_ids = ppisp_->ordered_camera_ids();
        mappings.camera_mapping.reserve(target_camera_ids.size());
        for (const int camera_id : target_camera_ids) {
            const auto it = source_camera_index_by_id.find(camera_id);
            if (it == source_camera_index_by_id.end()) {
                return std::unexpected(std::format(
                    "Frozen PPISP sidecar is missing camera id {} required by the current dataset",
                    camera_id));
            }
            mappings.camera_mapping.push_back(it->second);
        }
        if (target_camera_ids.size() != source_camera_index_by_id.size()) {
            return std::unexpected(std::format(
                "Frozen PPISP sidecar dataset mismatch: sidecar has {} camera ids but current training dataset uses {}",
                source_camera_index_by_id.size(),
                target_camera_ids.size()));
        }

        return mappings;
    }

    std::expected<void, std::string> Trainer::apply_ppisp_sidecar_if_configured() {
        if (!should_apply_ppisp_sidecar_on_init()) {
            return {};
        }
        if (!ppisp_ || !ppisp_->isFinalized()) {
            return std::unexpected("Cannot apply PPISP sidecar before PPISP initialization is complete");
        }

        PPISP loaded_ppisp(1);
        PPISPFileMetadata metadata;
        const auto sidecar_path = params_.optimization.ppisp_sidecar_path;

        if (auto result = load_ppisp_file(sidecar_path, loaded_ppisp, nullptr, &metadata); !result) {
            return std::unexpected(std::format(
                "Failed to load frozen PPISP sidecar '{}': {}",
                lfs::core::path_to_utf8(sidecar_path),
                result.error()));
        }

        auto mappings_result = build_ppisp_sidecar_mappings(loaded_ppisp, metadata, sidecar_path);
        if (!mappings_result) {
            return std::unexpected(mappings_result.error());
        }
        auto& mappings = *mappings_result;

        if (static_cast<int>(mappings.frame_mapping.size()) != ppisp_->num_frames()) {
            return std::unexpected(std::format(
                "Frozen PPISP sidecar frame mapping size mismatch: {} mappings for {} target frames",
                mappings.frame_mapping.size(),
                ppisp_->num_frames()));
        }
        if (static_cast<int>(mappings.camera_mapping.size()) != ppisp_->num_cameras()) {
            return std::unexpected(std::format(
                "Frozen PPISP sidecar camera mapping size mismatch: {} mappings for {} target cameras",
                mappings.camera_mapping.size(),
                ppisp_->num_cameras()));
        }

        if (auto result = ppisp_->copy_inference_weights_from(
                loaded_ppisp, mappings.frame_mapping, mappings.camera_mapping);
            !result) {
            return std::unexpected(std::format(
                "Failed to import frozen PPISP weights from '{}': {}",
                lfs::core::path_to_utf8(sidecar_path),
                result.error()));
        }

        LOG_INFO("Loaded frozen PPISP sidecar '{}' ({} cameras, {} frames{})",
                 lfs::core::path_to_utf8(sidecar_path),
                 loaded_ppisp.num_cameras(),
                 loaded_ppisp.num_frames(),
                 ", metadata-mapped");
        return {};
    }

    std::expected<void, std::string> Trainer::initialize_ppisp_controller() {
        if (!params_.optimization.ppisp_use_controller || !params_.optimization.use_ppisp) {
            return {};
        }

        if (!ppisp_) {
            return std::unexpected("PPISP must be initialized before controller");
        }

        try {
            const bool import_frozen_sidecar_controller = should_apply_ppisp_sidecar_on_init();
            const auto sidecar_path = params_.optimization.ppisp_sidecar_path;
            PPISPFileHeader sidecar_header{};
            if (import_frozen_sidecar_controller) {
                std::ifstream file;
                if (!lfs::core::open_file_for_read(sidecar_path, std::ios::binary, file)) {
                    return std::unexpected("Failed to open frozen PPISP sidecar: " +
                                           lfs::core::path_to_utf8(sidecar_path));
                }
                file.read(reinterpret_cast<char*>(&sidecar_header), sizeof(sidecar_header));
                if (!file) {
                    return std::unexpected("Failed to read frozen PPISP sidecar header: " +
                                           lfs::core::path_to_utf8(sidecar_path));
                }
                if (sidecar_header.magic != PPISP_FILE_MAGIC) {
                    return std::unexpected("Invalid frozen PPISP sidecar: wrong magic number");
                }
                if (sidecar_header.version > PPISP_FILE_VERSION) {
                    return std::unexpected("Unsupported frozen PPISP sidecar version: " +
                                           std::to_string(sidecar_header.version));
                }
                if (!has_flag(sidecar_header.flags, PPISPFileFlags::HAS_CONTROLLER)) {
                    LOG_INFO("Frozen PPISP sidecar '{}' has no controller pool; controller inference will remain disabled",
                             lfs::core::path_to_utf8(sidecar_path));
                    return {};
                }
            }

            PPISPControllerPool::Config config;
            config.lr = params_.optimization.ppisp_controller_lr;

            const int activation_step = params_.optimization.resolved_ppisp_controller_activation_step();
            if (params_.optimization.ppisp_controller_activation_step < 0) {
                params_.optimization.ppisp_controller_activation_step = activation_step;
            }
            int distillation_iters = static_cast<int>(params_.optimization.iterations) - activation_step;
            int num_cameras = ppisp_->num_cameras();

            ppisp_controller_pool_ = std::make_unique<PPISPControllerPool>(num_cameras, distillation_iters, config);

            size_t max_h = 0, max_w = 0;
            for (const auto& cam : train_dataset_->get_cameras()) {
                if (cam) {
                    max_h = std::max(max_h, static_cast<size_t>(cam->image_height()));
                    max_w = std::max(max_w, static_cast<size_t>(cam->image_width()));
                }
            }
            ppisp_controller_pool_->allocate_buffers(max_h, max_w);

            LOG_INFO("PPISP controller pool initialized: num_cameras={}, activation_step={}, lr={:.2e}, max_image={}x{}",
                     num_cameras, activation_step,
                     params_.optimization.ppisp_controller_lr, static_cast<int>(max_h), static_cast<int>(max_w));

            if (import_frozen_sidecar_controller) {
                PPISP loaded_ppisp(1);
                auto loaded_controller = std::make_unique<PPISPControllerPool>(
                    static_cast<int>(sidecar_header.num_cameras),
                    1);
                PPISPFileMetadata metadata;
                if (auto result = load_ppisp_file(sidecar_path, loaded_ppisp, loaded_controller.get(), &metadata); !result) {
                    return std::unexpected(std::format(
                        "Failed to load frozen PPISP controller sidecar '{}': {}",
                        lfs::core::path_to_utf8(sidecar_path),
                        result.error()));
                }
                auto mappings_result = build_ppisp_sidecar_mappings(loaded_ppisp, metadata, sidecar_path);
                if (!mappings_result) {
                    return std::unexpected(mappings_result.error());
                }
                if (const auto error = ppisp_controller_pool_->copy_inference_weights_from(
                        *loaded_controller, mappings_result->camera_mapping);
                    !error.empty()) {
                    return std::unexpected(std::format(
                        "Failed to import frozen PPISP controller weights from '{}': {}",
                        lfs::core::path_to_utf8(sidecar_path),
                        error));
                }
                LOG_INFO("Loaded frozen PPISP controller from '{}' ({} cameras)",
                         lfs::core::path_to_utf8(sidecar_path),
                         loaded_controller->num_cameras());
            }

            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to init PPISP controller pool: {}", e.what()));
        }
    }

    // Compute photometric loss AND gradient manually
    std::expected<std::pair<lfs::core::Tensor, lfs::core::Tensor>, std::string> Trainer::compute_photometric_loss_with_gradient(
        const lfs::core::Tensor& rendered,
        const lfs::core::Tensor& gt_image,
        const lfs::core::param::OptimizationParameters& opt_params) {
        lfs::training::losses::PhotometricLoss::Params params{.lambda_dssim = opt_params.lambda_dssim};
        auto result = photometric_loss_.forward(rendered, gt_image, params);
        if (!result) {
            return std::unexpected(result.error());
        }
        auto [loss_tensor, ctx] = *result;
        return std::make_pair(loss_tensor, ctx.grad_image);
    }

    std::expected<void, std::string> Trainer::validate_masks() {
        const auto& opt = params_.optimization;
        if (opt.mask_mode == lfs::core::param::MaskMode::None) {
            return {};
        }

        size_t alpha_count = 0;
        size_t masks_found = 0;
        for (const auto& cam : train_dataset_->get_cameras()) {
            if (cam && cam->has_alpha())
                ++alpha_count;
            if (cam && cam->has_mask())
                ++masks_found;
        }

        if (opt.use_alpha_as_mask && alpha_count > 0) {
            LOG_INFO("Using alpha channel as mask source ({}/{} cameras){}",
                     alpha_count, train_dataset_->get_cameras().size(),
                     opt.invert_masks ? " (inverted)" : "");
            return {};
        }

        if (masks_found == 0) {
            const auto path_str = lfs::core::path_to_utf8(params_.dataset.data_path);
            if (opt.use_alpha_as_mask) {
                return std::unexpected(std::format(
                    "Mask mode enabled with use_alpha_as_mask but no images have alpha and no mask files found in {}/masks/",
                    path_str));
            }
            return std::unexpected(std::format(
                "Mask mode enabled but no masks found in {}/masks/",
                path_str));
        }

        LOG_INFO("Found {} masks{}", masks_found, opt.invert_masks ? " (inverted)" : "");
        return {};
    }

    
std::expected<Trainer::MaskLossResult, std::string> Trainer::compute_photometric_loss_with_mask(
        const lfs::core::Tensor& rendered,
        const lfs::core::Tensor& gt_image,
        const lfs::core::Tensor& mask,
        const lfs::core::Tensor& alpha,
        const lfs::core::param::OptimizationParameters& opt_params,
        const lfs::core::Tensor* fg_core_2d,
        const lfs::core::Tensor* bg_core_2d) {

    
        // Guard: if mask is invalid or empty, fall back to standard photometric loss.
        // This can happen when a camera has no mask file or alpha extraction failed.
        if (!mask.is_valid() || mask.numel() == 0) {
            static std::atomic<bool> s_logged_mask_error{false};
            if (!s_logged_mask_error.exchange(true)) {
                LOG_ERROR("compute_photometric_loss_with_mask: mask tensor is {} - falling back to unmasked loss. "
                          "Check that mask files exist under <data_path>/masks/ or that alpha-as-mask is correctly configured. "
                          "(This message will not repeat.)",
                          !mask.is_valid() ? "invalid" : "empty (numel=0)");
            }
            auto fallback = compute_photometric_loss_with_gradient(rendered, gt_image, opt_params);
            if (!fallback)
                return std::unexpected(fallback.error());
            return MaskLossResult{.loss = fallback->first, .grad_image = fallback->second, .grad_alpha = {}};
        }

        using namespace lfs::core;
        constexpr float EPSILON = 1e-8f;
        constexpr float ALPHA_CONSISTENCY_WEIGHT = 10.0f;

        const auto mode = opt_params.mask_mode;
        const Tensor mask_2d = mask.ndim() == 3 ? mask.squeeze(0) : mask;

        Tensor loss, grad, grad_alpha;

        if (mode == param::MaskMode::Segment || mode == param::MaskMode::Ignore) {
            if (opt_params.lambda_dssim > 0.0f) {
                // Use FUSED masked L1+SSIM kernel
                auto [loss_tensor, ctx] = lfs::training::kernels::masked_fused_l1_ssim_forward(
                    rendered, gt_image, mask_2d, opt_params.lambda_dssim, masked_fused_workspace_);

                grad = lfs::training::kernels::masked_fused_l1_ssim_backward(ctx, masked_fused_workspace_);
                loss = loss_tensor;

                // Squeeze gradient to match input dimensions
                if (grad.ndim() == 4 && rendered.ndim() == 3) {
                    grad = grad.squeeze(0);
                }
            } else {
                // Pure L1 with mask (no SSIM)
                const Tensor mask_3d = mask_2d.unsqueeze(0);
                const Tensor mask_sum = mask_2d.sum() * static_cast<float>(rendered.shape()[0]) + EPSILON;
                const Tensor diff = rendered - gt_image;
                const Tensor masked_l1 = (diff.abs() * mask_3d).sum() / mask_sum;
                const Tensor sign_diff = diff.sign();
                grad = sign_diff * mask_3d / mask_sum;
                loss = masked_l1;
            }

            // Segment: opacity penalty for background
            if (mode == param::MaskMode::Segment && alpha.is_valid()) {
                const Tensor alpha_2d = alpha.ndim() == 3 ? alpha.squeeze(0) : alpha;
                const Tensor bg_mask = Tensor::full(mask_2d.shape(), 1.0f, mask_2d.device()) - mask_2d;
                const Tensor penalty_weights = bg_mask.pow(opt_params.mask_opacity_penalty_power);
                const Tensor penalty = (alpha_2d * penalty_weights).mean() * opt_params.mask_opacity_penalty_weight;

                const float inv_pixels = opt_params.mask_opacity_penalty_weight / static_cast<float>(alpha_2d.numel());
                grad_alpha = penalty_weights * inv_pixels;
                loss = loss + penalty;
            }

        } else if (mode == param::MaskMode::AlphaConsistent) {
            // Standard photometric loss
            const lfs::training::losses::PhotometricLoss::Params params{.lambda_dssim = opt_params.lambda_dssim};
            auto result = photometric_loss_.forward(rendered, gt_image, params);
            if (!result) {
                return std::unexpected(result.error());
            }
            auto [photo_loss, ctx] = *result;
            loss = photo_loss;
            grad = ctx.grad_image;

            // Alpha should match mask
            if (alpha.is_valid()) {
                const Tensor alpha_2d = alpha.ndim() == 3 ? alpha.squeeze(0) : alpha;
                const Tensor alpha_loss = (alpha_2d - mask_2d).abs().mean() * ALPHA_CONSISTENCY_WEIGHT;
                loss = loss + alpha_loss;
                grad_alpha = (alpha_2d - mask_2d).sign() * (ALPHA_CONSISTENCY_WEIGHT / static_cast<float>(alpha_2d.numel()));
            }
        } else if (mode == param::MaskMode::HardMatting || mode == param::MaskMode::SoftMatting) {
            // =====================================================================
            // QUALITY-FIRST MATTING (robust to imperfect binary masks)
            //
            // Motivation:
            // - With imperfect binary masks, "outside == background" is often wrong in some views.
            // - If we enforce BG too hard, we punch holes / kill details.
            // - If we ignore BG entirely, floaters grow.
            //
            // Strategy:
            // - Use a conservative trimap when cores are available:
            //     FG core: strong supervision (alpha->1 + photometric)
            //     BG core: light supervision (alpha->0 gated + photometric)
            //     Uncertain band: weak photometric (avoid detail collapse) and no alpha forcing.
            // - Avoid scaling the whole photometric loss/grad by the alpha penalty (stability).
            // =====================================================================

            const bool use_cores = (mode == param::MaskMode::SoftMatting);

            if (use_cores) {
                if (!fg_core_2d || !bg_core_2d || !fg_core_2d->is_valid() || !bg_core_2d->is_valid()) {
                    return std::unexpected(
                        "MaskMode::SoftMatting requires fg_core_2d and bg_core_2d (eroded cores).");
                }
            }

            // Conservative defaults
            constexpr float kHardBgPhotometricRatio = 0.10f;
            constexpr float kSoftBgPhotometricRatio = 0.10f;
            constexpr float kSoftUncertainFgRatio = 0.10f; // weak weight on FG band outside FG core
            constexpr float kGlobalAnchor = 0.02f;

            // Alpha penalty weights (quality-first)
            constexpr float kIn = 1.00f;  // FG: penalize transparency (1-alpha)
            constexpr float kOut = 0.25f; // BG: penalize opacity (alpha), weaker than FG

            // BG gating thresholds in alpha space
            constexpr float kBgGateLo = 0.20f; // alpha <= lo -> full BG penalty
            constexpr float kBgGateHi = 0.60f; // alpha >= hi -> zero BG penalty

            // Strict binary mask {0,1}
            const Tensor mask_bin = mask_2d.gt(0.5f).to(DataType::Float32);

            // Cached ones tensor (same shape/device as mask)
            static thread_local Tensor ones_2d;
            if (ones_2d.is_empty() || ones_2d.shape() != mask_2d.shape() || ones_2d.device() != mask_2d.device()) {
                ones_2d = Tensor::full(mask_2d.shape(), 1.0f, mask_2d.device());
            }

            Tensor weight_map;

            if (use_cores) {
                // Use conservative cores computed in Camera (erode-cross)
                const Tensor fg_core = fg_core_2d->gt(0.5f).to(DataType::Float32);
                const Tensor bg_core = bg_core_2d->gt(0.5f).to(DataType::Float32);

                // Uncertain FG band: mask_bin - fg_core, clamped to >= 0 using gt(0)
                Tensor fg_band = (mask_bin - fg_core);
                fg_band = fg_band * fg_band.gt(0.0f).to(DataType::Float32);

                // Photometric weight: FG core strong, FG band weak, BG core light, + tiny global anchor
                weight_map = fg_core + fg_band * kSoftUncertainFgRatio +
                                bg_core * kSoftBgPhotometricRatio + ones_2d * kGlobalAnchor;
            } else {
                // HardMatting: dense FG with light BG + tiny global anchor
                const Tensor bg_mask = ones_2d - mask_bin;
                weight_map = mask_bin + bg_mask * kHardBgPhotometricRatio + ones_2d * kGlobalAnchor;
            }

            // Photometric loss
            // For SoftMatting, prefer L1-only (SSIM windows can mix across mask boundaries and create artifacts).
            const bool allow_ssim = (!use_cores) && (opt_params.lambda_dssim > 0.0f);

            if (allow_ssim) {
                auto [loss_tensor, ctx] = lfs::training::kernels::masked_fused_l1_ssim_forward(
                    rendered, gt_image, weight_map, opt_params.lambda_dssim, masked_fused_workspace_);

                grad = lfs::training::kernels::masked_fused_l1_ssim_backward(ctx, masked_fused_workspace_);
                loss = loss_tensor;

                if (grad.ndim() == 4 && rendered.ndim() == 3) {
                    grad = grad.squeeze(0);
                }
            } else {
                // Pure L1 with weighted mask
                const Tensor weight_exp = weight_map.unsqueeze(0).expand({static_cast<int>(rendered.shape()[0]),
                                                                            static_cast<int>(mask_2d.shape()[0]),
                                                                            static_cast<int>(mask_2d.shape()[1])});
                const Tensor weight_sum = weight_exp.sum() + EPSILON;

                const Tensor diff = rendered - gt_image;
                const Tensor l1 = diff.abs();
                const Tensor sign = diff.sign();

                loss = (l1 * weight_exp).sum() / weight_sum;
                grad = sign * weight_exp / weight_sum;
            }

            // Alpha regularization (additive; do NOT scale photometric loss/grad)
            if (alpha.is_valid()) {
                Tensor alpha_2d = (alpha.ndim() == 3) ? alpha.squeeze(0) : alpha;

                const float penalty_weight = opt_params.mask_opacity_penalty_weight;
                if (penalty_weight > 0.0f) {

                    // If cores exist, use them for alpha penalty even in HardMatting (more robust).
                    Tensor fg_alpha_w;
                    Tensor bg_alpha_w;

                    if (use_cores) {
                        fg_alpha_w = fg_core_2d->gt(0.5f).to(DataType::Float32);
                        bg_alpha_w = bg_core_2d->gt(0.5f).to(DataType::Float32);
                    } else if (fg_core_2d && bg_core_2d && fg_core_2d->is_valid() && bg_core_2d->is_valid()) {
                        fg_alpha_w = fg_core_2d->gt(0.5f).to(DataType::Float32);
                        bg_alpha_w = bg_core_2d->gt(0.5f).to(DataType::Float32);
                    } else {
                        fg_alpha_w = mask_bin;
                        bg_alpha_w = ones_2d - mask_bin;
                    }

                    const Tensor ones_alpha = Tensor::full(alpha_2d.shape(), 1.0f, alpha_2d.device());

                    // Normalize per-region
                    const Tensor fg_norm = fg_alpha_w.sum() + EPSILON;
                    const Tensor bg_norm = bg_alpha_w.sum() + EPSILON;

                    // FG: (1 - alpha)
                    const Tensor inside_penalty = ((ones_alpha - alpha_2d) * fg_alpha_w).sum() / fg_norm;

                    // BG gating: bg_gate in [0,1] using only gt() (no clamp dependency)
                    Tensor gate = (alpha_2d - kBgGateLo) / (kBgGateHi - kBgGateLo);

                    // clamp to >= 0
                    gate = gate * gate.gt(0.0f).to(DataType::Float32);
                    // clamp to <= 1
                    const Tensor over = gate.gt(1.0f).to(DataType::Float32);
                    gate = gate - (gate - 1.0f) * over;

                    const Tensor bg_gate = ones_alpha - gate;

                    // BG: alpha, but gated
                    const Tensor outside_penalty = ((alpha_2d * bg_gate) * bg_alpha_w).sum() / bg_norm;

                    const Tensor total_penalty = inside_penalty * kIn + outside_penalty * kOut;

                    const float scale = penalty_weight * ALPHA_CONSISTENCY_WEIGHT;

                    loss = loss + total_penalty * scale;

                    // Gradient w.r.t alpha (ignore d(bg_gate)/d(alpha) for stability)
                    grad_alpha =
                        ((bg_alpha_w * bg_gate) * (kOut * scale) / bg_norm) -
                        (fg_alpha_w * (kIn * scale) / fg_norm);
                }
            }

        // MODO CHATI

        /*} else if (mode == param::MaskMode::FocusedSegment) {
            // FocusedSegment v2:
            // - Photometric loss really uses a weighted mask (not post-hoc gradient surgery)
            // - Darkness boost only reinforces the subject region
            // - Alpha penalty is additive and region-normalized for stability
            // - Background is allowed, but with a much smaller budget than foreground

            constexpr float kBgWeight = 0.05f;      // keep some background supervision
            constexpr float kDarknessBoost = 2.0f;  // reinforce dark subject regions
            constexpr float kAlphaFgWeight = 2.25f; // push solidity inside the subject
            constexpr float kAlphaBgWeight = 0.35f; // weaken opacity outside, but more gently
            constexpr float kMinMean = 1e-4f;

            const Tensor ones_2d = Tensor::full(mask_2d.shape(), 1.0f, mask_2d.device());

            // Use the incoming mask as soft FG prior if available.
            // This lets mask confidence shape where we spend resources.
            const Tensor fg_mask = mask_2d;
            const Tensor bg_mask = ones_2d - fg_mask;

            // Foreground-only darkness boost:
            // dark subject pixels get more photometric pressure,
            // but dark background does NOT steal optimization budget.
            Tensor fg_weight = fg_mask;
            if (kDarknessBoost > 0.0f) {
                const bool chw = (gt_image.ndim() == 3 && gt_image.shape()[0] == 3);
                const Tensor r = chw ? gt_image.slice(0, 0, 1).squeeze(0) : gt_image.slice(2, 0, 1).squeeze(2);
                const Tensor g = chw ? gt_image.slice(0, 1, 2).squeeze(0) : gt_image.slice(2, 1, 2).squeeze(2);
                const Tensor b = chw ? gt_image.slice(0, 2, 3).squeeze(0) : gt_image.slice(2, 2, 3).squeeze(2);

                const Tensor brightness = r * 0.299f + g * 0.587f + b * 0.114f;
                const Tensor darkness = ones_2d - brightness;

                // Only boost inside FG, never in BG
                fg_weight = fg_weight * (ones_2d + darkness * kDarknessBoost);
            }

            // Final spatial budget:
            // FG gets almost all the attention; BG is still seen, just with low weight.
            Tensor weight_map = fg_weight + bg_mask * kBgWeight;

            // Normalize to keep training scale stable across scenes/views
            const float weight_mean = weight_map.mean().item<float>();
            weight_map = weight_map * (1.0f / std::max(weight_mean, kMinMean));

            // Photometric loss with REAL weighted supervision
            if (opt_params.lambda_dssim > 0.0f) {
                auto [loss_tensor, ctx] = lfs::training::kernels::masked_fused_l1_ssim_forward(
                    rendered, gt_image, weight_map, opt_params.lambda_dssim, masked_fused_workspace_);

                grad = lfs::training::kernels::masked_fused_l1_ssim_backward(ctx, masked_fused_workspace_);
                loss = loss_tensor;

                if (grad.ndim() == 4 && rendered.ndim() == 3) {
                    grad = grad.squeeze(0);
                }
            } else {
                // Pure weighted L1 fallback
                const Tensor weight_3d =
                    (rendered.ndim() == 3 && rendered.shape()[0] == 3)
                        ? weight_map.unsqueeze(0)
                        : weight_map.unsqueeze(2);

                const Tensor diff = rendered - gt_image;
                const Tensor abs_diff = diff.abs();
                const Tensor sign_diff = diff.sign();
                const Tensor weight_sum = weight_3d.sum() + EPSILON;

                loss = (abs_diff * weight_3d).sum() / weight_sum;
                grad = sign_diff * weight_3d / weight_sum;
            }

            // Additive alpha penalty:
            // - inside FG -> push alpha up (solid subject)
            // - outside FG -> push alpha down (less external competition)
            //
            // Additive + region-normalized is much more stable than the current multiplicative form.
            if (alpha.is_valid() && opt_params.mask_opacity_penalty_weight > 0.0f) {
                const Tensor alpha_2d = (alpha.ndim() == 3) ? alpha.squeeze(0) : alpha;
                const Tensor ones_alpha = Tensor::full(alpha_2d.shape(), 1.0f, alpha_2d.device());

                const Tensor fg_alpha_w = fg_mask.pow(opt_params.mask_opacity_penalty_power);
                const Tensor bg_alpha_w = bg_mask.pow(opt_params.mask_opacity_penalty_power);

                const Tensor fg_norm = fg_alpha_w.sum() + EPSILON;
                const Tensor bg_norm = bg_alpha_w.sum() + EPSILON;

                const Tensor inside_penalty = ((ones_alpha - alpha_2d) * fg_alpha_w).sum() / fg_norm;
                const Tensor outside_penalty = (alpha_2d * bg_alpha_w).sum() / bg_norm;

                const float scale = opt_params.mask_opacity_penalty_weight;

                loss = loss + inside_penalty * (scale * kAlphaFgWeight) + outside_penalty * (scale * kAlphaBgWeight);

                grad_alpha =
                    (bg_alpha_w * (scale * kAlphaBgWeight) / bg_norm) -
                    (fg_alpha_w * (scale * kAlphaFgWeight) / fg_norm);
            }
        }*/

        // MODO ORIG    
           
        
        } else if (mode == param::MaskMode::FocusedSegment) {
            // FocusedSegment: full-image photometric loss (identical quality to None mode) with
            // spatial gradient weighting to focus learning on the masked foreground object.
            //
            // Design goals vs other mask modes:
            //   - None:         best object quality but no background removal
            //   - Segment:      hard mask truncates gradient, damages object reconstruction
            //   - FocusedSegment: full gradient signal everywhere, but FG weighted 20x over BG.
            //                   Darkness bonus reinforces opacity on dark pixels where L1 is weak.
            //                   Alpha penalty directly pushes Gaussians toward opaque inside mask
            //                   and transparent outside, independent of photometric loss magnitude.
            //
            // Training schedule (applied in train_step before calling this function):
            //   0-15%:  mask_mode = None       — geometry consolidation, no mask interference
            //   15-80%: FocusedSegment active  — spatial weighting + alpha pressure
            //   80-100%: opacity_penalty = 0   — allow natural transparencies to emerge freely

            constexpr float kBgWeight = 0.05f;     // BG gradient weight relative to FG (1.0)
            constexpr float kDarknessBoost = 2.0f; // extra FG weight on dark pixels; 0 = disabled
            constexpr float kAlphaFgWeight = 3.0f; // grad_alpha pressure to push FG alpha -> 1
            constexpr float kAlphaBgWeight = 1.5f; // grad_alpha pressure to push BG alpha -> 0

            const Tensor bg_mask = Tensor::full(mask_2d.shape(), 1.0f, mask_2d.device()) - mask_2d;

            // Step 1: full-image L1+SSIM forward — identical to None mode.
            // Produces correct grad_image and populates fused_workspace ssim_map for
            // pixel-error-based densification. No gradient approximation here.
            lfs::training::losses::PhotometricLoss::Params params{.lambda_dssim = opt_params.lambda_dssim};
            auto full_result = photometric_loss_.forward(rendered, gt_image, params);
            if (!full_result)
                return std::unexpected(full_result.error());
            auto [full_loss, ctx] = *full_result;
            grad = ctx.grad_image;
            loss = full_loss;

            // Step 2: spatial weight map — FG=1.0, BG=kBgWeight.
            // Darkness bonus applied to FG only (Rec.601 perceptual luminance from GT, not rendered).
            // Using GT keeps the weight map stable across iterations — rendered changes every step.
            // BG stays flat at kBgWeight regardless of darkness, avoiding spurious BG gradient boosts.
            Tensor weight_map;
            if (kDarknessBoost > 0.0f) {
                const bool chw = (gt_image.ndim() == 3 && gt_image.shape()[0] == 3);
                const Tensor r = chw ? gt_image.slice(0, 0, 1).squeeze(0) : gt_image.slice(2, 0, 1).squeeze(2);
                const Tensor g = chw ? gt_image.slice(0, 1, 2).squeeze(0) : gt_image.slice(2, 1, 2).squeeze(2);
                const Tensor b = chw ? gt_image.slice(0, 2, 3).squeeze(0) : gt_image.slice(2, 2, 3).squeeze(2);
                const Tensor brightness = r * 0.299f + g * 0.587f + b * 0.114f;
                const Tensor darkness = Tensor::full(brightness.shape(), 1.0f, brightness.device()) - brightness;
                weight_map = mask_2d * (Tensor::full(darkness.shape(), 1.0f, darkness.device()) + darkness * kDarknessBoost) + bg_mask * kBgWeight;
            } else {
                weight_map = mask_2d + bg_mask * kBgWeight;
            }

            // Normalize weight_map by its mean so the global gradient magnitude stays
            // comparable to None mode — prevents scale drift when mask size varies across scenes.
            const float weight_mean = weight_map.mean().item<float>();
            weight_map = weight_map * (1.0f / std::max(weight_mean, 1e-4f));

            // Step 3: apply spatial weights to gradient.
            // Multiplying grad post-hoc is exact for L1 (pixel-wise) and a good approximation
            // for SSIM (window-based). In practice this outperforms discarding SSIM gradient entirely.
            const Tensor weight_3d = (rendered.ndim() == 3 && rendered.shape()[0] == 3)
                                         ? weight_map.unsqueeze(0)
                                         : weight_map.unsqueeze(2);
            grad = grad * weight_3d;

            // Scale scalar loss to reflect FG-focused weighting.
            // This ensures the logged loss is representative of FG reconstruction quality,
            // not diluted by the large BG area. fg_pixels is cached to avoid a second GPU sync.
            const float total_pixels = static_cast<float>(mask_2d.numel());
            const float fg_pixels = std::max(mask_2d.sum().item<float>(), 1.0f);
            const float bg_pixels = std::max(total_pixels - fg_pixels, 1.0f);
            loss = loss * (fg_pixels / std::max(total_pixels * weight_mean, 1e-6f));

            // Step 4: alpha pressure via grad_alpha.
            // NOTE: modifying the scalar `loss` does NOT affect Gaussian parameters — only
            // grad_image and grad_alpha propagate through rasterize_backward to the optimizer.
            // Alpha pressure is therefore applied purely through grad_alpha.
            //
            // Each zone (FG/BG) is normalized independently by its pixel count so that pressure
            // is balanced regardless of how much of the image the mask occupies.
            //
            // Sign convention (gradient descent: param -= lr * grad):
            //   FG: negative grad_alpha  -> alpha_raw increases -> rendered alpha approaches 1
            //   BG: positive grad_alpha  -> alpha_raw decreases -> rendered alpha approaches 0
            if (alpha.is_valid() && opt_params.mask_opacity_penalty_weight > 0.0f) {
                const float w = opt_params.mask_opacity_penalty_weight;
                const Tensor alpha_2d = alpha.ndim() == 3 ? alpha.squeeze(0) : alpha;

                grad_alpha = bg_mask * (w * kAlphaBgWeight / bg_pixels)    // BG: push alpha -> 0
                             - mask_2d * (w * kAlphaFgWeight / fg_pixels); // FG: push alpha -> 1
            }
        }


        
        // MODO B

        /*
        } else if (mode == param::MaskMode::FocusedSegment) {
            constexpr float kBgWeight = 0.05f;
            constexpr float kDarknessBoost = 1.5f;
            constexpr float kAlphaFgWeight = 1.5f;
            constexpr float kAlphaBgWeight = 0.3f;

            const Tensor bg_mask = Tensor::full(mask_2d.shape(), 1.0f, mask_2d.device()) - mask_2d;

            // Step 1: full-image SSIM forward only — populates fused_workspace for densification error map.
            // Gradient is NOT used here (SSIM weighting post-hoc is mathematically incorrect).
            {
                lfs::training::losses::PhotometricLoss::Params params{.lambda_dssim = opt_params.lambda_dssim};
                auto full_result = photometric_loss_.forward(rendered, gt_image, params);
                if (!full_result)
                    return std::unexpected(full_result.error());
                // discard loss and grad — only ssim_map in fused_workspace is needed for densification
            }

            // Step 2: weighted L1 — mathematically correct to apply spatial weights post-hoc.
            const Tensor diff = rendered - gt_image;
            grad = diff.sign().contiguous(); // contiguous: materialize before spatial weighting
            const Tensor l1_map = diff.abs();
            const Tensor l1_map_reduced = (gt_image.ndim() == 3 && gt_image.shape()[0] == 3)
                                          ? l1_map.mean({0}, false)
                                          : l1_map.mean({2}, false);
            const float fg_sum = mask_2d.sum().item<float>();
            loss = (l1_map_reduced * mask_2d).sum() / std::max(fg_sum, 1e-6f);

            // Step 3: spatial weight map — FG=1.0, BG=kBgWeight
            Tensor weight_map = mask_2d + bg_mask * kBgWeight;

            // Step 4: darkness bonus (perceptual luminance Rec.601, from GT not rendered).
            // Normalized by mean to keep global gradient scale stable across scenes.
            if (kDarknessBoost > 0.0f) {
                const bool chw = (gt_image.ndim() == 3 && gt_image.shape()[0] == 3);
                const Tensor r = chw ? gt_image.slice(0, 0, 1).squeeze(0) : gt_image.slice(2, 0, 1).squeeze(2);
                const Tensor g = chw ? gt_image.slice(0, 1, 2).squeeze(0) : gt_image.slice(2, 1, 2).squeeze(2);
                const Tensor b = chw ? gt_image.slice(0, 2, 3).squeeze(0) : gt_image.slice(2, 2, 3).squeeze(2);
                const Tensor brightness = r * 0.299f + g * 0.587f + b * 0.114f;
                const Tensor darkness = Tensor::full(brightness.shape(), 1.0f, brightness.device()) - brightness;
                weight_map = weight_map * (Tensor::full(darkness.shape(), 1.0f, darkness.device()) + darkness * kDarknessBoost);
            }

            // Normalize by mean — keeps global gradient magnitude comparable to None mode
            const float weight_mean = weight_map.mean().item<float>();
            weight_map = weight_map * (1.0f / std::max(weight_mean, 1e-4f));

            // Step 5: apply weight map to L1 gradient
            const Tensor weight_3d = (gt_image.ndim() == 3 && gt_image.shape()[0] == 3)
                                         ? weight_map.unsqueeze(0)
                                         : weight_map.unsqueeze(2);
            grad = grad * weight_3d;
            const float wm = weight_map.mean().item<float>();
            loss = loss * wm;

            // Step 6: alpha penalty — multiplicative style from old codebase.
            // Pressure scales with photometric loss magnitude so it never dominates when converged.
            if (alpha.is_valid() && opt_params.mask_opacity_penalty_weight > 0.0f) {
                const float w = opt_params.mask_opacity_penalty_weight;
                const Tensor alpha_2d = alpha.ndim() == 3 ? alpha.squeeze(0) : alpha;
                const Tensor ones = Tensor::full(alpha_2d.shape(), 1.0f, alpha_2d.device());
                const float n_pixels = static_cast<float>(alpha_2d.numel());

                const Tensor fg_weights = mask_2d.pow(opt_params.mask_opacity_penalty_power);
                const Tensor fg_penalty = ((ones - alpha_2d) * fg_weights).sum() / n_pixels * (w * kAlphaFgWeight);

                const Tensor bg_weights = bg_mask.pow(opt_params.mask_opacity_penalty_power);
                const Tensor bg_penalty = (alpha_2d * bg_weights).sum() / n_pixels * (w * kAlphaBgWeight);

                const Tensor total_penalty = fg_penalty + bg_penalty;

                loss = loss * (Tensor::full({1}, 1.0f, loss.device()) + total_penalty) + total_penalty * 1e-2f;

                grad_alpha = bg_weights * (w * kAlphaBgWeight / n_pixels) -
                             fg_weights * (w * kAlphaFgWeight / n_pixels);
            }
        }*/

        
        // MODO C
        /*
        } else if (mode == param::MaskMode::FocusedSegment) {
            constexpr float kBgWeight      = 0.05f;
            constexpr float kDarknessBoost = 1.5f;
            constexpr float kAlphaFgWeight = 1.5f;
            constexpr float kAlphaBgWeight = 0.3f;

            const Tensor bg_mask = Tensor::full(mask_2d.shape(), 1.0f, mask_2d.device()) - mask_2d;

            // Step 1: full-image L1+SSIM forward — populates fused_workspace for densification.
            // SSIM grad used as-is (full-image, no spatial weighting — mathematically correct).
            // L1 grad discarded and recomputed with spatial weighting below.
            lfs::training::losses::PhotometricLoss::Params params{.lambda_dssim = opt_params.lambda_dssim};
            auto full_result = photometric_loss_.forward(rendered, gt_image, params);
            if (!full_result) return std::unexpected(full_result.error());
            auto [full_loss, ctx] = *full_result;
            const Tensor ssim_grad = ctx.grad_image.contiguous(); // force independent copy

            // Step 2: weighted L1 — spatial weights applied post-hoc (correct for pixel-wise loss).
            const Tensor diff = rendered - gt_image;
            Tensor l1_grad    = diff.sign().contiguous(); // contiguous: materialize before spatial weighting
            const Tensor l1_map = diff.abs();
            const Tensor l1_map_reduced = (gt_image.ndim() == 3 && gt_image.shape()[0] == 3)
                                          ? l1_map.mean({0}, false)
                                          : l1_map.mean({2}, false);
            const float fg_sum = mask_2d.sum().item<float>();
            loss = (l1_map_reduced * mask_2d).sum() / std::max(fg_sum, 1e-6f);

            // Step 3: spatial weight map — FG=1.0, BG=kBgWeight
            Tensor weight_map = mask_2d + bg_mask * kBgWeight;

            // Step 4: darkness bonus (perceptual luminance Rec.601, from GT not rendered).
            // Normalized by mean to keep global gradient scale stable.
            if (kDarknessBoost > 0.0f) {
                const bool chw = (gt_image.ndim() == 3 && gt_image.shape()[0] == 3);
                const Tensor r = chw ? gt_image.slice(0, 0, 1).squeeze(0) : gt_image.slice(2, 0, 1).squeeze(2);
                const Tensor g = chw ? gt_image.slice(0, 1, 2).squeeze(0) : gt_image.slice(2, 1, 2).squeeze(2);
                const Tensor b = chw ? gt_image.slice(0, 2, 3).squeeze(0) : gt_image.slice(2, 2, 3).squeeze(2);
                const Tensor brightness = r * 0.299f + g * 0.587f + b * 0.114f;
                const Tensor darkness   = Tensor::full(brightness.shape(), 1.0f, brightness.device()) - brightness;
                weight_map = weight_map * (Tensor::full(darkness.shape(), 1.0f, darkness.device()) + darkness * kDarknessBoost);
            }

            const float weight_mean = weight_map.mean().item<float>();
            weight_map = weight_map * (1.0f / std::max(weight_mean, 1e-4f));

            // Step 5: apply spatial weights to L1 gradient only
            const Tensor weight_3d = (gt_image.ndim() == 3 && gt_image.shape()[0] == 3)
                                     ? weight_map.unsqueeze(0)
                                     : weight_map.unsqueeze(2);
            l1_grad = l1_grad * weight_3d;

            // Step 6: combine weighted L1 + unweighted SSIM
            // (1-lambda)*L1_weighted + lambda*SSIM_full — each term mathematically correct
            const float lambda = opt_params.lambda_dssim;
            grad = l1_grad * (1.0f - lambda) + ssim_grad * lambda;
            const float wm = weight_map.mean().item<float>();
            loss = l1_loss * (1.0f - lambda) * wm + full_loss * lambda;

            // Step 7: alpha penalty — multiplicative, scales with loss magnitude
            if (alpha.is_valid() && opt_params.mask_opacity_penalty_weight > 0.0f) {
                const float  w        = opt_params.mask_opacity_penalty_weight;
                const Tensor alpha_2d = alpha.ndim() == 3 ? alpha.squeeze(0) : alpha;
                const Tensor ones     = Tensor::full(alpha_2d.shape(), 1.0f, alpha_2d.device());
                const float  n_pixels = static_cast<float>(alpha_2d.numel());

                const Tensor fg_weights = mask_2d.pow(opt_params.mask_opacity_penalty_power);
                const Tensor fg_penalty = ((ones - alpha_2d) * fg_weights).sum() / n_pixels * (w * kAlphaFgWeight);

                const Tensor bg_weights = bg_mask.pow(opt_params.mask_opacity_penalty_power);
                const Tensor bg_penalty = (alpha_2d * bg_weights).sum() / n_pixels * (w * kAlphaBgWeight);

                const Tensor total_penalty = fg_penalty + bg_penalty;

                loss = loss * (Tensor::full({1}, 1.0f, loss.device()) + total_penalty) + total_penalty * 1e-2f;

                grad_alpha = bg_weights * (w * kAlphaBgWeight / n_pixels) -
                             fg_weights * (w * kAlphaFgWeight / n_pixels);
            }
        }*/
        
        else {
            auto fallback = compute_photometric_loss_with_gradient(rendered, gt_image, opt_params);
            if (!fallback) {
                return std::unexpected(fallback.error());
            }
            return MaskLossResult{.loss = fallback->first, .grad_image = fallback->second, .grad_alpha = {}};
        }

        return MaskLossResult{.loss = loss, .grad_image = grad, .grad_alpha = grad_alpha};
    }

    // Returns GPU tensor for loss - NO SYNC!
    std::expected<lfs::core::Tensor, std::string> Trainer::compute_scale_reg_loss(
        lfs::core::SplatData& splatData,
        AdamOptimizer& optimizer,
        const lfs::core::param::OptimizationParameters& opt_params) {
        lfs::training::losses::ScaleRegularization::Params params{.weight = opt_params.scale_reg};
        return lfs::training::losses::ScaleRegularization::forward(splatData.scaling_raw(), optimizer.get_grad(ParamType::Scaling), params);
    }

    // Returns GPU tensor for loss - NO SYNC!
    std::expected<lfs::core::Tensor, std::string> Trainer::compute_opacity_reg_loss(
        lfs::core::SplatData& splatData,
        AdamOptimizer& optimizer,
        const lfs::core::param::OptimizationParameters& opt_params) {
        lfs::training::losses::OpacityRegularization::Params params{.weight = opt_params.opacity_reg};
        return lfs::training::losses::OpacityRegularization::forward(splatData.opacity_raw(), optimizer.get_grad(ParamType::Opacity), params);
    }

    std::expected<std::pair<lfs::core::Tensor, SparsityLossContext>, std::string>
    Trainer::compute_sparsity_loss_forward(const int iter, const lfs::core::SplatData& splat_data) {
        if (!sparsity_optimizer_ || !sparsity_optimizer_->should_apply_loss(iter)) {
            auto zero = lfs::core::Tensor::zeros({1}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
            return std::make_pair(std::move(zero), SparsityLossContext{});
        }

        if (!sparsity_optimizer_->is_initialized()) {
            if (auto result = sparsity_optimizer_->initialize(splat_data.opacity_raw()); !result) {
                return std::unexpected(result.error());
            }
            LOG_DEBUG("Sparsity optimizer initialized at iteration {}", iter);
        }

        return sparsity_optimizer_->compute_loss_forward(splat_data.opacity_raw());
    }

    std::expected<void, std::string> Trainer::handle_sparsity_update(const int iter, lfs::core::SplatData& splat_data) {
        if (!sparsity_optimizer_ || !sparsity_optimizer_->should_update(iter)) {
            return {};
        }
        return sparsity_optimizer_->update_state(splat_data.opacity_raw());
    }

    std::expected<void, std::string> Trainer::apply_sparsity_pruning(const int iter, lfs::core::SplatData& splat_data) {
        if (!sparsity_optimizer_ || !sparsity_optimizer_->should_prune(iter)) {
            return {};
        }

        auto mask_result = sparsity_optimizer_->get_prune_mask(splat_data.opacity_raw());
        if (!mask_result) {
            return std::unexpected(mask_result.error());
        }

        const int n_before = static_cast<int>(splat_data.size());
        strategy_->remove_gaussians(*mask_result);
        const int n_after = static_cast<int>(splat_data.size());

        LOG_INFO("Sparsity pruning: {} -> {} Gaussians ({}% reduction)",
                 n_before, n_after, static_cast<int>(100.0f * (n_before - n_after) / n_before));

        sparsity_optimizer_.reset();
        return {};
    }

    Trainer::Trainer(std::shared_ptr<CameraDataset> dataset,
                     std::unique_ptr<IStrategy> strategy,
                     std::optional<std::tuple<std::vector<std::string>, std::vector<std::string>>> provided_splits)
        : base_dataset_(std::move(dataset)),
          strategy_(std::move(strategy)),
          provided_splits_(std::move(provided_splits)) {
        // Check CUDA availability
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess || device_count == 0) {
            throw std::runtime_error("CUDA is not available – aborting.");
        }

        cudaStreamCreateWithFlags(&callback_stream_, cudaStreamNonBlocking);

        LOG_DEBUG("Trainer constructed with {} cameras", base_dataset_->get_cameras().size());
    }

    Trainer::Trainer(lfs::core::Scene& scene)
        : scene_(&scene) {
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess || device_count == 0) {
            throw std::runtime_error("CUDA is not available – aborting.");
        }

        cudaStreamCreateWithFlags(&callback_stream_, cudaStreamNonBlocking);

        if (!scene.hasTrainingData()) {
            throw std::runtime_error("Scene has no cameras");
        }

        LOG_DEBUG("Trainer constructed from Scene with {} cameras", scene.getAllCameras().size());
    }

    std::expected<void, std::string> Trainer::initialize(const lfs::core::param::TrainingParameters& params) {
        // Thread-safe initialization using mutex
        std::lock_guard<std::mutex> lock(init_mutex_);

        // Check again after acquiring lock (double-checked locking pattern)
        if (initialized_.load()) {
            LOG_INFO("Re-initializing trainer with new parameters");
            // Clean up existing state for re-initialization
            cleanup();
        }

        LOG_INFO("Initializing trainer with {} iterations", params.optimization.iterations);

        try {
            params_ = params;

            // Create DatasetConfig for lfs::training::CameraDataset
            lfs::training::DatasetConfig dataset_config;
            dataset_config.resize_factor = params.dataset.resize_factor;
            dataset_config.max_width = params.dataset.max_width;
            dataset_config.test_every = params.dataset.test_every;

            // Get source cameras from Scene nodes or base_dataset_
            std::vector<std::shared_ptr<lfs::core::Camera>> source_cameras;
            if (scene_) {
                source_cameras = scene_->getActiveCameras();
                if (source_cameras.empty()) {
                    return std::unexpected("Scene has no active cameras enabled for training");
                }
            } else if (base_dataset_) {
                source_cameras = base_dataset_->get_cameras();
            } else {
                return std::unexpected("No camera source available");
            }

            total_cameras_count_ = source_cameras.size();

            // Handle dataset split based on evaluation flag
            if (params.optimization.enable_eval) {
                // Create train/val split
                train_dataset_ = std::make_shared<CameraDataset>(
                    source_cameras, dataset_config, CameraDataset::Split::TRAIN,
                    provided_splits_ ? std::make_optional(std::get<0>(*provided_splits_)) : std::nullopt);
                val_dataset_ = std::make_shared<CameraDataset>(
                    source_cameras, dataset_config, CameraDataset::Split::VAL,
                    provided_splits_ ? std::make_optional(std::get<1>(*provided_splits_)) : std::nullopt);

                LOG_INFO("Created train/val split: {} train, {} val images",
                         train_dataset_->size(),
                         val_dataset_->size());
            } else {
                // Use all images for training
                train_dataset_ = std::make_shared<CameraDataset>(
                    source_cameras, dataset_config, CameraDataset::Split::ALL);
                val_dataset_ = nullptr;

                LOG_INFO("Using all {} images for training (no evaluation)",
                         train_dataset_->size());
            }

            train_dataset_size_ = train_dataset_->size();

            // If using Scene mode and no strategy yet, create one
            if (scene_ && !strategy_) {
                auto* model = scene_->getTrainingModel();
                if (!model) {
                    return std::unexpected("Scene has no training model set");
                }

                auto result = StrategyFactory::instance().create(params.optimization.strategy, *model);
                if (!result) {
                    return std::unexpected(result.error());
                }
                strategy_ = std::move(*result);
                LOG_DEBUG("Created {} strategy from Scene model", params.optimization.strategy);
            }

            auto& splat = strategy_->get_model();

            int max_cap = params.optimization.max_cap;
            if (max_cap < splat.size()) {
                LOG_WARN("Max cap is less than to {} initial splats {}. Choosing randomly {} splats", max_cap, splat.size(), max_cap);
                lfs::core::random_choose(splat, max_cap);
            }

            // Re-initialize strategy with new parameters
            strategy_->set_training_dataset(train_dataset_);
            strategy_->initialize(params.optimization);
            LOG_DEBUG("Strategy initialized");

            // Initialize bilateral grid if enabled
            if (auto result = initialize_bilateral_grid(); !result) {
                return std::unexpected(result.error());
            }

            // Initialize PPISP if enabled
            if (auto result = initialize_ppisp(); !result) {
                return std::unexpected(result.error());
            }

            // Initialize PPISP controller if enabled
            if (auto result = initialize_ppisp_controller(); !result) {
                return std::unexpected(result.error());
            }

            // Validate masks if mask mode is enabled
            if (auto result = validate_masks(); !result) {
                return std::unexpected(result.error());
            }

            // Apply undistortion to camera intrinsics (params already precomputed at load time)
            if (params.optimization.undistort) {
                int prepared = 0;
                for (auto& cam : train_dataset_->get_cameras()) {
                    if (cam && cam->has_distortion()) {
                        cam->prepare_undistortion();
                        ++prepared;
                    }
                }
                if (val_dataset_) {
                    for (auto& cam : val_dataset_->get_cameras()) {
                        if (cam && cam->has_distortion()) {
                            cam->prepare_undistortion();
                        }
                    }
                }
                if (prepared > 0) {
                    LOG_INFO("Prepared undistortion for {} cameras", prepared);
                }
            }

            // Initialize sparsity optimizer
            if (params.optimization.enable_sparsity) {
                constexpr int UPDATE_INTERVAL = 50;
                const int sparsify_steps = params.optimization.sparsify_steps;
                const int stored_iters = static_cast<int>(params.optimization.iterations);

                // Checkpoint already has total iterations; fresh start needs sparsify_steps added
                const bool is_resume = params.resume_checkpoint.has_value();
                const int base_iters = is_resume ? (stored_iters - sparsify_steps) : stored_iters;

                if (!is_resume) {
                    params_.optimization.iterations = static_cast<size_t>(base_iters + sparsify_steps);
                }

                const ADMMSparsityOptimizer::Config config{
                    .sparsify_steps = sparsify_steps,
                    .init_rho = params.optimization.init_rho,
                    .prune_ratio = params.optimization.prune_ratio,
                    .update_every = UPDATE_INTERVAL,
                    .start_iteration = base_iters};

                sparsity_optimizer_ = SparsityOptimizerFactory::create("admm", config);
                if (sparsity_optimizer_) {
                    LOG_INFO("Sparsity: base={}, steps={}, prune={:.0f}%",
                             base_iters, sparsify_steps, params.optimization.prune_ratio * 100);
                }
            }

            // Initialize background color tensor from params
            {
                const auto& bg_color = params.optimization.bg_color;
                background_ = lfs::core::Tensor::empty({3}, lfs::core::Device::CPU, lfs::core::DataType::Float32);
                auto* bg_ptr = background_.ptr<float>();
                bg_ptr[0] = bg_color[0];
                bg_ptr[1] = bg_color[1];
                bg_ptr[2] = bg_color[2];
                background_ = background_.to(lfs::core::Device::CUDA);
                LOG_INFO("Background color set to RGB({:.2f}, {:.2f}, {:.2f})", bg_color[0], bg_color[1], bg_color[2]);
            }

            // Initialize image cache loader before any code path that calls getInstance()
            auto& cache_loader = lfs::io::CacheLoader::getInstance(
                params_.dataset.loading_params.use_cpu_memory,
                params_.dataset.loading_params.use_fs_cache);
            cache_loader.update_cache_params(
                params_.dataset.loading_params.use_cpu_memory,
                params_.dataset.loading_params.use_fs_cache,
                train_dataset_size_,
                params_.dataset.loading_params.min_cpu_free_GB,
                params_.dataset.loading_params.min_cpu_free_memory_ratio,
                params_.dataset.loading_params.print_cache_status,
                params_.dataset.loading_params.print_status_freq_num);

            // Load background image if specified
            if (params.optimization.bg_mode == lfs::core::param::BackgroundMode::Image &&
                !params.optimization.bg_image_path.empty() &&
                std::filesystem::exists(params.optimization.bg_image_path)) {
                try {
                    auto& loader = lfs::io::CacheLoader::getInstance();
                    lfs::io::LoadParams load_params{
                        .resize_factor = 1,
                        .max_width = 0, // No max width limit
                        .cuda_stream = nullptr};
                    bg_image_base_ = loader.load_cached_image(params.optimization.bg_image_path, load_params);
                    if (bg_image_base_.device() != lfs::core::Device::CUDA) {
                        bg_image_base_ = bg_image_base_.to(lfs::core::Device::CUDA);
                    }
                    if (bg_image_base_.shape()[0] != 3) {
                        LOG_WARN("Background image has {} channels, expected 3 (RGB)", bg_image_base_.shape()[0]);
                        bg_image_base_ = {};
                        params_.optimization.bg_mode = lfs::core::param::BackgroundMode::SolidColor;
                    } else {
                        LOG_INFO("Background image: {} [{}x{}]",
                                 lfs::core::path_to_utf8(params.optimization.bg_image_path),
                                 bg_image_base_.shape()[2], bg_image_base_.shape()[1]);
                    }
                } catch (const std::exception& e) {
                    LOG_WARN("Failed to load background image: {}", e.what());
                    params_.optimization.bg_mode = lfs::core::param::BackgroundMode::SolidColor;
                }
            }

            // Create progress bar based on headless flag
            if (params.optimization.headless) {
                progress_ = std::make_unique<TrainingProgress>(
                    params_.optimization.iterations, // This now includes sparsity steps if enabled
                    /*update_frequency=*/100);
                LOG_DEBUG("Progress bar initialized for {} total iterations", params_.optimization.iterations);
            }

            // Initialize the evaluator - it handles all metrics internally
            evaluator_ = std::make_unique<lfs::training::MetricsEvaluator>(params_);
            LOG_DEBUG("Metrics evaluator initialized");

            // Resume from checkpoint if provided
            if (params_.resume_checkpoint.has_value()) {
                auto resume_result = load_checkpoint(*params_.resume_checkpoint);
                if (!resume_result) {
                    return std::unexpected(std::format("Failed to resume from checkpoint: {}", resume_result.error()));
                }
                LOG_INFO("Resumed training from checkpoint at iteration {}", *resume_result);

                // Reload bg_image if checkpoint restored different settings
                if (params_.optimization.bg_mode == lfs::core::param::BackgroundMode::Image &&
                    !params_.optimization.bg_image_path.empty() &&
                    std::filesystem::exists(params_.optimization.bg_image_path) &&
                    !bg_image_base_.is_valid()) {
                    try {
                        auto& loader = lfs::io::CacheLoader::getInstance();
                        lfs::io::LoadParams load_params{.resize_factor = 1, .max_width = 0, .cuda_stream = nullptr};
                        bg_image_base_ = loader.load_cached_image(params_.optimization.bg_image_path, load_params);
                        if (bg_image_base_.device() != lfs::core::Device::CUDA) {
                            bg_image_base_ = bg_image_base_.to(lfs::core::Device::CUDA);
                        }
                        if (bg_image_base_.shape()[0] != 3) {
                            LOG_WARN("Background image has {} channels, expected 3", bg_image_base_.shape()[0]);
                            bg_image_base_ = {};
                            params_.optimization.bg_mode = lfs::core::param::BackgroundMode::SolidColor;
                        } else {
                            LOG_INFO("Background image from checkpoint: {} [{}x{}]",
                                     lfs::core::path_to_utf8(params_.optimization.bg_image_path),
                                     bg_image_base_.shape()[2], bg_image_base_.shape()[1]);
                        }
                    } catch (const std::exception& e) {
                        LOG_WARN("Failed to load background image from checkpoint: {}", e.what());
                        params_.optimization.bg_mode = lfs::core::param::BackgroundMode::SolidColor;
                    }
                }
            }
            if (auto result = initialize_ngs(); !result) {
                return std::unexpected(result.error());
            }

            // Print configuration
            LOG_INFO("Visualization: {}", params.optimization.headless ? "disabled" : "enabled");
            LOG_INFO("Strategy: {}", params.optimization.strategy);
            if (params.optimization.mask_mode != lfs::core::param::MaskMode::None) {
                static constexpr const char* MASK_MODE_NAMES[] = {"none", "segment", "ignore", "alpha_consistent", "hardmatting", "softmatting", "focused_segment"};
                LOG_INFO("Mask mode: {}", MASK_MODE_NAMES[static_cast<int>(params.optimization.mask_mode)]);
            }
            if (current_iteration_ > 0) {
                LOG_INFO("Starting from iteration: {}", current_iteration_.load());
            }

            // Expose initial snapshot for Python control (iteration 0)
            {
                lfs::training::HookContext ctx{
                    .iteration = current_iteration_.load(),
                    .loss = current_loss_.load(),
                    .num_gaussians = strategy_ ? strategy_->get_model().size() : 0,
                    .is_refining = strategy_ ? strategy_->is_refining(current_iteration_.load()) : false,
                    .trainer = this};
                lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::SafeControl);
                lfs::training::CommandCenter::instance().update_snapshot(
                    ctx, params_.optimization.iterations, is_paused_.load(), is_running_.load(), stop_requested_.load(),
                    lfs::training::TrainingPhase::SafeControl);
            }

            // Execute configured Python scripts to register iteration callbacks
            if (!python_scripts_.empty()) {
                auto py_result = lfs::python::run_scripts(python_scripts_);
                if (!py_result) {
                    return std::unexpected(std::format("Failed to run Python scripts: {}", py_result.error()));
                }
            }

            initialized_ = true;
            LOG_INFO("Trainer initialization complete");
            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to initialize trainer: {}", e.what()));
        }
    }

    Trainer::~Trainer() {
        shutdown();
    }

    std::shared_ptr<lfs::io::PipelinedImageLoader> Trainer::getActiveImageLoader() const {
        std::lock_guard<std::mutex> lock(active_image_loader_mutex_);
        return active_image_loader_;
    }

    Trainer::GTLoadConfigSnapshot Trainer::getGTLoadConfigSnapshot() const {
        std::lock_guard<std::mutex> lock(gt_load_config_mutex_);
        return gt_load_config_snapshot_;
    }

    void Trainer::updateGTLoadConfigSnapshot() {
        GTLoadConfigSnapshot snapshot;
        if (train_dataset_) {
            snapshot.resize_factor = std::max(1, train_dataset_->get_resize_factor());
            snapshot.max_width = train_dataset_->get_max_width();

            for (const auto& cam : train_dataset_->get_cameras()) {
                if (cam && cam->is_undistort_prepared()) {
                    snapshot.undistort = true;
                    break;
                }
            }
        }

        std::lock_guard<std::mutex> lock(gt_load_config_mutex_);
        gt_load_config_snapshot_ = snapshot;
    }

    void Trainer::setActiveImageLoader(std::shared_ptr<lfs::io::PipelinedImageLoader> loader) {
        std::lock_guard<std::mutex> lock(active_image_loader_mutex_);
        active_image_loader_ = std::move(loader);
    }

    void Trainer::clearActiveImageLoader() {
        if (strategy_) {
            strategy_->set_image_loader(nullptr);
        }
        setActiveImageLoader(nullptr);
    }

    void Trainer::shutdown() {
        if (shutdown_complete_.exchange(true)) {
            return;
        }

        LOG_DEBUG("Trainer shutdown");
        stop_requested_ = true;

        lfs::core::image_io::BatchImageSaver::instance().wait_all();

        if (callback_stream_) {
            cudaStreamSynchronize(callback_stream_);
            cudaStreamDestroy(callback_stream_);
            callback_stream_ = nullptr;
        }
        callback_busy_ = false;

        cudaDeviceSynchronize();

        clearActiveImageLoader();
        strategy_.reset();
        bilateral_grid_.reset();
        ppisp_.reset();
        ppisp_controller_pool_.reset();
        sparsity_optimizer_.reset();
        evaluator_.reset();
        progress_.reset();
        train_dataset_.reset();
        val_dataset_.reset();

        // Release GPU memory pools back to system
        lfs::core::Tensor::trim_memory_pool();
        lfs::core::GlobalArenaManager::instance().get_arena().full_reset();
        cudaDeviceSynchronize();
        LOG_DEBUG("GPU memory released");

        initialized_ = false;
        is_running_ = false;
        training_complete_ = false;
    }

    void Trainer::setParams(const lfs::core::param::TrainingParameters& params) {
        // Check if background image path changed and needs to be (re)loaded
        const bool bg_image_path_changed =
            params.optimization.bg_image_path != params_.optimization.bg_image_path;
        const bool bg_mode_is_image =
            params.optimization.bg_mode == lfs::core::param::BackgroundMode::Image;

        // Update params first
        params_ = params;

        // Load/reload background image if needed
        if (bg_mode_is_image && bg_image_path_changed &&
            !params.optimization.bg_image_path.empty() &&
            std::filesystem::exists(params.optimization.bg_image_path)) {
            try {
                auto& loader = lfs::io::CacheLoader::getInstance();
                lfs::io::LoadParams load_params{
                    .resize_factor = 1,
                    .max_width = 0,
                    .cuda_stream = nullptr};
                bg_image_base_ = loader.load_cached_image(params.optimization.bg_image_path, load_params);
                if (bg_image_base_.device() != lfs::core::Device::CUDA) {
                    bg_image_base_ = bg_image_base_.to(lfs::core::Device::CUDA);
                }
                bg_image_cache_.clear();
                if (bg_image_base_.shape()[0] != 3) {
                    LOG_WARN("Background image has {} channels, expected 3 (RGB)", bg_image_base_.shape()[0]);
                    bg_image_base_ = {};
                    params_.optimization.bg_mode = lfs::core::param::BackgroundMode::SolidColor;
                } else {
                    LOG_INFO("Background image: {} [{}x{}]",
                             lfs::core::path_to_utf8(params.optimization.bg_image_path),
                             bg_image_base_.shape()[2], bg_image_base_.shape()[1]);
                }
            } catch (const std::exception& e) {
                LOG_WARN("Failed to load background image: {}", e.what());
                params_.optimization.bg_mode = lfs::core::param::BackgroundMode::SolidColor;
            }
        }

        if (!bg_mode_is_image && (bg_image_base_.is_valid() || !bg_image_cache_.empty())) {
            bg_image_cache_.clear();
            bg_image_base_ = {};
        }

        // Update background color tensor if changed
        const auto& bg_color = params.optimization.bg_color;
        if (background_.is_valid()) {
            auto bg_cpu = lfs::core::Tensor::empty({3}, lfs::core::Device::CPU, lfs::core::DataType::Float32);
            auto* bg_ptr = bg_cpu.ptr<float>();
            bg_ptr[0] = bg_color[0];
            bg_ptr[1] = bg_color[1];
            bg_ptr[2] = bg_color[2];
            background_ = bg_cpu.to(lfs::core::Device::CUDA);
        }
    }

    void Trainer::handle_control_requests(int iter, std::stop_token stop_token) {
        // Check stop token first
        if (stop_token.stop_requested()) {
            stop_requested_ = true;
            return;
        }

        // Handle pause/resume
        if (pause_requested_.load() && !is_paused_.load()) {
            is_paused_ = true;
            if (progress_) {
                progress_->pause();
            }
            LOG_INFO("Training paused at iteration {}", iter);
            LOG_DEBUG("Click 'Resume Training' to continue.");
        } else if (!pause_requested_.load() && is_paused_.load()) {
            is_paused_ = false;
            if (progress_) {
                progress_->resume(iter, current_loss_.load(), static_cast<int>(strategy_->get_model().size()));
            }
            LOG_INFO("Training resumed at iteration {}", iter);
        }

        if (save_requested_.exchange(false)) {
            LOG_INFO("Saving checkpoint and PLY at iteration {}...", iter);
            save_ply(params_.dataset.output_path, iter, /*join=*/false);
            auto result = save_checkpoint(iter);
            if (result) {
                const auto checkpoint_path = lfs::training::checkpoint_output_path(params_.dataset.output_path);
                LOG_INFO("Checkpoint and PLY saved to {} (checkpoint: {})",
                         lfs::core::path_to_utf8(params_.dataset.output_path),
                         lfs::core::path_to_utf8(checkpoint_path));
            } else {
                LOG_ERROR("Failed to save checkpoint: {}", result.error());
            }
        }

        // Handle stop request - this permanently stops training
        if (stop_requested_.load()) {
            LOG_INFO("Stopping training permanently at iteration {}...", iter);
            LOG_DEBUG("Saving final model...");
            save_ply(params_.dataset.output_path, iter, /*join=*/true);
            is_running_ = false;
        }
    }

    // Calculate noise intensity based on training progress.
    // Strategy:
    // 1. Grace Period: No noise initially to allow structure to solidify from SfM/Random points.
    // 2. Warmup: Ramp up noise to clear "floaters" and aggressively carve empty space.
    // 3. Decay: Reduce noise towards the end to allow fine-tuning of semi-transparent details (hair, edges).
    inline float inv_weight_piecewise(int step, int max_steps, bool isNoise) {
        if (max_steps <= 0)
            return 0.0f;

        const float phase = std::clamp(
            static_cast<float>(step) / static_cast<float>(max_steps),
            0.0f, 1.0f);

        // Schedule configuration
        constexpr float P_DELAY = 0.15f;      // First 10%: Grace period (Structural initialization)
        constexpr float P_WARMUP_LEN = 0.10f; // Next 10%: Linear ramp-up
        constexpr float P_RAMP_END = P_DELAY + P_WARMUP_LEN;
        constexpr float P_HOLD_END = P_RAMP_END + P_WARMUP_LEN * 2; // Hold peak intensity until
        const float P_DECAY_END = isNoise ? 0.8f : 0.60f; // Decay ends at 80%

        // Intensities
        const float MAX_INTENSITY = isNoise ? 0.85f : 0.6f;
        constexpr float MIN_INTENSITY = 0.01f; // Small floor to prevent late-training artifacts (halos)

        // Phase 1: Grace Period
        if (phase < P_DELAY) {
            return 0.0f; // Silence: Let the model learn the base geometry first.
        }
        // Phase 2: Warmup (Linear Ramp)
        else if (phase < P_RAMP_END) {
            // Normalize 't' from 0.0 to 1.0 within the warmup window
            const float t = (phase - P_DELAY) / P_WARMUP_LEN;
            return MAX_INTENSITY * t;
        }
        // Phase 3: Hold (Peak Cleaning)
        else if (phase < P_HOLD_END) {
            return MAX_INTENSITY; // Aggressive floater removal.
        }
        // Phase 4: Decay (Fine-tuning)
        else if (phase < P_DECAY_END) {
            // Linear decay from MAX to MIN to recover fine details/transparency
            const float t = (phase - P_HOLD_END) / (P_DECAY_END - P_HOLD_END);
            return MAX_INTENSITY + (MIN_INTENSITY - MAX_INTENSITY) * t;
        }
        // Phase 5: Floor (Stability)
        else {
            return MIN_INTENSITY;
        }
    }

    namespace {
        constexpr float TWO_PI = static_cast<float>(M_PI * 2.0);
        constexpr float PHASE_OFFSET_G = TWO_PI / 3.0f;
        constexpr float PHASE_OFFSET_B = TWO_PI * 2.0f / 3.0f;
        constexpr float CLAMP_EPS = 1e-4f;
        constexpr int BG_PERIOD_R = 37;
        constexpr int BG_PERIOD_G = 41;
        constexpr int BG_PERIOD_B = 43;
    } // anonymous namespace

    lfs::core::Tensor& Trainer::background_for_step(int iter) {
        if (!params_.optimization.bg_modulation || params_.optimization.bg_noise) {
            return background_;
        }

        const float w = inv_weight_piecewise(iter, params_.optimization.iterations, false);
        if (w <= 0.0f) {
            return background_;
        }

        // Sine-based RGB with prime periods for color diversity
        const float pr = TWO_PI * static_cast<float>(iter % BG_PERIOD_R) / BG_PERIOD_R;
        const float pg = TWO_PI * static_cast<float>(iter % BG_PERIOD_G) / BG_PERIOD_G;
        const float pb = TWO_PI * static_cast<float>(iter % BG_PERIOD_B) / BG_PERIOD_B;

        const float result[3] = {
            std::clamp(0.5f * (1.0f + std::sin(pr)) * w, CLAMP_EPS, 1.0f - CLAMP_EPS),
            std::clamp(0.5f * (1.0f + std::sin(pg + PHASE_OFFSET_G)) * w, CLAMP_EPS, 1.0f - CLAMP_EPS),
            std::clamp(0.5f * (1.0f + std::sin(pb + PHASE_OFFSET_B)) * w, CLAMP_EPS, 1.0f - CLAMP_EPS)};

        if (bg_mix_buffer_.is_empty()) {
            bg_mix_buffer_ = lfs::core::Tensor::empty({3}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        }

        cudaMemcpyAsync(bg_mix_buffer_.ptr<float>(), result, sizeof(result), cudaMemcpyHostToDevice, bg_mix_buffer_.stream());
        return bg_mix_buffer_;
    }

    
    lfs::core::Tensor Trainer::apply_background_noise(
        const lfs::core::Tensor& rendered, // [C, H, W] (usually C=3)
        const lfs::core::Tensor& alpha,    // [1, H, W]
        int iter) {
        // Compute noise intensity weight based on training progress
        const float w = inv_weight_piecewise(iter, params_.optimization.iterations, true);

        // Early exit: skip blending if noise contribution is negligible
        if (w <= CLAMP_EPS) {
            return rendered; // shallow return is fine
        }

        // Basic shape assumptions (keep lightweight; training path should be consistent)
        const size_t C = rendered.shape()[0];
        const size_t H = rendered.shape()[1];
        const size_t W = rendered.shape()[2];

        const lfs::core::TensorShape noise_shape({C, H, W});
        const lfs::core::TensorShape alpha_shape({1, H, W});
        const auto dev = rendered.device();

        // Persistent buffers to avoid per-iteration VRAM allocations.
        // thread_local keeps it safe if multiple trainers run in parallel on different threads.
        static thread_local lfs::core::Tensor noise_buffer;     // [C, H, W]
        static thread_local lfs::core::Tensor weighted_buffer;  // [C, H, W]
        static thread_local lfs::core::Tensor alpha_inv_buffer; // [1, H, W]

        // Re-allocate only if strictly necessary (shape/device changes)
        bool buffer_reset = false;
        if (noise_buffer.is_empty() || noise_buffer.shape() != noise_shape || noise_buffer.device() != dev) {
            noise_buffer = lfs::core::Tensor::empty(noise_shape, dev, lfs::core::DataType::Float32);
            buffer_reset = true;
        }
        if (weighted_buffer.is_empty() || weighted_buffer.shape() != noise_shape || weighted_buffer.device() != dev) {
            weighted_buffer = lfs::core::Tensor::empty(noise_shape, dev, lfs::core::DataType::Float32);
        }
        if (alpha_inv_buffer.is_empty() || alpha_inv_buffer.shape() != alpha_shape || alpha_inv_buffer.device() != dev) {
            alpha_inv_buffer = lfs::core::Tensor::empty(alpha_shape, dev, lfs::core::DataType::Float32);
        }

        // Constants
        constexpr int NOISE_RESET_INTERVAL = 100;
        constexpr float GOLDEN_RATIO = 0.61803398875f;

        // NOISE GENERATION STRATEGY:
        // 1) Periodic full reset: expensive RNG for statistical independence.
        // 2) Per-step cheap update: add constant offset and wrap to [0,1).
        //
        // IMPORTANT: In this tensor library, "a = a + b" materializes a new tensor.
        // Use in-place scalar ops to avoid allocations and extra kernels.
        if (buffer_reset || (iter % NOISE_RESET_INTERVAL == 0)) {
            // Expensive path
            noise_buffer.uniform_(0.0f, 1.0f);
        } else {
            // Cheap path: noise = fract(noise + phi)
            // In-place scalar add (no allocation)
            noise_buffer.add_(GOLDEN_RATIO);

            // Wrap around 1.0: fract(x) = x - floor(x)
            // NOTE: floor() currently materializes a temporary (1 allocation).
            // Still much cheaper than materializing (a + scalar) AND floor() AND subtraction.
            noise_buffer.sub_(noise_buffer.floor());
        }

        // Prepare alpha_inv = (1 - alpha) in a reusable buffer.
        // Avoid scalar Tensor "{}" (rank-0) which can be fragile depending on broadcast rules.
        alpha_inv_buffer.copy_(alpha); // alpha is expected to be Float32 on the training path
        alpha_inv_buffer.mul_(-1.0f);
        alpha_inv_buffer.add_(1.0f); // alpha_inv_buffer = 1 - alpha

        // weighted_buffer = noise * w (reused buffer, no allocation)
        weighted_buffer.copy_(noise_buffer);
        weighted_buffer.mul_(w);

        // Blend: out = rendered + (1 - alpha) * (noise * w)
        // This multiply will broadcast [1,H,W] over [C,H,W] and materialize ONE output tensor.
        lfs::core::Tensor out = alpha_inv_buffer * weighted_buffer;
        out.add_(rendered); // in-place add, shapes match exactly

        return out;
    }

    lfs::core::Tensor Trainer::get_background_image_for_camera(int width, int height) {
        // Return empty tensor if no background image is loaded
        if (!bg_image_base_.is_valid() || bg_image_base_.is_empty()) {
            return lfs::core::Tensor();
        }

        // Check cache first - key is (height << 32) | width
        const uint64_t cache_key = (static_cast<uint64_t>(height) << 32) | static_cast<uint64_t>(width);
        auto it = bg_image_cache_.find(cache_key);
        if (it != bg_image_cache_.end()) {
            return it->second;
        }

        // Resize background image to match camera dimensions
        const int src_h = static_cast<int>(bg_image_base_.shape()[1]);
        const int src_w = static_cast<int>(bg_image_base_.shape()[2]);
        const int channels = static_cast<int>(bg_image_base_.shape()[0]);

        // If dimensions match, use the original
        if (src_w == width && src_h == height) {
            bg_image_cache_[cache_key] = bg_image_base_;
            return bg_image_base_;
        }

        // Create resized tensor
        auto resized = lfs::core::Tensor::empty(
            {static_cast<size_t>(channels), static_cast<size_t>(height), static_cast<size_t>(width)},
            lfs::core::Device::CUDA,
            lfs::core::DataType::Float32);

        // Use bilinear resize kernel
        kernels::launch_bilinear_resize_chw(
            bg_image_base_.ptr<float>(),
            resized.ptr<float>(),
            channels,
            src_h, src_w,
            height, width,
            resized.stream());

        // Cache the resized image
        bg_image_cache_[cache_key] = resized;
        LOG_DEBUG("Background image resized: {}x{} -> {}x{}", src_w, src_h, width, height);

        return resized;
    }

    lfs::core::Tensor Trainer::get_random_background_for_camera(int width, int height, int iteration) {
        const size_t required_size = 3 * static_cast<size_t>(height) * static_cast<size_t>(width);

        if (!random_bg_buffer_.is_valid() || random_bg_buffer_.numel() != required_size) {
            random_bg_buffer_ = lfs::core::Tensor::empty(
                {3, static_cast<size_t>(height), static_cast<size_t>(width)},
                lfs::core::Device::CUDA,
                lfs::core::DataType::Float32);
        }

        kernels::launch_random_background(
            random_bg_buffer_.ptr<float>(),
            height, width,
            static_cast<uint64_t>(iteration),
            random_bg_buffer_.stream());

        return random_bg_buffer_;
    }

    std::expected<Trainer::StepResult, std::string> Trainer::train_step(
        int iter,
        lfs::core::Camera* cam,
        lfs::core::Tensor gt_image,
        RenderMode render_mode,
        std::stop_token stop_token) {
        try {
            //NGS Noise Guided Splatting transition
            handle_ngs_phase_transition(iter);

            // GUT mode enables Gaussian Unscented Transform for lens distortion handling
            if (params_.optimization.gut) {
                if (cam->camera_model_type() == core::CameraModelType::ORTHO) {
                    return std::unexpected("Training on cameras with ortho model is not supported yet.");
                }
            } else if (!params_.optimization.undistort || !cam->is_undistort_prepared()) {
                if (cam->radial_distortion().numel() != 0 ||
                    cam->tangential_distortion().numel() != 0) {
                    return std::unexpected("Distorted images detected. Use --gut or --undistort to train on cameras with distortion.");
                }
                if (cam->camera_model_type() != core::CameraModelType::PINHOLE) {
                    return std::unexpected("Use --gut or --undistort to train on cameras with non-pinhole model.");
                }
            }

            current_iteration_ = iter;

            // Check control requests at the beginning
            handle_control_requests(iter, stop_token);

            if (on_iteration_start_)
                on_iteration_start_();

            // Python hook: iteration start (safe, pre-forward)
            {
                lfs::training::HookContext ctx{
                    .iteration = iter,
                    .loss = current_loss_.load(),
                    .num_gaussians = strategy_ ? strategy_->get_model().size() : 0,
                    .is_refining = strategy_ ? strategy_->is_refining(iter) : false,
                    .trainer = this};
                lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::IterationStart);
                lfs::training::CommandCenter::instance().update_snapshot(
                    ctx, params_.optimization.iterations, is_paused_.load(), is_running_.load(), stop_requested_.load(),
                    lfs::training::TrainingPhase::IterationStart);
                lfs::training::ControlBoundary::instance().notify(lfs::training::ControlHook::IterationStart, ctx);
                auto view = lfs::training::CommandCenter::instance().snapshot();
                lfs::training::CommandCenter::instance().drain_enqueued(view);
            }

            // Training step entering forward/backward/optimizer region (commands blocked)
            lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::Forward);

            // If stop requested, return Stop
            if (stop_requested_.load() || stop_token.stop_requested()) {
                return StepResult::Stop;
            }

            // If paused, wait
            while (is_paused_.load() && !stop_requested_.load() && !stop_token.stop_requested()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                handle_control_requests(iter, stop_token);
            }

            // Check stop again after potential pause
            if (stop_requested_.load() || stop_token.stop_requested()) {
                return StepResult::Stop;
            }

            nvtxRangePush("background_for_step");
            lfs::core::Tensor& bg = background_for_step(iter);
            nvtxRangePop();

            lfs::core::Tensor bg_image;
            if (params_.optimization.bg_mode == lfs::core::param::BackgroundMode::Image) {
                bg_image = get_background_image_for_camera(cam->image_width(), cam->image_height());
            } else if (params_.optimization.bg_mode == lfs::core::param::BackgroundMode::Random) {
                bg_image = get_random_background_for_camera(cam->image_width(), cam->image_height(), iter);
            }

            // Configurable tile-based training to reduce peak memory
            const int full_width = cam->image_width();
            const int full_height = cam->image_height();

            // Read tile mode from parameters (1=1 tile, 2=2 tiles, 4=4 tiles)
            const TileMode tile_mode = static_cast<TileMode>(params_.optimization.tile_mode);

            // Determine tile configuration
            int tile_rows = 1, tile_cols = 1;
            switch (tile_mode) {
            case TileMode::One:
                tile_rows = 1;
                tile_cols = 1;
                break;
            case TileMode::Two:
                tile_rows = 2;
                tile_cols = 1;
                break;
            case TileMode::Four:
                tile_rows = 2;
                tile_cols = 2;
                break;
            }

            const int tile_width = full_width / tile_cols;
            const int tile_height = full_height / tile_rows;
            const int num_tiles = tile_rows * tile_cols;

            if (!loss_accumulator_.is_valid()) {
                loss_accumulator_ = core::Tensor::zeros({1}, core::Device::CUDA);
            } else {
                loss_accumulator_.zero_();
            }
            auto& loss_tensor_gpu = loss_accumulator_;
            RenderOutput r_output;
            int tiles_processed = 0;

            // Determine controller phase before tile loop (does not depend on tile results)
            const bool known_ppisp_camera = ppisp_ && ppisp_->is_known_camera(cam->camera_id());
            const int ppisp_cam_idx = known_ppisp_camera ? ppisp_->camera_index(cam->camera_id()) : -1;
            const int ppisp_activation_step = params_.optimization.resolved_ppisp_controller_activation_step();
            const bool ppisp_frozen = is_ppisp_frozen();
            const bool in_controller_phase = ppisp_controller_pool_ && known_ppisp_camera &&
                                             params_.optimization.ppisp_use_controller &&
                                             !ppisp_frozen &&
                                             params_.optimization.ppisp_freeze_gaussians_on_distill &&
                                             iter >= ppisp_activation_step &&
                                             ppisp_cam_idx >= 0 &&
                                             ppisp_cam_idx < ppisp_controller_pool_->num_cameras();
            const bool use_pixel_error_densification =
                (params_.optimization.strategy == "mcmc" ||
                 params_.optimization.strategy == "igs+");
            const bool use_ssim_error = use_pixel_error_densification;

            // Loop over tiles (row-major order)
            for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
                const int tile_row = tile_idx / tile_cols;
                const int tile_col = tile_idx % tile_cols;
                const int tile_x_offset = tile_col * tile_width;
                const int tile_y_offset = tile_row * tile_height;

                nvtxRangePush(std::format("tile_{}x{}", tile_row, tile_col).c_str());

                // Extract GT image tile
                lfs::core::Tensor gt_tile;
                if (num_tiles == 1) {
                    // No tiling - use full image
                    gt_tile = gt_image;
                } else if (gt_image.shape()[0] == 3) {
                    // CHW layout: gt_image is [3, H, W]
                    // Slice both height and width dimensions
                    auto tile_h = gt_image.slice(1, tile_y_offset, tile_y_offset + tile_height);
                    gt_tile = tile_h.slice(2, tile_x_offset, tile_x_offset + tile_width);
                } else {
                    // HWC layout: gt_image is [H, W, 3]
                    auto tile_h = gt_image.slice(0, tile_y_offset, tile_y_offset + tile_height);
                    gt_tile = tile_h.slice(1, tile_x_offset, tile_x_offset + tile_width);
                }

                // Extract background image tile (if using background image)
                lfs::core::Tensor bg_tile;
                if (bg_image.is_valid() && !bg_image.is_empty()) {
                    if (num_tiles == 1) {
                        // No tiling - use full image
                        bg_tile = bg_image;
                    } else {
                        // CHW layout: bg_image is [3, H, W]
                        // Slice both height and width dimensions
                        auto tile_h = bg_image.slice(1, tile_y_offset, tile_y_offset + tile_height);
                        bg_tile = tile_h.slice(2, tile_x_offset, tile_x_offset + tile_width);
                    }
                }

                // Render the tile
                nvtxRangePush("rasterize_forward");

                // Storage for render output (used by both paths)
                RenderOutput output;
                std::optional<FastRasterizeContext> fast_ctx;
                std::optional<GsplatRasterizeContext> gsplat_ctx;

                if (params_.optimization.gut) {
                    const int tw = (num_tiles > 1) ? tile_width : 0;
                    const int th = (num_tiles > 1) ? tile_height : 0;
                    auto rasterize_result = gsplat_rasterize_forward(
                        *cam, strategy_->get_model(), bg,
                        tile_x_offset, tile_y_offset, tw, th,
                        1.0f, false, GsplatRenderMode::RGB, true, bg_tile);

                    if (!rasterize_result) {
                        nvtxRangePop(); // rasterize_forward
                        nvtxRangePop(); // tile
                        return std::unexpected(rasterize_result.error());
                    }

                    output = std::move(rasterize_result->first);
                    gsplat_ctx.emplace(std::move(rasterize_result->second));
                } else {
                    // Standard mode: use fast rasterizer with tiling support
                    auto rasterize_result = fast_rasterize_forward(
                        *cam, strategy_->get_model(), bg,
                        tile_x_offset, tile_y_offset,
                        (num_tiles > 1) ? tile_width : 0, // 0 means full image
                        (num_tiles > 1) ? tile_height : 0,
                        params_.optimization.mip_filter, bg_tile);

                    // Check for OOM error
                    if (!rasterize_result) {
                        const std::string& error = rasterize_result.error();
                        if (error.find("OUT_OF_MEMORY") != std::string::npos) {
                            nvtxRangePop(); // rasterize_forward
                            nvtxRangePop(); // tile

                            // Handle OOM by switching tile mode
                            if (tile_mode == TileMode::Four) {
                                // Already at maximum tiling - can't tile further, return error
                                LOG_ERROR("OUT OF MEMORY at maximum tile mode (2x2). Cannot continue training.");
                                LOG_ERROR("Arena error: {}", error);
                                return std::unexpected(error);
                            } else {
                                // Upgrade to next tile mode
                                TileMode new_mode = (tile_mode == TileMode::One) ? TileMode::Two : TileMode::Four;
                                LOG_WARN("OUT OF MEMORY detected. Switching tile mode from {} to {}",
                                         static_cast<int>(tile_mode), static_cast<int>(new_mode));
                                LOG_WARN("Arena error: {}", error);
                                params_.optimization.tile_mode = static_cast<int>(new_mode);

                                // Retry this step with new tile mode
                                return std::unexpected("OOM_RETRY"); // Signal to retry the step
                            }
                        } else {
                            // Non-OOM error - propagate
                            nvtxRangePop();
                            nvtxRangePop();
                            return std::unexpected(error);
                        }
                    }

                    output = std::move(rasterize_result->first);
                    fast_ctx.emplace(std::move(rasterize_result->second));

                    if (fast_ctx->forward_ctx.n_visible_primitives == 0) {
                        auto& arena = lfs::core::GlobalArenaManager::instance().get_arena();
                        arena.end_frame(fast_ctx->forward_ctx.frame_id);
                        nvtxRangePop();
                        nvtxRangePop();
                        continue;
                    }
                }

                r_output = output; // Save last tile for densification
                nvtxRangePop();

                if (in_controller_phase) {
                    // Controller phase: forward through ISP with controller params, photometric loss,
                    // backward only through controller (base params frozen)
                    nvtxRangePush("controller_phase");
                    auto cleanup_controller_tile_context = [&]() {
                        auto& arena = lfs::core::GlobalArenaManager::instance().get_arena();
                        if (fast_ctx) {
                            arena.end_frame(fast_ctx->forward_ctx.frame_id);
                        } else if (gsplat_ctx) {
                            if (gsplat_ctx->isect_ids_ptr != nullptr) {
                                cudaFree(gsplat_ctx->isect_ids_ptr);
                                gsplat_ctx->isect_ids_ptr = nullptr;
                            }
                            if (gsplat_ctx->flatten_ids_ptr != nullptr) {
                                cudaFree(gsplat_ctx->flatten_ids_ptr);
                                gsplat_ctx->flatten_ids_ptr = nullptr;
                            }
                            arena.end_frame(gsplat_ctx->frame_id);
                        }
                    };

                    lfs::core::Tensor corrected_image = output.image;
                    if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
                        corrected_image = bilateral_grid_->apply(output.image, cam->uid());
                    }
                    auto ppisp_input = corrected_image;

                    auto pred = ppisp_controller_pool_->predict(ppisp_cam_idx, corrected_image.unsqueeze(0), 1.0f);
                    corrected_image = ppisp_->apply_with_controller_params(corrected_image, pred, ppisp_cam_idx);

                    // Photometric loss
                    nvtxRangePush("compute_photometric_loss");
                    lfs::core::Tensor tile_loss;
                    lfs::core::Tensor tile_grad;

                    const bool use_mask = params_.optimization.mask_mode != lfs::core::param::MaskMode::None &&
                                          (cam->has_mask() || (params_.optimization.use_alpha_as_mask && cam->has_alpha()));
                    if (use_mask) {
                        lfs::core::Tensor mask;
                        if (pipelined_mask_.is_valid() && pipelined_mask_.numel() > 0) {
                            mask = pipelined_mask_;
                        } else {
                            mask = cam->load_and_get_mask(
                                params_.dataset.resize_factor,
                                params_.dataset.max_width,
                                params_.optimization.invert_masks,
                                params_.optimization.mask_threshold);
                        }

                        lfs::core::Tensor mask_tile = mask;
                        if (num_tiles > 1 && mask.ndim() == 2) {
                            auto tile_h = mask.slice(0, tile_y_offset, tile_y_offset + tile_height);
                            mask_tile = tile_h.slice(1, tile_x_offset, tile_x_offset + tile_width);
                        }

                        // ------------- victor start
                        // === NUEVA LOGICA: HYBRID SWITCH ===
                        // Creamos una copia local de los parametros para este paso
                        lfs::core::param::OptimizationParameters step_params = params_.optimization;

                        const float progress = static_cast<float>(iter) / static_cast<float>(params_.optimization.iterations);

                        // ESTRATEGIA: "Cleaning Phase" (ultimo 20%)
                        // Si estamos acabando y estamos en modo Matting, cambiamos temporalmente a AlphaConsistent
                        // para limpiar los floaters.

                        if (step_params.mask_mode == lfs::core::param::MaskMode::SoftMatting ||
                            step_params.mask_mode == lfs::core::param::MaskMode::HardMatting) {
                            if (progress < 0.2f) {

                                // Forzamos el modo estricto para limpiar
                                step_params.mask_mode = lfs::core::param::MaskMode::None;

                                // Opcional: Aumentar lambda_dssim ligeramente en esta fase para preservar estructura
                                // step_params.lambda_dssim = 0.2f;
                            } else if (progress > 0.8f) {
                                step_params.mask_opacity_penalty_weight = 0.0f;
                            }
                        }

                        if (step_params.mask_mode == lfs::core::param::MaskMode::FocusedSegment) {
                            const float progress = static_cast<float>(iter) /
                                                   static_cast<float>(params_.optimization.iterations);
                             // NO SE LLAMA NUNCA SIN PPISP CONTROLLER                             
                            // Modo Orig
                            if (progress < 0.25f) {
                                step_params.mask_mode = lfs::core::param::MaskMode::None;
                            } else if (progress > 0.80f) {
                                step_params.mask_opacity_penalty_weight = 0.0f;
                            }
                        }

                        // ===================================
                        

                        // Matting uses precomputed eroded "core" masks to avoid fighting uncertain boundaries.
                        // These cores are computed lazily and cached per Camera, so this is NOT a per-iteration cost.
                        lfs::core::Tensor fg_core;
                        lfs::core::Tensor bg_core;
                        bool use_matting_cores = false;

                        if (step_params.mask_mode == lfs::core::param::MaskMode::SoftMatting) {
                            constexpr int kMattingCoreErodeRadiusPx = 2;

                            fg_core = cam->load_and_get_mask_fg_core(
                                params_.dataset.resize_factor,
                                params_.dataset.max_width,
                                step_params.invert_masks,
                                step_params.mask_threshold,
                                kMattingCoreErodeRadiusPx);

                            bg_core = cam->load_and_get_mask_bg_core(
                                params_.dataset.resize_factor,
                                params_.dataset.max_width,
                                step_params.invert_masks,
                                step_params.mask_threshold,
                                kMattingCoreErodeRadiusPx);

                            if (!fg_core.is_valid() || !bg_core.is_valid()) {
                                nvtxRangePop();
                                nvtxRangePop();
                                return std::unexpected("MaskMode::Matting: failed to load fg/bg core masks.");
                            }
                            use_matting_cores = true;
                        }
                        
                        // Extract core tiles if Matting is enabled (must match mask_tile spatially)
                        lfs::core::Tensor fg_core_tile;
                        lfs::core::Tensor bg_core_tile;
                        const lfs::core::Tensor* fg_core_ptr = nullptr;
                        const lfs::core::Tensor* bg_core_ptr = nullptr;

                        if (use_matting_cores) {
                            fg_core_tile = fg_core;
                            bg_core_tile = bg_core;

                            if (num_tiles > 1 && fg_core.ndim() == 2) {
                                auto fh = fg_core.slice(0, tile_y_offset, tile_y_offset + tile_height);
                                fg_core_tile = fh.slice(1, tile_x_offset, tile_x_offset + tile_width);
                                auto bh = bg_core.slice(0, tile_y_offset, tile_y_offset + tile_height);
                                bg_core_tile = bh.slice(1, tile_x_offset, tile_x_offset + tile_width);
                            }

                            fg_core_ptr = &fg_core_tile;
                            bg_core_ptr = &bg_core_tile;
                        }
                        // ------------- victor end

                        auto result = compute_photometric_loss_with_mask(
                            corrected_image, gt_tile, mask_tile, output.alpha, params_.optimization,
                            fg_core_ptr, bg_core_ptr);
                        if (!result) {
                            cleanup_controller_tile_context();
                            nvtxRangePop();
                            nvtxRangePop();
                            nvtxRangePop();
                            return std::unexpected(result.error());
                        }
                        tile_loss = result->loss;
                        tile_grad = result->grad_image;
                    } else {
                        auto result = compute_photometric_loss_with_gradient(
                            corrected_image, gt_tile, params_.optimization);
                        if (!result) {
                            cleanup_controller_tile_context();
                            nvtxRangePop();
                            nvtxRangePop();
                            nvtxRangePop();
                            return std::unexpected(result.error());
                        }
                        tile_loss = result->first;
                        tile_grad = result->second;
                    }

                    loss_tensor_gpu = loss_tensor_gpu + tile_loss;
                    tiles_processed++;
                    nvtxRangePop(); // compute_photometric_loss

                    // ISP backward for controller params
                    auto ctrl_grad = ppisp_->backward_with_controller_params(ppisp_input, tile_grad, pred, ppisp_cam_idx);
                    ppisp_controller_pool_->backward(ppisp_cam_idx, ctrl_grad);

                    // End arena frame explicitly (normally done inside rasterize_backward which we skip)
                    cleanup_controller_tile_context();

                    nvtxRangePop(); // controller_phase
                } else {

                    lfs::core::param::OptimizationParameters step_params = params_.optimization;

                    if (step_params.mask_mode == lfs::core::param::MaskMode::FocusedSegment) {
                        const float progress = static_cast<float>(iter) /
                                               static_cast<float>(params_.optimization.iterations);

                        // Modo chati
                        /* if (step_params.mask_mode == lfs::core::param::MaskMode::FocusedSegment) {
                            const float progress = static_cast<float>(iter) /
                                                   static_cast<float>(params_.optimization.iterations);

                            if (progress < 0.15f) {
                                step_params.mask_mode = lfs::core::param::MaskMode::None;
                            } else if (progress < 0.35f) {
                                // Learn appearance/coverage first, without extra alpha pressure
                                step_params.mask_opacity_penalty_weight = 0.0f;
                            } else if (progress > 0.85f) {
                                // Keep a little structural pressure at the end, but much softer
                                //step_params.mask_opacity_penalty_weight *= 0.25f;
                                step_params.mask_opacity_penalty_weight = 0.0f;
                            }
                        }*/

                        // Modo Orig
                        if (progress < 0.25f) {
                            step_params.mask_mode = lfs::core::param::MaskMode::None;
                        } else if (progress > 0.80f) {
                            step_params.mask_opacity_penalty_weight = 0.0f;
                        }

                        // Modo B
                        /* if (progress < 0.15f) {
                            step_params.mask_mode = lfs::core::param::MaskMode::None;
                        } else if (progress > 0.80f) {
                            step_params.mask_opacity_penalty_weight = 0.0f;
                        }*/
                        
                        // Modo C
                        /* if (progress < 0.15f) {
                            step_params.mask_opacity_penalty_weight = 0.0f;
                        } else if (progress > 0.80f) {
                            step_params.mask_opacity_penalty_weight = 0.0f;
                        }*/
                    }

                    // Normal phase: full forward + backward through all components
                    lfs::core::Tensor corrected_image = output.image;
                    if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
                        nvtxRangePush("bilateral_grid_forward");
                        corrected_image = bilateral_grid_->apply(output.image, cam->uid());
                        nvtxRangePop();
                    }

                    if (ppisp_ && params_.optimization.use_ppisp) {
                        nvtxRangePush("ppisp_forward");
                        corrected_image = ppisp_->apply(corrected_image, cam->camera_id(), cam->uid());
                        nvtxRangePop();
                    }

                    // Final tonemapping: clamp to [0, 1] for loss computation.
                    // This is redundant when PPISP is active (CRF already clamps), but ensures
                    // valid output range for bilateral grids and raw rasterizer output.
                    corrected_image = corrected_image.clamp(0.0f, 1.0f);

                    nvtxRangePush("compute_photometric_loss");
                    lfs::core::Tensor tile_loss;
                    lfs::core::Tensor tile_grad;
                    lfs::core::Tensor tile_grad_alpha;
                    lfs::core::Tensor tile_error_map;
                    lfs::core::Tensor mask_tile;

                    // 1) Compute photometric loss (populates ssim_map in workspace)
                    const bool use_mask = step_params.mask_mode != lfs::core::param::MaskMode::None &&
                                          (cam->has_mask() || (step_params.use_alpha_as_mask && cam->has_alpha()));
                    const bool used_masked_fused =
                        use_mask &&
                        (step_params.mask_mode == lfs::core::param::MaskMode::Segment ||
                         step_params.mask_mode == lfs::core::param::MaskMode::Ignore ||
                         step_params.mask_mode == lfs::core::param::MaskMode::HardMatting /* ||
                         step_params.mask_mode == lfs::core::param::MaskMode::FocusedSegment*/) &&
                        params_.optimization.lambda_dssim > 0.0f;
                    if (use_mask) {
                        lfs::core::Tensor mask;
                        if (pipelined_mask_.is_valid() && pipelined_mask_.numel() > 0) {
                            mask = pipelined_mask_;
                        } else {
                            mask = cam->load_and_get_mask(
                                params_.dataset.resize_factor,
                                params_.dataset.max_width,
                                params_.optimization.invert_masks,
                                params_.optimization.mask_threshold);
                        }

                        mask_tile = mask;
                        if (num_tiles > 1 && mask.ndim() == 2) {
                            auto tile_h = mask.slice(0, tile_y_offset, tile_y_offset + tile_height);
                            mask_tile = tile_h.slice(1, tile_x_offset, tile_x_offset + tile_width);
                        }
                        bool use_matting_cores = false;
                        lfs::core::Tensor fg_core, bg_core;

                        if (params_.optimization.mask_mode == lfs::core::param::MaskMode::SoftMatting) {
                            constexpr int kMattingCoreErodeRadiusPx = 2;
                            fg_core = cam->load_and_get_mask_fg_core(
                                params_.dataset.resize_factor, params_.dataset.max_width,
                                params_.optimization.invert_masks, params_.optimization.mask_threshold,
                                kMattingCoreErodeRadiusPx);
                            bg_core = cam->load_and_get_mask_bg_core(
                                params_.dataset.resize_factor, params_.dataset.max_width,
                                params_.optimization.invert_masks, params_.optimization.mask_threshold,
                                kMattingCoreErodeRadiusPx);

                            if (fg_core.is_valid() && bg_core.is_valid())
                                use_matting_cores = true;
                        }

                        lfs::core::Tensor fg_core_tile, bg_core_tile;
                        const lfs::core::Tensor* fg_core_ptr = nullptr;
                        const lfs::core::Tensor* bg_core_ptr = nullptr;

                        if (use_matting_cores) {
                            fg_core_tile = fg_core;
                            bg_core_tile = bg_core;

                            if (num_tiles > 1 && fg_core.ndim() == 2) {
                                auto fh = fg_core.slice(0, tile_y_offset, tile_y_offset + tile_height);
                                fg_core_tile = fh.slice(1, tile_x_offset, tile_x_offset + tile_width);
                                auto bh = bg_core.slice(0, tile_y_offset, tile_y_offset + tile_height);
                                bg_core_tile = bh.slice(1, tile_x_offset, tile_x_offset + tile_width);
                            }

                            fg_core_ptr = &fg_core_tile;
                            bg_core_ptr = &bg_core_tile;
                        }

                        auto result = compute_photometric_loss_with_mask(
                            corrected_image, gt_tile, mask_tile, output.alpha, step_params,
                            fg_core_ptr, bg_core_ptr);
                        if (!result) {
                            nvtxRangePop();
                            nvtxRangePop();
                            return std::unexpected(result.error());
                        }
                        tile_loss = result->loss;
                        tile_grad = result->grad_image;
                        tile_grad_alpha = result->grad_alpha;
                    } else {
                        auto result = compute_photometric_loss_with_gradient(
                            corrected_image, gt_tile, params_.optimization);
                        if (!result) {
                            nvtxRangePop();
                            nvtxRangePop();
                            return std::unexpected(result.error());
                        }
                        tile_loss = result->first;
                        tile_grad = result->second;
                    }

                    // 2) Extract error map from workspace's ssim_map
                    if (use_pixel_error_densification) {
                        if (use_ssim_error && params_.optimization.lambda_dssim > 0.0f &&
                            params_.optimization.mask_mode != lfs::core::param::MaskMode::SoftMatting) {
                            lfs::core::Tensor ssim_map;
                            if (used_masked_fused) {
                                ssim_map = masked_fused_workspace_.ssim_map;
                            } else if (params_.optimization.lambda_dssim < 1.0f) {
                                ssim_map = photometric_loss_.fused_workspace().ssim_map;
                            } else {
                                ssim_map = photometric_loss_.ssim_workspace().ssim_map;
                            }
                            {
                                const size_t H = ssim_map.shape()[2];
                                const size_t W = ssim_map.shape()[3];
                                if (!densification_error_map_.is_valid() ||
                                    densification_error_map_.shape()[0] != H ||
                                    densification_error_map_.shape()[1] != W) {
                                    densification_error_map_ = core::Tensor::empty({H, W}, core::Device::CUDA);
                                }
                                lfs::training::kernels::launch_ssim_to_error_map(ssim_map, densification_error_map_);
                                tile_error_map = densification_error_map_;
                            }
                        } else if (use_ssim_error) {
                            // lambda_dssim == 0 but error-priority densification still needs SSIM error
                            lfs::core::Tensor pred_chw = corrected_image;
                            lfs::core::Tensor gt_chw = gt_tile;
                            if (pred_chw.ndim() == 3 && pred_chw.shape()[2] == 3 &&
                                gt_chw.ndim() == 3 && gt_chw.shape()[2] == 3) {
                                pred_chw = pred_chw.permute({2, 0, 1}).contiguous();
                                gt_chw = gt_chw.permute({2, 0, 1}).contiguous();
                            }
                            lfs::training::kernels::ssim_error_map_forward(
                                pred_chw, gt_chw, densification_ssim_workspace_, densification_error_map_);
                            tile_error_map = densification_error_map_;
                        } else {
                            const lfs::core::Tensor abs_diff = (corrected_image - gt_tile).abs();
                            if (abs_diff.ndim() == 3 && abs_diff.shape()[0] == 3) {
                                tile_error_map = abs_diff.mean({0}, false);
                            } else if (abs_diff.ndim() == 3 && abs_diff.shape()[2] == 3) {
                                tile_error_map = abs_diff.mean({2}, false);
                            } else {
                                tile_error_map = abs_diff;
                            }
                            tile_error_map = tile_error_map.contiguous();
                        }

                        if (use_mask &&
                            (params_.optimization.mask_mode == lfs::core::param::MaskMode::Segment ||
                             params_.optimization.mask_mode == lfs::core::param::MaskMode::Ignore ||
                             params_.optimization.mask_mode == lfs::core::param::MaskMode::HardMatting ||
                             params_.optimization.mask_mode == lfs::core::param::MaskMode::SoftMatting ||
                             params_.optimization.mask_mode == lfs::core::param::MaskMode::FocusedSegment)) {
                            tile_error_map = (tile_error_map * mask_tile).contiguous();
                        }
                    }

                    loss_tensor_gpu = loss_tensor_gpu + tile_loss;
                    tiles_processed++;
                    nvtxRangePop();

                    lfs::core::Tensor raster_grad = tile_grad;
                    if (ppisp_ && params_.optimization.use_ppisp) {
                        nvtxRangePush("ppisp_backward");
                        lfs::core::Tensor ppisp_input = output.image;
                        if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
                            ppisp_input = bilateral_grid_->apply(output.image, cam->uid());
                        }
                        raster_grad = ppisp_->backward(ppisp_input, raster_grad, cam->camera_id(), cam->uid());
                        if (ppisp_frozen) {
                            ppisp_->zero_grad();
                        }
                        nvtxRangePop();
                    }

                    if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
                        nvtxRangePush("bilateral_grid_backward");
                        raster_grad = bilateral_grid_->backward(output.image, raster_grad, cam->uid());
                        nvtxRangePop();
                    }

                    nvtxRangePush("rasterize_backward");
                    if (gsplat_ctx) {
                        auto grad_alpha = tile_grad_alpha.is_valid()
                                              ? tile_grad_alpha
                                              : lfs::core::Tensor::zeros_like(output.alpha);
                        gsplat_rasterize_backward(*gsplat_ctx, raster_grad, grad_alpha,
                                                  strategy_->get_model(), strategy_->get_optimizer(),
                                                  use_pixel_error_densification ? tile_error_map : lfs::core::Tensor{});
                    } else {
                        fast_rasterize_backward(*fast_ctx, raster_grad, strategy_->get_model(),
                                                strategy_->get_optimizer(), tile_grad_alpha,
                                                use_pixel_error_densification ? tile_error_map : lfs::core::Tensor{});
                    }
                    nvtxRangePop();
                }

                nvtxRangePop(); // End tile
            }

            if (tiles_processed > 1)
                loss_tensor_gpu = loss_tensor_gpu / static_cast<float>(tiles_processed);

            if (tiles_processed == 0) {
                LOG_DEBUG("Skipping iteration {} - no visible primitives", iter);
                return iter < params_.optimization.iterations && !stop_requested_.load() && !stop_token.stop_requested()
                           ? StepResult::Continue
                           : StepResult::Stop;
            }

            if (in_controller_phase) {
                // Controller phase: only update controller weights
                nvtxRangePush("controller_optimizer_step");
                ppisp_controller_pool_->optimizer_step(ppisp_cam_idx);
                ppisp_controller_pool_->zero_grad();
                ppisp_controller_pool_->scheduler_step(ppisp_cam_idx);
                nvtxRangePop();
            } else {
                // Normal phase: regularization losses + optimizer steps for all components

                if (params_.optimization.scale_reg > 0.0f) {
                    nvtxRangePush("compute_scale_reg_loss");
                    auto scale_loss_result = compute_scale_reg_loss(strategy_->get_model(), strategy_->get_optimizer(), params_.optimization);
                    if (!scale_loss_result) {
                        return std::unexpected(scale_loss_result.error());
                    }
                    loss_tensor_gpu = loss_tensor_gpu + *scale_loss_result;
                    nvtxRangePop();
                }

                if (params_.optimization.opacity_reg > 0.0f) {
                    nvtxRangePush("compute_opacity_reg_loss");
                    auto opacity_loss_result = compute_opacity_reg_loss(strategy_->get_model(), strategy_->get_optimizer(), params_.optimization);
                    if (!opacity_loss_result) {
                        return std::unexpected(opacity_loss_result.error());
                    }
                    loss_tensor_gpu = loss_tensor_gpu + *opacity_loss_result;
                    nvtxRangePop();
                }

                if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
                    nvtxRangePush("bilateral_grid_tv_and_step");
                    const float tv_weight = params_.optimization.tv_loss_weight;

                    loss_tensor_gpu = loss_tensor_gpu + bilateral_grid_->tv_loss_gpu() * tv_weight;
                    bilateral_grid_->tv_backward(tv_weight);
                    bilateral_grid_->optimizer_step();
                    bilateral_grid_->zero_grad();
                    bilateral_grid_->scheduler_step();

                    nvtxRangePop();
                }

                if (ppisp_ && params_.optimization.use_ppisp && !ppisp_frozen) {
                    nvtxRangePush("ppisp_reg_and_step");

                    loss_tensor_gpu = loss_tensor_gpu + ppisp_->reg_loss_gpu();
                    ppisp_->reg_backward();
                    ppisp_->optimizer_step();
                    ppisp_->zero_grad();
                    ppisp_->scheduler_step();

                    nvtxRangePop();
                }
            }

            // Sparsity loss - ALL ON GPU, no CPU sync here
            lfs::core::Tensor sparsity_loss_gpu;
            if (sparsity_optimizer_ && sparsity_optimizer_->should_apply_loss(iter)) {
                nvtxRangePush("sparsity_loss");
                auto sparsity_result = compute_sparsity_loss_forward(iter, strategy_->get_model());
                if (!sparsity_result) {
                    nvtxRangePop();
                    return std::unexpected(sparsity_result.error());
                }
                auto& [loss_tensor, ctx] = *sparsity_result;
                sparsity_loss_gpu = std::move(loss_tensor);

                if (ctx.n > 0) {
                    if (auto result = sparsity_optimizer_->compute_loss_backward(
                            ctx, 1.0f, strategy_->get_optimizer().get_grad(ParamType::Opacity));
                        !result) {
                        nvtxRangePop();
                        return std::unexpected(result.error());
                    }
                }
                nvtxRangePop();
            }

            // Sparsification phase logging (once per phase transition)
            if (params_.optimization.enable_sparsity) {
                const int base_iterations = params_.optimization.iterations - params_.optimization.sparsify_steps;
                if (iter == base_iterations + 1) {
                    LOG_INFO("Entering sparsification: {} Gaussians, target prune={}%",
                             strategy_->get_model().size(), params_.optimization.prune_ratio * 100);
                }
            }

            // Sync loss to CPU only at intervals - single sync point
            constexpr int LOSS_SYNC_INTERVAL = 10;
            float loss_value = 0.0f;
            if (iter % LOSS_SYNC_INTERVAL == 0 || iter == 1) {
                // Accumulate on GPU then sync once
                auto total_loss = sparsity_loss_gpu.numel() > 0
                                      ? (loss_tensor_gpu + sparsity_loss_gpu)
                                      : loss_tensor_gpu;
                loss_value = total_loss.item<float>();

                if (std::isnan(loss_value) || std::isinf(loss_value)) {
                    return std::unexpected(std::format("NaN/Inf loss at iteration {}", iter));
                }

                current_loss_ = loss_value;
                if (progress_) {
                    progress_->update(iter, loss_value,
                                      static_cast<int>(strategy_->get_model().size()),
                                      strategy_->is_refining(iter));
                }
                lfs::core::events::state::TrainingProgress{
                    .iteration = iter,
                    .loss = loss_value,
                    .num_gaussians = static_cast<int>(strategy_->get_model().size()),
                    .is_refining = strategy_->is_refining(iter)}
                    .emit();
            }

            const bool in_sparsification = params_.optimization.enable_sparsity &&
                                           iter > (params_.optimization.iterations - params_.optimization.sparsify_steps);

            if (!in_sparsification) {
                strategy_->pre_step(iter, r_output);
            }

            {
                DeferredEvents deferred;
                {
                    std::unique_lock<std::shared_mutex> lock(render_mutex_);

                    // Python hook: pre-optimizer-step (post-backward, pre-step)
                    {
                        lfs::training::HookContext ctx{
                            .iteration = iter,
                            .loss = current_loss_.load(),
                            .num_gaussians = strategy_ ? strategy_->get_model().size() : 0,
                            .is_refining = strategy_ ? strategy_->is_refining(iter) : false,
                            .trainer = this};
                        lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::OptimizerStep);
                        lfs::training::CommandCenter::instance().update_snapshot(
                            ctx, params_.optimization.iterations, is_paused_.load(), is_running_.load(), stop_requested_.load(),
                            lfs::training::TrainingPhase::OptimizerStep);
                        lfs::training::ControlBoundary::instance().notify(lfs::training::ControlHook::PreOptimizerStep, ctx);
                    }

                    // Skip post_backward during sparsification phase
                    const bool in_sparsification = params_.optimization.enable_sparsity &&
                                                   iter > (params_.optimization.iterations - params_.optimization.sparsify_steps);
                                                   
                    // Skip topology-changing ops during sparsification OR NGS noise phase
                    const bool ngs_block_topology = (ngs_current_phase_ == NGSPhase::WithNoise);

                    // Freeze noise gaussians: means/scaling/rotation/SH, keep opacity trainable
                    // NGS quality-first freezing schedule (applied AFTER backward, BEFORE any optimizer step):
                    // - FineTune [start, finetune_end): surface fully frozen, train ONLY noise opacity
                    // - Guided  [finetune_end, end): noise fully frozen, freeze surface SH (avoid color/light drift)
                    if (ngs_block_topology && ngs_surface_count_ > 0 && ngs_noise_count_ > 0) {
                        auto& opt = strategy_->get_optimizer();
                        const size_t noise_start = ngs_surface_count_;
                        const size_t noise_count = ngs_noise_count_;
                        const size_t surface_count = ngs_surface_count_;
                    
                        const int end_iter = ngs_phase_manager_ ? ngs_phase_manager_->noise_end() : params_.optimization.iterations;
                        const bool in_finetune = (iter < ngs_finetune_end_iter_);
                    
                        // Always freeze noise geometry + SH (noise exists only to block / guide, never to move).
                        opt.zero_grad_range(ParamType::Means, noise_start, noise_count);
                        opt.zero_grad_range(ParamType::Sh0, noise_start, noise_count);
                        opt.zero_grad_range(ParamType::ShN, noise_start, noise_count);
                        opt.zero_grad_range(ParamType::Scaling, noise_start, noise_count);
                        opt.zero_grad_range(ParamType::Rotation, noise_start, noise_count);
                    
                        if (in_finetune) {
                            // FineTune: surface is completely frozen; ONLY noise opacity can change.
                            opt.zero_grad_range(ParamType::Means, 0, surface_count);
                            opt.zero_grad_range(ParamType::Sh0, 0, surface_count);
                            opt.zero_grad_range(ParamType::ShN, 0, surface_count);
                            opt.zero_grad_range(ParamType::Scaling, 0, surface_count);
                            opt.zero_grad_range(ParamType::Rotation, 0, surface_count);
                            opt.zero_grad_range(ParamType::Opacity, 0, surface_count);
                            // NOTE: do NOT zero noise opacity here (we want it trainable).
                        } else if (iter < end_iter) {
                            // GuidedSurface: freeze noise opacity too, and freeze surface SH to avoid texture drift under random noise colors.
                            opt.zero_grad_range(ParamType::Opacity, noise_start, noise_count);
                            opt.zero_grad_range(ParamType::Sh0, 0, surface_count);
                            opt.zero_grad_range(ParamType::ShN, 0, surface_count);
                        }
                    }
                    if (!in_sparsification && !ngs_block_topology) {
                        strategy_->post_backward(iter, r_output);
                    }

                    // Skip strategy step if we're in controller distillation phase and freeze is enabled
                    const int ppisp_activation_step = params_.optimization.resolved_ppisp_controller_activation_step();
                    const bool freeze_gaussians = ppisp_controller_pool_ &&
                                                  params_.optimization.ppisp_use_controller &&
                                                  params_.optimization.ppisp_freeze_gaussians_on_distill &&
                                                  iter >= ppisp_activation_step;
                    if (!freeze_gaussians) {
                        strategy_->step(iter);
                    }
                }
                
                // During NGS noise phase we keep ordering/size stable (no pruning/densification).
                if (ngs_current_phase_ != NGSPhase::WithNoise) {
                    if (auto result = handle_sparsity_update(iter, strategy_->get_model()); !result) {
                        LOG_ERROR("Sparsity update: {}", result.error());
                    }
                    if (auto result = apply_sparsity_pruning(iter, strategy_->get_model()); !result) {
                        LOG_ERROR("Sparsity pruning: {}", result.error());
                    }
                }

                // Clean evaluation - let the evaluator handle everything
                if (evaluator_->is_enabled() && evaluator_->should_evaluate(iter)) {
                    evaluator_->print_evaluation_header(iter);
                    auto metrics = evaluator_->evaluate(iter,
                                                        strategy_->get_model(),
                                                        val_dataset_,
                                                        background_);
                    LOG_INFO("{}", metrics.to_string());
                }

                // Save checkpoint (not PLY) at specified steps                
                if (!params_.optimization.skip_intermediate) {
                    for (size_t save_step : params_.optimization.save_steps) {
                        if (iter == static_cast<int>(save_step) && iter != params_.optimization.iterations) {
                            auto result = save_checkpoint(iter);
                            if (!result) {
                                LOG_WARN("Failed to save checkpoint at iteration {}: {}", iter, result.error());
                            }
                        }
                    }
                }

                if (!params_.dataset.timelapse_images.empty() && iter % params_.dataset.timelapse_every == 0) {
                    for (const auto& img_name : params_.dataset.timelapse_images) {
                        auto train_cam = train_dataset_->get_camera_by_filename(img_name);
                        auto val_cam = val_dataset_ ? val_dataset_->get_camera_by_filename(img_name) : std::nullopt;
                        if (train_cam.has_value() || val_cam.has_value()) {
                            lfs::core::Camera* cam_to_use = train_cam.has_value() ? train_cam.value() : val_cam.value();

                            // Image size isn't correct until the image has been loaded once
                            // If we use the camera before it's loaded, it will render images at the non-scaled size
                            if ((cam_to_use->camera_height() == cam_to_use->image_height() && params_.dataset.resize_factor != 1) ||
                                cam_to_use->image_height() > params_.dataset.max_width ||
                                cam_to_use->image_width() > params_.dataset.max_width) {
                                cam_to_use->load_image_size(params_.dataset.resize_factor, params_.dataset.max_width);
                            }

                            RenderOutput rendered_timelapse_output;
                            if (params_.optimization.gut) {
                                rendered_timelapse_output = gsplat_rasterize(*cam_to_use, strategy_->get_model(), background_,
                                                                             1.0f, false, GsplatRenderMode::RGB, true);
                            } else {
                                rendered_timelapse_output = fast_rasterize(*cam_to_use, strategy_->get_model(), background_);
                            }

                            // Get folder name to save in by stripping file extension
                            std::string folder_name = lfs::io::strip_extension(img_name);

                            auto output_path = params_.dataset.output_path / "timelapse" / folder_name;
                            std::filesystem::create_directories(output_path);

                            lfs::core::image_io::save_image_async(output_path / std::format("{:06d}.jpg", iter),
                                                                  rendered_timelapse_output.image);
                        } else {
                            LOG_WARN("Timelapse image '{}' not found in dataset.", img_name);
                        }
                    }
                }
            }

            // Python hook: post-step (after optimizer and side-effects)
            {
                lfs::training::HookContext ctx{
                    .iteration = iter,
                    .loss = current_loss_.load(),
                    .num_gaussians = strategy_ ? strategy_->get_model().size() : 0,
                    .is_refining = strategy_ ? strategy_->is_refining(iter) : false,
                    .trainer = this};
                lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::SafeControl);
                lfs::training::CommandCenter::instance().update_snapshot(
                    ctx, params_.optimization.iterations, is_paused_.load(), is_running_.load(), stop_requested_.load(),
                    lfs::training::TrainingPhase::SafeControl);
                lfs::training::ControlBoundary::instance().notify(lfs::training::ControlHook::PostStep, ctx);
            }

            // Return Continue if we should continue training
            if (iter < params_.optimization.iterations && !stop_requested_.load() && !stop_token.stop_requested()) {
                return StepResult::Continue;
            } else {
                return StepResult::Stop;
            }
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Training step failed: {}", e.what()));
        }
    }

    std::expected<void, std::string> Trainer::train(std::stop_token stop_token) {
        // Check if initialized
        if (!initialized_.load()) {
            return std::unexpected("Trainer not initialized. Call initialize() before train()");
        }

        is_running_ = false;
        training_complete_ = false;
        ready_to_start_ = false; // Reset the flag
        lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::SafeControl);

        ready_to_start_ = true; // Skip GUI wait for now

        is_running_ = true; // Now we can start
        LOG_INFO("Starting training loop");
        auto& cache_loader = lfs::io::CacheLoader::getInstance();
        cache_loader.reset_cache();
        cache_loader.update_cache_params(params_.dataset.loading_params.use_cpu_memory,
                                         params_.dataset.loading_params.use_fs_cache,
                                         train_dataset_size_,
                                         params_.dataset.loading_params.min_cpu_free_GB,
                                         params_.dataset.loading_params.min_cpu_free_memory_ratio,
                                         params_.dataset.loading_params.print_cache_status,
                                         params_.dataset.loading_params.print_status_freq_num);

        // Notify Python control layer that training is starting
        {
            lfs::training::HookContext ctx{
                .iteration = 0,
                .loss = current_loss_.load(),
                .num_gaussians = strategy_ ? strategy_->get_model().size() : 0,
                .is_refining = strategy_ ? strategy_->is_refining(0) : false,
                .trainer = this};
            lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::SafeControl);
            lfs::training::CommandCenter::instance().update_snapshot(
                ctx, params_.optimization.iterations, is_paused_.load(), is_running_.load(), stop_requested_.load(),
                lfs::training::TrainingPhase::SafeControl);
            lfs::training::ControlBoundary::instance().notify(lfs::training::ControlHook::TrainingStart, ctx);
        }

        try {
            // Start from current_iteration_ (allows resume from checkpoint)
            int iter = current_iteration_.load() > 0 ? current_iteration_.load() + 1 : 1;
            const RenderMode render_mode = RenderMode::RGB;

            if (progress_) {
                progress_->update(iter, current_loss_.load(),
                                  static_cast<int>(strategy_->get_model().size()),
                                  strategy_->is_refining(iter));
            }

            // Conservative prefetch to avoid VRAM exhaustion
            lfs::io::PipelinedLoaderConfig pipelined_config;
            pipelined_config.jpeg_batch_size = 8;
            pipelined_config.prefetch_count = 8;
            pipelined_config.output_queue_size = 4;
            pipelined_config.io_threads = 2;

            // Non-JPEG images (PNG, WebP) need CPU decoding - use more threads until cache warms
            constexpr float NON_JPEG_THRESHOLD = 0.1f;
            constexpr size_t MIN_COLD_THREADS = 4;
            constexpr size_t COLD_PREFETCH_COUNT = 16;
            const float non_jpeg_ratio = train_dataset_->get_non_jpeg_ratio();
            if (non_jpeg_ratio > NON_JPEG_THRESHOLD) {
                const size_t cold_threads = std::max(MIN_COLD_THREADS,
                                                     static_cast<size_t>(std::thread::hardware_concurrency() / 2));
                pipelined_config.cold_process_threads = cold_threads;
                pipelined_config.prefetch_count = COLD_PREFETCH_COUNT;
                LOG_INFO("{:.0f}% non-JPEG images, using {} cold threads", non_jpeg_ratio * 100.0f, cold_threads);
            }

            const bool alpha_available = scene_ && scene_->imagesHaveAlpha();
            PipelinedMaskConfig mask_pipeline_config;
            if (params_.optimization.mask_mode != lfs::core::param::MaskMode::None) {
                mask_pipeline_config.invert_masks = params_.optimization.invert_masks;
                mask_pipeline_config.mask_threshold = params_.optimization.mask_threshold;
                if (params_.optimization.use_alpha_as_mask && alpha_available) {
                    mask_pipeline_config.use_alpha_as_mask = true;
                    mask_pipeline_config.load_masks = true;
                    LOG_INFO("Alpha-as-mask enabled (invert={}, threshold={})",
                             mask_pipeline_config.invert_masks, mask_pipeline_config.mask_threshold);
                } else {
                    mask_pipeline_config.load_masks = true;
                    LOG_INFO("Mask file loading enabled (invert={}, threshold={})",
                             mask_pipeline_config.invert_masks, mask_pipeline_config.mask_threshold);
                }
            }

            auto train_dataloader = create_infinite_pipelined_dataloader(
                train_dataset_, pipelined_config, mask_pipeline_config);
            auto active_image_loader_guard = makeScopeGuard([this]() {
                clearActiveImageLoader();
            });
            updateGTLoadConfigSnapshot();
            setActiveImageLoader(train_dataloader->get_loader_shared());
            strategy_->set_image_loader(train_dataloader->get_loader());

            LOG_DEBUG("Starting training iterations");
            while (iter <= params_.optimization.iterations) {
                lfs::core::Tensor::set_memory_pool_iteration(iter);

                if (stop_token.stop_requested() || stop_requested_.load())
                    break;
                if (callback_busy_.load(std::memory_order_acquire)) {
                    const cudaError_t callback_status = cudaStreamQuery(callback_stream_);
                    if (callback_status == cudaSuccess) {
                        callback_busy_.store(false, std::memory_order_release);
                    } else if (callback_status != cudaErrorNotReady) {
                        LOG_WARN("Callback stream query failed: {}", cudaGetErrorString(callback_status));
                        callback_busy_.store(false, std::memory_order_release);
                    }
                }

                lfs::core::Camera* cam = nullptr;
                lfs::core::Tensor gt_image;
                auto example_opt = train_dataloader->next();
                if (!example_opt) {
                    LOG_ERROR("DataLoader returned nullopt unexpectedly");
                    break;
                }
                auto& example = *example_opt;
                cam = example.data.camera;
                gt_image = std::move(example.data.image);

                // Store pipelined mask for use in train_step
                pipelined_mask_ = example.mask.has_value() ? std::move(*example.mask) : lfs::core::Tensor();

                auto step_result = train_step(iter, cam, gt_image, render_mode, stop_token);
                if (!step_result) {
                    // Check if this is an OOM_RETRY signal
                    if (step_result.error() == "OOM_RETRY") {
                        cudaDeviceSynchronize();
                        cudaGetLastError();

                        lfs::core::GlobalArenaManager::instance().get_arena().full_reset();
                        lfs::core::Tensor::trim_memory_pool();

                        cudaDeviceSynchronize();
                        cudaGetLastError();

                        LOG_INFO("OOM recovery: retrying iteration {}", iter);
                        step_result = train_step(iter, cam, gt_image, render_mode, stop_token);
                        if (!step_result) {
                            return std::unexpected(step_result.error());
                        }
                    } else {
                        return std::unexpected(step_result.error());
                    }
                }

                // Transition to safe control phase and execute deferred Python callbacks
                lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::SafeControl);
                lfs::training::ControlBoundary::instance().drain_callbacks();

                if (*step_result == StepResult::Stop) {
                    break;
                }

                // Launch callback for async progress update (except first iteration)
                if (iter > 1 && callback_ && !callback_busy_.load(std::memory_order_acquire)) {
                    callback_busy_.store(true, std::memory_order_release);
                    auto err = cudaLaunchHostFunc(
                        callback_stream_,
                        [](void* self) {
                            auto* trainer = static_cast<Trainer*>(self);
                            if (trainer->callback_) {
                                trainer->callback_();
                            }
                            trainer->callback_busy_.store(false, std::memory_order_release);
                        },
                        this);
                    if (err != cudaSuccess) {
                        LOG_WARN("Failed to launch callback: {}", cudaGetErrorString(err));
                        callback_busy_.store(false, std::memory_order_release);
                    }
                }

                ++iter;
            }

            clearActiveImageLoader();
            active_image_loader_guard.release();

            // Ensure callback is finished before final save
            if (callback_busy_.load()) {
                cudaStreamSynchronize(callback_stream_);
            }

            // -----------------------------------------------------------------
            // Post-training mask-based pruning
            //
            // For matting modes, remove splats that don't align with masks:
            // - Center-vote: splats whose center is outside mask in most views
            // - Leakage: splats whose footprint extends outside mask boundary
            // -----------------------------------------------------------------
            if (params_.optimization.mask_mode == lfs::core::param::MaskMode::HardMatting ||
                params_.optimization.mask_mode == lfs::core::param::MaskMode::SoftMatting ||
                params_.optimization.mask_mode == lfs::core::param::MaskMode::FocusedSegment)
            {
                mask_pruning::GeometricDomePruningConfig geomdome_cfg;
                //default values are ok

                mask_pruning::CenterVotePruningConfig center_cfg;
                center_cfg.enabled = true;

                // Conservative threshold — protects legitimate border splats (feet, arms).
                // Only removes Gaussians clearly outside the mask in a large majority of views.
                // In a dome with 104 cameras, a splat visible in 30 views can afford 8 bad-mask
                // views and still pass (73% good > 0.72).
                center_cfg.vote_ratio_threshold = 0.75f;
                // Moderate margin — avoids penalizing splats near frustum edges without
                // being as permissive as the original 0.25 that missed lateral floaters.
                center_cfg.border_safe_margin = 0.33f;
                center_cfg.enable_depth_filtering = true;
                center_cfg.min_visibility_count = 3;
                center_cfg.invert_masks = params_.optimization.invert_masks;

                mask_pruning::LeakagePruningConfig leak_cfg;
                leak_cfg.enabled = true;
                // Main tool for halos — removes elongated Gaussians whose footprint
                // extends outside the mask. Center vote is too permissive for these.
                leak_cfg.leak_keep_threshold = 0.70f;
                // 2 of 8 sample points outside counts as a leak per view.
                // Catches elongated splats extending above heads without being too strict
                // on border splats that legitimately straddle the mask edge.
                leak_cfg.per_view_leak_fraction = 0.25f;
                leak_cfg.min_visibility_count = 3;
                // Low radius — evaluate small/thin elongated splats that caused halos.
                // Original 2.0f missed these entirely.
                leak_cfg.min_pixel_radius = 1.0f;
                leak_cfg.sample_points = 8;
                // Small dilation — tolerates 3px at mask boundary to protect
                // extremities that straddle the mask edge in some views.
                leak_cfg.dilate_px = 2;
                leak_cfg.invert_masks = params_.optimization.invert_masks;

                
                mask_pruning::IsolationPruningConfig iso_cfg;
                iso_cfg.enabled = true;

                #ifdef NeedsPruningDiagnostics
                    std::filesystem::path diag_dir = params_.dataset.output_path / "pruning_diagnostics";

                    LOG_INFO("===== MASK-BASED PRUNING WITH VISUALIZATION =====");
                    bool viz_success = mask_pruning::visualizer::generate_pruning_diagnostics(
                        *strategy_,
                        *train_dataset_,
                        center_cfg,
                        diag_dir,
                        25 // Max 25 cámaras
                    );

                    if (viz_success) {
                        LOG_INFO("✓ Diagnostic images saved to: {}", diag_dir.string());
                        LOG_INFO("  Generated 6 layers per camera:");
                        LOG_INFO("    01_mask.jpg          - Mask only");
                        LOG_INFO("    02_centers_all.jpg   - All splat centers");
                        LOG_INFO("    03_centers_classified.jpg - Green=inside, Red=outside");
                        LOG_INFO("    04_footprints.jpg    - Splat ellipses");
                        LOG_INFO("    05_problem_splats.jpg - Outside + large radius");
                        LOG_INFO("    06_summary.jpg       - Summary overlay");
                    } else {
                        LOG_WARN("Failed to generate diagnostic visualizations");
                    }
                #endif
                LOG_INFO("Running post-training mask-based pruning...");


                auto pruning_result = mask_pruning::prune_after_training(
                    *strategy_,
                    *train_dataset_,
                    geomdome_cfg,
                    center_cfg,
                    leak_cfg,
                    iso_cfg);


                if (!pruning_result) {
                    LOG_WARN("Post-training pruning failed: {}", pruning_result.error());
                } else if (pruning_result->splats_removed > 0) {
                    LOG_INFO("Pruning complete: removed {} splats ({:.1f}%)",
                             pruning_result->splats_removed,
                             pruning_result->removal_ratio() * 100.0f);
                }
            }

            // Final save if not already saved by stop request
            if (!stop_requested_.load() && !stop_token.stop_requested()) {
                auto final_path = params_.dataset.output_path;
                save_ply(final_path, params_.optimization.iterations, /*join=*/true);
            }

            if (progress_) {
                progress_->complete();
            }
            evaluator_->save_report();
            if (progress_) {
                progress_->print_final_summary(static_cast<int>(strategy_->get_model().size()));
            }

            is_running_ = false;
            training_complete_ = true;

            cache_loader.clear_cpu_cache();
            lfs::core::image_io::wait_for_pending_saves();

            // Notify training end
            {
                lfs::training::HookContext ctx{
                    .iteration = current_iteration_.load(),
                    .loss = current_loss_.load(),
                    .num_gaussians = strategy_ ? strategy_->get_model().size() : 0,
                    .is_refining = strategy_ ? strategy_->is_refining(current_iteration_.load()) : false,
                    .trainer = this};
                lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::SafeControl);
                lfs::training::CommandCenter::instance().update_snapshot(
                    ctx, params_.optimization.iterations, is_paused_.load(), is_running_.load(), stop_requested_.load(),
                    lfs::training::TrainingPhase::SafeControl);
                lfs::training::ControlBoundary::instance().notify(lfs::training::ControlHook::TrainingEnd, ctx);
            }

            lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::Idle);

            LOG_INFO("Training completed successfully");
            return {};
        } catch (const std::exception& e) {
            is_running_ = false;
            cache_loader.clear_cpu_cache();
            lfs::core::image_io::wait_for_pending_saves();
            lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::Idle);

            return std::unexpected(std::format("Training failed: {}", e.what()));
        }
    }

    void Trainer::save_ply(const std::filesystem::path& save_path, const int iter_num, const bool join_threads) {
        const lfs::io::PlySaveOptions ply_options{
            .output_path = save_path / ("splat_" + std::to_string(iter_num) + ".ply"),
            .binary = true,
            .async = !join_threads};

        const auto ply_result = lfs::io::save_ply(strategy_->get_model(), ply_options);
        if (!ply_result) {
            if (ply_result.error().code == lfs::io::ErrorCode::INSUFFICIENT_DISK_SPACE) {
                lfs::core::events::state::DiskSpaceSaveFailed{
                    .iteration = iter_num,
                    .path = ply_options.output_path,
                    .error = ply_result.error().message,
                    .required_bytes = ply_result.error().required_bytes,
                    .available_bytes = ply_result.error().available_bytes,
                    .is_disk_space_error = true,
                    .is_checkpoint = false}
                    .emit();
            }
            LOG_WARN("Failed to save PLY: {}", ply_result.error().message);
            return; // Don't save checkpoint if PLY failed
        }

        
        // Save checkpoint alongside PLY for training resumption (skip if --skip-intermediate)
        if (!params_.optimization.skip_intermediate) {
            // Only save controller if training has reached activation step
            PPISPControllerPool* controller_to_save = nullptr;
            if (ppisp_controller_pool_ && iter_num >= params_.optimization.ppisp_controller_activation_step) {
                controller_to_save = controller_pool_for_save(iter_num);
            }

            // Save checkpoint alongside PLY for training resumption
            auto ckpt_result = lfs::training::save_checkpoint(save_path, iter_num, *strategy_, params_,
                                                              bilateral_grid_.get(), ppisp_.get(), controller_to_save);
            if (!ckpt_result) {
                LOG_WARN("Failed to save checkpoint: {}", ckpt_result.error());
            }

            if (ppisp_) {
                const auto ppisp_path = get_ppisp_companion_path(ply_options.output_path);
                std::optional<PPISPFileMetadata> metadata;
                if (auto metadata_result = build_ppisp_sidecar_metadata(); metadata_result) {
                    metadata = std::move(*metadata_result);
                } else {
                    LOG_WARN("Failed to build PPISP sidecar metadata for '{}': {}. Saving sidecar without metadata.",
                             lfs::core::path_to_utf8(ppisp_path), metadata_result.error());
                }
                const auto ppisp_result = save_ppisp_file(ppisp_path, *ppisp_, controller_to_save,
                                                          metadata ? &*metadata : nullptr);
                if (!ppisp_result) {
                    LOG_WARN("Failed to save PPISP file: {}", ppisp_result.error());
                }
            }
        }

        LOG_DEBUG("PLY save initiated: {} (sync={})", lfs::core::path_to_utf8(save_path), join_threads);
    }

    std::expected<void, std::string> Trainer::save_checkpoint(int iteration) {
        if (!strategy_) {
            return std::unexpected("Cannot save checkpoint: no strategy initialized");
        }

        PPISPControllerPool* controller_to_save = controller_pool_for_save(iteration);

        return lfs::training::save_checkpoint(params_.dataset.output_path, iteration, *strategy_, params_,
                                              bilateral_grid_.get(), ppisp_.get(), controller_to_save);
    }

    std::expected<void, std::string> Trainer::save_checkpoint_to(const std::filesystem::path& output_path,
                                                                 int iteration) {
        if (!strategy_) {
            return std::unexpected("Cannot save checkpoint: no strategy initialized");
        }

        PPISPControllerPool* controller_to_save = controller_pool_for_save(iteration);

        return lfs::training::save_checkpoint(output_path, iteration, *strategy_, params_,
                                              bilateral_grid_.get(), ppisp_.get(), controller_to_save);
    }

    lfs::core::Tensor Trainer::applyPPISPForViewport(const lfs::core::Tensor& rgb, const int camera_uid,
                                                     const PPISPViewportOverrides& overrides,
                                                     const bool use_controller) const {
        if (!ppisp_ || !params_.optimization.use_ppisp || rgb.shape().rank() != 3) {
            return rgb;
        }

        const bool is_chw = (rgb.shape()[0] == 3);
        const auto rgb_chw = is_chw ? rgb : rgb.permute({2, 0, 1}).contiguous();
        const bool is_training_camera = ppisp_->is_known_frame(camera_uid);
        const bool has_controller = ppisp_controller_pool_ && params_.optimization.ppisp_use_controller;

        lfs::core::Tensor result;

        if (use_controller && has_controller) {
            constexpr int CONTROLLER_IDX = 0;
            const auto controller_params = ppisp_controller_pool_->predict(CONTROLLER_IDX, rgb_chw.unsqueeze(0), 1.0f);
            result = overrides.isIdentity()
                         ? ppisp_->apply_with_controller_params(rgb_chw, controller_params, CONTROLLER_IDX)
                         : ppisp_->apply_with_controller_params_and_overrides(rgb_chw, controller_params, CONTROLLER_IDX,
                                                                              toRenderOverrides(overrides));
        } else if (is_training_camera) {
            const int camera_id = ppisp_->camera_for_frame(camera_uid);
            result = overrides.isIdentity() ? ppisp_->apply(rgb_chw, camera_id, camera_uid)
                                            : ppisp_->apply_with_overrides(rgb_chw, camera_id, camera_uid,
                                                                           toRenderOverrides(overrides));
        } else {
            const int fallback_camera = ppisp_->any_camera_id();
            const int fallback_frame = ppisp_->any_frame_uid();
            result = overrides.isIdentity() ? ppisp_->apply(rgb_chw, fallback_camera, fallback_frame)
                                            : ppisp_->apply_with_overrides(rgb_chw, fallback_camera, fallback_frame,
                                                                           toRenderOverrides(overrides));
        }

        return is_chw ? result : result.permute({1, 2, 0}).contiguous();
    }

    PPISPControllerPool* Trainer::controller_pool_for_save(const int iteration) const {
        if (!ppisp_controller_pool_) {
            return nullptr;
        }
        if (is_ppisp_frozen()) {
            return ppisp_controller_pool_.get();
        }
        return iteration >= params_.optimization.resolved_ppisp_controller_activation_step()
                   ? ppisp_controller_pool_.get()
                   : nullptr;
    }

    void Trainer::save_final_ply_and_checkpoint(const int iteration) {
        save_ply(params_.dataset.output_path, iteration, /*join=*/true);
    }

    std::expected<int, std::string> Trainer::load_checkpoint(const std::filesystem::path& checkpoint_path) {
        if (!strategy_) {
            return std::unexpected("Cannot load checkpoint: no strategy initialized");
        }

        // Create bilateral grid before loading if needed (checkpoint may contain grid state)
        if (params_.optimization.use_bilateral_grid && !bilateral_grid_) {
            if (auto init_result = initialize_bilateral_grid(); !init_result) {
                LOG_WARN("Failed to init bilateral grid for resume: {}", init_result.error());
            }
        }

        // Create PPISP before loading if needed
        if (params_.optimization.use_ppisp && !ppisp_) {
            if (auto init_result = initialize_ppisp(); !init_result) {
                LOG_WARN("Failed to init PPISP for resume: {}", init_result.error());
            }
        }

        // Create PPISP controller pool before loading if needed
        if (params_.optimization.ppisp_use_controller && !ppisp_controller_pool_) {
            bool should_initialize_controller = true;
            if (is_ppisp_frozen()) {
                const auto checkpoint_header = lfs::core::load_checkpoint_header(checkpoint_path);
                if (!checkpoint_header) {
                    LOG_WARN("Failed to inspect checkpoint header for PPISP controller state: {}",
                             checkpoint_header.error());
                    should_initialize_controller = false;
                } else {
                    should_initialize_controller =
                        lfs::core::has_flag(checkpoint_header->flags, lfs::core::CheckpointFlags::HAS_PPISP_CONTROLLER);
                    if (!should_initialize_controller) {
                        LOG_INFO("Checkpoint has no PPISP controller pool; frozen controller state remains disabled");
                    }
                }
            }
            if (should_initialize_controller) {
                if (auto init_result = initialize_ppisp_controller(); !init_result) {
                    LOG_WARN("Failed to init PPISP controller pool for resume: {}", init_result.error());
                }
            }
        }

        auto result = lfs::training::load_checkpoint(
            checkpoint_path, *strategy_, params_, bilateral_grid_.get(), ppisp_.get(),
            ppisp_controller_pool_.get());
        if (!result) {
            return result;
        }
        current_iteration_ = *result;

        LOG_INFO("Restored training state from checkpoint at iteration {}", *result);
        return result;
    }
        
    
    // ----------------------------------------------------------------------------
    // initialize_ngs() - Call from Trainer::initialize() after strategy init
    // ----------------------------------------------------------------------------

    std::expected<void, std::string> Trainer::initialize_ngs() {
        const auto& opt = params_.optimization;

        if (opt.ngs_noise_path.empty()) {
            ngs_enabled_ = false;
            return {};
        }

        LOG_INFO("Initializing NGS");

        // Create config with just the path
        NGSConfig config;
        config.noise_ply_path = opt.ngs_noise_path;

        // Create phase manager
        ngs_phase_manager_ = std::make_unique<NGSPhaseManager>(config, opt.iterations);

        // Load noise (but don't inject yet)
        ngs_noise_ = std::make_unique<NoiseGaussians>();
        if (auto result = ngs_noise_->load(opt.ngs_noise_path); !result) {
            return std::unexpected(result.error());
        }

        ngs_enabled_ = true;
        ngs_current_phase_ = NGSPhase::StandardTraining;
        ngs_surface_count_ = 0;
        ngs_noise_count_ = 0;

        const int start_iter = ngs_phase_manager_->noise_start();
        const int end_iter = ngs_phase_manager_->noise_end();

         // Conservative NGS schedule:
         // - [start_iter, ngs_finetune_end_iter_) : train ONLY noise opacity (surface fully frozen)
         // - [ngs_finetune_end_iter_, end_iter)   : train surface (noise fully frozen) and freeze surface SH to avoid color/light drift
         // This is intentionally quality-first and avoids NGS-induced texture degradation.
         const int finetune_len = std::max(1, static_cast<int>(0.03f * opt.iterations)); // ~3% of total iters
         ngs_finetune_end_iter_ = std::min(end_iter, start_iter + finetune_len);

        LOG_INFO("NGS ready: {} noise Gaussians, inject at iter {}, finetune_end {}, remove at iter {}",
                ngs_noise_->size(), start_iter, ngs_finetune_end_iter_, end_iter);

        return {};
    }

    // ----------------------------------------------------------------------------
    // handle_ngs_phase_transition() - Call at START of train_step()
    // ----------------------------------------------------------------------------

    void Trainer::handle_ngs_phase_transition(int iter) {
        if (!ngs_enabled_ || !ngs_phase_manager_)
            return;

        const NGSPhase new_phase = ngs_phase_manager_->get_phase(iter);

        // Phase transition
        if (new_phase != ngs_current_phase_) {
            const NGSPhase old_phase = ngs_current_phase_;
            ngs_current_phase_ = new_phase;

            if (old_phase == NGSPhase::StandardTraining && new_phase == NGSPhase::WithNoise) {
                LOG_INFO("NGS: Injecting {} noise Gaussians at iteration {}", ngs_noise_ ? ngs_noise_->size() : 0, iter);
                inject_noise_into_model();
            } else if (old_phase == NGSPhase::WithNoise && new_phase == NGSPhase::Cleanup) {
                LOG_INFO("NGS: Removing noise Gaussians at iteration {}", iter);
                remove_noise_from_model();
            }
        }

        // Per-iteration operations during noise phase
        if (ngs_current_phase_ == NGSPhase::WithNoise && ngs_surface_count_ > 0 && ngs_noise_count_ > 0) {
            auto& model = strategy_->get_model();
            auto& optimizer = strategy_->get_optimizer();

                        // Randomize noise colors (reduced frequency to stabilize optimization)
                        constexpr int kColorRandomizePeriod = 8; // quality-first: less jitter
                        if ((iter % kColorRandomizePeriod) == 0 || (ngs_phase_manager_ && iter == ngs_phase_manager_->noise_start())) {
                            ngs_randomize_sh0_range_inplace(model.sh0(), ngs_surface_count_, ngs_noise_count_, static_cast<uint32_t>(iter));
                        }

                        // NOTE: actual freezing logic is applied after backward (see train_step).
        }
    }

    // ----------------------------------------------------------------------------
    // inject_noise_into_model() - Add noise to SplatData and expand optimizer
    // ----------------------------------------------------------------------------

    void Trainer::inject_noise_into_model() {
        if (!ngs_noise_) {
            LOG_WARN("NGS: No noise loaded, cannot inject");
            return;
        }

        auto& model = strategy_->get_model();
        auto& optimizer = strategy_->get_optimizer();

        ngs_surface_count_ = model.size();
        ngs_noise_count_ = ngs_noise_->size();
        
        const size_t noise_count = ngs_noise_count_;
        if (noise_count == 0) {
            LOG_WARN("NGS: Noise PLY contains 0 gaussians");
            return;
        }

        LOG_DEBUG("NGS inject: {} surface + {} noise", ngs_surface_count_, noise_count);

        // Concatenate parameter tensors (noise provides means/scaling/rotation/opacity only)
        auto new_means = lfs::core::Tensor::cat({model.means(), ngs_noise_->means()}, 0);
        auto new_scaling = lfs::core::Tensor::cat({model.scaling_raw(), ngs_noise_->scaling_raw()}, 0);
        auto new_rotation = lfs::core::Tensor::cat({model.rotation_raw(), ngs_noise_->rotation_raw()}, 0);
        auto new_opacity = lfs::core::Tensor::cat({model.opacity_raw(), ngs_noise_->opacity_raw()}, 0);
        
        // SH0: allocate zeros for noise, then randomize in-place per-iteration (model-side)
        lfs::core::Tensor new_sh0;
        {
            const auto& shape = model.sh0().shape();
            std::vector<size_t> dims = {noise_count};
            for (size_t i = 1; i < shape.rank(); ++i) dims.push_back(shape[i]);
            lfs::core::TensorShape noise_shape(dims);

            auto noise_sh0 = lfs::core::Tensor::zeros(noise_shape, model.sh0().device(), model.sh0().dtype());
            new_sh0 = lfs::core::Tensor::cat({model.sh0(), noise_sh0}, 0);
        }

        // SHN: noise has no higher-order SH; append zeros if surface uses SHN
        lfs::core::Tensor new_shN;
        if (model.shN().is_valid() && model.shN().numel() > 0) {
            const auto& shape = model.shN().shape();
            std::vector<size_t> dims = {noise_count};
            for (size_t i = 1; i < shape.rank(); ++i) dims.push_back(shape[i]);
            lfs::core::TensorShape noise_shape(dims);

            auto noise_shN = lfs::core::Tensor::zeros(noise_shape, model.shN().device(), model.shN().dtype());             
            new_shN = lfs::core::Tensor::cat({model.shN(), noise_shN}, 0);
        }

        // Update model tensors
        model.set_means_internal(new_means);
        model.set_sh0_internal(new_sh0);
        model.set_scaling_internal(new_scaling);
        model.set_rotation_internal(new_rotation);
        model.set_opacity_internal(new_opacity);
        if (new_shN.is_valid()) {
            model.set_shN_internal(new_shN);
        }

        // Expand optimizer state for new Gaussians
        optimizer.extend_for_new_gaussians(noise_count);

        LOG_INFO("NGS: Model now has {} Gaussians ({} surface + {} noise)",
                 model.size(), ngs_surface_count_, noise_count);
    }

    // ----------------------------------------------------------------------------
    // remove_noise_from_model() - Remove noise from SplatData and shrink optimizer
    // ----------------------------------------------------------------------------

    // Helper: remove [s,e) along dim0 without ever calling slice on an empty range.
    static lfs::core::Tensor remove_range_dim0_no_empty(const lfs::core::Tensor& src, int s, int e, int total) {
        // Preconditions: 0 <= s < e <= total
        const bool keep_prefix = (s > 0);
        const bool keep_suffix = (e < total);

        if (keep_prefix && keep_suffix) {
            // Keep [0,s) + [e,total)
            return lfs::core::Tensor::cat({src.slice(0, 0, s), src.slice(0, e, total)}, 0).contiguous();
        } else if (keep_prefix) {
            // Keep [0,s)
            return src.slice(0, 0, s).contiguous();
        } else if (keep_suffix) {
            // Keep [e,total)
            return src.slice(0, e, total).contiguous();
        }

        // Would remove everything -> shouldn't happen
        LOG_ERROR("NGS: remove_range_dim0 would remove entire tensor (total={})", total);
        return src;
    }

    void Trainer::remove_noise_from_model() {
        auto& model = strategy_->get_model();
        auto& optimizer = strategy_->get_optimizer();

        if (ngs_surface_count_ == 0 || ngs_noise_count_ == 0) {
            LOG_WARN("NGS: Cannot remove noise - nothing injected");
            return;
        }

        const size_t total_sz = model.size();
        const size_t start_sz = ngs_surface_count_;
        const size_t end_sz = std::min(total_sz, start_sz + ngs_noise_count_);

        if (start_sz >= total_sz || end_sz <= start_sz) {
            LOG_WARN("NGS: Cannot remove noise - invalid range (start={}, end={}, total={})",
                     start_sz, end_sz, total_sz);
            return;
        }

        const int t = static_cast<int>(total_sz);
        const int s = static_cast<int>(start_sz);
        const int e = static_cast<int>(end_sz);

        // IMPORTANT: do NOT call slice() with [t,t)
        model.set_means_internal(remove_range_dim0_no_empty(model.means(), s, e, t));
        model.set_sh0_internal(remove_range_dim0_no_empty(model.sh0(), s, e, t));
        model.set_scaling_internal(remove_range_dim0_no_empty(model.scaling_raw(), s, e, t));
        model.set_rotation_internal(remove_range_dim0_no_empty(model.rotation_raw(), s, e, t));
        model.set_opacity_internal(remove_range_dim0_no_empty(model.opacity_raw(), s, e, t));

        if (model.shN().is_valid() && model.shN().numel() > 0) {
            model.set_shN_internal(remove_range_dim0_no_empty(model.shN(), s, e, t));
        }

        // Remove optimizer state for the same range (also must avoid empty slices inside it!)
        optimizer.remove_range(start_sz, end_sz - start_sz);

        LOG_INFO("NGS: Removed {} noise Gaussians, model now has {} Gaussians",
                 (end_sz - start_sz), model.size());

        ngs_noise_.reset();
        ngs_surface_count_ = 0;
        ngs_noise_count_ = 0;
    }


} // namespace lfs::training
