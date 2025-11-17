/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "components/bilateral_grid.hpp"
#include "components/poseopt.hpp"
#include "components/sparsity_optimizer.hpp"
#include "core/events.hpp"
#include "core/parameters.hpp"
#include "dataset.hpp"
#include "metrics/metrics.hpp"
#include "optimizers/scheduler.hpp"
#include "progress.hpp"
#include "project/project.hpp"
#include "rasterization/rasterizer.hpp"
#include "strategies/istrategy.hpp"
#include <ATen/cuda/CUDAEvent.h>
#include <atomic>
#include <expected>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stop_token>
#include <torch/torch.h>

// Uncomment ONE of these to select background mode:
//#define USE_SINGLE_COLOR_BACKGROUND
#define USE_FULL_NOISE_BACKGROUND

// Default to sine if nothing is defined
#if !defined(USE_SINGLE_COLOR_BACKGROUND) && !defined(USE_FULL_NOISE_BACKGROUND)
#define USE_SINGLE_COLOR_BACKGROUND
#endif

// Ensure only one is defined
#if defined(USE_SINGLE_COLOR_BACKGROUND) && defined(USE_FULL_NOISE_BACKGROUND)
#error "Cannot define both USE_SINGLE_COLOR_BACKGROUND and USE_FULL_NOISE_BACKGROUND"
#endif

// Forward declaration
class Camera;

namespace gs::training {
    class Trainer {
    public:
        // Constructor that takes ownership of strategy and shares datasets
        Trainer(std::shared_ptr<CameraDataset> dataset,
                std::unique_ptr<IStrategy> strategy,
                std::optional<std::tuple<std::vector<std::string>, std::vector<std::string>>> provided_splits);

        // Delete copy operations
        Trainer(const Trainer&) = delete;

        Trainer& operator=(const Trainer&) = delete;

        // Allow move operations
        Trainer(Trainer&&) = default;

        Trainer& operator=(Trainer&&) = default;

        ~Trainer();

        // Initialize trainer - must be called before training
        std::expected<void, std::string> initialize(const param::TrainingParameters& params);

        // Check if trainer is initialized
        bool isInitialized() const { return initialized_.load(); }

        // Main training method with stop token support
        std::expected<void, std::string> train(std::stop_token stop_token = {});

        // Control methods for GUI interaction
        void request_pause() { pause_requested_ = true; }
        void request_resume() { pause_requested_ = false; }
        void request_save() { save_requested_ = true; }
        void request_stop() { stop_requested_ = true; }

        bool is_paused() const { return is_paused_.load(); }
        bool is_running() const { return is_running_.load(); }
        bool is_training_complete() const { return training_complete_.load(); }
        bool has_stopped() const { return stop_requested_.load(); }

        // Get current training state
        int get_current_iteration() const { return current_iteration_.load(); }
        float get_current_loss() const { return current_loss_.load(); }

        // just for viewer to get model
        const IStrategy& get_strategy() const { return *strategy_; }

        // Allow viewer to lock for rendering
        std::shared_mutex& getRenderMutex() const { return render_mutex_; }

        const param::TrainingParameters& getParams() const { return params_; }

        std::shared_ptr<const Camera> getCamById(int camId) const;

        std::vector<std::shared_ptr<const Camera>> getCamList() const;

        void setProject(std::shared_ptr<gs::management::Project> project) { lf_project_ = project; }

        void load_cameras_info();

    private:
        // Helper for deferred event emission to prevent deadlocks
        struct DeferredEvents {
            std::vector<std::function<void()>> events;

            template <typename Event>
            void add(Event&& e) {
                events.push_back([e = std::move(e)]() { e.emit(); });
            }

            ~DeferredEvents() {
                for (auto& e : events)
                    e();
            }
        };

        // Training step result
        enum class StepResult {
            Continue,
            Stop,
            Error
        };

        // Returns the background color to use at a given iteration
        torch::Tensor& background_for_step(int iter, int width, int height);

        // Protected method for processing a single training step
        std::expected<StepResult, std::string> train_step(
            int iter,
            Camera* cam,
            torch::Tensor gt_image,
            torch::Tensor weights,
            torch::Tensor inv_frequency,
            RenderMode render_mode,
            bool out_of_mask_penalty,
            std::stop_token stop_token = {});

        // Protected methods for computing loss
        std::expected<torch::Tensor, std::string> compute_photometric_loss(
            const RenderOutput& render_output,
            const torch::Tensor& gt_image,
            const torch::Tensor& inv_freq,
            const SplatData& splatData,
            const param::OptimizationParameters& opt_params);
            
        std::expected<torch::Tensor, std::string> compute_photometric_loss(
            const RenderOutput& render_output,
            const torch::Tensor& gt_image,
            const torch::Tensor& weights,
            const torch::Tensor& inv_freq,
            const float outOfMaskAlphaPenalty,
            const SplatData& splatData,
            const param::OptimizationParameters& opt_params);

        void maybe_alpha_boost(const float boost_value, const float alpha_min, const int min_visible_views);

        std::expected<float, std::string> compute_scale_reg_loss(
            SplatData& splatData,
            const param::OptimizationParameters& opt_params);

        std::expected<float, std::string> compute_opacity_reg_loss(
            SplatData& splatData,
            const param::OptimizationParameters& opt_params);

        std::expected<torch::Tensor, std::string> compute_bilateral_grid_tv_loss(
            const std::unique_ptr<BilateralGrid>& bilateral_grid,
            const param::OptimizationParameters& opt_params);

        // Sparsity-related methods
        std::expected<torch::Tensor, std::string> compute_sparsity_loss(
            int iter,
            const SplatData& splatData);

        std::expected<void, std::string> handle_sparsity_update(
            int iter,
            SplatData& splatData);

        std::expected<void, std::string> apply_sparsity_pruning(
            int iter,
            SplatData& splatData);

        // Cleanup method for re-initialization
        void cleanup();

        std::expected<void, std::string> initialize_bilateral_grid();

        // Handle control requests
        void handle_control_requests(int iter, std::stop_token stop_token = {});

        // Prune gaussians using masks
        void prune_by_center_vote(float center_keep_threshold, int min_visibility_count = 3);
        
        
        /**
        * @brief Prune by mask leakage (success-rate semantics) with optional mask dilation.
        *
        * For each visible Gaussian whose 2D center is inside the (dilated) mask and whose
        * projected footprint is non-trivial, sample P points on its projected ellipse.
        * We count a per-view "leak" if the fraction of sampled points that fall outside the
        * (dilated) mask exceeds per_view_leak_frac. A Gaussian is KEPT only if:
        *     (views_without_leak / candidate_views) >= leak_keep_threshold.
        *
        * - Visibility: views where the Gaussian is not visible do NOT contribute to ratios.
        * - Mask dilation: if dilate_px > 0, the mask is dilated by that many pixels (tolerance).
        *
        * @param leak_keep_threshold   Required success rate of "no-leak" to KEEP a Gaussian. [0..1]
        * @param min_pixel_radius      Minimum pixel radius to consider a footprint (filters micro-splats).
        * @param min_center_mask       Minimum mask value at the center to consider "center inside".
        * @param sample_points         4 (NESW) or 8 (NESW+diagonals).
        * @param dilate_px             Mask dilation in pixels (tolerance at boundary). 0 = no dilation.
        * @param per_view_leak_frac    Per-view leak threshold in [0..1]. Count a view as "leak" only if
        *                              the fraction of sampled contour points outside the (dilated) mask
        *                              exceeds this value. 0.0 means "any point outside" counts as leak.
        */
        void prune_by_mask_leakage(float leak_keep_threshold,
                                         float min_pixel_radius = 1.5f,
                                         float min_center_mask = 0.5f,
                                         int sample_points = 8,
                                         int dilate_px = 25,
                                         float per_view_leak_frac = 0.0f);

        void prune_after_training(float center_keep_threshold, float leak_keep_threshold);
        
        void save_ply(const std::filesystem::path& save_path, int iter_num, bool join_threads = true);

        // Member variables
        std::shared_ptr<CameraDataset> base_dataset_;
        std::shared_ptr<CameraDataset> train_dataset_;
        std::shared_ptr<CameraDataset> val_dataset_;
        std::unique_ptr<IStrategy> strategy_;
        param::TrainingParameters params_;
        std::optional<std::tuple<std::vector<std::string>, std::vector<std::string>>> provided_splits_;

        torch::Tensor background_{};
        torch::Tensor bg_mix_buffer_;
        torch::Tensor noise_buffer_; // Full HxWx3 noise buffer
        std::unique_ptr<TrainingProgress> progress_;
        size_t train_dataset_size_ = 0;

        // Bilateral grid components
        std::unique_ptr<BilateralGrid> bilateral_grid_;
        std::unique_ptr<torch::optim::Adam> bilateral_grid_optimizer_;
        std::unique_ptr<WarmupExponentialLR> bilateral_grid_scheduler_;

        std::unique_ptr<PoseOptimizationModule> poseopt_module_; // Pose optimization module
        std::unique_ptr<torch::optim::Adam> poseopt_optimizer_;  // Optimizer for pose optimization

        // Sparsity optimizer
        std::unique_ptr<ISparsityOptimizer> sparsity_optimizer_;

        // Metrics evaluator - handles all evaluation logic
        std::unique_ptr<MetricsEvaluator> evaluator_;

        // Single mutex that protects the model during training
        mutable std::shared_mutex render_mutex_;

        // Mutex for initialization to ensure thread safety
        mutable std::mutex init_mutex_;

        // Control flags for thread communication
        std::atomic<bool> pause_requested_{false};
        std::atomic<bool> save_requested_{false};
        std::atomic<bool> stop_requested_{false};
        std::atomic<bool> is_paused_{false};
        std::atomic<bool> is_running_{false};
        std::atomic<bool> training_complete_{false};
        std::atomic<bool> ready_to_start_{false};
        std::atomic<bool> initialized_{false};

        // Current training state
        std::atomic<int> current_iteration_{0};
        std::atomic<float> current_loss_{0.0f};

        // Callback system for async operations
        std::function<void()> callback_;
        std::atomic<bool> callback_busy_{false};
        at::cuda::CUDAStream callback_stream_ = at::cuda::getStreamFromPool(false);
        at::cuda::CUDAEvent callback_launch_event_;

        // camera id to cam
        std::map<int, std::shared_ptr<const Camera>> m_cam_id_to_cam;

        // LichtFeld project
        std::shared_ptr<gs::management::Project> lf_project_ = nullptr;
    };
} // namespace gs::training
