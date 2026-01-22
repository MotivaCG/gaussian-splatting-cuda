/* ScanMeNow file */

#pragma once

/**
 * @file mask_pruning.hpp
 * @brief Post-training pruning for masked Gaussian splatting WITH DEPTH FILTERING.
 */

#include "core/camera.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "dataset.hpp"
#include "strategies/istrategy.hpp"

#include <expected>
#include <string>

namespace lfs::training {
    namespace mask_pruning {

        // =============================================================================
        // Internal structures
        // =============================================================================

        /**
         * @brief Result of projecting splats to 2D.
         */
        struct ProjectionResult {
            lfs::core::Tensor radii;   // [N, 2] int32
            lfs::core::Tensor means2d; // [N, 2] float32
            lfs::core::Tensor depths;  // [N] float32
            lfs::core::Tensor conics;  // [N, 3] float32
        };

        // =============================================================================
        // Configuration
        // =============================================================================

        /**
         * @brief Configuration for center-vote pruning.
         */
        struct CenterVotePruningConfig {
            /// Minimum ratio of views where center must be inside mask to keep splat
            float vote_ratio_threshold = 0.80f;

            /// Minimum number of views where splat must be visible to be considered
            int min_visibility_count = 3;

            /// Projection parameters
            float eps2d = 0.3f;
            float near_plane = 0.01f;
            float far_plane = 10000.0f;
            float radius_clip = 0.0f;
            float scaling_modifier = 1.0f;
            // Margin relative to image size (0.25 = 25%) to ignore splats just outside the frustum.
            // Splats in this "grey zone" are neither penalized nor rewarded (ignored).
            float border_safe_margin = 0.25f; // margin to avoid mark splats like out of the mask

            bool invert_masks = false;

            // ========= DEPTH FILTERING PARAMETERS =========

            /// Enable depth-based occlusion filtering
            bool enable_depth_filtering = true;

            /// Number of robust standard deviations (MAD-based) for acceptable depth range
            float depth_filter_sigma_multiplier = 2.5f;

            /// Minimum number of inside-mask splats required to compute depth statistics
            int min_splats_for_depth_stats = 10;
        };

        /// Default configuration
        inline constexpr CenterVotePruningConfig DEFAULT_CONFIG{};

        /**
         * @brief Configuration for leakage pruning (footprint outside mask).
         */
        struct LeakagePruningConfig {
            /// Enable/disable leakage pass
            bool enabled = true;

            /// Keep threshold for "non-leak ratio" = 1 - leak_views / evaluated_views
            float leak_keep_threshold = 0.90f;

            /// Per-view: if outside_samples_ratio > this => that view counts as leaking
            float per_view_leak_fraction = 0.50f;

            /// Minimum number of evaluated views
            int min_visibility_count = 1;

            /// Minimum projected pixel radius to consider for leakage eval
            float min_pixel_radius = 2.0f;

            /// Number of boundary sample points: 4 or 8
            int sample_points = 8;

            /// Dilate mask by this many pixels for a "tolerant region"
            int dilate_px = 0;

            /// Projection parameters (typically match CenterVote)
            float eps2d = 0.3f;
            float near_plane = 0.01f;
            float far_plane = 10000.0f;
            float radius_clip = 0.0f;
            float scaling_modifier = 1.0f;

            bool invert_masks = false;
        };

        // =============================================================================
        // Results
        // =============================================================================

        struct PruningResult {
            int splats_before = 0;
            int splats_after = 0;
            int splats_removed = 0;
            bool success = false;

            float removal_ratio() const {
                return (splats_before > 0) ? static_cast<float>(splats_removed) / static_cast<float>(splats_before) : 0.0f;
            }
        };

        // =============================================================================
        // Internal operations (used by visualizer)
        // =============================================================================

        /**
         * @brief Project splats to 2D for one camera.
         * Internal function exposed for use by visualizer.
         */
        std::expected<ProjectionResult, std::string> project_splats(
            const lfs::core::Camera& camera,
            const lfs::core::SplatData& splat_data,
            const CenterVotePruningConfig& config);

        // =============================================================================
        // Main operations
        // =============================================================================

        /**
         * @brief Remove splats whose projected center doesn't fall inside mask consistently.
         */
        std::expected<PruningResult, std::string> prune_by_center_vote(
            IStrategy& strategy,
            const CameraDataset& dataset,
            const CenterVotePruningConfig& config = DEFAULT_CONFIG);

        /**
         * @brief Remove splats whose footprint leaks outside the mask too often.
         */
        std::expected<PruningResult, std::string> prune_by_mask_leakage(
            IStrategy& strategy,
            const CameraDataset& dataset,
            const LeakagePruningConfig& config);

        /**
         * @brief Run both center-vote and leakage pruning (recommended workflow).
         */
        std::expected<PruningResult, std::string> prune_after_training(
            IStrategy& strategy,
            const CameraDataset& dataset,
            const CenterVotePruningConfig& center_config = DEFAULT_CONFIG,
            const LeakagePruningConfig& leakage_config = {});

    } // namespace mask_pruning
} // namespace lfs::training