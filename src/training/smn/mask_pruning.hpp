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
            float vote_ratio_threshold = 0.90f;

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

        /**
         * @brief Configuration for isolation pruning (3D nearest-neighbor outliers).
         *
         * Goal:
         *   Remove splats that are spatially isolated in 3D (typical "flyers" / small outlier islands).
         *
         * Core idea:
         *   For each splat i, compute d_k(i) = distance to its kth nearest neighbor (excluding self),
         *   obtained from a k-NN query.
         *
         * Robust global threshold (applies to ALL splats):
         *   Compute global_median = median( d_k(i) ) over ALL splats.
         *   Then remove splat i if:
         *
         *     d_k(i) > max(abs_distance_min, threshold_multiplier * global_median)
         *
         * Notes:
         * - This pass is independent from masks/frustum; it works purely in 3D space.
         * - Designed to catch very small outlier groups (1-3 splats). Using kth_neighbor=4 makes those groups detectable.
         */
        struct IsolationPruningConfig {
            /**
             * @brief Enable/disable this pass.
             */
            bool enabled = true;

            /**
             * @brief Number of neighbors to query in the k-NN search (excluding self).
             *
             * Constraints:
             * - Must be >= kth_neighbor.
             * - Typical values: 8 or 16.
             *
             * Performance:
             * - Larger values increase query cost, but can stabilize neighbor-distance selection.
             */
            int k_neighbors = 8;

            /**
             * @brief Which neighbor distance is used as the per-splat isolation metric (excluding self).
             *
             * Example:
             * - kth_neighbor = 4 uses d4 = distance to the 4th nearest neighbor.
             *
             * Rationale:
             * - d1 (nearest) can fail for small outlier clusters (2-3 flyers "protect" each other).
             * - d4 is robust for clusters of size up to 3, because the 4th neighbor typically lies in the main body.
             *
             * Constraints:
             * - 1 <= kth_neighbor <= k_neighbors
             */
            int kth_neighbor = 4;

            /**
             * @brief Global relative threshold multiplier (applies equally to all splats).
             *
             * Removal rule (per splat i):
             *   d_k(i) > threshold_multiplier * global_median
             *
             * Typical stable ranges (people, meters):
             * - 6.0  : more aggressive (removes more isolated points)
             * - 8.0  : conservative / stable default
             * - 16+  : very conservative
             */
            float threshold_multiplier = 16.0f;

            /**
             * @brief Optional absolute minimum distance threshold in meters.
             *
             * Removal uses:
             *   d_k(i) > max(abs_distance_min, threshold_multiplier * global_median)
             *
             * This prevents edge cases where global_median becomes extremely small and the relative threshold
             * becomes too strict.
             *
             * Set to 0 to disable. Typical values if enabled: 0.03-0.05 (3-5 cm).
             */
            float abs_distance_min = 0.0f;
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
         * @brief Isolation pruning in 3D space as neighbor outliers
         */
        std::expected<PruningResult, std::string> prune_by_isolation_distance(
            IStrategy& strategy,
            const IsolationPruningConfig& config);

        /**
         * @brief Run both center-vote and leakage pruning (recommended workflow).
         */
        std::expected<PruningResult, std::string> prune_after_training(
            IStrategy& strategy,
            const CameraDataset& dataset,
            const CenterVotePruningConfig& center_config = DEFAULT_CONFIG,
            const LeakagePruningConfig& leakage_config = {},
            const IsolationPruningConfig& isolation_config = {});

    } // namespace mask_pruning
} // namespace lfs::training