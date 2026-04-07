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
         * @brief Configuration for geometric pruning passes that don't require mask evaluation.
         *
         * Groups two cheap geometric filters that can run before any mask-based pruning:
         *
         * 1. Behind-camera pruning: In a dome capture rig, all cameras point inward toward
         *    the subject. Any Gaussian that lies behind the optical plane of any camera is
         *    by definition outside the dome volume — it cannot have been seen from that
         *    camera during training and is therefore a floater or reconstruction artifact.
         *    This test reduces to a single dot product per Gaussian per camera and requires
         *    no masks, no projection, and no GPU synchronization.
         *
         * 2. Floor pruning: Removes Gaussians below a world-space Y threshold. In a dome
         *    setup the floor plane is well defined — anything reconstructed below ground
         *    level is an artifact of floor reflections, shadow Gaussians, or noise from
         *    the lower camera ring. The threshold should be set slightly below the lowest
         *    legitimate point of the subject (e.g. -0.01f for 1cm below floor level).
         *
         * Both passes operate entirely in 3D world space on CPU, making them the cheapest
         * possible pre-filter before the more expensive mask-based passes (center vote,
         * leakage, isolation). Running them first reduces the splat count for subsequent
         * passes and avoids false positives from floaters that could skew voting statistics.
         *
         * @note The behind-camera test assumes all cameras point inward, which is true for
         *       dome rigs but may not hold for general capture setups. Disable
         *       enable_behind_camera if using outward-facing or free-placement cameras.
         *
         * @note The floor_y threshold depends on the world-space coordinate convention of
         *       the COLMAP reconstruction. Verify the Y axis direction before setting this
         *       value — in some COLMAP outputs Y points downward, which would invert the
         *       expected threshold sign.
         */
        struct GeometricDomePruningConfig {
            /// Enable/disable this entire pass
            bool enabled = true;

            /// Enable behind-camera geometric filter.
            /// Removes any Gaussian whose position lies behind the optical plane of at
            /// least one camera. In an inward-facing dome rig this is equivalent to
            /// "outside the dome volume" — a necessary condition for a floater.
            bool enable_behind_camera = true;

            /// Tolerance for the behind-camera dot product test (world units).
            /// A Gaussian at dot product distance <= behind_tolerance from the camera
            /// plane is considered behind that camera. The small negative default (-0.01f)
            /// avoids false positives for Gaussians that lie exactly on the plane due to
            /// floating-point precision, without creating a meaningful blind zone.
            float behind_tolerance = -0.1f;

            /// Enable floor pruning based on world-space Y coordinate.
            /// Removes Gaussians reconstructed below the capture floor, which are
            /// typically caused by floor reflections, dark shadow regions, or
            /// reconstruction noise from the lowest camera ring.
            bool enable_floorandceil_pruning = true;

            /// World-space Y coordinate threshold for floor pruning.
            /// Gaussians with y <= floor_y are removed.
            ///
            /// Guidelines:
            /// - Set to slightly below the lowest legitimate point of the subject.
            ///   For a standing person with feet at y=0, a value of -0.01f (1cm) is safe.
            /// - Verify coordinate convention: COLMAP may reconstruct with Y pointing
            ///   down, in which case the sign of this threshold must be inverted.
            /// - Set enable_floorandceil_pruning = false rather than using -FLT_MAX to disable.
            /// - If ceil < floor it gets disabled
            float floor_y = -0.1f;
            float ceil_y = 2.6f;
            float max_scale_in_meters = 0.7;
        };


        /**
         * @brief Configuration for center-vote pruning.
         */
        struct CenterVotePruningConfig {
            
            bool enabled = true;

            /// Minimum ratio of views where center must be inside mask to keep splat
            float vote_ratio_threshold = 0.90f;

            /// Minimum fraction of usable cameras in which the splat must be visible
            /// to be considered by the ratio test.
            ///
            /// "Usable cameras" means cameras that were actually processed by the
            /// center-vote pass after excluding missing masks, size mismatches and
            /// projection failures. This is a pure ratio threshold:
            ///
            ///   required_views = ceil(min_visibility_ratio * usable_cameras)
            ///
            /// Example:
            ///   0.10f with 104 usable cameras requires visibility in at least 11 views.
            float min_visibility_ratio = 0.1f;

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
            float depth_filter_sigma_multiplier = 5.0f;

            /// Minimum number of inside-mask splats required to compute depth statistics
            int min_splats_for_depth_stats = 7;
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

            /// Minimum fraction of usable cameras that must provide decisive
            /// good/bad evidence before the splat is judged by leakage.
            ///
            /// "Usable cameras" means cameras that were actually processed by the
            /// leakage pass after excluding missing masks, size mismatches and
            /// projection failures. This is a pure ratio threshold:
            ///
            ///   required_decisive_views = ceil(min_visibility_ratio * usable_cameras)
            float min_visibility_ratio = 0.1f;

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


        /**
         * @brief Configuration for alpha-consensus pruning.
         *
         * For each Gaussian, projects it to 2D across all cameras where it is visible
         * and measures what fraction of its Gaussian mass (weighted by the 2D Gaussian
         * function) falls inside the mask. The ratio is averaged across cameras to form
         * a consensus. Gaussians whose mass is predominantly outside the mask are removed.
         *
         * Advantages over leakage pruning:
         * - Uses the actual 2D Gaussian weighting (exp(-0.5 * conic * d^2)) so the
         *   center contributes more than the edges, matching real visual contribution.
         * - No min_pixel_radius threshold: thin elongated splats are evaluated correctly
         *   regardless of their projected size.
         * - Mask errors in isolated cameras are diluted by the multi-camera average.
         * - Correctly handles elongated splats: a splat with center inside the mask but
         *   significant Gaussian mass outside will fail the consensus test.
         */
        struct AlphaConsensusPruningConfig {
            /// Enable/disable this pass
            bool enabled = true;

            /// Minimum fraction of Gaussian mass that must fall inside the mask.
            /// Gaussians with consensus_ratio < this threshold are removed.
            /// 0.50 = more than half the mass must be inside the mask.
            float consensus_threshold = 0.75f;

            /// Minimum number of cameras where the splat must be visible
            /// to be evaluated. Splats seen in fewer cameras are kept (not enough evidence).
            int min_visibility_count = 3;

            /// NxN sample grid within the projected ellipse bounding box.
            /// Higher values give more accurate mass estimation at the cost of compute.
            /// 7 is a good balance: 49 samples per splat per camera.
            int sample_grid = 7;

            /// Projection parameters (match CenterVotePruningConfig for consistency)
            float eps2d = 0.3f;
            float near_plane = 0.01f;
            float far_plane = 10000.0f;
            float radius_clip = 0.0f;
            float scaling_modifier = 1.0f;

            bool invert_masks = false;
        };

        /**
         * @brief Configuration for ellipse boundary pruning.
         *
         * For each surviving Gaussian (center already validated by center-vote),
         * projects the 2D ellipse boundary into every camera and samples 8 points
         * (4 cardinal + 4 diagonal). Each point is checked against a slightly
         * expanded version of the mask. If any in-frame boundary point falls
         * outside the expanded mask, that camera votes negative for the splat.
         * Splats accumulating negative votes in enough cameras are removed.
         *
         * Advantages over alpha-consensus and leakage pruning:
         * - No Gaussian weighting or mass estimation: a simple binary per-point
         *   check that is faster, easier to reason about, and has fewer parameters.
         * - Frame-border safe by design: points that fall outside the image are
         *   individually ignored, not penalized. A camera only votes if at least
         *   one boundary point is in-frame.
         * - The expanded mask provides uniform tolerance for minor mask
         *   imperfections and natural Gaussian bleed at the object silhouette.
         * - The low negative-vote threshold (10%) combined with the generous
         *   mask expansion means only splats with clearly excessive leakage
         *   across multiple cameras are removed.
         */
        struct EllipseBoundaryPruningConfig {
            /// Enable/disable this pass
            bool enabled = true;
            /// Mask expansion as a fraction of the longest image side in pixels.
            /// The binary mask is dilated by this many pixels before checking
            /// boundary points, providing tolerance for minor mask errors and
            /// natural Gaussian bleed at the object silhouette.
            /// 0.05 = expand by 1% of max(W, H). At 2K this is ~20px.
            float mask_expansion_fraction = 0.05f;
            /// Fraction of evaluating cameras that must vote negative to remove.
            /// A camera votes negative if any of the 8 in-frame boundary points
            /// falls outside the expanded mask.
            /// 0.05 = remove if >= 5% of evaluating cameras flagged leakage.
            float negative_vote_threshold = 0.1f;
            /// Minimum number of cameras that could evaluate the splat (had at
            /// least one boundary point in-frame). Splats evaluated by fewer
            /// cameras are kept — not enough evidence to judge.
            int min_evaluating_cameras = 3;
            /// Projection parameters (match CenterVotePruningConfig for consistency)
            float eps2d = 0.3f;
            float near_plane = 0.01f;
            float far_plane = 10000.0f;
            float radius_clip = 0.0f;
            float scaling_modifier = 1.0f;
            bool invert_masks = false;
        };


        /**
         * @brief Configuration for 3D cluster + extreme-point pruning.
         *
         * Pass 1:
         *   Compute the median nearest-neighbor distance between Gaussian centers.
         *   Build epsilon-connected 3D clusters with:
         *
         *     eps_cluster = cluster_eps_multiplier * median_nn_distance
         *
         *   Then remove clusters whose size is smaller than min_cluster_size.
         *
         * Pass 2:
         *   For the surviving splats, compute the 6 oriented extremes of the splat
         *   from its center, scaling and rotation:
         *
         *     center ± R * (sx, 0, 0)
         *     center ± R * (0, sy, 0)
         *     center ± R * (0, 0, sz)
         *
         *   Any extreme with world Y < burial_y_ignore_threshold is ignored
         *   (not penalized), because buried geometry is often useful for legs/feet.
         *
         *   For every remaining extreme, find the nearest surviving center
         *   (including the splat's own center). If any evaluated extreme is farther than:
         *
         *     eps_extreme = extreme_eps_multiplier * median_nn_distance
         *
         *   the entire splat is removed.
         */
        struct ClusterExtremePruningConfig {
            /// Enable / disable this pass
            bool enabled = true;

            /// Epsilon multiplier for 3D center clustering
            /// multiplier over the mean distance to separate in clusters
            float cluster_eps_multiplier = 25.0f;

            /// Minimum connected-component size to keep
            int min_cluster_size = 25;

            /// Epsilon multiplier for extreme-point validation
            /// splat extremes that are far from the main point cloud
            float extreme_eps_multiplier = 25.0f;

            /// Multiplier applied to splat scale when generating the 6 extremes.
            /// Use 1.0f to use the scale exactly as stored in SplatData.
            float extreme_axis_multiplier = 1.0f;

            /// Extremes with y < burial_y_ignore_threshold are ignored (not penalized)
            /// ie do not introduce penalty if they are under the floor.
            float burial_y_ignore_threshold = -0.1f;
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
             * - 8.0  : more aggressive (removes more isolated points)
             * - 16.0  : conservative / stable default
             * - 32+  : very conservative
             */
            float threshold_multiplier = 32.0f;

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
         * @brief Remove splats outside the dome volume using geometric tests only.
         *
         * Runs up to two passes depending on config:
         *   1. Behind-camera: dot product test against each camera's optical plane
         *   2. Floor: world-space Y threshold
         *
         * This is always the first pruning pass — it is the cheapest possible filter
         * and reduces splat count before the more expensive mask-based passes.
         */
        std::expected<PruningResult, std::string> prune_by_geometric_dome(
            IStrategy& strategy,
            const CameraDataset& dataset,
            const GeometricDomePruningConfig& config = {});

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
         * @brief Remove splats whose Gaussian mass is predominantly outside the mask.
         */
        std::expected<PruningResult, std::string> prune_by_alpha_consensus(
            IStrategy& strategy,
            const CameraDataset& dataset,
            const AlphaConsensusPruningConfig& config = {});

        /**
         * @brief Remove splats whose Gaussian boundary goes out of the extended mask.
         */
        std::expected<PruningResult, std::string> prune_by_ellipse_boundary(
            IStrategy& strategy,
            const CameraDataset& dataset,
            const EllipseBoundaryPruningConfig& config = {});

        /**
         * @brief Remove tiny 3D clusters and splats whose oriented 3D extremes are too far
         *        from any surviving center.
         */
        std::expected<PruningResult, std::string> prune_by_cluster_and_extremes(
            IStrategy& strategy,
            const ClusterExtremePruningConfig& config = {});
        
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
            const GeometricDomePruningConfig& geometric_config = {},
            const CenterVotePruningConfig& center_config = DEFAULT_CONFIG,
            const LeakagePruningConfig& leakage_config = {},
            const IsolationPruningConfig& isolation_config = {});

    } // namespace mask_pruning
} // namespace lfs::training
