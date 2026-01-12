/* ScanMeNow file */

#pragma once

/**
 * @file mask_pruning.hpp
 * @brief Post-training pruning for masked Gaussian splatting.
 *
 * Provides pruning operations that remove splats which don't align well with
 * attention masks across multiple views. Designed to be called after training
 * completes, before the final save.
 *
 * @section operations Operations
 * - prune_by_center_vote(): Removes splats whose projected center doesn't
 *   consistently fall inside the mask across views.
 * - prune_by_mask_leakage(): Removes splats whose projected footprint leaks
 *   outside the mask in too many views.
 *
 * @section requirements Requirements
 * Uses gsplat_lfs::launch_projection_ut_3dgs_fused_kernel for fast 2D projection.
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

    bool invert_masks = false;
};

/// Default configuration
inline constexpr CenterVotePruningConfig DEFAULT_CONFIG{};

/**
 * @brief Configuration for leakage pruning (footprint outside mask).
 *
 * Definition used here:
 * - For each visible splat, we sample several points on its projected footprint
 *   (using projected radii) and check how many land outside the (optionally tolerant) mask.
 * - If in a view too many samples are outside => that view "leaks" for that splat.
 * - Across views, if the splat leaks too often => remove it.
 */
struct LeakagePruningConfig {
    /// Enable/disable leakage pass
    bool enabled = true;

    /// Keep threshold for "non-leak ratio" = 1 - leak_views / evaluated_views
    /// Example: 0.90 => require >= 90% of evaluated views to be non-leaking.
    float leak_keep_threshold = 0.90f;

    /// Per-view: if outside_samples_ratio > this => that view counts as leaking
    float per_view_leak_fraction = 0.50f;

    /// Minimum evaluated views required to apply leakage decision
    /// (splats with fewer evaluated views are NOT removed by leakage pass).
    int min_visibility_count = 1;

    /// Ignore leakage evaluation for splats with tiny projected footprint
    float min_pixel_radius = 2.0f;

    /// "Inside" threshold for mask sampling
    float mask_inside_threshold = 0.5f;

    /// Number of sample points around the footprint boundary (4 or 8)
    int sample_points = 8;

    /// Optional boundary tolerance:
    /// if >0 use (1 - bg_core) as a tolerant mask (bg_core is eroded background).
    int dilate_px = 2;

    /// Projection parameters (same meaning as CenterVotePruningConfig)
    float eps2d = 0.3f;
    float near_plane = 0.01f;
    float far_plane = 10000.0f;
    float radius_clip = 0.0f;
    float scaling_modifier = 1.0f;

    bool invert_masks = false;
};

inline constexpr LeakagePruningConfig DEFAULT_LEAKAGE_CONFIG{};

// =============================================================================
// Result Types
// =============================================================================

/**
 * @brief Result of a pruning operation.
 */
struct PruningResult {
    int splats_before = 0;
    int splats_after = 0;
    int splats_removed = 0;
    bool success = false;
    std::string error;

    [[nodiscard]] float removal_ratio() const {
        return splats_before > 0
                   ? static_cast<float>(splats_removed) / static_cast<float>(splats_before)
                   : 0.0f;
    }
};

// =============================================================================
// Projection Result
// =============================================================================

/**
 * @brief Result of fast 2D projection.
 */
struct ProjectionResult {
    lfs::core::Tensor radii;    ///< Projected radii [N, 2], int32, CUDA
    lfs::core::Tensor means2d;  ///< Projected 2D centers [N, 2], float32, CUDA
    lfs::core::Tensor depths;   ///< Depths [N], float32, CUDA
    lfs::core::Tensor conics;   ///< Conics [N, 3], float32, CUDA
};

// =============================================================================
// Core Functions
// =============================================================================

[[nodiscard]] std::expected<ProjectionResult, std::string> project_splats(
    const lfs::core::Camera& camera,
    const lfs::core::SplatData& splat_data,
    const CenterVotePruningConfig& config = DEFAULT_CONFIG);

[[nodiscard]] std::expected<PruningResult, std::string> prune_by_center_vote(
    IStrategy& strategy,
    const CameraDataset& dataset,
    const CenterVotePruningConfig& config = DEFAULT_CONFIG);

[[nodiscard]] std::expected<PruningResult, std::string> prune_by_mask_leakage(
    IStrategy& strategy,
    const CameraDataset& dataset,
    const LeakagePruningConfig& config = DEFAULT_LEAKAGE_CONFIG);

/**
 * @brief Main entry point for post-training pruning.
 *
 * Order:
 * 1) center-vote
 * 2) leakage (optional)
 */
[[nodiscard]] std::expected<PruningResult, std::string> prune_after_training(
    IStrategy& strategy,
    const CameraDataset& dataset,
    const CenterVotePruningConfig& center_config = DEFAULT_CONFIG,
    const LeakagePruningConfig& leakage_config = DEFAULT_LEAKAGE_CONFIG);

} // namespace mask_pruning
} // namespace lfs::training
