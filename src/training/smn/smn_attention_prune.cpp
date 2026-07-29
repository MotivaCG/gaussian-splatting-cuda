/* SPDX-FileCopyrightText: 2026 SMN | Víctor M. Feliz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "smn_attention_prune.hpp"

#include "smn_attention_constants.h"
#include "mask_pruning.hpp"

#include "core/logger.hpp"

namespace lfs::training::smn {

    void run_attention_prune(lfs::training::IStrategy& strategy,
                             const lfs::training::CameraDataset& dataset,
                             const bool invert_masks) {

        if (!SMN_ATTENTION_PRUNE_ENABLED) {
            return;
        }

        namespace mp = mask_pruning;

        const int before = static_cast<int>(strategy.get_model().size());
        if (before == 0) {
            return;
        }

        // Configs from our constants (SMNV2's validated tuning). invert_masks flows
        // to the mask passes.
        mp::GeometricDomePruningConfig geom;
        geom.enabled = SMN_PRUNE_GEOMETRIC_ENABLED;
        geom.behind_tolerance = SMN_PRUNE_BEHIND_TOLERANCE;
        geom.floor_y = SMN_PRUNE_FLOOR_Y;
        geom.ceil_y = SMN_PRUNE_CEIL_Y;
        geom.max_scale_in_meters = SMN_PRUNE_MAX_SCALE_M;

        mp::CenterVotePruningConfig center;
        center.enabled = SMN_PRUNE_CENTER_VOTE_ENABLED;
        center.vote_ratio_threshold = SMN_PRUNE_CENTER_VOTE_RATIO;
        center.min_visibility_ratio = SMN_PRUNE_CENTER_MIN_VIS_RATIO;
        center.border_safe_margin = SMN_PRUNE_CENTER_BORDER_MARGIN;
        center.enable_depth_filtering = SMN_PRUNE_CENTER_DEPTH_FILTER;
        center.invert_masks = invert_masks;

        mp::LeakagePruningConfig leak;
        leak.enabled = SMN_PRUNE_LEAKAGE_ENABLED;
        leak.leak_keep_threshold = SMN_PRUNE_LEAK_KEEP_THRESHOLD;
        leak.per_view_leak_fraction = SMN_PRUNE_LEAK_PER_VIEW_FRACTION;
        leak.min_visibility_ratio = SMN_PRUNE_LEAK_MIN_VIS_RATIO;
        leak.min_pixel_radius = SMN_PRUNE_LEAK_MIN_PIXEL_RADIUS;
        leak.sample_points = SMN_PRUNE_LEAK_SAMPLE_POINTS;
        leak.dilate_px = SMN_PRUNE_LEAK_DILATE_PX;
        leak.invert_masks = invert_masks;

        mp::AlphaConsensusPruningConfig consensus;
        consensus.enabled = SMN_PRUNE_CONSENSUS_ENABLED;
        consensus.consensus_threshold = SMN_PRUNE_CONSENSUS_THRESHOLD;
        consensus.min_visibility_count = SMN_PRUNE_CONSENSUS_MIN_VIS;
        consensus.sample_grid = SMN_PRUNE_CONSENSUS_GRID;
        consensus.invert_masks = invert_masks;

        mp::EllipseBoundaryPruningConfig ellipse;
        ellipse.enabled = SMN_PRUNE_ELLIPSE_ENABLED;
        ellipse.mask_expansion_fraction = SMN_PRUNE_ELLIPSE_EXPANSION_FRACTION;
        ellipse.negative_vote_threshold = SMN_PRUNE_ELLIPSE_NEG_VOTE_THRESHOLD;
        ellipse.min_evaluating_cameras = SMN_PRUNE_ELLIPSE_MIN_CAMERAS;
        ellipse.invert_masks = invert_masks;

        mp::IsolationPruningConfig isolation;
        isolation.enabled = SMN_PRUNE_ISOLATION_ENABLED;
        isolation.k_neighbors = SMN_PRUNE_ISOLATION_K;
        isolation.kth_neighbor = SMN_PRUNE_ISOLATION_KTH;
        isolation.threshold_multiplier = SMN_PRUNE_ISOLATION_MULTIPLIER;

        mp::SORPruningConfig sor;
        sor.enabled = SMN_PRUNE_SOR_ENABLED;
        sor.k_neighbors = SMN_PRUNE_SOR_NEIGHBORS;
        sor.std_ratio = SMN_PRUNE_SOR_STD_RATIO;
        sor.guard_kth = SMN_PRUNE_SOR_GUARD_KTH;
        sor.guard_multiplier = SMN_PRUNE_SOR_GUARD_MULTIPLIER;

        LOG_INFO("[SMN prune] Post-training multi-pass prune: classifying {} splats...", before);

        // MCMC soft-deletes (size() unchanged) so compact after every pass or the
        // passes never compound; apply_deleted is a no-op for MRNF.
        const auto run = [&strategy](const char* name, auto&& fn) {
            const size_t before_pass = strategy.get_model().size();
            auto result = fn();
            if (!result) {
                LOG_WARN("[SMN prune] {} failed: {}", name, result.error());
                return;
            }
            strategy.get_model().apply_deleted();
            const size_t after_pass = strategy.get_model().size();
            const size_t removed = (before_pass > after_pass) ? before_pass - after_pass : 0;
            if (removed > 0) {
                LOG_INFO("[SMN prune] {}: removed {} splats ({:.1f}%)",
                         name, removed,
                         before_pass > 0 ? 100.0f * static_cast<float>(removed) /
                                               static_cast<float>(before_pass)
                                         : 0.0f);
            }
        };

        run("Geometric dome", [&] { return mp::prune_by_geometric_dome(strategy, dataset, geom); });
        run("Center vote", [&] { return mp::prune_by_center_vote(strategy, dataset, center); });
        run("Mask leakage", [&] { return mp::prune_by_mask_leakage(strategy, dataset, leak); });
        run("Alpha consensus", [&] { return mp::prune_by_alpha_consensus(strategy, dataset, consensus); });
        run("Isolation", [&] { return mp::prune_by_isolation_distance(strategy, isolation); });
        run("Ellipse boundary", [&] { return mp::prune_by_ellipse_boundary(strategy, dataset, ellipse); });
        // SOR last: it removes whatever the mask/geometry passes leave isolated.
        run("SOR", [&] { return mp::prune_by_sor(strategy, sor); });

        const int after = static_cast<int>(strategy.get_model().size());
        LOG_INFO("[SMN prune] done: {} -> {} splats ({} removed, {:.1f}%)",
                 before, after, before - after,
                 before > 0 ? 100.0f * static_cast<float>(before - after) / static_cast<float>(before)
                            : 0.0f);
    }

} // namespace lfs::training::smn
