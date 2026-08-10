/* SPDX-FileCopyrightText: 2026 SMN | Víctor M. Feliz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "smn_hybrid.hpp"

#include "core/parameters.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace lfs::training::smn {

    namespace {

        void install_mcmc_to_mrnf_safety(
            lfs::core::param::OptimizationParameters& target,
            const int /*switch_iteration*/) {
            const int regular_horizon = regular_training_horizon(target);

            target.strategy = std::string(lfs::core::param::kStrategyMRNF);
            target.stop_refine = std::max(
                target.start_refine,
                std::min(target.stop_refine, static_cast<size_t>(regular_horizon)));

            // Preserve the proven MCMC hand-off profile for LR, scale/rotation,
            // opacity/scale regularization and refinement cadence. Replacing it with
            // fresh MRNF defaults made large edge splats survive as crest-like geometry.
            target.smn_hybrid_mcmc_densification = true;
        }

    } // namespace

    int regular_training_horizon(
        const lfs::core::param::OptimizationParameters& params) noexcept {
        return static_cast<int>(std::min<size_t>(
            params.iterations,
            static_cast<size_t>(std::numeric_limits<int>::max())));
    }

    int resolve_strategy_switch_iteration(
        const lfs::core::param::OptimizationParameters& params) noexcept {
        if (params.smn_switch_strategy_to.empty() ||
            params.smn_switch_at_fraction <= 0.0f) {
            return -1;
        }
        // Match the original Hybrid schedule exactly: its fractional handoff was
        // resolved against the effective runtime total, including a fixed sparse tail.
        // `iterations` has already been transformed by steps_scaler at this point.
        const int horizon = params.resolved_total_iterations();
        if (horizon <= 0) {
            return -1;
        }
        return std::clamp(
            static_cast<int>(std::lround(
                static_cast<double>(horizon) * params.smn_switch_at_fraction)),
            1,
            horizon);
    }

    bool uses_mcmc_densification_signal(
        const lfs::core::param::OptimizationParameters& params) noexcept {
        return params.smn_hybrid_mcmc_densification &&
               lfs::core::param::is_mrnf_strategy(params.strategy);
    }

    lfs::core::param::OptimizationParameters make_in_process_target_params(
        const lfs::core::param::OptimizationParameters& source,
        const std::string_view target_type,
        const int switch_iteration) {
        auto target = source;
        const std::string canonical(
            lfs::core::param::canonical_strategy_name(target_type));
        if (canonical.empty()) {
            throw std::invalid_argument(
                "Unsupported in-process target strategy: " + std::string(target_type));
        }

        if (lfs::core::param::is_mrnf_strategy(canonical) &&
            source.strategy == lfs::core::param::kStrategyMCMC) {
            install_mcmc_to_mrnf_safety(target, switch_iteration);
        } else {
            target.strategy = canonical;
            target.smn_hybrid_mcmc_densification = false;
        }

        target.smn_switch_strategy_to.clear();
        target.smn_switch_at_fraction = 0.0f;
        return target;
    }

} // namespace lfs::training::smn
