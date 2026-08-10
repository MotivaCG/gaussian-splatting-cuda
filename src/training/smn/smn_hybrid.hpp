/* SPDX-FileCopyrightText: 2026 SMN | Víctor M. Feliz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/parameters.hpp"
#include <string_view>

namespace lfs::training::smn {

    // The regular horizon remains useful for topology/refinement limits. Hybrid's
    // switch fraction itself intentionally follows the original effective runtime
    // total, including the optional fixed sparsification tail.
    [[nodiscard]] int regular_training_horizon(
        const lfs::core::param::OptimizationParameters& params) noexcept;

    [[nodiscard]] int resolve_strategy_switch_iteration(
        const lfs::core::param::OptimizationParameters& params) noexcept;

    // Hybrid deliberately retains the unnormalized MCMC error signal after MRNF
    // takes ownership of refinement. This is persisted in OptimizationParameters
    // so checkpoints and GUI updates cannot silently change the handoff behavior.
    [[nodiscard]] bool uses_mcmc_densification_signal(
        const lfs::core::param::OptimizationParameters& params) noexcept;

    // Builds the persistent (regular-iteration) parameter set for a hot-swapped
    // target strategy. The caller may extend `iterations` with a runtime-only
    // sparsification tail before initializing the concrete strategy.
    [[nodiscard]] lfs::core::param::OptimizationParameters make_in_process_target_params(
        const lfs::core::param::OptimizationParameters& source,
        std::string_view target_type,
        int switch_iteration);

} // namespace lfs::training::smn
