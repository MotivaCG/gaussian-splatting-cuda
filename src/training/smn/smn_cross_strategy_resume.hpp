/* SPDX-FileCopyrightText: 2026 SMN | Víctor M. Feliz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include "strategies/istrategy.hpp"
#include <memory>
#include <string>

namespace lfs::training::smn {

    // Cross-strategy checkpoint resume (--allow-strategy-switch): builds and
    // initializes a fresh strategy of `target_type` around `loaded_model`, so the
    // checkpoint loader's existing ICheckpointStateAdopter path can adopt it exactly
    // like a same-type resume. Only the Gaussian model transfers between the two
    // strategies; optimizer momentum, LR schedule progress, and strategy-internal
    // buffers (MCMC noise, MRNF decay/edge-score state, ...) do not, since none of
    // that is meaningful across different concrete strategies. Throws on failure —
    // call before the checkpoint loader's commit boundary so an OOM here still
    // surfaces as a clean error return.
    std::unique_ptr<lfs::training::IStrategy> build_cross_strategy_target(
        const std::string& target_type,
        lfs::core::SplatData& loaded_model,
        const lfs::core::param::OptimizationParameters& optim_params);

} // namespace lfs::training::smn
