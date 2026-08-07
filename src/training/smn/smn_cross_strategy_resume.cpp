/* SPDX-FileCopyrightText: 2026 SMN | Víctor M. Feliz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "smn_cross_strategy_resume.hpp"

#include "strategies/strategy_factory.hpp"
#include <stdexcept>

namespace lfs::training::smn {

    std::unique_ptr<lfs::training::IStrategy> build_cross_strategy_target(
        const std::string& target_type,
        lfs::core::SplatData& loaded_model,
        const lfs::core::param::OptimizationParameters& optim_params) {
        auto target_result = lfs::training::StrategyFactory::instance().create(target_type, loaded_model);
        if (!target_result) {
            throw std::runtime_error("Cannot construct target strategy: " + target_result.error());
        }
        auto target = std::move(*target_result);
        target->initialize(optim_params);
        return target;
    }

} // namespace lfs::training::smn
