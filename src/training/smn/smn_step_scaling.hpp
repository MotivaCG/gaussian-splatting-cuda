/* SPDX-FileCopyrightText: 2026 SMN | Víctor M. Feliz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <algorithm>
#include <cmath>

namespace lfs::training::smn {

    // Scale hidden component warmups which are not OptimizationParameters fields.
    // A non-positive scaler means that automatic step scaling is disabled.
    [[nodiscard]] inline int scaled_component_steps(
        const int steps,
        const float scaler) noexcept {
        if (steps <= 0 || scaler <= 0.0f) {
            return steps;
        }
        return std::max(1, static_cast<int>(std::lround(
                               static_cast<double>(steps) * scaler)));
    }

} // namespace lfs::training::smn
