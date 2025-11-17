/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <memory>

namespace lfs::core {

    namespace param {
        struct TrainingParameters;
    } // namespace param

    struct TrainingParameters;

    class Application {
    public:
        int run(std::unique_ptr<param::TrainingParameters> params);
    };

} // namespace lfs::core
