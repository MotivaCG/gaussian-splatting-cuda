/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/ui_context.hpp"

namespace lfs::vis::gui::panels {

    void DrawEllipsoidControls(const UIContext& ctx);

    struct EllipsoidState {
        bool show_ellipsoid = false;
        bool use_ellipsoid = false;

        static EllipsoidState& getInstance() {
            static EllipsoidState instance;
            return instance;
        }
    };

    const EllipsoidState& getEllipsoidState();

} // namespace lfs::vis::gui::panels
