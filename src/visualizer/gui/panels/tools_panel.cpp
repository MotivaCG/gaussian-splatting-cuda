/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panels/tools_panel.hpp"
#include "gui/gui_manager.hpp"
#include "gui/panels/transform_panel.hpp"
#include "visualizer_impl.hpp"
#include <imgui.h>

namespace lfs::vis::gui::panels {

    void DrawToolsPanel(const UIContext& ctx) {
        auto* const gui_manager = ctx.viewer->getGuiManager();
        if (!gui_manager)
            return;

        const ToolType current_tool = gui_manager->getCurrentToolMode();
        const TransformSpace transform_space = gui_manager->getGizmoToolbarState().transform_space;

        DrawTransformControls(ctx, current_tool, transform_space, gui_manager->getTransformPanelState());
    }

} // namespace lfs::vis::gui::panels
