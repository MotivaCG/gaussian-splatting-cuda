/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/gizmo_transform.hpp"
#include "gui/panels/gizmo_toolbar.hpp"
#include "gui/ui_context.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <string>
#include <vector>

namespace lfs::vis::gui::panels {

    struct TransformPanelState {
        // Single-node editing state
        bool editing_active = false;
        std::string editing_node_name;
        glm::mat4 transform_before_edit{1.0f};

        glm::vec3 initial_translation{0.0f};
        glm::vec3 initial_scale{1.0f};

        // Euler display state - avoids gimbal lock from constant decomposition
        glm::vec3 euler_display{0.0f};
        std::string euler_display_node;
        glm::quat euler_display_rotation{1.0f, 0.0f, 0.0f, 0.0f};

        // Multi-node editing state
        bool multi_editing_active = false;
        gizmo_ops::MultiNodeCapture multi_capture;
        glm::vec3 pivot_world{0.0f};

        // Display values for multi-selection (offsets from original)
        glm::vec3 display_translation{0.0f};
        glm::vec3 display_euler{0.0f};
        glm::vec3 display_scale{1.0f, 1.0f, 1.0f};

        // Track which tool was active when multi-edit started
        ToolType multi_edit_tool = ToolType::None;

        void resetMultiEdit() {
            multi_editing_active = false;
            multi_capture = {};
            pivot_world = glm::vec3(0.0f);
            display_translation = glm::vec3(0.0f);
            display_euler = glm::vec3(0.0f);
            display_scale = glm::vec3(1.0f);
            multi_edit_tool = ToolType::None;
        }
    };

    void DrawTransformControls(const UIContext& ctx, ToolType current_tool,
                               TransformSpace transform_space, TransformPanelState& state);

} // namespace lfs::vis::gui::panels
