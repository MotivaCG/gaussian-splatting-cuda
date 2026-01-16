/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/ui_context.hpp"
#include "scene/scene.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace lfs::vis::gui::panels {

    void DrawCropBoxControls(const UIContext& ctx);

    // Crop box state
    struct CropBoxState {
        bool show_crop_box = false;
        bool use_crop_box = false;

        // Euler display state - avoids gimbal lock from constant decomposition
        glm::vec3 euler_display{0.0f};
        NodeId euler_display_node{NULL_NODE};
        glm::quat euler_display_rotation{1.0f, 0.0f, 0.0f, 0.0f};

        static CropBoxState& getInstance() {
            static CropBoxState instance;
            return instance;
        }
    };

    // Access for GuiManager - Add this declaration
    const CropBoxState& getCropBoxState();
} // namespace lfs::vis::gui::panels