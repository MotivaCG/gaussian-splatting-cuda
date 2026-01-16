/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "ellipsoid_command.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"

namespace lfs::vis::command {

    EllipsoidCommand::EllipsoidCommand(SceneManager* scene_manager,
                                       const std::string& ellipsoid_node_name,
                                       const EllipsoidState& old_state,
                                       const EllipsoidState& new_state)
        : scene_manager_(scene_manager),
          ellipsoid_node_name_(ellipsoid_node_name),
          old_state_(old_state),
          new_state_(new_state) {}

    void EllipsoidCommand::undo() {
        applyState(old_state_);
    }

    void EllipsoidCommand::redo() {
        applyState(new_state_);
    }

    void EllipsoidCommand::applyState(const EllipsoidState& state) {
        if (!scene_manager_)
            return;

        auto* node = scene_manager_->getScene().getMutableNode(ellipsoid_node_name_);
        if (!node || !node->ellipsoid)
            return;

        node->ellipsoid->radii = state.radii;
        node->ellipsoid->inverse = state.inverse;
        node->local_transform = state.local_transform;
        node->transform_dirty = true;

        scene_manager_->getScene().invalidateCache();
        scene_manager_->syncEllipsoidToRenderSettings();
    }

} // namespace lfs::vis::command
