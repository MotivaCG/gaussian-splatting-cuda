/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "command/command.hpp"
#include "scene/scene.hpp"
#include <glm/glm.hpp>
#include <string>

namespace lfs::vis {
    class SceneManager;
}

namespace lfs::vis::command {

    struct EllipsoidState {
        glm::vec3 radii{1.0f};
        glm::mat4 local_transform{1.0f};
        bool inverse = false;
    };

    class EllipsoidCommand : public Command {
    public:
        EllipsoidCommand(SceneManager* scene_manager,
                         const std::string& ellipsoid_node_name,
                         const EllipsoidState& old_state,
                         const EllipsoidState& new_state);

        void undo() override;
        void redo() override;
        std::string getName() const override { return "Ellipsoid"; }

    private:
        void applyState(const EllipsoidState& state);

        SceneManager* scene_manager_;
        std::string ellipsoid_node_name_;
        EllipsoidState old_state_;
        EllipsoidState new_state_;
    };

} // namespace lfs::vis::command
