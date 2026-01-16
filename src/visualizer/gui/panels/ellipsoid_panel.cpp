/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panels/ellipsoid_panel.hpp"
#include "command/command_history.hpp"
#include "command/commands/ellipsoid_command.hpp"
#include "gui/dpi_scale.hpp"
#include "gui/localization_manager.hpp"
#include "gui/string_keys.hpp"
#include "gui/ui_widgets.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "visualizer_impl.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <optional>
#include <imgui.h>

namespace lfs::vis::gui::panels {

    using namespace lfs::core::events;
    using namespace lichtfeld::Strings;

    namespace {
        constexpr float POSITION_STEP = 0.01f;
        constexpr float POSITION_STEP_FAST = 0.1f;
        constexpr float ROTATION_STEP = 1.0f;
        constexpr float ROTATION_STEP_FAST = 15.0f;
        constexpr float MIN_RADIUS = 0.001f;
        constexpr float BASE_INPUT_WIDTH_PADDING = 40.0f;

        std::optional<command::EllipsoidState> s_state_before_edit;
        std::string s_editing_ellipsoid_name;
        bool s_editing_active = false;

        command::EllipsoidState captureState(const SceneNode* node) {
            if (!node || !node->ellipsoid) {
                return {};
            }
            return {
                .radii = node->ellipsoid->radii,
                .local_transform = node->local_transform.get(),
                .inverse = node->ellipsoid->inverse};
        }

        void commitUndoIfChanged(VisualizerImpl* viewer, SceneManager* sm, const std::string& node_name) {
            if (!s_state_before_edit.has_value())
                return;

            const auto* node = sm->getScene().getNode(node_name);
            if (!node)
                return;

            const auto new_state = captureState(node);
            const bool changed = (s_state_before_edit->radii != new_state.radii ||
                                  s_state_before_edit->inverse != new_state.inverse ||
                                  s_state_before_edit->local_transform != new_state.local_transform);

            if (changed) {
                auto cmd = std::make_unique<command::EllipsoidCommand>(
                    sm, node_name, *s_state_before_edit, new_state);
                viewer->getCommandHistory().execute(std::move(cmd));
            }
            s_state_before_edit.reset();
        }

        glm::vec3 matrixToEulerDegrees(const glm::mat3& rot) {
            float pitch, yaw, roll;
            glm::extractEulerAngleXYZ(glm::mat4(rot), pitch, yaw, roll);
            return glm::degrees(glm::vec3(pitch, yaw, roll));
        }

        glm::mat3 eulerDegreesToMatrix(const glm::vec3& euler) {
            const glm::vec3 rad = glm::radians(euler);
            return glm::mat3(glm::eulerAngleXYZ(rad.x, rad.y, rad.z));
        }
    } // namespace

    void DrawEllipsoidControls(const UIContext& ctx) {
        auto* const sm = ctx.viewer->getSceneManager();
        auto* const rm = ctx.viewer->getRenderingManager();
        if (!sm || !rm)
            return;

        if (!ImGui::CollapsingHeader(LOC(Ellipsoid::TITLE), ImGuiTreeNodeFlags_DefaultOpen))
            return;

        const auto& settings = rm->getSettings();
        if (!settings.show_ellipsoid) {
            ImGui::TextDisabled("%s", LOC(Ellipsoid::NOT_VISIBLE));
            return;
        }

        const NodeId ellipsoid_id = sm->getSelectedNodeEllipsoidId();
        if (ellipsoid_id == NULL_NODE) {
            ImGui::TextDisabled("%s", LOC(Ellipsoid::NO_SELECTION));
            return;
        }

        auto* node = sm->getScene().getMutableNode(sm->getScene().getNodeById(ellipsoid_id)->name);
        if (!node || !node->ellipsoid) {
            ImGui::TextDisabled("%s", LOC(Ellipsoid::INVALID));
            return;
        }

        bool changed = false;
        bool any_active = false;
        bool any_deactivated = false;

        const float width = ImGui::CalcTextSize("-000.000").x + ImGui::GetStyle().FramePadding.x * 2.0f + BASE_INPUT_WIDTH_PADDING * getDpiScale();

        glm::vec3 translation, scale, skew;
        glm::quat rotation;
        glm::vec4 perspective;
        glm::decompose(node->local_transform.get(), scale, rotation, translation, skew, perspective);
        glm::vec3 euler = matrixToEulerDegrees(glm::mat3_cast(rotation));

        if (ImGui::TreeNodeEx(LOC(Ellipsoid::POSITION), ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("X:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(width);
            changed |= ImGui::InputFloat("##PosX", &translation.x, POSITION_STEP, POSITION_STEP_FAST, "%.3f");
            any_active |= ImGui::IsItemActive();
            any_deactivated |= ImGui::IsItemDeactivatedAfterEdit();

            ImGui::Text("Y:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(width);
            changed |= ImGui::InputFloat("##PosY", &translation.y, POSITION_STEP, POSITION_STEP_FAST, "%.3f");
            any_active |= ImGui::IsItemActive();
            any_deactivated |= ImGui::IsItemDeactivatedAfterEdit();

            ImGui::Text("Z:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(width);
            changed |= ImGui::InputFloat("##PosZ", &translation.z, POSITION_STEP, POSITION_STEP_FAST, "%.3f");
            any_active |= ImGui::IsItemActive();
            any_deactivated |= ImGui::IsItemDeactivatedAfterEdit();
            ImGui::TreePop();
        }

        if (ImGui::TreeNodeEx(LOC(Ellipsoid::ROTATION), ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("X:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(width);
            changed |= ImGui::InputFloat("##RotX", &euler.x, ROTATION_STEP, ROTATION_STEP_FAST, "%.1f");
            any_active |= ImGui::IsItemActive();
            any_deactivated |= ImGui::IsItemDeactivatedAfterEdit();

            ImGui::Text("Y:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(width);
            changed |= ImGui::InputFloat("##RotY", &euler.y, ROTATION_STEP, ROTATION_STEP_FAST, "%.1f");
            any_active |= ImGui::IsItemActive();
            any_deactivated |= ImGui::IsItemDeactivatedAfterEdit();

            ImGui::Text("Z:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(width);
            changed |= ImGui::InputFloat("##RotZ", &euler.z, ROTATION_STEP, ROTATION_STEP_FAST, "%.1f");
            any_active |= ImGui::IsItemActive();
            any_deactivated |= ImGui::IsItemDeactivatedAfterEdit();
            ImGui::TreePop();
        }

        if (ImGui::TreeNodeEx(LOC(Ellipsoid::RADII), ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("X:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(width);
            changed |= ImGui::InputFloat("##RadX", &node->ellipsoid->radii.x, POSITION_STEP, POSITION_STEP_FAST, "%.3f");
            any_active |= ImGui::IsItemActive();
            any_deactivated |= ImGui::IsItemDeactivatedAfterEdit();

            ImGui::Text("Y:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(width);
            changed |= ImGui::InputFloat("##RadY", &node->ellipsoid->radii.y, POSITION_STEP, POSITION_STEP_FAST, "%.3f");
            any_active |= ImGui::IsItemActive();
            any_deactivated |= ImGui::IsItemDeactivatedAfterEdit();

            ImGui::Text("Z:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(width);
            changed |= ImGui::InputFloat("##RadZ", &node->ellipsoid->radii.z, POSITION_STEP, POSITION_STEP_FAST, "%.3f");
            any_active |= ImGui::IsItemActive();
            any_deactivated |= ImGui::IsItemDeactivatedAfterEdit();

            node->ellipsoid->radii = glm::max(node->ellipsoid->radii, glm::vec3(MIN_RADIUS));
            ImGui::TreePop();
        }

        if (ImGui::TreeNode(LOC(Ellipsoid::APPEARANCE))) {
            float color[3] = {node->ellipsoid->color.x, node->ellipsoid->color.y, node->ellipsoid->color.z};
            if (ImGui::ColorEdit3(LOC(MainPanel::COLOR), color)) {
                node->ellipsoid->color = glm::vec3(color[0], color[1], color[2]);
                changed = true;
            }
            changed |= ImGui::SliderFloat(LOC(Ellipsoid::LINE_WIDTH), &node->ellipsoid->line_width, 0.5f, 10.0f);
            ImGui::TreePop();
        }

        if (any_active && !s_editing_active) {
            s_editing_active = true;
            s_editing_ellipsoid_name = node->name;
            s_state_before_edit = captureState(node);
        }

        if (changed) {
            const glm::mat3 rot_mat = eulerDegreesToMatrix(euler);
            node->local_transform = glm::translate(glm::mat4(1.0f), translation) * glm::mat4(rot_mat);
            node->transform_dirty = true;
            sm->getScene().invalidateCache();
            rm->markDirty();
        }

        if (any_deactivated && s_editing_active) {
            s_editing_active = false;
            commitUndoIfChanged(ctx.viewer, sm, s_editing_ellipsoid_name);
        }

        if (!any_active && s_editing_active) {
            s_editing_active = false;
            commitUndoIfChanged(ctx.viewer, sm, s_editing_ellipsoid_name);
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::TextDisabled("%s", LOC(Ellipsoid::INSTRUCTIONS));
    }

    const EllipsoidState& getEllipsoidState() {
        return EllipsoidState::getInstance();
    }

} // namespace lfs::vis::gui::panels
