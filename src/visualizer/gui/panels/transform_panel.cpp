/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panels/transform_panel.hpp"
#include "command/command_history.hpp"
#include "command/commands/transform_command.hpp"
#include "gui/dpi_scale.hpp"
#include "gui/localization_manager.hpp"
#include "gui/string_keys.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "visualizer_impl.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <imgui.h>

namespace lfs::vis::gui::panels {

    using namespace lichtfeld::Strings;

    namespace {
        constexpr float TRANSLATE_STEP = 0.01f;
        constexpr float TRANSLATE_STEP_FAST = 0.1f;
        constexpr float TRANSLATE_STEP_CTRL = 0.1f;
        constexpr float TRANSLATE_STEP_CTRL_FAST = 1.0f;
        constexpr float ROTATE_STEP = 1.0f;
        constexpr float ROTATE_STEP_FAST = 15.0f;
        constexpr float SCALE_STEP = 0.01f;
        constexpr float SCALE_STEP_FAST = 0.1f;
        constexpr float MIN_SCALE = 0.001f;
        constexpr float BASE_INPUT_WIDTH_PADDING = 40.0f;

        struct DecomposedTransform {
            glm::vec3 translation;
            glm::quat rotation;
            glm::vec3 scale;
        };

        constexpr float QUAT_EQUIV_EPSILON = 1e-4f;

        DecomposedTransform decompose(const glm::mat4& m) {
            DecomposedTransform d;
            glm::vec3 skew;
            glm::vec4 perspective;
            glm::decompose(m, d.scale, d.rotation, d.translation, skew, perspective);
            return d;
        }

        bool sameRotation(const glm::quat& a, const glm::quat& b) {
            const float dot = glm::dot(a, b);
            return std::abs(std::abs(dot) - 1.0f) < QUAT_EQUIV_EPSILON;
        }
    } // namespace

    void DrawTransformControls(const UIContext& ctx, const ToolType current_tool,
                               const TransformSpace transform_space, TransformPanelState& state) {
        if (current_tool != ToolType::Translate &&
            current_tool != ToolType::Rotate &&
            current_tool != ToolType::Scale) {
            return;
        }

        auto* const scene_manager = ctx.viewer->getSceneManager();
        auto* const render_manager = ctx.viewer->getRenderingManager();
        if (!scene_manager || !render_manager)
            return;

        const auto selected_names = scene_manager->getSelectedNodeNames();
        if (selected_names.empty())
            return;

        const bool is_multi_selection = (selected_names.size() > 1);

        const char* header_label = nullptr;
        switch (current_tool) {
        case ToolType::Translate: header_label = LOC(Toolbar::TRANSLATE); break;
        case ToolType::Rotate: header_label = LOC(Toolbar::ROTATE); break;
        case ToolType::Scale: header_label = LOC(Toolbar::SCALE); break;
        default: return;
        }

        if (!ImGui::CollapsingHeader(header_label, ImGuiTreeNodeFlags_DefaultOpen))
            return;

        // Multi-selection: same UI as single node, values represent selection center
        if (is_multi_selection) {
            const bool ctrl_pressed = ImGui::GetIO().KeyCtrl;
            const float translate_step = ctrl_pressed ? TRANSLATE_STEP_CTRL : TRANSLATE_STEP;
            const float translate_step_fast = ctrl_pressed ? TRANSLATE_STEP_CTRL_FAST : TRANSLATE_STEP_FAST;
            const float text_width = ImGui::CalcTextSize("-000.000").x +
                                     ImGui::GetStyle().FramePadding.x * 2.0f + BASE_INPUT_WIDTH_PADDING * getDpiScale();

            const glm::vec3 current_center = scene_manager->getSelectionWorldCenter();

            if (!state.multi_editing_active) {
                state.display_translation = current_center;
                state.display_euler = glm::vec3(0.0f);
                state.display_scale = glm::vec3(1.0f);
            }

            ImGui::Text(LOC(Transform::NODES_SELECTED), selected_names.size());
            ImGui::Text(LOC(Transform::SPACE), LOC(Transform::WORLD));
            ImGui::Separator();

            bool changed = false;
            bool any_active = false;

            // Detect selection change during active edit -> commit and reset
            const bool selection_changed = state.multi_editing_active &&
                                           state.multi_capture.node_names != selected_names;
            if (selection_changed) {
                if (!state.multi_capture.empty()) {
                    std::vector<glm::mat4> final_transforms;
                    for (const auto& name : state.multi_capture.node_names) {
                        final_transforms.push_back(scene_manager->getNodeTransform(name));
                    }
                    bool any_changed = false;
                    for (size_t i = 0; i < state.multi_capture.size(); ++i) {
                        if (state.multi_capture.local_transforms[i] != final_transforms[i]) {
                            any_changed = true;
                            break;
                        }
                    }
                    if (any_changed) {
                        auto cmd = std::make_unique<command::MultiTransformCommand>(
                            state.multi_capture.node_names,
                            state.multi_capture.local_transforms, std::move(final_transforms));
                        services().commands().execute(std::move(cmd));
                    }
                }
                state.resetMultiEdit();
                state.display_translation = current_center;
            }

            if (current_tool == ToolType::Translate) {
                ImGui::Text("%s", LOC(Transform::POSITION));
                ImGui::Text("X:");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(text_width);
                changed |= ImGui::InputFloat("##MultiPosX", &state.display_translation.x, translate_step, translate_step_fast, "%.3f");
                any_active |= ImGui::IsItemActive();

                ImGui::Text("Y:");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(text_width);
                changed |= ImGui::InputFloat("##MultiPosY", &state.display_translation.y, translate_step, translate_step_fast, "%.3f");
                any_active |= ImGui::IsItemActive();

                ImGui::Text("Z:");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(text_width);
                changed |= ImGui::InputFloat("##MultiPosZ", &state.display_translation.z, translate_step, translate_step_fast, "%.3f");
                any_active |= ImGui::IsItemActive();
            }

            if (current_tool == ToolType::Rotate) {
                ImGui::Text("%s", LOC(Transform::ROTATION_DEGREES));

                ImGui::Text("X:");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(text_width);
                changed |= ImGui::InputFloat("##MultiRotX", &state.display_euler.x, ROTATE_STEP, ROTATE_STEP_FAST, "%.1f");
                any_active |= ImGui::IsItemActive();

                ImGui::Text("Y:");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(text_width);
                changed |= ImGui::InputFloat("##MultiRotY", &state.display_euler.y, ROTATE_STEP, ROTATE_STEP_FAST, "%.1f");
                any_active |= ImGui::IsItemActive();

                ImGui::Text("Z:");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(text_width);
                changed |= ImGui::InputFloat("##MultiRotZ", &state.display_euler.z, ROTATE_STEP, ROTATE_STEP_FAST, "%.1f");
                any_active |= ImGui::IsItemActive();
            }

            if (current_tool == ToolType::Scale) {
                ImGui::Text("%s", LOC(Transform::SCALE));

                float uniform = (state.display_scale.x + state.display_scale.y + state.display_scale.z) / 3.0f;
                ImGui::Text("U:");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(text_width);
                if (ImGui::InputFloat("##MultiScaleU", &uniform, SCALE_STEP, SCALE_STEP_FAST, "%.3f")) {
                    uniform = std::max(uniform, MIN_SCALE);
                    state.display_scale = glm::vec3(uniform);
                    changed = true;
                }
                any_active |= ImGui::IsItemActive();

                ImGui::Separator();

                ImGui::Text("X:");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(text_width);
                changed |= ImGui::InputFloat("##MultiScaleX", &state.display_scale.x, SCALE_STEP, SCALE_STEP_FAST, "%.3f");
                any_active |= ImGui::IsItemActive();

                ImGui::Text("Y:");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(text_width);
                changed |= ImGui::InputFloat("##MultiScaleY", &state.display_scale.y, SCALE_STEP, SCALE_STEP_FAST, "%.3f");
                any_active |= ImGui::IsItemActive();

                ImGui::Text("Z:");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(text_width);
                changed |= ImGui::InputFloat("##MultiScaleZ", &state.display_scale.z, SCALE_STEP, SCALE_STEP_FAST, "%.3f");
                any_active |= ImGui::IsItemActive();

                state.display_scale = glm::max(state.display_scale, glm::vec3(MIN_SCALE));
            }

            if ((any_active || changed) && !state.multi_editing_active) {
                state.multi_editing_active = true;
                state.multi_edit_tool = current_tool;
                state.pivot_world = current_center;
                state.multi_capture = gizmo_ops::captureNodes(
                    scene_manager->getScene(), selected_names);
            }

            if (changed && state.multi_editing_active && !state.multi_capture.empty()) {
                Scene& scene = scene_manager->getScene();
                if (current_tool == ToolType::Translate) {
                    const glm::vec3 delta = state.display_translation - state.pivot_world;
                    gizmo_ops::applyMultiTranslation(state.multi_capture, scene, delta);
                } else if (current_tool == ToolType::Rotate) {
                    const glm::mat3 rotation = glm::mat3(glm::quat(glm::radians(state.display_euler)));
                    gizmo_ops::applyMultiRotation(state.multi_capture, scene, rotation, state.pivot_world);
                } else if (current_tool == ToolType::Scale) {
                    gizmo_ops::applyMultiScale(state.multi_capture, scene, state.display_scale, state.pivot_world);
                }
            }

            if (!any_active && state.multi_editing_active) {
                if (!state.multi_capture.empty()) {
                    std::vector<glm::mat4> final_transforms;
                    for (const auto& name : state.multi_capture.node_names) {
                        final_transforms.push_back(scene_manager->getNodeTransform(name));
                    }
                    bool any_changed = false;
                    for (size_t i = 0; i < state.multi_capture.size(); ++i) {
                        if (state.multi_capture.local_transforms[i] != final_transforms[i]) {
                            any_changed = true;
                            break;
                        }
                    }
                    if (any_changed) {
                        auto cmd = std::make_unique<command::MultiTransformCommand>(
                            state.multi_capture.node_names,
                            state.multi_capture.local_transforms, std::move(final_transforms));
                        services().commands().execute(std::move(cmd));
                    }
                }
                state.resetMultiEdit();
            }

            ImGui::Separator();
            if (ImGui::Button(LOC(Transform::RESET_ALL))) {
                if (state.multi_editing_active && !state.multi_capture.empty()) {
                    std::vector<glm::mat4> final_transforms;
                    for (const auto& name : state.multi_capture.node_names) {
                        final_transforms.push_back(scene_manager->getNodeTransform(name));
                    }
                    bool any_changed = false;
                    for (size_t i = 0; i < state.multi_capture.size(); ++i) {
                        if (state.multi_capture.local_transforms[i] != final_transforms[i]) {
                            any_changed = true;
                            break;
                        }
                    }
                    if (any_changed) {
                        auto cmd = std::make_unique<command::MultiTransformCommand>(
                            state.multi_capture.node_names,
                            state.multi_capture.local_transforms, std::move(final_transforms));
                        services().commands().execute(std::move(cmd));
                    }
                    state.resetMultiEdit();
                }

                static const glm::mat4 IDENTITY(1.0f);
                const auto capture = gizmo_ops::captureNodes(scene_manager->getScene(), selected_names);
                if (!capture.empty()) {
                    std::vector<glm::mat4> new_transforms(capture.size(), IDENTITY);
                    for (const auto& name : capture.node_names) {
                        scene_manager->setNodeTransform(name, IDENTITY);
                    }
                    auto cmd = std::make_unique<command::MultiTransformCommand>(
                        capture.node_names, capture.local_transforms, std::move(new_transforms));
                    services().commands().execute(std::move(cmd));
                }
            }
            return;
        }

        const std::string& node_name = selected_names[0];
        const bool use_world_space = (transform_space == TransformSpace::World);
        const glm::mat4 current_transform = scene_manager->getSelectedNodeTransform();
        auto [translation, rotation, scale] = decompose(current_transform);

        // Only decompose euler on selection change or external modification to avoid gimbal lock
        const bool selection_changed = (node_name != state.euler_display_node);
        const bool external_change = !sameRotation(rotation, state.euler_display_rotation);
        if (selection_changed || external_change) {
            state.euler_display = glm::degrees(glm::eulerAngles(rotation));
            state.euler_display_node = node_name;
            state.euler_display_rotation = rotation;
        }
        glm::vec3 euler = state.euler_display;

        const bool ctrl_pressed = ImGui::GetIO().KeyCtrl;
        const float translate_step = ctrl_pressed ? TRANSLATE_STEP_CTRL : TRANSLATE_STEP;
        const float translate_step_fast = ctrl_pressed ? TRANSLATE_STEP_CTRL_FAST : TRANSLATE_STEP_FAST;
        const float text_width = ImGui::CalcTextSize("-000.000").x +
                                 ImGui::GetStyle().FramePadding.x * 2.0f + BASE_INPUT_WIDTH_PADDING * getDpiScale();

        ImGui::Text(LOC(Transform::NODE), node_name.c_str());
        ImGui::Text(LOC(Transform::SPACE), use_world_space ? LOC(Transform::WORLD) : LOC(Transform::LOCAL));
        ImGui::Separator();

        bool changed = false;
        bool any_active = false;

        if (current_tool == ToolType::Translate) {
            ImGui::Text("%s", LOC(Transform::POSITION));
            ImGui::Text("X:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(text_width);
            changed |= ImGui::InputFloat("##PosX", &translation.x, translate_step, translate_step_fast, "%.3f");
            any_active |= ImGui::IsItemActive();

            ImGui::Text("Y:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(text_width);
            changed |= ImGui::InputFloat("##PosY", &translation.y, translate_step, translate_step_fast, "%.3f");
            any_active |= ImGui::IsItemActive();

            ImGui::Text("Z:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(text_width);
            changed |= ImGui::InputFloat("##PosZ", &translation.z, translate_step, translate_step_fast, "%.3f");
            any_active |= ImGui::IsItemActive();
        }

        if (current_tool == ToolType::Rotate) {
            ImGui::Text("%s", LOC(Transform::ROTATION_DEGREES));

            ImGui::Text("X:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(text_width);
            changed |= ImGui::InputFloat("##RotX", &euler.x, ROTATE_STEP, ROTATE_STEP_FAST, "%.1f");
            any_active |= ImGui::IsItemActive();

            ImGui::Text("Y:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(text_width);
            changed |= ImGui::InputFloat("##RotY", &euler.y, ROTATE_STEP, ROTATE_STEP_FAST, "%.1f");
            any_active |= ImGui::IsItemActive();

            ImGui::Text("Z:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(text_width);
            changed |= ImGui::InputFloat("##RotZ", &euler.z, ROTATE_STEP, ROTATE_STEP_FAST, "%.1f");
            any_active |= ImGui::IsItemActive();

            if (changed && !use_world_space) {
                rotation = glm::quat(glm::radians(euler));
            }
        }

        if (current_tool == ToolType::Scale) {
            ImGui::Text("%s", LOC(Transform::SCALE));

            float uniform = (scale.x + scale.y + scale.z) / 3.0f;
            ImGui::Text("U:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(text_width);
            if (ImGui::InputFloat("##ScaleU", &uniform, SCALE_STEP, SCALE_STEP_FAST, "%.3f")) {
                uniform = std::max(uniform, MIN_SCALE);
                scale = glm::vec3(uniform);
                changed = true;
            }
            any_active |= ImGui::IsItemActive();

            ImGui::Separator();

            ImGui::Text("X:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(text_width);
            changed |= ImGui::InputFloat("##ScaleX", &scale.x, SCALE_STEP, SCALE_STEP_FAST, "%.3f");
            any_active |= ImGui::IsItemActive();

            ImGui::Text("Y:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(text_width);
            changed |= ImGui::InputFloat("##ScaleY", &scale.y, SCALE_STEP, SCALE_STEP_FAST, "%.3f");
            any_active |= ImGui::IsItemActive();

            ImGui::Text("Z:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(text_width);
            changed |= ImGui::InputFloat("##ScaleZ", &scale.z, SCALE_STEP, SCALE_STEP_FAST, "%.3f");
            any_active |= ImGui::IsItemActive();

            scale = glm::max(scale, glm::vec3(MIN_SCALE));
        }

        // Capture initial state before first change
        if ((any_active || changed) && !state.editing_active) {
            state.editing_active = true;
            state.editing_node_name = node_name;
            state.transform_before_edit = current_transform;
            state.initial_translation = translation;
            state.initial_scale = scale;
        }

        if (current_tool == ToolType::Rotate && changed) {
            state.euler_display = euler;
        }

        if (changed) {
            glm::mat4 new_transform;

            if (use_world_space) {
                if (current_tool == ToolType::Translate) {
                    new_transform = state.transform_before_edit;
                    new_transform[3] = glm::vec4(translation, 1.0f);
                } else if (current_tool == ToolType::Rotate) {
                    rotation = glm::quat(glm::radians(euler));
                    new_transform = glm::translate(glm::mat4(1.0f), translation) *
                                    glm::mat4_cast(rotation) *
                                    glm::scale(glm::mat4(1.0f), scale);
                } else {
                    const glm::mat3 initial_rs(state.transform_before_edit);
                    const glm::vec3 scale_ratio = scale / state.initial_scale;
                    new_transform = glm::mat4(glm::mat3(glm::scale(glm::mat4(1.0f), scale_ratio)) * initial_rs);
                    new_transform[3] = state.transform_before_edit[3];
                }
            } else {
                new_transform = glm::translate(glm::mat4(1.0f), translation) *
                                glm::mat4_cast(rotation) *
                                glm::scale(glm::mat4(1.0f), scale);
            }

            scene_manager->setSelectedNodeTransform(new_transform);

            if (current_tool == ToolType::Rotate) {
                state.euler_display_rotation = rotation;
            }
        }

        // Commit undo command when editing ends
        if (!any_active && state.editing_active) {
            state.editing_active = false;
            const glm::mat4 final_transform = scene_manager->getSelectedNodeTransform();
            if (state.transform_before_edit != final_transform) {
                auto cmd = std::make_unique<command::TransformCommand>(
                    state.editing_node_name,
                    state.transform_before_edit, final_transform);
                services().commands().execute(std::move(cmd));
            }
        }

        ImGui::Separator();
        if (ImGui::Button(LOC(Transform::RESET_TRANSFORM))) {
            auto cmd = std::make_unique<command::TransformCommand>(
                node_name, current_transform, glm::mat4(1.0f));
            services().commands().execute(std::move(cmd));
            scene_manager->setSelectedNodeTransform(glm::mat4(1.0f));
            state.euler_display = glm::vec3(0.0f);
            state.euler_display_rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        }
    }

} // namespace lfs::vis::gui::panels
