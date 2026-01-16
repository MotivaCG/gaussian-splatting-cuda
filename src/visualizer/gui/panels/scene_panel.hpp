/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/events.hpp"
#include "gui/ui_context.hpp"
#include "scene/scene.hpp"
#include "training/training_manager.hpp"

#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace lfs::vis {

    class SceneManager;

    namespace gui {

        // Scene panel that integrates with existing GUI
        // Queries Scene and SceneManager directly - no duplicate state
        class ScenePanel {
        public:
            explicit ScenePanel(std::shared_ptr<const TrainerManager> trainer_manager);
            ~ScenePanel();

            void render(bool* p_open, const UIContext* ctx);
            void renderContent(const UIContext* ctx);
            void setOnDatasetLoad(std::function<void(const std::filesystem::path&)> callback);

        private:
            // Panel state
            float m_panelWidth = 300.0f;

            // Image list data for dataset mode
            std::vector<std::filesystem::path> m_imagePaths;
            using CamId = int;
            std::map<std::filesystem::path, CamId> m_pathToCamId;
            int m_selectedImageIndex = -1;
            std::filesystem::path m_currentDatasetPath;
            bool m_needsScrollToSelection = false;

            // Camera highlighting in scene graph (synced with GoToCamView)
            int m_highlightedCamUid = -1;
            bool m_needsScrollToCam = false;

            // Drag-drop state
            std::string m_dragPayload;

            // Filter
            char m_filterText[128] = {};
            mutable int m_rowIndex = 0;                     // For alternating row backgrounds
            mutable float m_selectionFlashIntensity = 0.0f; // Selection flash effect (0-1)

            // Rename state
            struct RenameState {
                bool is_renaming = false;
                std::string renaming_node_name;
                char buffer[256] = {};
                bool focus_input = false;
                bool input_was_active = false;
                bool escape_pressed = false;
            } m_renameState;

            // Tab management
            enum class TabType { Images,
                                 PLYs };
            TabType m_activeTab = TabType::PLYs;

            // Callbacks
            std::function<void(const std::filesystem::path&)> m_onDatasetLoad;

            // Image preview integration
            std::unique_ptr<class ImagePreview> m_imagePreview;
            bool m_showImagePreview = false;

            // For loading training cameras in scene panel
            std::shared_ptr<const TrainerManager> m_trainerManager;

            // Scene icons (Tabler Icons - MIT license)
            struct SceneIcons {
                unsigned int visible = 0;
                unsigned int hidden = 0;
                unsigned int group = 0;
                unsigned int dataset = 0;
                unsigned int camera = 0;
                unsigned int splat = 0;
                unsigned int cropbox = 0;
                unsigned int ellipsoid = 0;
                unsigned int pointcloud = 0;
                unsigned int mask = 0;
                unsigned int trash = 0;
                unsigned int grip = 0;
                bool initialized = false;
            } m_icons;

            void initIcons();
            void shutdownIcons();

            // Helper methods
            void setupEventHandlers();
            bool hasImages() const;
            bool hasPLYs(const UIContext* ctx) const;

            // Event handlers for images
            void handleGoToCamView(const lfs::core::events::cmd::GoToCamView& event);
            void loadImageCams(const std::filesystem::path& path);
            void onImageSelected(const std::filesystem::path& imagePath);
            void onImageDoubleClicked(size_t imageIndex);

            // PLY scene graph rendering - queries Scene directly
            void renderPLYSceneGraph(const UIContext* ctx);
            void renderModelsFolder(const Scene& scene, const std::unordered_set<std::string>& selected_names);
            void renderBackgroundSection();
            void renderModelNode(const SceneNode& node, const Scene& scene,
                                 const std::unordered_set<std::string>& selected_names, int depth = 0);
            void renderNodeChildren(NodeId parent_id, const Scene& scene,
                                    const std::unordered_set<std::string>& selected_names, int depth);
            void renderIndentGuides(int depth) const;
            void renderImageList();

            // Drag-drop with constraints
            bool handleDragDrop(const std::string& target_name, bool is_group_target);
            static bool canReparent(const SceneNode& node, const SceneNode* target, const Scene& scene);

            // Rename functionality
            void startRenaming(const std::string& node_name);
            void finishRenaming(SceneManager* scene_manager);
            void cancelRenaming();

            // Training protection
            bool isNodeProtectedDuringTraining(const SceneNode& node, const Scene& scene) const;
        };

    } // namespace gui
} // namespace lfs::vis
