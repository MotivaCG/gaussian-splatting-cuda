/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gl_resources.hpp"
#include "shader_manager.hpp"
#include <glm/glm.hpp>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace lfs::rendering {

    class TextRenderer;

    struct OverlayColor {
        float r, g, b, a;
    };

    class ScreenOverlayRenderer {
    public:
        ScreenOverlayRenderer();
        ~ScreenOverlayRenderer();

        Result<void> initialize();
        void shutdown();

        void beginFrame(int window_pixel_w, int window_pixel_h);
        void endFrame();
        bool isFrameActive() const { return frame_active_; }

        void pushClipRect(glm::vec2 min, glm::vec2 max, bool intersect_with_current = true);
        void popClipRect();

        void addLine(glm::vec2 a, glm::vec2 b, OverlayColor c, float thickness = 1.0f);
        void addPolyline(std::span<const glm::vec2> pts, OverlayColor c, bool closed, float thickness);
        void addConvexPolyFilled(std::span<const glm::vec2> pts, OverlayColor c);
        void addCircle(glm::vec2 center, float radius, OverlayColor c, int segments, float thickness);
        void addCircleFilled(glm::vec2 center, float radius, OverlayColor c, int segments = 0);
        void addRect(glm::vec2 a, glm::vec2 b, OverlayColor c, float thickness = 1.0f);
        void addRectFilled(glm::vec2 a, glm::vec2 b, OverlayColor c);
        void addTriangleFilled(glm::vec2 p0, glm::vec2 p1, glm::vec2 p2, OverlayColor c);

        void addText(glm::vec2 top_left_px, std::string_view text, OverlayColor c, float size_px);
        void addTextWithShadow(glm::vec2 top_left_px, std::string_view text,
                               OverlayColor c, OverlayColor shadow_c, float size_px,
                               glm::vec2 shadow_offset = {1.0f, 1.0f});
        [[nodiscard]] glm::vec2 measureText(std::string_view text, float size_px) const;

        void flush();

        class ScopedClipRect {
        public:
            ScopedClipRect(ScreenOverlayRenderer& r, glm::vec2 min, glm::vec2 max,
                           bool intersect_with_current = true)
                : r_(r) {
                r_.pushClipRect(min, max, intersect_with_current);
            }
            ~ScopedClipRect() { r_.popClipRect(); }
            ScopedClipRect(const ScopedClipRect&) = delete;
            ScopedClipRect& operator=(const ScopedClipRect&) = delete;

        private:
            ScreenOverlayRenderer& r_;
        };

    private:
        struct Vertex {
            glm::vec2 pos_px;
            glm::vec4 color_premul;
        };
        struct ClipRect {
            glm::vec2 min;
            glm::vec2 max;
        };

        void uploadAndDraw();
        void applyClipScissor() const;
        std::vector<glm::vec2> circlePerimeter(glm::vec2 center, float radius, int segments) const;
        glm::vec4 toPremul(OverlayColor c) const;
        int adaptiveSegments(float radius) const;
        void emitTriangle(glm::vec2 p0, glm::vec2 p1, glm::vec2 p2, glm::vec4 c);
        void emitLineSegment(glm::vec2 a, glm::vec2 b, glm::vec4 c, float thickness);
        Result<void> ensureFontLoaded();

        bool frame_active_ = false;
        int window_w_ = 0;
        int window_h_ = 0;

        VAO vao_;
        VBO vbo_;
        std::size_t vbo_capacity_bytes_ = 0;
        ManagedShader shader_;
        bool initialized_ = false;

        std::vector<Vertex> verts_;
        std::vector<ClipRect> clip_stack_;

        std::unique_ptr<TextRenderer> text_renderer_;
        bool text_renderer_failed_ = false;
        unsigned int text_atlas_size_ = 32;
    };

} // namespace lfs::rendering
