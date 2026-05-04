/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gl_resources.hpp"
#include "shader_manager.hpp"
#include <filesystem>
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <map>
#include <string>
#include <string_view>

namespace lfs::rendering {

    struct Character {
        GLuint textureID;
        glm::ivec2 size;
        glm::ivec2 bearing;
        GLuint advance;
    };

    class TextRenderer {
    public:
        TextRenderer(unsigned int width, unsigned int height);
        ~TextRenderer();

        Result<void> LoadFont(const std::filesystem::path& fontPath, unsigned int fontSize);
        Result<void> LoadAsciiFont(const std::filesystem::path& fontPath, unsigned int fontSize);
        Result<void> RenderText(const std::string& text, float x, float y, float scale,
                                const glm::vec3& color = glm::vec3(1.0f),
                                float alpha = 1.0f);
        void updateScreenSize(unsigned int width, unsigned int height);

        // Get character dimensions for centering calculations
        [[nodiscard]] glm::vec2 getCharacterSize(char c, float scale) const;
        [[nodiscard]] glm::vec2 getCharacterBearing(char c, float scale) const;
        [[nodiscard]] glm::vec2 measureString(std::string_view text, float scale) const;
        [[nodiscard]] unsigned int getBasePixelSize() const { return base_pixel_size_; }
        [[nodiscard]] float getAscender(float scale) const { return ascender_px_ * scale; }
        [[nodiscard]] float getLineHeight(float scale) const { return line_height_px_ * scale; }

    private:
        unsigned int screenWidth, screenHeight;
        unsigned int base_pixel_size_ = 0;
        float ascender_px_ = 0.0f;
        float line_height_px_ = 0.0f;
        VAO vao_;
        VBO vbo_;
        ManagedShader shader_;
        std::map<char, Character> characters;

        Result<void> initRenderData();
    };

} // namespace lfs::rendering
