#include "window_manager.hpp"
// clang-format off
// CRITICAL: GLAD must be included before GLFW to avoid OpenGL header conflicts
#include <glad/glad.h>
#include <GLFW/glfw3.h>
// clang-format on
#include <iostream>
#include <print>

namespace gs {

    WindowManager::WindowManager(const std::string& title, int width, int height)
        : title_(title),
          window_size_(width, height),
          framebuffer_size_(width, height) {
    }

    WindowManager::~WindowManager() {
        if (window_) {
            glfwDestroyWindow(window_);
        }
        glfwTerminate();
    }

    bool WindowManager::init() {
        if (!glfwInit()) {
            std::cerr << "Failed to initialize GLFW!" << std::endl;
            return false;
        }

        glfwWindowHint(GLFW_SAMPLES, 8);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, GLFW_FALSE);
        glfwWindowHint(GLFW_DEPTH_BITS, 24);

        window_ = glfwCreateWindow(
            window_size_.x,
            window_size_.y,
            title_.c_str(),
            nullptr,
            nullptr);

        if (!window_) {
            std::cerr << "Failed to create GLFW window!" << std::endl;
            glfwTerminate();
            return false;
        }

        glfwMakeContextCurrent(window_);

        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
            std::cerr << "GLAD init failed" << std::endl;
            glfwTerminate();
            return false;
        }

        // Enable vsync by default
        glfwSwapInterval(1);

        // Set up OpenGL state
        glEnable(GL_LINE_SMOOTH);
        glDepthFunc(GL_LEQUAL);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glBlendEquation(GL_FUNC_ADD);
        glEnable(GL_PROGRAM_POINT_SIZE);

        return true;
    }

    void WindowManager::updateWindowSize() {
        int winW, winH, fbW, fbH;
        glfwGetWindowSize(window_, &winW, &winH);
        glfwGetFramebufferSize(window_, &fbW, &fbH);
        window_size_ = glm::ivec2(winW, winH);
        framebuffer_size_ = glm::ivec2(fbW, fbH);
        glViewport(0, 0, fbW, fbH);
    }

    void WindowManager::swapBuffers() {
        glfwSwapBuffers(window_);
    }

    void WindowManager::pollEvents() {
        glfwPollEvents();
    }

    bool WindowManager::shouldClose() const {
        return glfwWindowShouldClose(window_);
    }

    void WindowManager::setVSync(bool enabled) {
        glfwSwapInterval(enabled ? 1 : 0);
    }

} // namespace gs
