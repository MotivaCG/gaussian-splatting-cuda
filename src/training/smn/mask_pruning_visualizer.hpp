/* ScanMeNow file */

#pragma once

/**
 * @file mask_pruning_visualizer.hpp
 * @brief Pure C++ visualization for mask-based pruning diagnostics.
 *
 * Generates diagnostic images showing masks, splat projections, and pruning decisions
 * using only existing LFS infrastructure (no Python, no OpenCV).
 */

#include "core/camera.hpp"
#include "core/image_io.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "mask_pruning.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace lfs::training::mask_pruning {
    namespace visualizer {

        /**
         * @brief Color for drawing (RGB).
         */
        struct Color {
            uint8_t r, g, b;

            static constexpr Color Red() { return {255, 0, 0}; }
            static constexpr Color Green() { return {0, 255, 0}; }
            static constexpr Color Blue() { return {0, 0, 255}; }
            static constexpr Color Yellow() { return {255, 255, 0}; }
            static constexpr Color Cyan() { return {0, 255, 255}; }
            static constexpr Color Magenta() { return {255, 0, 255}; }
            static constexpr Color White() { return {255, 255, 255}; }
            static constexpr Color Black() { return {0, 0, 0}; }
            static constexpr Color Gray() { return {128, 128, 128}; }
        };

        /**
         * @brief Simple image canvas for drawing diagnostic overlays.
         */
        class Canvas {
        public:
            Canvas(int width, int height);

            // Initialize with mask as grayscale background
            void set_mask_background(const lfs::core::Tensor& mask, float alpha = 0.3f);

            // Drawing primitives
            void draw_point(int x, int y, Color color, int size = 1);
            void draw_circle(int cx, int cy, int radius, Color color, int thickness = 1);
            void draw_ellipse(int cx, int cy, int rx, int ry, Color color, int thickness = 1);
            void draw_cross(int cx, int cy, int size, Color color, int thickness = 1);

            // Text rendering (simple, using raster patterns)
            void draw_text(int x, int y, const std::string& text, Color color);

            // Save to file
            void save(const std::filesystem::path& path);

            int width() const { return width_; }
            int height() const { return height_; }

        private:
            int width_;
            int height_;
            std::vector<uint8_t> data_; // RGB, row-major

            void set_pixel(int x, int y, Color color, float alpha = 1.0f);
            bool in_bounds(int x, int y) const;
        };

        /**
         * @brief Generate diagnostic visualizations for one camera view.
         */
        class CameraViewVisualizer {
        public:
            CameraViewVisualizer(
                const lfs::core::Camera& camera,
                const lfs::core::Tensor& mask,
                const ProjectionResult& proj_result,
                int camera_index);

            /**
             * @brief Generate all visualization layers for this camera.
             * Creates 6 images showing different aspects of the pruning process.
             */
            void generate_all_layers(const std::filesystem::path& output_dir);

            /**
             * @brief Generate individual layers (for more control).
             */
            void generate_layer_1_mask_only(const std::filesystem::path& path);
            void generate_layer_2_all_centers(const std::filesystem::path& path);
            void generate_layer_3_centers_classified(const std::filesystem::path& path);
            void generate_layer_4_sample_footprints(const std::filesystem::path& path, int max_splats = 500);
            void generate_layer_5_problem_splats(const std::filesystem::path& path);
            void generate_layer_6_summary_overlay(const std::filesystem::path& path);

            // Statistics
            int num_visible() const;
            int num_inside() const;
            int num_outside() const;
            int num_problem_splats() const; // Outside + large radius

            // Access to classification data (for summary generation)
            const std::vector<bool>& get_visible() const { return visible_; }
            const std::vector<bool>& get_inside_mask() const { return inside_mask_; }

        private:
            const lfs::core::Camera& camera_;
            mutable lfs::core::Tensor mask_; // mutable to allow accessor() in const methods
            ProjectionResult proj_;
            int camera_index_;
            std::string camera_name_;

            int width_;
            int height_;
            int num_splats_;

            // Cached classification
            std::vector<bool> visible_;
            std::vector<bool> inside_mask_;
            std::vector<bool> is_problem_;

            void classify_splats();
            bool is_inside_mask(int px, int py) const;
        };

        /**
         * @brief Generate summary visualizations across multiple cameras.
         */
        class SummaryVisualizer {
        public:
            SummaryVisualizer() = default;

            /**
             * @brief Add statistics from one camera view.
             */
            void add_camera_stats(
                int num_splats,
                const std::vector<bool>& visible,
                const std::vector<bool>& inside);

            /**
             * @brief Generate summary text file with statistics.
             */
            void generate_summary_text(const std::filesystem::path& path) const;

            /**
             * @brief Generate summary images (histograms, pie charts).
             * Note: This creates simple visualizations without external plotting libraries.
             */
            void generate_summary_images(const std::filesystem::path& output_dir) const;

        private:
            int num_cameras_ = 0;
            int total_splats_ = 0;
            std::vector<int> per_splat_visibility_;
            std::vector<int> per_splat_inside_count_;

            void generate_visibility_histogram(const std::filesystem::path& path) const;
            void generate_ratio_histogram(const std::filesystem::path& path) const;
        };

        /**
         * @brief Main entry point for diagnostic visualization.
         *
         * Call this after training to generate diagnostic images showing
         * how the mask-based pruning would work.
         *
         * @param strategy Training strategy with model
         * @param dataset Dataset with cameras and masks
         * @param config Pruning configuration
         * @param output_dir Where to save diagnostic images
         * @param max_cameras Maximum number of cameras to visualize (default 10)
         * @return true if successful
         */
        bool generate_pruning_diagnostics(
            IStrategy& strategy,
            const CameraDataset& dataset,
            const CenterVotePruningConfig& config,
            const std::filesystem::path& output_dir,
            int max_cameras = 10);

    } // namespace visualizer
} // namespace lfs::training::mask_pruning