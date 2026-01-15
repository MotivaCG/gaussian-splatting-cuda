/* ScanMeNow file */

// =============================================================================
// IMPORTANT: Visual Studio PCH Configuration
// =============================================================================
// If you get error C1010 about precompiled headers, you have two options:
//
// OPTION 1 (Recommended): Disable PCH for this file in CMakeLists.txt:
//   set_source_files_properties(mask_pruning_visualizer.cpp
//                                PROPERTIES SKIP_PRECOMPILE_HEADERS ON)
//
// OPTION 2: Uncomment the line below and adjust the path:
//   #include "pch.hpp"  // or your PCH filename
// =============================================================================

#include "mask_pruning_visualizer.hpp"
#include "core/logger.hpp"

#include <algorithm>
#include <cmath>
#include <expected>
#include <format>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <random>
#include <sstream>

namespace lfs::training::mask_pruning::visualizer {

    namespace {

        static inline int round_to_int(float x) {
            return static_cast<int>(std::lround(static_cast<double>(x)));
        }

        static inline int clamp_int(int v, int lo, int hi) {
            return (v < lo) ? lo : (v > hi ? hi : v);
        }

        static inline bool is_visible_radii_strict(int rx, int ry) {
            return (rx > 0) && (ry > 0);
        }

        // Dataset sizing helper (copied from mask_pruning.cpp)
        struct DatasetSizing {
            int resize_factor = 1;
            int max_width = 0;
        };

        static std::expected<DatasetSizing, std::string> get_dataset_sizing_or_error(const CameraDataset& dataset) {
            DatasetSizing out;
            out.resize_factor = dataset.get_resize_factor();
            out.max_width = dataset.get_max_width();

            if (out.resize_factor <= 0) {
                return std::unexpected("CameraDataset::get_resize_factor() is invalid (<= 0).");
            }
            if (out.max_width < 0) {
                return std::unexpected("CameraDataset::get_max_width() is invalid (< 0).");
            }
            return out;
        }

    } // namespace

    // ============================================================================
    // Canvas Implementation
    // ============================================================================

    Canvas::Canvas(int width, int height)
        : width_(width), height_(height) {
        data_.resize(static_cast<size_t>(width) * static_cast<size_t>(height) * 3, 0);
    }

    void Canvas::set_mask_background(const lfs::core::Tensor& mask, float alpha) {
        using namespace lfs::core;

        Tensor mask_cpu = mask.cpu().contiguous();
        auto mask_acc = mask_cpu.accessor<float, 2>();

        for (int y = 0; y < height_; ++y) {
            for (int x = 0; x < width_; ++x) {
                float mask_val = std::fmin(1.0f,mask_acc(y, x));
                uint8_t gray = static_cast<uint8_t>(mask_val * 255.0f * alpha);

                size_t idx = (static_cast<size_t>(y) * static_cast<size_t>(width_) + static_cast<size_t>(x)) * 3;
                data_[idx + 0] = gray;
                data_[idx + 1] = gray;
                data_[idx + 2] = gray;
            }
        }
    }

    bool Canvas::in_bounds(int x, int y) const {
        return x >= 0 && x < width_ && y >= 0 && y < height_;
    }

    void Canvas::set_pixel(int x, int y, Color color, float alpha) {
        if (!in_bounds(x, y))
            return;

        size_t idx = (static_cast<size_t>(y) * static_cast<size_t>(width_) + static_cast<size_t>(x)) * 3;

        if (alpha >= 1.0f) {
            data_[idx + 0] = color.r;
            data_[idx + 1] = color.g;
            data_[idx + 2] = color.b;
        } else {
            // Alpha blending
            data_[idx + 0] = static_cast<uint8_t>(data_[idx + 0] * (1.0f - alpha) + color.r * alpha);
            data_[idx + 1] = static_cast<uint8_t>(data_[idx + 1] * (1.0f - alpha) + color.g * alpha);
            data_[idx + 2] = static_cast<uint8_t>(data_[idx + 2] * (1.0f - alpha) + color.b * alpha);
        }
    }

    void Canvas::draw_point(int x, int y, Color color, int size) {
        if (size == 1) {
            set_pixel(x, y, color);
        } else {
            int half = size / 2;
            for (int dy = -half; dy <= half; ++dy) {
                for (int dx = -half; dx <= half; ++dx) {
                    set_pixel(x + dx, y + dy, color);
                }
            }
        }
    }

    void Canvas::draw_circle(int cx, int cy, int radius, Color color, int thickness) {
        // Bresenham's circle algorithm
        int x = 0;
        int y = radius;
        int d = 3 - 2 * radius;

        auto draw_circle_points = [&](int xc, int yc, int x, int y) {
            for (int t = 0; t < thickness; ++t) {
                set_pixel(xc + x, yc + y, color);
                set_pixel(xc - x, yc + y, color);
                set_pixel(xc + x, yc - y, color);
                set_pixel(xc - x, yc - y, color);
                set_pixel(xc + y, yc + x, color);
                set_pixel(xc - y, yc + x, color);
                set_pixel(xc + y, yc - x, color);
                set_pixel(xc - y, yc - x, color);
            }
        };

        draw_circle_points(cx, cy, x, y);

        while (y >= x) {
            x++;
            if (d > 0) {
                y--;
                d = d + 4 * (x - y) + 10;
            } else {
                d = d + 4 * x + 6;
            }
            draw_circle_points(cx, cy, x, y);
        }
    }

    void Canvas::draw_ellipse(int cx, int cy, int rx, int ry, Color color, int thickness) {
        // Simple ellipse using parametric form
        const int num_points = std::max(rx, ry) * 4;
        const float dt = 2.0f * M_PI / static_cast<float>(num_points);

        int prev_x = cx + rx;
        int prev_y = cy;

        for (int i = 1; i <= num_points; ++i) {
            float t = static_cast<float>(i) * dt;
            int x = cx + static_cast<int>(rx * std::cos(t));
            int y = cy + static_cast<int>(ry * std::sin(t));

            // Draw line from prev to current
            int dx = x - prev_x;
            int dy = y - prev_y;
            int steps = std::max(std::abs(dx), std::abs(dy));

            for (int s = 0; s <= steps; ++s) {
                float frac = static_cast<float>(s) / static_cast<float>(std::max(1, steps));
                int ix = prev_x + static_cast<int>(dx * frac);
                int iy = prev_y + static_cast<int>(dy * frac);
                draw_point(ix, iy, color, thickness);
            }

            prev_x = x;
            prev_y = y;
        }
    }

    void Canvas::draw_cross(int cx, int cy, int size, Color color, int thickness) {
        for (int i = -size; i <= size; ++i) {
            draw_point(cx + i, cy, color, thickness);
            draw_point(cx, cy + i, color, thickness);
        }
    }

    void Canvas::draw_text(int x, int y, const std::string& text, Color color) {
        // Simple 5x7 bitmap font for basic text
        // For now, just draw a small rectangle as placeholder
        // (Full implementation would require a bitmap font)
        draw_point(x, y, color, 2);
    }

    void Canvas::save(const std::filesystem::path& path) {
        using namespace lfs::core;

        // Convert to Tensor [H, W, 3]
        Tensor image = Tensor::empty(
            {static_cast<size_t>(height_), static_cast<size_t>(width_), 3UL},
            Device::CPU,
            DataType::UInt8);

        auto acc = image.accessor<uint8_t, 3>();
        for (int y = 0; y < height_; ++y) {
            for (int x = 0; x < width_; ++x) {
                size_t idx = (static_cast<size_t>(y) * static_cast<size_t>(width_) + static_cast<size_t>(x)) * 3;
                acc(y, x, 0) = data_[idx + 0];
                acc(y, x, 1) = data_[idx + 1];
                acc(y, x, 2) = data_[idx + 2];
            }
        }

        // Use existing image saving
        lfs::core::image_io::save_image_async(path, image);
    }

    // ============================================================================
    // CameraViewVisualizer Implementation
    // ============================================================================

    CameraViewVisualizer::CameraViewVisualizer(
        const lfs::core::Camera& camera,
        const lfs::core::Tensor& mask,
        const ProjectionResult& proj_result,
        int camera_index)
        : camera_(camera), mask_(mask.cpu().contiguous()) // Remove const, copy to CPU
          ,
          proj_(proj_result),
          camera_index_(camera_index),
          camera_name_("cam_" + std::to_string(camera_index)),
          width_(static_cast<int>(camera.image_width())),
          height_(static_cast<int>(camera.image_height())) {
        using namespace lfs::core;

        // Get number of splats
        Tensor radii_cpu = proj_.radii.cpu().contiguous();
        num_splats_ = static_cast<int>(radii_cpu.shape()[0]);

        classify_splats();
    }

    void CameraViewVisualizer::classify_splats() {
        using namespace lfs::core;

        Tensor radii_cpu = proj_.radii.cpu().contiguous();
        Tensor means2d_cpu = proj_.means2d.cpu().contiguous();

        auto r_acc = radii_cpu.accessor<int32_t, 2>();
        auto m_acc = means2d_cpu.accessor<float, 2>();

        visible_.resize(static_cast<size_t>(num_splats_));
        inside_mask_.resize(static_cast<size_t>(num_splats_));
        is_problem_.resize(static_cast<size_t>(num_splats_));

        for (int i = 0; i < num_splats_; ++i) {
            const int rx = static_cast<int>(r_acc(i, 0));
            const int ry = static_cast<int>(r_acc(i, 1));
            const float cx = m_acc(i, 0);
            const float cy = m_acc(i, 1);

            visible_[static_cast<size_t>(i)] = is_visible_radii_strict(rx, ry);

            bool inside = false;
            if (visible_[static_cast<size_t>(i)]) {
                int px = clamp_int(round_to_int(cx), 0, width_ - 1);
                int py = clamp_int(round_to_int(cy), 0, height_ - 1);
                inside = is_inside_mask(px, py);
            }
            inside_mask_[static_cast<size_t>(i)] = inside;

            // Problem splat: visible, outside, and large radius
            const float max_r = std::max(std::abs(static_cast<float>(rx)), std::abs(static_cast<float>(ry)));
            is_problem_[static_cast<size_t>(i)] = visible_[static_cast<size_t>(i)] &&
                                                  !inside &&
                                                  (max_r >= 2.0f);
        }
    }

    bool CameraViewVisualizer::is_inside_mask(int px, int py) const {
        auto mask_acc = mask_.accessor<float, 2>();
        return mask_acc(py, px) >= 0.5f;
    }

    int CameraViewVisualizer::num_visible() const {
        return std::count(visible_.begin(), visible_.end(), true);
    }

    int CameraViewVisualizer::num_inside() const {
        int count = 0;
        for (size_t i = 0; i < visible_.size(); ++i) {
            if (visible_[i] && inside_mask_[i])
                count++;
        }
        return count;
    }

    int CameraViewVisualizer::num_outside() const {
        int count = 0;
        for (size_t i = 0; i < visible_.size(); ++i) {
            if (visible_[i] && !inside_mask_[i])
                count++;
        }
        return count;
    }

    int CameraViewVisualizer::num_problem_splats() const {
        return std::count(is_problem_.begin(), is_problem_.end(), true);
    }

    void CameraViewVisualizer::generate_all_layers(const std::filesystem::path& output_dir) {
        std::filesystem::create_directories(output_dir);

        //generate_layer_1_mask_only(output_dir / (camera_name_ + "_01_mask.png"));
        //generate_layer_2_all_centers(output_dir / (camera_name_ + "_02_centers_all.png"));
        //generate_layer_3_centers_classified(output_dir / (camera_name_ + "_03_centers_classified.png"));
        //generate_layer_4_sample_footprints(output_dir / (camera_name_ + "_04_footprints.png"));
        //generate_layer_5_problem_splats(output_dir / (camera_name_ + "_05_problem_splats.png"));
        generate_layer_6_summary_overlay(output_dir / (camera_name_ + "_06_summary.png"));
    }

    void CameraViewVisualizer::generate_layer_1_mask_only(const std::filesystem::path& path) {
        Canvas canvas(width_, height_);
        canvas.set_mask_background(mask_, 0.5f);
        canvas.save(path);
    }

    void CameraViewVisualizer::generate_layer_2_all_centers(const std::filesystem::path& path) {
        using namespace lfs::core;

        Canvas canvas(width_, height_);
        canvas.set_mask_background(mask_, 0.3f);

        Tensor means2d_cpu = proj_.means2d.cpu().contiguous();
        auto m_acc = means2d_cpu.accessor<float, 2>();

        for (int i = 0; i < num_splats_; ++i) {
            if (!visible_[static_cast<size_t>(i)])
                continue;

            int x = round_to_int(m_acc(i, 0));
            int y = round_to_int(m_acc(i, 1));
            canvas.draw_point(x, y, Color::Cyan(), 1);
        }

        canvas.save(path);
    }

    void CameraViewVisualizer::generate_layer_3_centers_classified(const std::filesystem::path& path) {
        using namespace lfs::core;

        Canvas canvas(width_, height_);
        canvas.set_mask_background(mask_, 0.3f);

        Tensor means2d_cpu = proj_.means2d.cpu().contiguous();
        auto m_acc = means2d_cpu.accessor<float, 2>();

        for (int i = 0; i < num_splats_; ++i) {
            if (!visible_[static_cast<size_t>(i)])
                continue;

            int x = round_to_int(m_acc(i, 0));
            int y = round_to_int(m_acc(i, 1));

            Color color = inside_mask_[static_cast<size_t>(i)] ? Color::Green() : Color::Red();
            int size = inside_mask_[static_cast<size_t>(i)] ? 1 : 2;
            canvas.draw_point(x, y, color, size);
        }

        canvas.save(path);
    }

    void CameraViewVisualizer::generate_layer_4_sample_footprints(const std::filesystem::path& path, int max_splats) {
        using namespace lfs::core;

        Canvas canvas(width_, height_);
        canvas.set_mask_background(mask_, 0.3f);

        Tensor radii_cpu = proj_.radii.cpu().contiguous();
        Tensor means2d_cpu = proj_.means2d.cpu().contiguous();
        auto r_acc = radii_cpu.accessor<int32_t, 2>();
        auto m_acc = means2d_cpu.accessor<float, 2>();

        // Sample visible splats
        std::vector<int> visible_indices;
        for (int i = 0; i < num_splats_; ++i) {
            if (visible_[static_cast<size_t>(i)]) {
                visible_indices.push_back(i);
            }
        }

        if (visible_indices.size() > static_cast<size_t>(max_splats)) {
            // Random sample
            std::mt19937 rng(42);
            std::shuffle(visible_indices.begin(), visible_indices.end(), rng);
            visible_indices.resize(static_cast<size_t>(max_splats));
        }

        for (int idx : visible_indices) {
            int cx = round_to_int(m_acc(idx, 0));
            int cy = round_to_int(m_acc(idx, 1));
            int rx = std::abs(static_cast<int>(r_acc(idx, 0)));
            int ry = std::abs(static_cast<int>(r_acc(idx, 1)));

            Color color = inside_mask_[static_cast<size_t>(idx)] ? Color::Green() : Color::Red();
            canvas.draw_ellipse(cx, cy, rx, ry, color, 1);
            canvas.draw_point(cx, cy, color, 1);
        }

        canvas.save(path);
    }

    void CameraViewVisualizer::generate_layer_5_problem_splats(const std::filesystem::path& path) {
        using namespace lfs::core;

        Canvas canvas(width_, height_);
        canvas.set_mask_background(mask_, 0.3f);

        Tensor radii_cpu = proj_.radii.cpu().contiguous();
        Tensor means2d_cpu = proj_.means2d.cpu().contiguous();
        auto r_acc = radii_cpu.accessor<int32_t, 2>();
        auto m_acc = means2d_cpu.accessor<float, 2>();

        for (int i = 0; i < num_splats_; ++i) {
            if (!is_problem_[static_cast<size_t>(i)])
                continue;

            int cx = round_to_int(m_acc(i, 0));
            int cy = round_to_int(m_acc(i, 1));
            int rx = std::abs(static_cast<int>(r_acc(i, 0)));
            int ry = std::abs(static_cast<int>(r_acc(i, 1)));

            canvas.draw_ellipse(cx, cy, rx, ry, Color::Red(), 2);
            canvas.draw_cross(cx, cy, 3, Color::Yellow(), 2);
        }

        canvas.save(path);
    }

    void CameraViewVisualizer::generate_layer_6_summary_overlay(const std::filesystem::path& path) {
        using namespace lfs::core;

        Canvas canvas(width_, height_);
        canvas.set_mask_background(mask_, 0.3f);

        // Draw all classified centers (like layer 3)
        Tensor means2d_cpu = proj_.means2d.cpu().contiguous();
        auto m_acc = means2d_cpu.accessor<float, 2>();

        for (int i = 0; i < num_splats_; ++i) {
            if (!visible_[static_cast<size_t>(i)])
                continue;

            int x = round_to_int(m_acc(i, 0));
            int y = round_to_int(m_acc(i, 1));

            Color color = inside_mask_[static_cast<size_t>(i)] ? Color::Green() : Color::Magenta();
            int size = inside_mask_[static_cast<size_t>(i)] ? 1 : 2;
            canvas.draw_point(x, y, color, size);
        }

        // Highlight problem splats with yellow crosses
        Tensor radii_cpu = proj_.radii.cpu().contiguous();
        auto r_acc = radii_cpu.accessor<int32_t, 2>();

        for (int i = 0; i < num_splats_; ++i) {
            if (!is_problem_[static_cast<size_t>(i)])
                continue;

            int cx = round_to_int(m_acc(i, 0));
            int cy = round_to_int(m_acc(i, 1));

            canvas.draw_cross(cx, cy, 4, Color::Red(), 2);
        }

        // Draw text summary at top (simple text overlay would require a font)
        // For now, just mark corners to show this is the summary layer
        canvas.draw_point(10, 10, Color::Cyan(), 5);
        canvas.draw_point(width_ - 10, 10, Color::Cyan(), 5);
        canvas.draw_point(10, height_ - 10, Color::Cyan(), 5);
        canvas.draw_point(width_ - 10, height_ - 10, Color::Cyan(), 5);

        canvas.save(path);
    }

    // ============================================================================
    // SummaryVisualizer Implementation
    // ============================================================================

    void SummaryVisualizer::add_camera_stats(
        int num_splats,
        const std::vector<bool>& visible,
        const std::vector<bool>& inside) {

        if (num_cameras_ == 0) {
            total_splats_ = num_splats;
            per_splat_visibility_.resize(static_cast<size_t>(num_splats), 0);
            per_splat_inside_count_.resize(static_cast<size_t>(num_splats), 0);
        }

        for (int i = 0; i < num_splats; ++i) {
            if (visible[static_cast<size_t>(i)]) {
                per_splat_visibility_[static_cast<size_t>(i)]++;
                if (inside[static_cast<size_t>(i)]) {
                    per_splat_inside_count_[static_cast<size_t>(i)]++;
                }
            }
        }

        num_cameras_++;
    }

    void SummaryVisualizer::generate_summary_text(const std::filesystem::path& path) const {
        std::ofstream out(path);

        int zero_vis = 0, low_vis = 0, adequate_vis = 0;
        int would_remove = 0;

        for (int i = 0; i < total_splats_; ++i) {
            const int v = per_splat_visibility_[static_cast<size_t>(i)];
            const int ins = per_splat_inside_count_[static_cast<size_t>(i)];

            if (v == 0) {
                zero_vis++;
                would_remove++;
            } else if (v < 3) {
                low_vis++;
                would_remove++;
            } else {
                adequate_vis++;
                const float ratio = static_cast<float>(ins) / static_cast<float>(v);
                if (ratio < 0.8f) {
                    would_remove++;
                }
            }
        }

        out << "PRUNING DIAGNOSTIC SUMMARY\n";
        out << "==========================\n\n";
        out << "Total Cameras: " << num_cameras_ << "\n";
        out << "Total Splats: " << total_splats_ << "\n\n";
        out << "VISIBILITY DISTRIBUTION:\n";
        out << std::format("  - Zero visibility (0 cams):        {:6d} ({:5.1f}%)\n",
                           zero_vis, 100.0 * zero_vis / total_splats_);
        out << std::format("  - Low visibility (1-2 cams):       {:6d} ({:5.1f}%)\n",
                           low_vis, 100.0 * low_vis / total_splats_);
        out << std::format("  - Adequate visibility (3+ cams):   {:6d} ({:5.1f}%)\n\n",
                           adequate_vis, 100.0 * adequate_vis / total_splats_);
        out << std::format("WOULD BE REMOVED BY PRUNING:         {:6d} ({:5.1f}%)\n",
                           would_remove, 100.0 * would_remove / total_splats_);

        out.close();
    }

    // ============================================================================
    // Main Entry Point
    // ============================================================================

    bool generate_pruning_diagnostics(
        IStrategy& strategy,
        const CameraDataset& dataset,
        const CenterVotePruningConfig& config,
        const std::filesystem::path& output_dir,
        int max_cameras) {

        using namespace lfs::core;

        LOG_INFO("[Visualizer] Generating pruning diagnostics...");

        auto& model = strategy.get_model();
        const int N = static_cast<int>(model.size());

        if (N <= 0) {
            LOG_WARN("[Visualizer] No splats to visualize");
            return false;
        }

        // Get dataset sizing
        auto sizing_result = get_dataset_sizing_or_error(dataset);
        if (!sizing_result) {
            LOG_ERROR("[Visualizer] {}", sizing_result.error());
            return false;
        }
        auto sizing = *sizing_result;

        std::filesystem::create_directories(output_dir);

        const auto& cams = dataset.get_cameras();
        const int n_cams = std::min(static_cast<int>(cams.size()), max_cameras);

        SummaryVisualizer summary;
        int generated = 0;

        for (int ci = 0; ci < n_cams; ++ci) {
            const auto& cam_ptr = cams[static_cast<size_t>(ci)];
            if (!cam_ptr || !cam_ptr->has_mask()) {
                continue;
            }

            auto& cam = *cam_ptr;

            Tensor mask = cam.load_and_get_mask(sizing.resize_factor, sizing.max_width,
                                                config.invert_masks, 0.5f);
            if (!mask.is_valid() || mask.numel() == 0) {
                continue;
            }

            if (mask.ndim() == 3 && mask.shape()[0] == 1) {
                mask = mask.squeeze(0);
            }

            const int H = static_cast<int>(cam.image_height());
            const int W = static_cast<int>(cam.image_width());

            if (mask.ndim() != 2 || static_cast<int>(mask.shape()[0]) != H ||
                static_cast<int>(mask.shape()[1]) != W) {
                continue;
            }

            auto proj = project_splats(cam, model, config);
            if (!proj) {
                continue;
            }

            CameraViewVisualizer visualizer(cam, mask, *proj, ci);
            visualizer.generate_all_layers(output_dir);

            LOG_INFO("[Visualizer] Camera {} - Visible: {}, Inside: {}, Outside: {}, Problem: {}",
                     ci, visualizer.num_visible(), visualizer.num_inside(),
                     visualizer.num_outside(), visualizer.num_problem_splats());

            // Add to summary
            summary.add_camera_stats(N, visualizer.get_visible(), visualizer.get_inside_mask());

            generated++;
        }

        // Generate summary
        summary.generate_summary_text(output_dir / "summary.txt");

        // Wait for async saves
        lfs::core::image_io::wait_for_pending_saves();

        LOG_INFO("[Visualizer] Generated {} camera visualizations in {}", generated, output_dir.string());
        return true;
    }

} // namespace lfs::training::mask_pruning::visualizer