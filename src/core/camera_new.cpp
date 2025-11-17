/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera_new.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include <cuda_runtime.h>
#include <sstream>

namespace gs {

    // ========== HELPER FUNCTION ==========
    namespace {
        // ========== UTILITY FUNCTIONS ==========

        /**
         * @brief Convert focal length to field of view
         * @param focal Focal length in pixels
         * @param pixels Image dimension in pixels
         * @return Field of view in radians
         */
        inline float focal2fov(float focal, int pixels) {
            return 2.0f * std::atan(pixels / (2.0f * focal));
        }

        /**
         * @brief Convert field of view to focal length
         * @param fov Field of view in radians
         * @param pixels Image dimension in pixels
         * @return Focal length in pixels
         */
        inline float fov2focal(float fov, int pixels) {
            float tan_fov = std::tan(fov * 0.5f);
            return pixels / (2.0f * tan_fov);
        }

    } // namespace

    Tensor CameraNew::world_to_view(const Tensor& R, const Tensor& t) {
        // Create 4x4 identity matrix
        auto w2c = Tensor::eye(4, R.device());

        // Set rotation part [0:3, 0:3] = R
        auto w2c_cpu = w2c.cpu();
        auto R_cpu = R.cpu();
        auto t_cpu = t.cpu();

        auto w2c_acc = w2c_cpu.accessor<float, 2>();
        auto R_acc = R_cpu.accessor<float, 2>();
        auto t_acc = t_cpu.accessor<float, 1>();

        // Copy rotation
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                w2c_acc(i, j) = R_acc(i, j);
            }
        }

        // Copy translation [0:3, 3] = t
        for (size_t i = 0; i < 3; ++i) {
            w2c_acc(i, 3) = t_acc(i);
        }

        // Return as [1, 4, 4] on CUDA
        return w2c_cpu.cuda().unsqueeze(0).contiguous();
    }

    // ========== CONSTRUCTORS ==========

    CameraNew::CameraNew(const Tensor& R,
                         const Tensor& T,
                         float focal_x, float focal_y,
                         float center_x, float center_y,
                         const Tensor& radial_distortion,
                         const Tensor& tangential_distortion,
                         gsplat::CameraModelType camera_model_type,
                         const std::string& image_name,
                         const std::filesystem::path& image_path,
                         int camera_width, int camera_height,
                         int uid)
        : uid_(uid),
          focal_x_(focal_x),
          focal_y_(focal_y),
          center_x_(center_x),
          center_y_(center_y),
          R_(R),
          T_(T),
          radial_distortion_(radial_distortion),
          tangential_distortion_(tangential_distortion),
          camera_model_type_(camera_model_type),
          image_name_(image_name),
          image_path_(image_path),
          camera_width_(camera_width),
          camera_height_(camera_height),
          image_width_(camera_width),
          image_height_(camera_height) {

        // Validate inputs
        if (R.ndim() != 2 || R.size(0) != 3 || R.size(1) != 3) {
            LOG_ERROR("CameraNew: R must be [3, 3], got {}", R.shape().str());
        }
        if (T.ndim() != 1 || T.size(0) != 3) {
            LOG_ERROR("CameraNew: T must be [3], got {}", T.shape().str());
        }

        // Compute world-to-view transform
        world_view_transform_ = world_to_view(R, T);

        // Compute camera position (inverse of world-to-view)
        // c2w = inverse(w2c)
        // cam_position = c2w[:3, 3]

        auto w2v_cpu = world_view_transform_.squeeze(0).cpu();
        auto w2v_acc = w2v_cpu.accessor<float, 2>();

        // Create 3x3 rotation part
        std::vector<float> rot_data(9);
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                rot_data[i * 3 + j] = w2v_acc(i, j);
            }
        }
        auto R_part = Tensor::from_vector(rot_data, TensorShape({3, 3}), Device::CPU);

        // Create translation part
        std::vector<float> t_data = {w2v_acc(0, 3), w2v_acc(1, 3), w2v_acc(2, 3)};
        auto t_part = Tensor::from_vector(t_data, TensorShape({3}), Device::CPU);

        // Compute inverse: c2w = [R^T | -R^T * t]
        auto R_T = R_part.transpose(0, 1);
        auto cam_pos = R_T.mm(t_part.unsqueeze(1)).squeeze(1).neg();

        cam_position_ = cam_pos.cuda().contiguous();

        // Compute field of view
        FoVx_ = focal2fov(focal_x_, camera_width_);
        FoVy_ = focal2fov(focal_y_, camera_height_);

        // Create CUDA stream for async operations
        // cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);

        LOG_DEBUG("Created CameraNew: uid={}, {}x{}, focal=({:.2f}, {:.2f})",
                  uid_, camera_width_, camera_height_, focal_x_, focal_y_);
    }

    CameraNew::CameraNew(const CameraNew& other, const Tensor& transform)
        : uid_(other.uid_),
          focal_x_(other.focal_x_),
          focal_y_(other.focal_y_),
          center_x_(other.center_x_),
          center_y_(other.center_y_),
          R_(other.R_),
          T_(other.T_),
          radial_distortion_(other.radial_distortion_),
          tangential_distortion_(other.tangential_distortion_),
          camera_model_type_(other.camera_model_type_),
          image_name_(other.image_name_),
          image_path_(other.image_path_),
          camera_width_(other.camera_width_),
          camera_height_(other.camera_height_),
          image_width_(other.image_width_),
          image_height_(other.image_height_),
          cam_position_(other.cam_position_),
          FoVx_(other.FoVx_),
          FoVy_(other.FoVy_),
          world_view_transform_(transform) {

        // Create new CUDA stream
        // cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    }

    CameraNew::~CameraNew() {
        // if (stream_ != nullptr) {
        //     cudaStreamDestroy(stream_);
        //     stream_ = nullptr;
        // }
    }

    // ========== INTRINSICS ==========

    Tensor CameraNew::K() const {
        auto K = Tensor::zeros({1, 3, 3}, world_view_transform_.device());

        auto [fx, fy, cx, cy] = get_intrinsics();

        auto K_cpu = K.cpu();
        auto K_acc = K_cpu.accessor<float, 3>();

        K_acc(0, 0, 0) = fx;
        K_acc(0, 1, 1) = fy;
        K_acc(0, 0, 2) = cx;
        K_acc(0, 1, 2) = cy;
        K_acc(0, 2, 2) = 1.0f;

        return K_cpu.to(world_view_transform_.device());
    }

    std::tuple<float, float, float, float> CameraNew::get_intrinsics() const {
        float x_scale_factor = static_cast<float>(image_width_) / static_cast<float>(camera_width_);
        float y_scale_factor = static_cast<float>(image_height_) / static_cast<float>(camera_height_);

        float fx = focal_x_ * x_scale_factor;
        float fy = focal_y_ * y_scale_factor;
        float cx = center_x_ * x_scale_factor;
        float cy = center_y_ * y_scale_factor;

        return std::make_tuple(fx, fy, cx, cy);
    }

    // ========== IMAGE LOADING ==========

    Tensor CameraNew::load_and_get_image(int resize_factor, int max_width) {
        LOG_TIMER("CameraNew::load_and_get_image");

        // Load image data
        auto [data, w, h, c] = load_image(image_path_, resize_factor);

        image_width_ = w;
        image_height_ = h;

        if (data == nullptr) {
            LOG_ERROR("Failed to load image: {}", image_path_.string());
            return Tensor();
        }

        // Create tensor from image data on CPU first
        std::vector<unsigned char> img_vec(data, data + w * h * c);
        free_image(data);

        // Convert to float vector and normalize to [0, 1]
        std::vector<float> img_float(w * h * c);
        for (size_t i = 0; i < img_vec.size(); ++i) {
            img_float[i] = static_cast<float>(img_vec[i]) / 255.0f;
        }

        // Create tensor [H, W, C]
        auto image = Tensor::from_vector(img_float, TensorShape({static_cast<size_t>(h), static_cast<size_t>(w), static_cast<size_t>(c)}),
                                         Device::CPU);

        // Transfer to CUDA and permute to [C, H, W]
        image = image.cuda().permute({2, 0, 1}).contiguous();

        LOG_DEBUG("Loaded image: {}x{}x{} from {}", c, h, w, image_path_.filename().string());

        return image;
    }

    void CameraNew::load_image_size(int resize_factor, int max_width) {
        auto [w, h, c] = get_image_info(image_path_);

        if (resize_factor > 0) {
            if (w % resize_factor || h % resize_factor) {
                LOG_WARN("Width {} or height {} not divisible by resize_factor {}",
                         w, h, resize_factor);
            }
            image_width_ = w / resize_factor;
            image_height_ = h / resize_factor;
        } else {
            image_width_ = w;
            image_height_ = h;
        }

        if (max_width > 0 && (image_width_ > max_width || image_height_ > max_width)) {
            if (image_width_ > image_height_) {
                image_height_ = (image_height_ * max_width) / image_width_;
                image_width_ = max_width;
            } else {
                image_width_ = (image_width_ * max_width) / image_height_;
                image_height_ = max_width;
            }
        }

        LOG_DEBUG("Image size: {}x{}", image_width_, image_height_);
    }

    size_t CameraNew::get_num_bytes_from_file(int resize_factor, int max_width) const {
        auto [w, h, c] = get_image_info(image_path_);

        if (resize_factor > 0) {
            w = w / resize_factor;
            h = h / resize_factor;
        }

        if (max_width > 0 && (w > max_width || h > max_width)) {
            if (w > h) {
                h = (h * max_width) / w;
                w = max_width;
            } else {
                w = (w * max_width) / h;
                h = max_width;
            }
        }

        size_t num_bytes = w * h * c * sizeof(float);
        return num_bytes;
    }

    size_t CameraNew::get_num_bytes_from_file() const {
        auto [w, h, c] = get_image_info(image_path_);
        size_t num_bytes = w * h * c * sizeof(float);
        return num_bytes;
    }

    // ========== VALIDATION ==========

    bool CameraNew::is_valid() const {
        if (!R_.is_valid() || !T_.is_valid()) {
            LOG_ERROR("CameraNew: R or T is invalid");
            return false;
        }

        if (!world_view_transform_.is_valid() || !cam_position_.is_valid()) {
            LOG_ERROR("CameraNew: world_view_transform or cam_position is invalid");
            return false;
        }

        if (R_.ndim() != 2 || R_.size(0) != 3 || R_.size(1) != 3) {
            LOG_ERROR("CameraNew: R must be [3, 3], got {}", R_.shape().str());
            return false;
        }

        if (T_.ndim() != 1 || T_.size(0) != 3) {
            LOG_ERROR("CameraNew: T must be [3], got {}", T_.shape().str());
            return false;
        }

        if (world_view_transform_.ndim() != 3 ||
            world_view_transform_.size(0) != 1 ||
            world_view_transform_.size(1) != 4 ||
            world_view_transform_.size(2) != 4) {
            LOG_ERROR("CameraNew: world_view_transform must be [1, 4, 4], got {}",
                      world_view_transform_.shape().str());
            return false;
        }

        if (cam_position_.ndim() != 1 || cam_position_.size(0) != 3) {
            LOG_ERROR("CameraNew: cam_position must be [3], got {}",
                      cam_position_.shape().str());
            return false;
        }

        return true;
    }

    std::string CameraNew::str() const {
        std::ostringstream oss;
        oss << "CameraNew(";
        oss << "uid=" << uid_;
        oss << ", size=" << image_width_ << "x" << image_height_;
        oss << ", focal=(" << focal_x_ << ", " << focal_y_ << ")";
        oss << ", center=(" << center_x_ << ", " << center_y_ << ")";
        oss << ", FoV=(" << FoVx_ << ", " << FoVy_ << ")";
        oss << ", image=" << image_name_;
        oss << ")";
        return oss.str();
    }

} // namespace gs