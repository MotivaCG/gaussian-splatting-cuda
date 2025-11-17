/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core_new/camera.hpp"
#include "core_new/image_io.hpp"
#include "core_new/logger.hpp"
#include "loader_new/cache_image_loader.hpp"
#include <cuda_runtime.h>

namespace lfs::core {
    static Tensor world_to_view(const Tensor& R, const Tensor& t) {
        // Create 4x4 identity matrix
        auto w2c = Tensor::eye(4, R.device());

        // Set rotation and translation parts
        auto w2c_cpu = w2c.cpu();
        auto R_cpu = R.cpu();
        auto t_cpu = t.cpu();

        auto w2c_acc = w2c_cpu.accessor<float, 2>();
        auto R_acc = R_cpu.accessor<float, 2>();
        auto t_acc = t_cpu.accessor<float, 1>();

        // Copy rotation [0:3, 0:3] = R
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
        return w2c_cpu.to(Device::CUDA).unsqueeze(0).contiguous();
    }

    Camera::Camera(const Tensor& R,
                   const Tensor& T,
                   float focal_x, float focal_y,
                   float center_x, float center_y,
                   const Tensor radial_distortion,
                   const Tensor tangential_distortion,
                   gsplat::CameraModelType camera_model_type,
                   const std::string& image_name,
                   const std::filesystem::path& image_path,
                   int camera_width, int camera_height,
                   int uid)
        : _uid(uid),
          _focal_x(focal_x),
          _focal_y(focal_y),
          _center_x(center_x),
          _center_y(center_y),
          _R(R),
          _T(T),
          _radial_distortion(radial_distortion),
          _tangential_distortion(tangential_distortion),
          _camera_model_type(camera_model_type),
          _image_name(image_name),
          _image_path(image_path),
          _camera_width(camera_width),
          _camera_height(camera_height),
          _image_width(camera_width),
          _image_height(camera_height) {

        // Validate inputs
        if (!R.is_valid() || R.numel() == 0) {
            LOG_ERROR("Camera constructor: R tensor is invalid or empty");
            throw std::runtime_error("Camera constructor: R tensor is invalid or empty");
        }
        if (!T.is_valid() || T.numel() == 0) {
            LOG_ERROR("Camera constructor: T tensor is invalid or empty");
            throw std::runtime_error("Camera constructor: T tensor is invalid or empty");
        }

        // Compute world-to-view transform
        _world_view_transform = world_to_view(R, T);

        // Compute camera position: inverse of w2v gives c2w, position is c2w[:3, 3]
        // For transformation matrix [R|t], inverse is [R^T | -R^T*t]
        auto w2v_cpu = _world_view_transform.squeeze(0).cpu();
        auto w2v_acc = w2v_cpu.accessor<float, 2>();

        // Extract 3x3 rotation part
        std::vector<float> rot_data(9);
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                rot_data[i * 3 + j] = w2v_acc(i, j);
            }
        }
        auto R_part = Tensor::from_vector(rot_data, TensorShape({3, 3}), Device::CPU);

        // Extract translation part
        std::vector<float> t_data = {w2v_acc(0, 3), w2v_acc(1, 3), w2v_acc(2, 3)};
        auto t_part = Tensor::from_vector(t_data, TensorShape({3}), Device::CPU);

        // Compute camera position: -R^T * t
        auto R_T = R_part.transpose(0, 1);
        auto cam_pos = R_T.matmul(t_part.unsqueeze(1)).squeeze(1).neg();

        _cam_position = cam_pos.to(Device::CUDA).contiguous();

        _FoVx = focal2fov(_focal_x, _camera_width);
        _FoVy = focal2fov(_focal_y, _camera_height);
    }
    Camera::Camera(const Camera& other, const Tensor& transform)
        : _uid(other._uid),
          _focal_x(other._focal_x),
          _focal_y(other._focal_y),
          _center_x(other._center_x),
          _center_y(other._center_y),
          _R(other._R),
          _T(other._T),
          _radial_distortion(other._radial_distortion),
          _tangential_distortion(other._tangential_distortion),
          _camera_model_type(other._camera_model_type),
          _image_name(other._image_name),
          _image_path(other._image_path),
          _camera_width(other._camera_width),
          _camera_height(other._camera_height),
          _image_width(other._image_width),
          _image_height(other._image_height),
          _cam_position(other._cam_position),
          _FoVx(other._FoVx),
          _FoVy(other._FoVy) {
        _world_view_transform = transform;
    }
    Tensor Camera::K() const {
        // Create [1, 3, 3] zero matrix on same device as world_view_transform
        auto K = Tensor::zeros({1, 3, 3}, _world_view_transform.device());
        auto [fx, fy, cx, cy] = get_intrinsics();

        // Fill in the intrinsic matrix on CPU then move to device
        auto K_cpu = K.cpu();
        auto K_acc = K_cpu.accessor<float, 3>();
        K_acc(0, 0, 0) = fx;
        K_acc(0, 1, 1) = fy;
        K_acc(0, 0, 2) = cx;
        K_acc(0, 1, 2) = cy;
        K_acc(0, 2, 2) = 1.0f;

        return K_cpu.to(_world_view_transform.device()).contiguous();
    }

    std::tuple<float, float, float, float> Camera::get_intrinsics() const {
        float x_scale_factor = float(_image_width) / float(_camera_width);
        float y_scale_factor = float(_image_height) / float(_camera_height);
        float fx = _focal_x * x_scale_factor;
        float fy = _focal_y * y_scale_factor;
        float cx = _center_x * x_scale_factor;
        float cy = _center_y * y_scale_factor;
        return std::make_tuple(fx, fy, cx, cy);
    }

    Tensor Camera::load_and_get_image(int resize_factor, int max_width) {
        unsigned char* data;
        int w, h, c;
        auto& loader = lfs::loader::CacheLoader::getInstance();
        // Load image synchronously
        lfs::loader::LoadParams params{.resize_factor = resize_factor, .max_width = max_width};

        auto result = loader.load_cached_image(_image_path, params);

        data = std::get<0>(result);
        w = std::get<1>(result);
        h = std::get<2>(result);
        c = std::get<3>(result);

        _image_width = w;
        _image_height = h;

        // Create tensor from raw data [H, W, C] uint8
        auto image = Tensor::from_blob(
            data,
            TensorShape({static_cast<size_t>(h), static_cast<size_t>(w), static_cast<size_t>(c)}),
            Device::CPU,
            DataType::UInt8);

        // Convert to float and normalize, then permute to [C, H, W]
        image = image.to(DataType::Float32) / 255.0f;
        image = image.permute({2, 0, 1});

        // Transfer to CUDA
        image = image.to(Device::CUDA).contiguous();

        // Free the original data
        free_image(data);

        return image;
    }

    void Camera::load_image_size(int resize_factor, int max_width) {
        auto result = get_image_info(_image_path);

        int w = std::get<0>(result);
        int h = std::get<1>(result);

        if (resize_factor > 0) {
            if (w % resize_factor || h % resize_factor) {
                LOG_WARN("width or height are not divisible by resize_factor w {} h {} resize_factor {}", w, h, resize_factor);
            }
            _image_width = w / resize_factor;
            _image_height = h / resize_factor;
        } else {
            _image_width = w;
            _image_height = h;
        }

        if (max_width > 0 && (_image_width > max_width || _image_height > max_width)) {
            if (_image_width > _image_height) {
                _image_width = max_width;
                _image_height = (_image_height * max_width) / _image_width;
            } else {
                _image_height = max_width;
                _image_width = (_image_width * max_width) / _image_height;
            }
        }
    }

    size_t Camera::get_num_bytes_from_file(int resize_factor, int max_width) const {
        auto result = get_image_info(_image_path);

        int w = std::get<0>(result);
        int h = std::get<1>(result);
        int c = std::get<2>(result);

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

    size_t Camera::get_num_bytes_from_file() const {
        auto [w, h, c] = get_image_info(_image_path);
        size_t num_bytes = w * h * c * sizeof(float);
        return num_bytes;
    }
} // namespace lfs::core