/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/splat_data_transform.hpp"
#include "core/logger.hpp"
#include "core/point_cloud.hpp"
#include "core/splat_data.hpp"
#include "geometry/bounding_box.hpp"

#include <algorithm>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <numeric>
#include <random>
#include <vector>

namespace lfs::core {

    SplatData& transform(SplatData& splat_data, const glm::mat4& transform_matrix) {
        LOG_TIMER("transform");

        if (!splat_data._means.is_valid() || splat_data._means.size(0) == 0) {
            LOG_WARN("Cannot transform invalid or empty SplatData");
            return splat_data;
        }

        const int num_points = splat_data._means.size(0);
        auto device = splat_data._means.device();

        // GLM uses column-major storage: mat[col][row], so mat[3] is the translation column.
        // Our tensor MM expects row-major, so we transpose during construction.
        // Final transform: M * p^T where p is [N,4] homogeneous points.
        const std::vector<float> transform_data = {
            transform_matrix[0][0], transform_matrix[1][0], transform_matrix[2][0], transform_matrix[3][0],
            transform_matrix[0][1], transform_matrix[1][1], transform_matrix[2][1], transform_matrix[3][1],
            transform_matrix[0][2], transform_matrix[1][2], transform_matrix[2][2], transform_matrix[3][2],
            transform_matrix[0][3], transform_matrix[1][3], transform_matrix[2][3], transform_matrix[3][3]};

        const auto transform_tensor = Tensor::from_vector(transform_data, TensorShape({4, 4}), device);
        const auto ones = Tensor::ones({static_cast<size_t>(num_points), 1}, device);
        const auto means_homo = splat_data._means.cat(ones, 1);
        const auto transformed_means = transform_tensor.mm(means_homo.t()).t();

        splat_data._means = transformed_means.slice(1, 0, 3).contiguous();

        // 2. Extract rotation from transform matrix
        glm::mat3 rot_mat(transform_matrix);
        glm::vec3 scale;
        for (int i = 0; i < 3; ++i) {
            scale[i] = glm::length(rot_mat[i]);
            if (scale[i] > 0.0f) {
                rot_mat[i] /= scale[i];
            }
        }

        glm::quat rotation_quat = glm::quat_cast(rot_mat);

        // 3. Transform rotations (quaternions) if there's rotation
        if (std::abs(rotation_quat.w - 1.0f) > 1e-6f) {
            std::vector<float> rot_data = {rotation_quat.w, rotation_quat.x, rotation_quat.y, rotation_quat.z};
            auto rot_tensor = Tensor::from_vector(rot_data, TensorShape({4}), device);

            auto q = splat_data._rotation;
            std::vector<int> expand_shape = {num_points, 4};
            auto q_rot = rot_tensor.unsqueeze(0).expand(std::span<const int>(expand_shape));

            auto w1 = q_rot.slice(1, 0, 1).squeeze(1);
            auto x1 = q_rot.slice(1, 1, 2).squeeze(1);
            auto y1 = q_rot.slice(1, 2, 3).squeeze(1);
            auto z1 = q_rot.slice(1, 3, 4).squeeze(1);

            auto w2 = q.slice(1, 0, 1).squeeze(1);
            auto x2 = q.slice(1, 1, 2).squeeze(1);
            auto y2 = q.slice(1, 2, 3).squeeze(1);
            auto z2 = q.slice(1, 3, 4).squeeze(1);

            auto w_new = w1.mul(w2).sub(x1.mul(x2)).sub(y1.mul(y2)).sub(z1.mul(z2));
            auto x_new = w1.mul(x2).add(x1.mul(w2)).add(y1.mul(z2)).sub(z1.mul(y2));
            auto y_new = w1.mul(y2).sub(x1.mul(z2)).add(y1.mul(w2)).add(z1.mul(x2));
            auto z_new = w1.mul(z2).add(x1.mul(y2)).sub(y1.mul(x2)).add(z1.mul(w2));

            std::vector<Tensor> components = {
                w_new.unsqueeze(1),
                x_new.unsqueeze(1),
                y_new.unsqueeze(1),
                z_new.unsqueeze(1)};
            splat_data._rotation = Tensor::cat(components, 1);
        }

        // 4. Transform scaling
        if (std::abs(scale.x - 1.0f) > 1e-6f ||
            std::abs(scale.y - 1.0f) > 1e-6f ||
            std::abs(scale.z - 1.0f) > 1e-6f) {

            float avg_scale = (scale.x + scale.y + scale.z) / 3.0f;
            splat_data._scaling = splat_data._scaling.add(std::log(avg_scale));
        }

        // 5. Update scene scale
        Tensor scene_center = splat_data._means.mean({0}, false);
        Tensor dists = splat_data._means.sub(scene_center).norm(2.0f, {1}, false);
        auto sorted_dists = dists.sort(0, false);
        float new_scene_scale = sorted_dists.first[num_points / 2].item();

        if (std::abs(new_scene_scale - splat_data._scene_scale) > splat_data._scene_scale * 0.1f) {
            splat_data._scene_scale = new_scene_scale;
        }

        LOG_DEBUG("Transformed {} gaussians", num_points);
        return splat_data;
    }

    // Helper: compute inside-cropbox mask for given means and bounding box
    static Tensor compute_cropbox_mask(const Tensor& means,
                                       const lfs::geometry::BoundingBox& bounding_box) {
        const auto bbox_min = bounding_box.getMinBounds();
        const auto bbox_max = bounding_box.getMaxBounds();

        const int num_points = means.size(0);

        // Use full mat4 if available (preserves scale), otherwise fall back to EuclideanTransform
        const glm::mat4 world_to_bbox_matrix = bounding_box.hasFullTransform()
                                                   ? bounding_box.getworld2BBoxMat4()
                                                   : bounding_box.getworld2BBox().toMat4();

        const std::vector<float> transform_data = {
            world_to_bbox_matrix[0][0], world_to_bbox_matrix[1][0], world_to_bbox_matrix[2][0], world_to_bbox_matrix[3][0],
            world_to_bbox_matrix[0][1], world_to_bbox_matrix[1][1], world_to_bbox_matrix[2][1], world_to_bbox_matrix[3][1],
            world_to_bbox_matrix[0][2], world_to_bbox_matrix[1][2], world_to_bbox_matrix[2][2], world_to_bbox_matrix[3][2],
            world_to_bbox_matrix[0][3], world_to_bbox_matrix[1][3], world_to_bbox_matrix[2][3], world_to_bbox_matrix[3][3]};
        auto transform_tensor = Tensor::from_vector(
            transform_data,
            TensorShape({4, 4}),
            means.device());

        auto ones = Tensor::ones({static_cast<size_t>(num_points), 1}, means.device());
        auto means_homo = means.cat(ones, 1);

        const auto transformed_points = transform_tensor.mm(means_homo.t()).t();
        const auto local_points = transformed_points.slice(1, 0, 3);

        const std::vector<float> bbox_min_data = {bbox_min.x, bbox_min.y, bbox_min.z};
        const std::vector<float> bbox_max_data = {bbox_max.x, bbox_max.y, bbox_max.z};

        auto bbox_min_tensor = Tensor::from_vector(bbox_min_data, TensorShape({3}), means.device());
        auto bbox_max_tensor = Tensor::from_vector(bbox_max_data, TensorShape({3}), means.device());

        auto inside_min = local_points.ge(bbox_min_tensor.unsqueeze(0));
        auto inside_max = local_points.le(bbox_max_tensor.unsqueeze(0));

        auto inside_both = inside_min && inside_max;
        std::vector<int> reduce_dims = {1};
        return inside_both.all(std::span<const int>(reduce_dims), false);
    }

    SplatData crop_by_cropbox(const SplatData& splat_data,
                              const lfs::geometry::BoundingBox& bounding_box,
                              const bool inverse) {
        LOG_TIMER("crop_by_cropbox");

        if (!splat_data._means.is_valid() || splat_data._means.size(0) == 0) {
            LOG_WARN("Cannot crop invalid or empty SplatData");
            return SplatData();
        }

        const int num_points = splat_data._means.size(0);

        auto inside_mask = compute_cropbox_mask(splat_data._means, bounding_box);

        // Invert mask if inverse mode
        auto selection_mask = inverse ? inside_mask.logical_not() : inside_mask;
        const int points_selected = selection_mask.sum_scalar();

        if (points_selected == 0) {
            LOG_WARN("No points selected, returning empty SplatData");
            return SplatData();
        }

        auto indices = selection_mask.nonzero();
        if (indices.ndim() == 2) {
            indices = indices.squeeze(1);
        }

        auto cropped_means = splat_data._means.index_select(0, indices).contiguous();
        auto cropped_sh0 = splat_data._sh0.index_select(0, indices).contiguous();
        Tensor cropped_shN = splat_data._shN.is_valid()
                                 ? splat_data._shN.index_select(0, indices).contiguous()
                                 : Tensor{};
        auto cropped_scaling = splat_data._scaling.index_select(0, indices).contiguous();
        auto cropped_rotation = splat_data._rotation.index_select(0, indices).contiguous();
        auto cropped_opacity = splat_data._opacity.index_select(0, indices).contiguous();

        Tensor scene_center = cropped_means.mean({0}, false);
        Tensor dists = cropped_means.sub(scene_center).norm(2.0f, {1}, false);

        float new_scene_scale = splat_data._scene_scale;
        if (points_selected > 1) {
            auto sorted_dists = dists.sort(0, false);
            new_scene_scale = sorted_dists.first[points_selected / 2].item();
        }

        SplatData cropped_splat(
            splat_data._max_sh_degree,
            std::move(cropped_means),
            std::move(cropped_sh0),
            std::move(cropped_shN),
            std::move(cropped_scaling),
            std::move(cropped_rotation),
            std::move(cropped_opacity),
            new_scene_scale);

        cropped_splat.set_active_sh_degree(splat_data._active_sh_degree);

        if (splat_data._densification_info.is_valid() && splat_data._densification_info.size(0) == num_points) {
            cropped_splat._densification_info =
                splat_data._densification_info.index_select(0, indices).contiguous();
        }

        LOG_DEBUG("Cropped SplatData: {} -> {} points (inverse={})", num_points, points_selected, inverse);
        return cropped_splat;
    }

    Tensor soft_crop_by_cropbox(SplatData& splat_data,
                                const lfs::geometry::BoundingBox& bounding_box,
                                const bool inverse) {
        LOG_TIMER("soft_crop_by_cropbox");

        const auto& means = splat_data.means();
        if (!means.is_valid() || means.size(0) == 0) {
            return Tensor();
        }

        const auto inside_mask = compute_cropbox_mask(means, bounding_box);
        const auto delete_mask = inverse ? inside_mask : inside_mask.logical_not();
        const int points_to_delete = delete_mask.sum_scalar();

        if (points_to_delete == 0) {
            return Tensor();
        }

        return splat_data.soft_delete(delete_mask);
    }

    Tensor soft_crop_by_ellipsoid(SplatData& splat_data,
                                  const glm::mat4& transform,
                                  const glm::vec3& radii,
                                  const bool inverse) {
        LOG_TIMER("soft_crop_by_ellipsoid");

        const auto& means = splat_data.means();
        if (!means.is_valid() || means.size(0) == 0) {
            return Tensor();
        }

        const size_t num_points = static_cast<size_t>(means.size(0));
        const auto device = means.device();

        // Build transformation tensor (GLM column-major to row-major)
        const auto transform_tensor = Tensor::from_vector(
            {transform[0][0], transform[1][0], transform[2][0], transform[3][0],
             transform[0][1], transform[1][1], transform[2][1], transform[3][1],
             transform[0][2], transform[1][2], transform[2][2], transform[3][2],
             transform[0][3], transform[1][3], transform[2][3], transform[3][3]},
            {4, 4}, device);

        // Transform to ellipsoid local space
        const auto ones = Tensor::ones({num_points, 1}, device);
        const auto local_pos = transform_tensor.mm(means.cat(ones, 1).t()).t();

        // Compute normalized distances: (x/rx)^2 + (y/ry)^2 + (z/rz)^2
        const auto x = local_pos.slice(1, 0, 1).squeeze(1) / radii.x;
        const auto y = local_pos.slice(1, 1, 2).squeeze(1) / radii.y;
        const auto z = local_pos.slice(1, 2, 3).squeeze(1) / radii.z;

        const auto dist_sq = x * x + y * y + z * z;
        const auto inside_mask = dist_sq <= 1.0f;
        const auto delete_mask = inverse ? inside_mask : inside_mask.logical_not();
        const int points_to_delete = delete_mask.sum_scalar();

        if (points_to_delete == 0) {
            return Tensor();
        }

        return splat_data.soft_delete(delete_mask);
    }

    void random_choose(SplatData& splat_data, int num_required_splat, int seed) {
        LOG_TIMER("random_choose");

        if (!splat_data._means.is_valid() || splat_data._means.size(0) == 0) {
            LOG_WARN("Cannot choose from invalid or empty SplatData");
            return;
        }

        const int num_points = splat_data._means.size(0);

        if (num_required_splat <= 0) {
            LOG_WARN("num_splat must be positive, got {}", num_required_splat);
            return;
        }

        if (num_required_splat >= num_points) {
            LOG_DEBUG("num_splat ({}) >= total points ({}), keeping all data",
                      num_required_splat, num_points);
            return;
        }

        LOG_DEBUG("Randomly selecting {} points from {} total points (seed: {})",
                  num_required_splat, num_points, seed);

        std::vector<int> all_indices(num_points);
        std::iota(all_indices.begin(), all_indices.end(), 0);

        std::mt19937 rng(seed);
        std::shuffle(all_indices.begin(), all_indices.end(), rng);

        std::vector<int> selected_indices(all_indices.begin(),
                                          all_indices.begin() + num_required_splat);

        auto indices_tensor = Tensor::from_vector(
            selected_indices,
            TensorShape({static_cast<size_t>(num_required_splat)}),
            splat_data._means.device());

        splat_data._means = splat_data._means.index_select(0, indices_tensor).contiguous();
        splat_data._sh0 = splat_data._sh0.index_select(0, indices_tensor).contiguous();
        if (splat_data._shN.is_valid()) {
            splat_data._shN = splat_data._shN.index_select(0, indices_tensor).contiguous();
        }
        splat_data._scaling = splat_data._scaling.index_select(0, indices_tensor).contiguous();
        splat_data._rotation = splat_data._rotation.index_select(0, indices_tensor).contiguous();
        splat_data._opacity = splat_data._opacity.index_select(0, indices_tensor).contiguous();

        if (splat_data._densification_info.is_valid() && splat_data._densification_info.size(0) == num_points) {
            splat_data._densification_info = splat_data._densification_info.index_select(0, indices_tensor).contiguous();
        }

        Tensor scene_center = splat_data._means.mean({0}, false);
        Tensor dists = splat_data._means.sub(scene_center).norm(2.0f, {1}, false);

        float old_scene_scale = splat_data._scene_scale;
        if (num_required_splat > 1) {
            auto sorted_dists = dists.sort(0, false);
            splat_data._scene_scale = sorted_dists.first[num_required_splat / 2].item();
        }

        LOG_DEBUG("Successfully selected {} random splats in-place (scale: {:.4f} -> {:.4f})",
                  num_required_splat, old_scene_scale, splat_data._scene_scale);
    }

    bool compute_bounds(const SplatData& splat_data,
                        glm::vec3& min_bounds,
                        glm::vec3& max_bounds,
                        const float padding,
                        const bool use_percentile) {
        const auto& means = splat_data.means();
        if (!means.is_valid() || means.size(0) == 0) {
            return false;
        }

        // Filter deleted gaussians (index_select preserves [N,3] shape)
        Tensor visible_means = means;
        if (splat_data.has_deleted_mask()) {
            const auto visible_indices = splat_data.deleted().logical_not().nonzero().squeeze(1);
            if (visible_indices.size(0) == 0)
                return false;
            visible_means = means.index_select(0, visible_indices);
        }

        if (visible_means.size(0) == 0) {
            return false;
        }

        const int64_t n = visible_means.size(0);

        if (use_percentile && n > 100) {
            // Exclude 2% outliers (1% each end)
            const int64_t lo = n / 100;
            const int64_t hi = n - 1 - lo;
            for (int i = 0; i < 3; ++i) {
                const auto sorted = visible_means.slice(1, i, i + 1).squeeze(1).sort(0, false).first;
                min_bounds[i] = sorted[lo].item() - padding;
                max_bounds[i] = sorted[hi].item() + padding;
            }
        } else {
            for (int i = 0; i < 3; ++i) {
                const auto col = visible_means.slice(1, i, i + 1).squeeze(1);
                min_bounds[i] = col.min().item() - padding;
                max_bounds[i] = col.max().item() + padding;
            }
        }

        return true;
    }

    bool compute_bounds(const PointCloud& point_cloud,
                        glm::vec3& min_bounds,
                        glm::vec3& max_bounds,
                        const float padding,
                        const bool use_percentile) {
        const auto& means = point_cloud.means;
        if (!means.is_valid() || means.size(0) == 0) {
            return false;
        }

        const int64_t n = means.size(0);

        if (use_percentile && n > 100) {
            // Exclude 2% outliers (1% each end)
            const int64_t lo = n / 100;
            const int64_t hi = n - 1 - lo;
            for (int i = 0; i < 3; ++i) {
                const auto sorted = means.slice(1, i, i + 1).squeeze(1).sort(0, false).first;
                min_bounds[i] = sorted[lo].item() - padding;
                max_bounds[i] = sorted[hi].item() + padding;
            }
        } else {
            for (int i = 0; i < 3; ++i) {
                const auto col = means.slice(1, i, i + 1).squeeze(1);
                min_bounds[i] = col.min().item() - padding;
                max_bounds[i] = col.max().item() + padding;
            }
        }

        return true;
    }

    SplatData extract_by_mask(const SplatData& splat_data, const Tensor& mask) {
        if (!splat_data._means.is_valid() || splat_data._means.size(0) == 0) {
            return SplatData();
        }
        if (!mask.is_valid() || mask.size(0) != splat_data._means.size(0)) {
            return SplatData();
        }

        const auto selection_mask = mask.to(DataType::Bool);
        const int count = selection_mask.sum_scalar();
        if (count == 0) {
            return SplatData();
        }

        auto indices = selection_mask.nonzero();
        if (indices.ndim() == 2) {
            indices = indices.squeeze(1);
        }

        Tensor shN_selected = splat_data._shN.is_valid()
                                  ? splat_data._shN.index_select(0, indices).contiguous()
                                  : Tensor{};

        SplatData result(
            splat_data._max_sh_degree,
            splat_data._means.index_select(0, indices).contiguous(),
            splat_data._sh0.index_select(0, indices).contiguous(),
            std::move(shN_selected),
            splat_data._scaling.index_select(0, indices).contiguous(),
            splat_data._rotation.index_select(0, indices).contiguous(),
            splat_data._opacity.index_select(0, indices).contiguous(),
            splat_data._scene_scale);
        result.set_active_sh_degree(splat_data._active_sh_degree);
        return result;
    }

} // namespace lfs::core
