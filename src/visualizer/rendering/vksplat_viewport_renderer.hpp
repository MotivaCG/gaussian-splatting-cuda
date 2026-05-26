/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/splat_data.hpp"
#include "rendering/cuda_vulkan_interop.hpp"
#include "rendering/rasterizer/vulkan/src/gs_renderer.h"
#include "rendering/rendering.hpp"
#include "window/vulkan_context.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <expected>
#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <vector>

namespace lfs::vis {

    class VksplatViewportRenderer {
    public:
        struct RenderResult {
            VkImage image = VK_NULL_HANDLE;
            VkImageView image_view = VK_NULL_HANDLE;
            VkImageLayout image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
            std::uint64_t generation = 0;
            VkImage depth_image = VK_NULL_HANDLE;
            VkImageView depth_image_view = VK_NULL_HANDLE;
            VkImageLayout depth_image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
            std::uint64_t depth_generation = 0;
            glm::ivec2 size{0, 0};
            bool flip_y = false;
        };

        struct ModelInputSnapshot {
            const lfs::core::SplatData* model = nullptr;
            std::size_t count = 0;
            int max_sh_degree = -1;
            const void* means = nullptr;
            const void* scaling = nullptr;
            const void* rotation = nullptr;
            const void* opacity = nullptr;
            const void* sh0 = nullptr;
            const void* shn = nullptr;
            // Tracking the deleted mask pointer + byte count is enough to invalidate
            // the resident-input cache when the user soft-deletes (or undoes a
            // delete). The renderer then re-runs the copy path so the opacity
            // upload applies the latest mask.
            const void* deleted = nullptr;
            std::size_t means_bytes = 0;
            std::size_t scaling_bytes = 0;
            std::size_t rotation_bytes = 0;
            std::size_t opacity_bytes = 0;
            std::size_t sh0_bytes = 0;
            std::size_t shn_bytes = 0;
            std::size_t deleted_bytes = 0;

            [[nodiscard]] bool valid() const { return model != nullptr && count > 0; }
            [[nodiscard]] friend bool operator==(const ModelInputSnapshot& a,
                                                 const ModelInputSnapshot& b) = default;
        };

        enum class SelectionMaskShape : std::uint32_t {
            Brush = 0,
            Rectangle = 1,
            Polygon = 2,
        };

        enum class OutputSlot : std::size_t {
            Main = 0,
            SplitLeft = 1,
            SplitRight = 2,
            Preview = 3,
        };

        struct SelectionMaskRequest {
            lfs::rendering::FrameView frame_view;
            lfs::rendering::GaussianSceneState scene;
            SelectionMaskShape shape = SelectionMaskShape::Brush;
            std::vector<glm::vec4> primitives;
            std::vector<glm::vec2> polygon_vertices;
            bool gut = false;
            bool equirectangular = false;
            bool synchronize_input_upload = false;
        };

        VksplatViewportRenderer();
        ~VksplatViewportRenderer();

        VksplatViewportRenderer(const VksplatViewportRenderer&) = delete;
        VksplatViewportRenderer& operator=(const VksplatViewportRenderer&) = delete;

        [[nodiscard]] std::expected<RenderResult, std::string> render(
            VulkanContext& context,
            const lfs::core::SplatData& splat_data,
            const lfs::rendering::ViewportRenderRequest& request,
            bool force_input_upload,
            OutputSlot output_slot = OutputSlot::Main,
            bool synchronize_input_upload = false);
        [[nodiscard]] std::expected<RenderResult, std::string> rerenderSelectionOverlay(
            VulkanContext& context,
            const lfs::core::SplatData& splat_data,
            const lfs::rendering::ViewportRenderRequest& request,
            OutputSlot output_slot = OutputSlot::Main);
        [[nodiscard]] std::expected<std::shared_ptr<lfs::core::Tensor>, std::string> readOutputImage(
            VulkanContext& context,
            OutputSlot output_slot = OutputSlot::Main) const;
        [[nodiscard]] std::expected<float, std::string> sampleDepthAtPixel(
            VulkanContext& context,
            int x,
            int y,
            OutputSlot output_slot = OutputSlot::Main) const;
        [[nodiscard]] std::expected<lfs::core::Tensor, std::string> buildSelectionMask(
            VulkanContext& context,
            const lfs::core::SplatData& splat_data,
            const SelectionMaskRequest& request,
            bool force_input_upload);

        void reset();

    private:
        struct ComposePipeline;
        struct InputBindingResult {
            bool uses_temporary_upload_slot = false;
        };

        [[nodiscard]] std::expected<void, std::string> ensureInitialized(VulkanContext& context);
        [[nodiscard]] std::expected<InputBindingResult, std::string> prepareInputs(
            VulkanContext& context,
            const lfs::core::SplatData& splat_data,
            std::size_t ring_slot,
            bool force_upload,
            bool synchronize_upload = false);
        struct OverlayBindingViews {
            _VulkanBuffer selection_mask{};
            _VulkanBuffer preview_mask{};
            _VulkanBuffer selection_colors{};
            _VulkanBuffer transform_indices{};
            _VulkanBuffer node_mask{};
            _VulkanBuffer overlay_params{};
            _VulkanBuffer model_transforms{};
        };
        [[nodiscard]] std::expected<OverlayBindingViews, std::string> uploadSelectionOverlay(
            VulkanContext& context,
            const lfs::rendering::ViewportRenderRequest& request,
            std::size_t num_splats,
            std::size_t ring_slot);
        [[nodiscard]] bool inputsResident(const lfs::core::SplatData& splat_data,
                                          std::size_t ring_slot) const;
        [[nodiscard]] std::expected<void, std::string> ensureOutputImages(
            VulkanContext& context,
            glm::ivec2 size,
            OutputSlot output_slot);
        [[nodiscard]] std::expected<void, std::string> ensureComposePipeline(VulkanContext& context);
        [[nodiscard]] std::expected<void, std::string> composePixelState(
            VulkanContext& context,
            VkCommandBuffer cmd,
            const VulkanGSRendererUniforms& uniforms,
            const glm::vec3& background,
            OutputSlot output_slot,
            bool transparent_background,
            bool depth_view,
            float depth_min,
            float depth_max);

        // Fallback coalesced CUDA-imported VkBuffer per ring slot, holding raw
        // SplatData input regions back-to-back. Training tensors created as
        // Vulkan-external buffers bypass this allocation and are bound directly.
        static constexpr std::size_t kInputRegionCount = 6;
        static constexpr std::size_t kOverlayRegionCount = 7;
        static constexpr std::size_t kSelectionQueryRegionCount = 7;
        static constexpr std::size_t kRegionAlignment = 256; // VK minStorageBufferOffsetAlignment upper bound on common HW
        struct CudaInputSlot {
            VulkanContext::ExternalBuffer buffer{};
            lfs::rendering::CudaVulkanBufferInterop interop{};
            std::array<std::size_t, kInputRegionCount> region_offset{};
            std::array<std::size_t, kInputRegionCount> region_bytes{};
        };
        struct CudaOverlaySlot {
            VulkanContext::ExternalBuffer buffer{};
            lfs::rendering::CudaVulkanBufferInterop interop{};
            std::array<std::size_t, kOverlayRegionCount> region_offset{};
            std::array<std::size_t, kOverlayRegionCount> region_bytes{};
            lfs::core::Tensor selection_source;
            lfs::core::Tensor preview_source;
            lfs::core::Tensor color_table_source;
            // Fingerprint of the palette currently staged in color_table_source.
            // Hits on lasso-drag frames where the theme/palette is unchanged,
            // letting us skip the ~5 ms CPU-tensor build + H2D transfer.
            // Validity gate is color_table_source.is_valid().
            std::array<glm::vec4, lfs::rendering::kSelectionColorTableCount> cached_color_palette{};
            lfs::core::Tensor transform_indices_source;
            lfs::core::Tensor node_mask_source;
            // Fingerprint of emphasized_node_mask currently staged in
            // node_mask_source. Skips the CPU-tensor build + H2D when the user
            // hasn't changed the selected node set (common during lasso drag).
            std::vector<bool> cached_emphasized_node_mask;
            lfs::core::Tensor overlay_params_source;
            // Mirror of the CPU bytes currently staged in overlay_params_source.
            // The full overlay-params table is built on CPU each frame (cheap),
            // then memcmp'd against this to decide whether the ~6 ms H2D is
            // needed. Output-byte fingerprint is robust: any input change that
            // matters is reflected in the bytes, no field-by-field hashing.
            lfs::core::Tensor cached_overlay_params_cpu;
            lfs::core::Tensor model_transforms_source;
            // Same output-bytes fingerprint cache as overlay_params: skips the
            // ~5 ms NULL-stream H2D when the transforms haven't changed (the
            // common case during a lasso drag on a static scene).
            lfs::core::Tensor cached_model_transforms_cpu;
        };
        struct CudaSelectionQuerySlot {
            VulkanContext::ExternalBuffer buffer{};
            lfs::rendering::CudaVulkanBufferInterop interop{};
            std::array<std::size_t, kSelectionQueryRegionCount> region_offset{};
            std::array<std::size_t, kSelectionQueryRegionCount> region_bytes{};
            lfs::core::Tensor transform_indices_source;
            lfs::core::Tensor node_mask_source;
            lfs::core::Tensor primitive_source;
            lfs::core::Tensor model_transforms_source;
            lfs::core::Tensor polygon_vertices_source;
            lfs::core::Tensor output_tensor;
        };

        void detachManagedBuffers();
        void plugRingInputs(std::size_t ring_slot, std::size_t num_splats, bool reset_cached_raster_state);
        void aliasSortScratchToInputSlot(std::size_t ring_slot);
        void releaseInputSlot(VulkanContext& context, std::size_t ring_slot);

        VulkanContext* context_ = nullptr;
        bool initialized_ = false;
        // Dedicated non-blocking CUDA stream for overlay-source H2D uploads.
        // Created with cudaStreamNonBlocking so it does NOT implicitly
        // serialize with the legacy default (NULL) stream where the rest of
        // the project's CUDA work runs — otherwise sub-KB uploads would still
        // wait for unrelated CUDA work to drain. Downstream Vulkan compute
        // observes the upload via the per-slot timeline semaphore signal, so
        // cross-API ordering is preserved without per-frame sync.
        cudaStream_t overlay_upload_stream_ = nullptr;
        VulkanGSRenderer renderer_;
        VulkanGSPipelineBuffers buffers_;
        std::unique_ptr<ComposePipeline> compose_;
        struct OutputImageSlot {
            VulkanContext::ExternalImage image{};
            VulkanContext::ExternalImage depth_image{};
            glm::ivec2 size{0, 0};
            VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
            VkImageLayout depth_layout = VK_IMAGE_LAYOUT_UNDEFINED;
            std::uint64_t generation = 0;
        };
        static constexpr std::size_t kOutputSlotCount = 4;
        std::array<OutputImageSlot, kOutputSlotCount> output_slots_{};

        // Fallback CUDA-backed input buffers for models that are not already
        // backed by Vulkan-external tensor storage. Direct Vulkan-external
        // training tensors bypass this ring and bind their VkBuffers directly.
        static constexpr std::size_t kInputRingSize = 1;
        std::array<CudaInputSlot, kInputRingSize> cuda_inputs_{};
        std::array<CudaOverlaySlot, kInputRingSize> cuda_overlays_{};
        CudaSelectionQuerySlot cuda_selection_query_{};
        std::array<ModelInputSnapshot, kInputRingSize> ring_uploaded_{};

        // Per-ring-slot timeline semaphore used to gate Vulkan compute on the
        // CUDA upload completing; eliminates the per-frame
        // cudaStreamSynchronize that previously blocked the CPU after every
        // upload (P15). Values are monotonic; on each upload we bump the slot's
        // counter, signal CUDA-side, and queue a Vulkan-side wait.
        struct UploadTimeline {
            VulkanContext::ExternalSemaphore vk_semaphore{};
            lfs::rendering::CudaTimelineSemaphore cuda_semaphore{};
            std::uint64_t value = 0;
        };
        std::array<UploadTimeline, kInputRingSize> upload_timelines_{};
        std::array<UploadTimeline, kInputRingSize> overlay_upload_timelines_{};
    };

} // namespace lfs::vis
