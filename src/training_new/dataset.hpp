/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core_new/camera.hpp"
#include "core_new/image_io.hpp"
#include "core_new/logger.hpp"
#include "core_new/parameters.hpp"
#include "core_new/splat_data.hpp"
#include "core_new/tensor.hpp"
#include "loader_new/loader.hpp"
#include <atomic>
#include <condition_variable>
#include <cuda_runtime.h>
#include <expected>
#include <format>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <random>
#include <thread>
#include <vector>

namespace lfs::training {

    // Simple CUDA image buffer - no torch dependency
    class CUDAImageBuffer {
    private:
        float* data_ = nullptr;
        size_t width_ = 0;
        size_t height_ = 0;
        size_t channels_ = 3;
        bool allocated_ = false;

    public:
        CUDAImageBuffer() = default;

        ~CUDAImageBuffer() {
            if (data_) {
                cudaFree(data_);
            }
        }

        // Move only
        CUDAImageBuffer(const CUDAImageBuffer&) = delete;
        CUDAImageBuffer& operator=(const CUDAImageBuffer&) = delete;

        CUDAImageBuffer(CUDAImageBuffer&& other) noexcept
            : data_(other.data_),
              width_(other.width_),
              height_(other.height_),
              channels_(other.channels_),
              allocated_(other.allocated_) {
            other.data_ = nullptr;
            other.allocated_ = false;
        }

        CUDAImageBuffer& operator=(CUDAImageBuffer&& other) noexcept {
            if (this != &other) {
                if (data_)
                    cudaFree(data_);
                data_ = other.data_;
                width_ = other.width_;
                height_ = other.height_;
                channels_ = other.channels_;
                allocated_ = other.allocated_;
                other.data_ = nullptr;
                other.allocated_ = false;
            }
            return *this;
        }

        // Allocate or reallocate if size changes
        void ensure_size(size_t w, size_t h, size_t c = 3) {
            size_t required_size = w * h * c;
            size_t current_size = width_ * height_ * channels_;

            if (!allocated_ || required_size != current_size) {
                if (data_) {
                    cudaFree(data_);
                }

                cudaMalloc(&data_, required_size * sizeof(float));
                width_ = w;
                height_ = h;
                channels_ = c;
                allocated_ = true;
            }
        }

        // Load image from file directly to this buffer
        void load_from_file(const std::filesystem::path& path, int resize_factor) {
            // Load image using existing image_io
            unsigned char* cpu_data;
            int w, h, c;
            std::tie(cpu_data, w, h, c) = load_image(path, resize_factor);

            // Ensure buffer is right size
            ensure_size(w, h, c);

            // Convert to float and upload to GPU in one step
            // Create temporary float buffer on CPU
            std::vector<float> float_buffer(w * h * c);

            // Convert uint8 to float in CHW format (channels first)
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    for (int ch = 0; ch < c; ++ch) {
                        size_t src_idx = (y * w + x) * c + ch;
                        size_t dst_idx = ch * h * w + y * w + x;
                        float_buffer[dst_idx] = cpu_data[src_idx] / 255.0f;
                    }
                }
            }

            // Copy to GPU
            cudaMemcpy(data_, float_buffer.data(),
                       float_buffer.size() * sizeof(float),
                       cudaMemcpyHostToDevice);

            // Free CPU image
            free_image(cpu_data);
        }

        float* data() { return data_; }
        const float* data() const { return data_; }
        size_t width() const { return width_; }
        size_t height() const { return height_; }
        size_t channels() const { return channels_; }
    };

    // Image buffer pool for reusing memory
    class ImageBufferPool {
    private:
        std::vector<std::unique_ptr<CUDAImageBuffer>> buffers_;
        std::queue<CUDAImageBuffer*> available_;
        mutable std::mutex mutex_;
        size_t total_buffers_ = 0;

    public:
        explicit ImageBufferPool(size_t initial_size = 8) {
            for (size_t i = 0; i < initial_size; ++i) {
                auto buffer = std::make_unique<CUDAImageBuffer>();
                available_.push(buffer.get());
                buffers_.push_back(std::move(buffer));
            }
            total_buffers_ = initial_size;
        }

        CUDAImageBuffer* acquire() {
            std::lock_guard<std::mutex> lock(mutex_);

            if (available_.empty()) {
                // Create new buffer if pool is exhausted
                auto buffer = std::make_unique<CUDAImageBuffer>();
                CUDAImageBuffer* ptr = buffer.get();
                buffers_.push_back(std::move(buffer));
                total_buffers_++;
                LOG_DEBUG("ImageBufferPool expanded to {} buffers", total_buffers_);
                return ptr;
            }

            CUDAImageBuffer* buffer = available_.front();
            available_.pop();
            return buffer;
        }

        void release(CUDAImageBuffer* buffer) {
            std::lock_guard<std::mutex> lock(mutex_);
            available_.push(buffer);
        }

        size_t size() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return total_buffers_;
        }

        size_t available_count() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return available_.size();
        }
    };

    // Wrapper for camera with raw image data
    struct CameraWithImage {
        lfs::core::Camera* camera;
        CUDAImageBuffer* image_buffer; // Borrowed from pool

        // For compatibility - return raw pointer
        const float* image_data() const {
            return image_buffer ? image_buffer->data() : nullptr;
        }

        size_t image_width() const {
            return image_buffer ? image_buffer->width() : 0;
        }

        size_t image_height() const {
            return image_buffer ? image_buffer->height() : 0;
        }

        size_t image_channels() const {
            return image_buffer ? image_buffer->channels() : 3;
        }

        // Raw pointer accessors for fast access
        const float* world_view_transform_ptr() const {
            return camera->world_view_transform().ptr<float>();
        }

        const float* cam_position_ptr() const {
            return camera->cam_position().ptr<float>();
        }
    };

    // Forward declarations
    class InfiniteRandomSampler;
    class RandomSampler;

    class CameraDataset {
    public:
        enum class Split {
            TRAIN,
            VAL,
            ALL
        };

        CameraDataset(std::vector<std::shared_ptr<lfs::core::Camera>> cameras,
                      const lfs::core::param::DatasetConfig& params,
                      Split split = Split::ALL)
            : _cameras(std::move(cameras)),
              _datasetConfig(params),
              _split(split),
              _buffer_pool(std::make_shared<ImageBufferPool>(16)) {

            // Create indices based on split
            _indices.clear();
            for (size_t i = 0; i < _cameras.size(); ++i) {
                const bool is_test = (i % params.test_every) == 0;

                if (_split == Split::ALL ||
                    (_split == Split::TRAIN && !is_test) ||
                    (_split == Split::VAL && is_test)) {
                    _indices.push_back(i);
                }
            }

            std::cout << "Dataset created with " << _indices.size()
                      << " images (split: " << static_cast<int>(_split) << ")" << std::endl;
        }

        // Default copy constructor works with shared_ptr
        CameraDataset(const CameraDataset&) = default;
        CameraDataset(CameraDataset&&) noexcept = default;
        CameraDataset& operator=(CameraDataset&&) noexcept = default;
        CameraDataset& operator=(const CameraDataset&) = default;

        // Share buffer pool between datasets
        void share_buffer_pool(std::shared_ptr<ImageBufferPool> pool) {
            _buffer_pool = pool;
        }

        CameraWithImage get(size_t index) {
            if (index >= _indices.size()) {
                throw std::out_of_range("Dataset index out of range");
            }

            size_t camera_idx = _indices[index];
            auto& cam = _cameras[camera_idx];

            // Get buffer from pool
            CUDAImageBuffer* buffer = _buffer_pool->acquire();

            // Load image directly into buffer
            buffer->load_from_file(cam->image_path(), _datasetConfig.resize_factor);

            // Update camera's image dimensions if needed
            if (cam->image_width() != static_cast<int>(buffer->width()) ||
                cam->image_height() != static_cast<int>(buffer->height())) {
                cam->load_image_size(_datasetConfig.resize_factor);
            }

            return {cam.get(), buffer};
        }

        // Return buffer to pool after use
        void release_buffer(CUDAImageBuffer* buffer) {
            if (buffer && _buffer_pool) {
                _buffer_pool->release(buffer);
            }
        }

        size_t size() const {
            return _indices.size();
        }

        const std::vector<std::shared_ptr<lfs::core::Camera>>& get_cameras() const {
            return _cameras;
        }

        Split get_split() const { return _split; }

        size_t get_num_bytes() const {
            if (_cameras.empty()) {
                return 0;
            }
            size_t total_bytes = 0;
            // for (const auto& cam : _cameras) {
            //     //total_bytes += cam->get_num_bytes_from_file();
            // }
            //  Adjust for resolution factor if specified
            if (_datasetConfig.resize_factor > 0) {
                total_bytes /= _datasetConfig.resize_factor * _datasetConfig.resize_factor;
            }
            return total_bytes;
        }

        [[nodiscard]] std::optional<lfs::core::Camera*> get_camera_by_filename(const std::string& filename) const {
            for (const auto& cam : _cameras) {
                if (cam->image_name() == filename) {
                    return cam.get();
                }
            }
            return std::nullopt;
        }

        void set_resize_factor(int resize_factor) {
            _datasetConfig.resize_factor = resize_factor;
        }

        std::shared_ptr<ImageBufferPool> get_buffer_pool() const {
            return _buffer_pool;
        }

    private:
        std::vector<std::shared_ptr<lfs::core::Camera>> _cameras;
        lfs::core::param::DatasetConfig _datasetConfig;
        Split _split;
        std::vector<size_t> _indices;
        std::shared_ptr<ImageBufferPool> _buffer_pool;
    };

    // Regular random sampler (non-infinite)
    class RandomSampler {
    public:
        explicit RandomSampler(size_t dataset_size)
            : size_(dataset_size),
              current_position_(0),
              gen_(std::random_device{}()) {

            if (size_ == 0) {
                throw std::invalid_argument("Dataset size must be positive");
            }

            // Generate initial permutation
            generate_permutation();
        }

        // Get next batch of indices (returns nullopt when epoch is done)
        std::optional<std::vector<size_t>> next(size_t batch_size) {
            if (current_position_ >= indices_.size()) {
                return std::nullopt; // End of epoch
            }

            std::vector<size_t> batch;
            batch.reserve(batch_size);

            for (size_t i = 0; i < batch_size && current_position_ < indices_.size(); ++i) {
                batch.push_back(indices_[current_position_]);
                current_position_++;
            }

            return batch;
        }

        void reset() {
            generate_permutation();
            current_position_ = 0;
        }

        size_t size() const { return size_; }

    private:
        void generate_permutation() {
            indices_.resize(size_);
            for (size_t i = 0; i < size_; ++i) {
                indices_[i] = i;
            }
            std::shuffle(indices_.begin(), indices_.end(), gen_);
        }

        size_t size_;
        size_t current_position_;
        std::vector<size_t> indices_;
        std::mt19937 gen_;
    };

    // Torch-free infinite random sampler
    class InfiniteRandomSampler {
    public:
        explicit InfiniteRandomSampler(size_t dataset_size)
            : size_(dataset_size),
              current_position_(0),
              epoch_(0),
              gen_(std::random_device{}()) {

            if (size_ == 0) {
                throw std::invalid_argument("Dataset size must be positive");
            }

            // Generate initial permutation
            generate_permutation();
        }

        // Get next batch of indices
        std::vector<size_t> next(size_t batch_size) {
            std::vector<size_t> batch;
            batch.reserve(batch_size);

            for (size_t i = 0; i < batch_size; ++i) {
                // Check if we need to generate new permutation
                if (current_position_ >= indices_.size()) {
                    generate_permutation();
                    current_position_ = 0;
                    epoch_++;
                }

                batch.push_back(indices_[current_position_]);
                current_position_++;
            }

            return batch;
        }

        // Get single index
        size_t next_single() {
            std::lock_guard<std::mutex> lock(mutex_);
            if (current_position_ >= indices_.size()) {
                generate_permutation();
                current_position_ = 0;
                epoch_++;
            }

            return indices_[current_position_++];
        }

        void reset() {
            std::lock_guard<std::mutex> lock(mutex_);
            generate_permutation();
            current_position_ = 0;
            epoch_++;
        }

        size_t size() const { return size_; }
        size_t epoch() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return epoch_;
        }
        size_t position() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return current_position_;
        }

    private:
        void generate_permutation() {
            indices_.resize(size_);
            for (size_t i = 0; i < size_; ++i) {
                indices_[i] = i;
            }
            std::shuffle(indices_.begin(), indices_.end(), gen_);
        }

        size_t size_;
        size_t current_position_;
        size_t epoch_;
        std::vector<size_t> indices_;
        std::mt19937 gen_;
        mutable std::mutex mutex_;
    };

    // Simple batch structure
    struct CameraBatch {
        std::vector<CameraWithImage> data;
        std::shared_ptr<ImageBufferPool> pool; // Keep reference to pool

        CameraBatch() = default;

        ~CameraBatch() {
            // Return all buffers to pool when batch is destroyed
            if (pool) {
                for (auto& item : data) {
                    if (item.image_buffer) {
                        pool->release(item.image_buffer);
                    }
                }
            }
        }

        // Move only
        CameraBatch(const CameraBatch&) = delete;
        CameraBatch& operator=(const CameraBatch&) = delete;

        CameraBatch(CameraBatch&& other) noexcept
            : data(std::move(other.data)),
              pool(std::move(other.pool)) {
            other.data.clear();
        }

        CameraBatch& operator=(CameraBatch&& other) noexcept {
            if (this != &other) {
                // Return current buffers first
                if (pool) {
                    for (auto& item : data) {
                        if (item.image_buffer) {
                            pool->release(item.image_buffer);
                        }
                    }
                }
                data = std::move(other.data);
                pool = std::move(other.pool);
                other.data.clear();
            }
            return *this;
        }

        CameraWithImage& operator[](size_t idx) {
            if (idx >= data.size()) {
                throw std::out_of_range("Batch index out of range");
            }
            return data[idx];
        }

        const CameraWithImage& operator[](size_t idx) const {
            if (idx >= data.size()) {
                throw std::out_of_range("Batch index out of range");
            }
            return data[idx];
        }
    };

    // Regular DataLoader (non-infinite) - for evaluation, single-threaded
    class DataLoader {
    public:
        DataLoader(std::shared_ptr<CameraDataset> dataset,
                   size_t batch_size = 1,
                   [[maybe_unused]] int num_workers = 0) // num_workers ignored for evaluation
            : dataset_(dataset),
              batch_size_(batch_size),
              sampler_(dataset->size()) {
        }

        // Iterator for range-based for loops
        class Iterator {
        public:
            Iterator(DataLoader* loader, bool is_end = false)
                : loader_(loader),
                  is_end_(is_end) {
                if (!is_end_) {
                    advance();
                }
            }

            CameraBatch operator*() {
                return std::move(current_batch_);
            }

            Iterator& operator++() {
                advance();
                return *this;
            }

            bool operator!=(const Iterator& other) const {
                return is_end_ != other.is_end_;
            }

        private:
            void advance() {
                auto indices = loader_->sampler_.next(loader_->batch_size_);
                if (!indices) {
                    is_end_ = true;
                    return;
                }

                current_batch_ = CameraBatch();
                current_batch_.pool = loader_->dataset_->get_buffer_pool();
                current_batch_.data.clear();
                for (size_t idx : *indices) {
                    current_batch_.data.push_back(loader_->dataset_->get(idx));
                }
            }

            DataLoader* loader_;
            CameraBatch current_batch_;
            bool is_end_;
        };

        Iterator begin() {
            sampler_.reset(); // Reset sampler for new epoch
            return Iterator(this, false);
        }

        Iterator end() {
            return Iterator(this, true);
        }

    private:
        std::shared_ptr<CameraDataset> dataset_;
        size_t batch_size_;
        RandomSampler sampler_;
    };

    // Torch-free Infinite DataLoader with worker threads for training
    class InfiniteDataLoader {
    public:
        InfiniteDataLoader(std::shared_ptr<CameraDataset> dataset,
                           size_t batch_size = 1,
                           int num_workers = 4)
            : dataset_(dataset),
              batch_size_(batch_size),
              num_workers_(std::max(0, num_workers)),
              sampler_(dataset->size()),
              stop_workers_(false),
              prefetch_factor_(2) {

            if (num_workers_ > 0) {
                // Start worker threads
                for (int i = 0; i < num_workers_; ++i) {
                    workers_.emplace_back(&InfiniteDataLoader::worker_thread, this, i);
                }
                LOG_DEBUG("Started {} worker threads for training dataloader", num_workers_);
            } else {
                LOG_WARN("Running training dataloader without workers - expect memory issues!");
            }
        }

        ~InfiniteDataLoader() {
            stop_workers_ = true;
            cv_workers_.notify_all();

            for (auto& worker : workers_) {
                if (worker.joinable()) {
                    worker.join();
                }
            }
        }

        // Iterator for range-based for loops
        class Iterator {
        public:
            Iterator(InfiniteDataLoader* loader, bool is_end = false)
                : loader_(loader),
                  is_end_(is_end) {
                if (!is_end_) {
                    advance();
                }
            }

            CameraBatch operator*() {
                return std::move(current_batch_);
            }

            Iterator& operator++() {
                advance();
                return *this;
            }

            bool operator!=(const Iterator& other) const {
                // Infinite iterator never equals end
                return !is_end_ || !other.is_end_;
            }

        private:
            void advance() {
                if (loader_->num_workers_ > 0) {
                    // Get from queue (workers are loading in background)
                    current_batch_ = loader_->get_next_batch();
                } else {
                    // Single-threaded fallback
                    size_t idx = loader_->sampler_.next_single();
                    current_batch_ = CameraBatch();
                    current_batch_.pool = loader_->dataset_->get_buffer_pool();
                    current_batch_.data.clear();
                    current_batch_.data.push_back(loader_->dataset_->get(idx));
                }
            }

            InfiniteDataLoader* loader_;
            CameraBatch current_batch_;
            bool is_end_;
        };

        Iterator begin() {
            return Iterator(this, false);
        }

        Iterator end() {
            return Iterator(this, true);
        }

    private:
        void worker_thread(int worker_id) {
            // Set CUDA device for this worker thread (CRITICAL for GUI mode)
            cudaSetDevice(0);

            // Set CPU affinity if possible (Linux/Unix only)
#if defined(__linux__) || defined(__unix__)
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(worker_id % std::thread::hardware_concurrency(), &cpuset);
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif

            while (!stop_workers_) {
                // Check queue size
                {
                    std::unique_lock<std::mutex> lock(queue_mutex_);
                    cv_workers_.wait(lock, [this] {
                        return stop_workers_ ||
                               loaded_batches_.size() < prefetch_factor_ * num_workers_;
                    });
                }

                if (stop_workers_)
                    break;

                // Get next index
                size_t idx = sampler_.next_single();

                // Load data (this happens outside the lock)
                CameraBatch batch;
                batch.pool = dataset_->get_buffer_pool();
                batch.data.push_back(dataset_->get(idx));

                // Add to queue
                {
                    std::lock_guard<std::mutex> lock(queue_mutex_);
                    loaded_batches_.push(std::move(batch));
                }
                cv_main_.notify_one();
            }
        }

        CameraBatch get_next_batch() {
            std::unique_lock<std::mutex> lock(queue_mutex_);

            // Notify workers if queue is getting low
            if (loaded_batches_.size() < prefetch_factor_) {
                cv_workers_.notify_all();
            }

            // Wait for batch
            cv_main_.wait(lock, [this] {
                return !loaded_batches_.empty() || stop_workers_;
            });

            if (stop_workers_ && loaded_batches_.empty()) {
                return CameraBatch();
            }

            CameraBatch batch = std::move(loaded_batches_.front());
            loaded_batches_.pop();

            // Notify workers to produce more
            cv_workers_.notify_one();

            return batch;
        }

        std::shared_ptr<CameraDataset> dataset_;
        size_t batch_size_;
        int num_workers_;
        InfiniteRandomSampler sampler_;

        // Worker management
        std::vector<std::thread> workers_;
        std::queue<CameraBatch> loaded_batches_;
        std::mutex queue_mutex_;
        std::condition_variable cv_workers_;
        std::condition_variable cv_main_;
        std::atomic<bool> stop_workers_;
        size_t prefetch_factor_;
    };

    // Factory functions to match existing interface
    inline std::unique_ptr<DataLoader> create_dataloader_from_dataset(
        std::shared_ptr<CameraDataset> dataset,
        int num_workers = 4) {

        return std::make_unique<DataLoader>(dataset, 1, num_workers);
    }

    inline std::unique_ptr<InfiniteDataLoader> create_infinite_dataloader_from_dataset(
        std::shared_ptr<CameraDataset> dataset,
        int num_workers = 4) {

        return std::make_unique<InfiniteDataLoader>(dataset, 1, num_workers);
    }

    // Keep the existing create_dataset
    inline std::expected<std::tuple<std::shared_ptr<CameraDataset>, lfs::core::Tensor>, std::string>
    create_dataset_from_colmap(const lfs::core::param::DatasetConfig& datasetConfig) {
        try {
            if (!std::filesystem::exists(datasetConfig.data_path)) {
                return std::unexpected(std::format("Data path does not exist: {}",
                                                   datasetConfig.data_path.string()));
            }

            // Create loader
            auto loader = lfs::loader::Loader::create();

            // Set up load options
            lfs::loader::LoadOptions options{
                .resize_factor = datasetConfig.resize_factor,
                .images_folder = datasetConfig.images,
                .validate_only = false};

            // Load the data
            auto result = loader->load(datasetConfig.data_path, options);
            if (!result) {
                return std::unexpected(std::format("Failed to load COLMAP dataset: {}", result.error()));
            }

            // Handle the result
            return std::visit(
                [&result, &datasetConfig](
                    auto&& data) -> std::expected<std::tuple<std::shared_ptr<CameraDataset>, lfs::core::Tensor>, std::string> {
                    using T = std::decay_t<decltype(data)>;

                    if constexpr (std::is_same_v<T, std::shared_ptr<lfs::core::SplatData>>) {
                        return std::unexpected("Expected COLMAP dataset but got PLY file");
                    } else if constexpr (std::is_same_v<T, lfs::loader::LoadedScene>) {
                        if (!data.cameras) {
                            return std::unexpected("Loaded scene has no cameras");
                        }

                        // Return the camera dataset that's already been created
                        return std::make_tuple(data.cameras, result->scene_center);
                    } else {
                        return std::unexpected("Unknown data type returned from loader");
                    }
                },
                result->data);
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to create dataset from COLMAP: {}", e.what()));
        }
    }

    inline std::expected<std::tuple<std::shared_ptr<CameraDataset>, lfs::core::Tensor>, std::string>
    create_dataset_from_transforms(const lfs::core::param::DatasetConfig& datasetConfig) {
        try {
            if (!std::filesystem::exists(datasetConfig.data_path)) {
                return std::unexpected(std::format("Data path does not exist: {}",
                                                   datasetConfig.data_path.string()));
            }

            // Create loader
            auto loader = lfs::loader::Loader::create();

            // Set up load options
            lfs::loader::LoadOptions options{
                .resize_factor = datasetConfig.resize_factor,
                .images_folder = datasetConfig.images,
                .validate_only = false};

            // Load the data
            auto result = loader->load(datasetConfig.data_path, options);
            if (!result) {
                return std::unexpected(std::format("Failed to load transforms dataset: {}", result.error()));
            }

            // Handle the result
            return std::visit(
                [&datasetConfig, &result](
                    auto&& data) -> std::expected<std::tuple<std::shared_ptr<CameraDataset>, lfs::core::Tensor>, std::string> {
                    using T = std::decay_t<decltype(data)>;

                    if constexpr (std::is_same_v<T, std::shared_ptr<lfs::core::SplatData>>) {
                        return std::unexpected("Expected transforms.json dataset but got PLY file");
                    } else if constexpr (std::is_same_v<T, lfs::loader::LoadedScene>) {
                        if (!data.cameras) {
                            return std::unexpected("Loaded scene has no cameras");
                        }

                        // Return the camera dataset that's already been created
                        return std::make_tuple(data.cameras, result->scene_center);
                    } else {
                        return std::unexpected("Unknown data type returned from loader");
                    }
                },
                result->data);
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to create dataset from transforms: {}", e.what()));
        }
    }

} // namespace lfs::training