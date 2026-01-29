/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

/**
 * @file checkpoint.hpp
 * @brief Training checkpoint format for LichtFeld Studio (.resume files)
 *
 * Binary Format (v1):
 * ┌─────────────────────────────────────────────────────────────────┐
 * │ CheckpointHeader (48 bytes)                                    │
 * │   - magic: uint32 = 0x4C464B50 ("LFKP")                        │
 * │   - version: uint32 = 1                                        │
 * │   - iteration: int32                                           │
 * │   - num_gaussians: uint32                                      │
 * │   - sh_degree: int32                                           │
 * │   - flags: uint32 (HAS_BILATERAL_GRID=1, HAS_PPISP=2)           │
 * │   - params_json_offset: uint64                                 │
 * │   - params_json_size: uint64                                   │
 * ├─────────────────────────────────────────────────────────────────┤
 * │ Strategy Type (variable)                                       │
 * │   - type_len: uint32                                           │
 * │   - type_str: char[type_len] ("mcmc" or "default")            │
 * ├─────────────────────────────────────────────────────────────────┤
 * │ SplatData (via serialize)                                      │
 * │   - means, sh0, scaling, rotation, opacity tensors             │
 * │   - shN (if max_sh_degree > 0)                                 │
 * │   - deleted mask, densification_info (optional)                │
 * ├─────────────────────────────────────────────────────────────────┤
 * │ Strategy State (via serialize)                                 │
 * │   - Adam optimizer states (m, v for each param group)          │
 * │   - Strategy-specific state (e.g., MCMC binoms)                │
 * ├─────────────────────────────────────────────────────────────────┤
 * │ [Optional] BilateralGrid (if HAS_BILATERAL_GRID flag set)      │
 * │   - Grid magic: uint32 = 0x4C464247 ("LFBG")                   │
 * │   - Grid version: uint32 = 1                                   │
 * │   - Dimensions: num_images, grid_W, grid_H, grid_L             │
 * │   - Config struct                                              │
 * │   - Scheduler state: step, current_lr, initial_lr, total_iter  │
 * │   - grids tensor [N, 12, L, H, W]                              │
 * │   - exp_avg tensor (Adam first moment)                         │
 * │   - exp_avg_sq tensor (Adam second moment)                     │
 * ├─────────────────────────────────────────────────────────────────┤
 * │ [Optional] PPISP (if HAS_PPISP flag set)                       │
 * │   - Magic: uint32 = 0x4C465050 ("LFPP")                        │
 * │   - Version: uint32 = 1                                        │
 * │   - num_cameras, num_frames, config, scheduler state           │
 * │   - exposure, vignetting, color, crf params + Adam states      │
 * ├─────────────────────────────────────────────────────────────────┤
 * │ Training Parameters JSON (at params_json_offset)               │
 * │   - optimization: OptimizationParameters                       │
 * │   - dataset: DatasetConfig                                     │
 * └─────────────────────────────────────────────────────────────────┘
 */

#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <string>

namespace lfs::training {

    class IStrategy;
    class BilateralGrid;
    class PPISP;
    class PPISPControllerPool;

    constexpr uint32_t CHECKPOINT_MAGIC = 0x4C464B50; // "LFKP"
    constexpr uint32_t CHECKPOINT_VERSION = 1;

    enum class CheckpointFlags : uint32_t {
        NONE = 0,
        HAS_BILATERAL_GRID = 1 << 0,
        HAS_PPISP = 1 << 1,
        HAS_PPISP_CONTROLLER = 1 << 2,
    };

    inline CheckpointFlags operator|(CheckpointFlags a, CheckpointFlags b) {
        return static_cast<CheckpointFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
    }

    inline bool has_flag(CheckpointFlags flags, CheckpointFlags flag) {
        return (static_cast<uint32_t>(flags) & static_cast<uint32_t>(flag)) != 0;
    }

    struct CheckpointHeader {
        uint32_t magic = CHECKPOINT_MAGIC;
        uint32_t version = CHECKPOINT_VERSION;
        int32_t iteration = 0;
        uint32_t num_gaussians = 0;
        int32_t sh_degree = 0;
        CheckpointFlags flags = CheckpointFlags::NONE;
        uint64_t params_json_offset = 0;
        uint64_t params_json_size = 0;
    };

    /// Save complete training checkpoint (strategy + optional appearance components)
    std::expected<void, std::string> save_checkpoint(
        const std::filesystem::path& path,
        int iteration,
        const IStrategy& strategy,
        const lfs::core::param::TrainingParameters& params,
        const BilateralGrid* bilateral_grid = nullptr,
        const PPISP* ppisp = nullptr,
        const PPISPControllerPool* ppisp_controller_pool = nullptr);

    /// Load checkpoint header only
    std::expected<CheckpointHeader, std::string> load_checkpoint_header(
        const std::filesystem::path& path);

    /// Load complete training checkpoint (strategy + optional appearance components)
    std::expected<int, std::string> load_checkpoint(
        const std::filesystem::path& path,
        IStrategy& strategy,
        lfs::core::param::TrainingParameters& params,
        BilateralGrid* bilateral_grid = nullptr,
        PPISP* ppisp = nullptr,
        PPISPControllerPool* ppisp_controller_pool = nullptr);

    /// Load only SplatData from checkpoint
    std::expected<lfs::core::SplatData, std::string> load_checkpoint_splat_data(
        const std::filesystem::path& path);

    /// Load only training parameters from checkpoint
    std::expected<lfs::core::param::TrainingParameters, std::string> load_checkpoint_params(
        const std::filesystem::path& path);

} // namespace lfs::training
