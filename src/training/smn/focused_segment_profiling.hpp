/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// =============================================================================
// SMN / FocusedSegment per-subtask timing infrastructure.
//
// THIS HEADER BELONGS TO THE SMN FORK (training/smn/).
//
// Cumulative wall-clock timing for each FocusedSegment helper plus the
// post-training prune passes. Uses `cudaDeviceSynchronize` around every scoped
// region so the measured time reflects GPU work, not just CPU launch overhead.
// Overhead: ~10-50 us per scoped region (one extra sync at begin and end);
// negligible against the per-iter cost of the regions we time.
//
// Owning container: lfs::training::Trainer::fs_timings_ (declared in
// trainer.hpp inside the FocusedSegment helpers block, clearly marked SMN-only).
//
// Output: `Trainer::focused_segment_print_timings()` is called once at the end
// of `Trainer::train()` (after the post-training prune) and dumps a table to
// the log via LOG_INFO.
// =============================================================================

#include <array>
#include <chrono>
#include <cstdint>
#include <cuda_runtime.h>
#include <string_view>

namespace lfs::training::smn {

    // Introduces a penalty for Device Sync so it should be false for production
    constexpr bool ENABLE_TRAINING_PRINT_DEBUG = true; 

    enum class FsTimerSlot : int {
        ApplySchedule = 0,
        ComputeLossTotal,
        ComputeLossPhoto,        // L1+SSIM forward + backward
        ComputeLossSpatialWeight,// build weight map + apply to grad + rescale loss
        ComputeLossGradAlpha,    // build grad_alpha (FG + BG terms)
        CacheBuildMask,          // first-hit cost for mask_bool + pixel counts
        CacheBuildDarkness,      // first-hit cost for darkness FP16
        ApplyDensifyMask,
        PostBackward,            // full body of focused_segment_post_backward
        ApplyCenterPenalty,      // per-splat opacity penalty (called from post_backward)
        PruneGeometricDome,
        PruneCenterVote,
        PruneMaskLeakage,
        PruneEllipseBoundary,
        PruneClusterExtreme,
        PruneIsolation,
        Count
    };

    constexpr int kFsTimerSlotCount = static_cast<int>(FsTimerSlot::Count);

    struct FsTimings {
        std::array<double, kFsTimerSlotCount> total_ms{};
        std::array<std::uint64_t, kFsTimerSlotCount> calls{};
        bool enabled = ENABLE_TRAINING_PRINT_DEBUG; // Disable to make ScopedFsTimer a no-op (avoids syncs).
    };

    inline constexpr std::string_view fs_timer_slot_name(FsTimerSlot s) {
        switch (s) {
        case FsTimerSlot::ApplySchedule:            return "apply_schedule";
        case FsTimerSlot::ComputeLossTotal:         return "compute_loss (total)";
        case FsTimerSlot::ComputeLossPhoto:         return "  L1+SSIM fwd+bwd";
        case FsTimerSlot::ComputeLossSpatialWeight: return "  spatial weight map";
        case FsTimerSlot::ComputeLossGradAlpha:     return "  grad_alpha";
        case FsTimerSlot::CacheBuildMask:           return "  cache build: mask (1st hit)";
        case FsTimerSlot::CacheBuildDarkness:       return "  cache build: darkness (1st hit)";
        case FsTimerSlot::ApplyDensifyMask:         return "apply_densify_mask";
        case FsTimerSlot::PostBackward:             return "post_backward (total)";
        case FsTimerSlot::ApplyCenterPenalty:       return "  center penalty";
        case FsTimerSlot::PruneGeometricDome:       return "prune: geometric dome";
        case FsTimerSlot::PruneCenterVote:          return "prune: center vote";
        case FsTimerSlot::PruneMaskLeakage:         return "prune: mask leakage";
        case FsTimerSlot::PruneEllipseBoundary:     return "prune: ellipse boundary";
        case FsTimerSlot::PruneClusterExtreme:      return "prune: cluster + extremes";
        case FsTimerSlot::PruneIsolation:           return "prune: isolation";
        case FsTimerSlot::Count:                    return "<count>";
        }
        return "?";
    }

    /// RAII timer: cudaDeviceSynchronize at begin and end, chrono for wall time.
    /// Begin/end overhead is one sync each (~10-50 us), so do not place around
    /// trivial CPU-only regions where the overhead would dominate.
    class ScopedFsTimer {
    public:
        ScopedFsTimer(FsTimings& t, FsTimerSlot slot) noexcept
            : t_(t), slot_(static_cast<int>(slot)) {
            if (!t_.enabled) return;
            cudaDeviceSynchronize();
            start_ = std::chrono::steady_clock::now();
        }

        ~ScopedFsTimer() noexcept {
            if (!t_.enabled) return;
            cudaDeviceSynchronize();
            const auto end = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(end - start_).count();
            t_.total_ms[slot_] += ms;
            t_.calls[slot_] += 1;
        }

        ScopedFsTimer(const ScopedFsTimer&) = delete;
        ScopedFsTimer& operator=(const ScopedFsTimer&) = delete;

    private:
        FsTimings& t_;
        int slot_;
        std::chrono::steady_clock::time_point start_;
    };

} // namespace lfs::training::smn
