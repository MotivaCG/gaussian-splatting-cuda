/* SPDX-FileCopyrightText: 2026 SMN | Víctor M. Feliz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// =============================================================================
// SMN — Attention mask mode: post-training projection-vote prune
// =============================================================================
//
// Runs once, after the final optimization step, when MaskMode::Attention is
// active and SMN_ATTENTION_PRUNE_ENABLED is set. Every Gaussian center is
// projected into each masked training view; Gaussians that consistently land
// outside the mask (across the views that see them) are removed through the
// strategy's own remove_gaussians() path, so optimizer state and frozen splats
// are handled correctly.
//
// This is intentionally separate from the training-time loss/penalty code
// (smn_attention_penalty.*): the penalty shapes the model during training, the
// prune is a one-shot geometric cleanup at the end.
// =============================================================================

#include "strategies/istrategy.hpp"

#include "core/camera.hpp"

#include <expected>
#include <memory>
#include <string>
#include <vector>

namespace lfs::training::smn {

    // Mask-loading parameters needed to reproduce the training-time mask for a
    // camera. These mirror the dataset/optimization fields the trainer already
    // passes to Camera::load_and_get_mask.
    struct AttentionPruneConfig {
        int resize_factor = -1;
        int max_width = 0;
        bool invert_masks = false;
        float mask_threshold = 0.5f;
    };

    // Prune background floaters from the model using the attention masks.
    // No-op (returns success) when pruning is disabled by constant, when the
    // model is empty, or when no usable masks are available. The model is
    // mutated in place through `strategy`.
    std::expected<void, std::string> run_attention_prune(
        lfs::training::IStrategy& strategy,
        const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras,
        const AttentionPruneConfig& config);

} // namespace lfs::training::smn
