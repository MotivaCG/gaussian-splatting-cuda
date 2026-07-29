/* SPDX-FileCopyrightText: 2026 SMN | Víctor M. Feliz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// =============================================================================
// SMN — Attention mask mode: post-training multi-pass prune orchestrator
// =============================================================================
//
// Runs once, right before the final save, when MaskMode::Attention is active and
// SMN_ATTENTION_PRUNE_ENABLED is set. Runs the mask_pruning passes enabled in
// smn_attention_constants.h (section 5): geometric dome, center vote (with depth),
// mask leakage, alpha consensus, ellipse boundary, isolation and SOR. Each pass
// removes splats through the strategy's own soft-delete path, so optimizer state
// and frozen splats are handled correctly. Strategy-agnostic (MRNF, MCMC, igs+).
//
// Never fails the training run: a failed pass is logged as a warning and the
// remaining passes continue.
// =============================================================================

#include "strategies/istrategy.hpp"
#include "training/dataset.hpp"

namespace lfs::training::smn {

    void run_attention_prune(lfs::training::IStrategy& strategy,
                             const lfs::training::CameraDataset& dataset,
                             bool invert_masks);

} // namespace lfs::training::smn
