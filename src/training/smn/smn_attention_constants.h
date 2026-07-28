/* SPDX-FileCopyrightText: 2026 SMN | Víctor M. Feliz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <string_view>

// =============================================================================
// SMN — Attention mask mode: fixed tuning constants
// =============================================================================
//
// This header centralizes every compile-time constant used by the "attention"
// mask mode (see MaskMode::Attention). Values are grouped by the stage that
// consumes them:
//
//   1. Photometric weighting  — how strongly out-of-mask pixels are trained.
//   2. Opacity penalty        — the bidirectional alpha guidance term.
//   3. Penalty schedule       — how that penalty ramps in and out over training.
//   4. Post-training prune     — the projection-vote cleanup pass.
//
// All names are UPPER_CASE and prefixed with `SMN_`. Nothing here changes at
// runtime; these are the knobs of the technique. To disable the prune stage
// entirely, flip SMN_ATTENTION_PRUNE_ENABLED to false — no other change needed.
// =============================================================================

namespace lfs::training::smn {

    // -------------------------------------------------------------------------
    // 1. Photometric weighting
    // -------------------------------------------------------------------------

    // Per-pixel loss weight applied OUTSIDE the mask. Inside the mask the weight
    // is always 1.0. A small value (< 1) keeps the background loosely supervised
    // instead of ignoring it outright, which stabilizes geometry near the ROI
    // border while still concentrating detail on the masked object.
    inline constexpr float SMN_ATTENTION_OUT_MASK_WEIGHT = 1.0f / 20.0f; // 0.05

    // -------------------------------------------------------------------------
    // 2. Opacity penalty (bidirectional, per pixel)
    // -------------------------------------------------------------------------
    //
    // The penalty pushes the rendered alpha towards the mask: opaque inside,
    // transparent outside. It is expressed in additive form so it composes with
    // the engine's manual-gradient loss pipeline:
    //
    //   penalty = SCALE * ( IN_WEIGHT  * mean( (1 - alpha) * mask )      // want alpha high inside
    //                     + OUT_WEIGHT * mean(  alpha      * (1-mask) ) ) // want alpha low outside
    //
    // The inside term is weighted more heavily than the outside term because
    // filling the object is the primary objective; suppressing background
    // floaters is secondary and also handled by the post-training prune.

    // Weight of the "be opaque inside the mask" term.
    inline constexpr float SMN_ATTENTION_PENALTY_INSIDE_WEIGHT = 0.667f;

    // Weight of the "be transparent outside the mask" term.
    inline constexpr float SMN_ATTENTION_PENALTY_OUTSIDE_WEIGHT = 0.333f;

    // Global multiplier on the whole opacity penalty. Scales its gradient
    // contribution relative to the photometric loss.
    inline constexpr float SMN_ATTENTION_PENALTY_SCALE = 1.0f;

    // Couple the penalty magnitude to the current photometric loss, i.e. multiply
    // it by (photometric_loss + SMN_ATTENTION_PENALTY_LOSS_FLOOR). This reproduces
    // oldmode's multiplicative form `loss*(1+p) + eps*p`, whose alpha gradient is
    // implicitly scaled by the (small) photometric loss. It keeps the penalty
    // GENTLE: the background is only lightly suppressed during training and the
    // post-training prune does the final cleanup. Without this coupling a fixed
    // SCALE=1 penalty is ~10-30x stronger, clears the background too aggressively,
    // and the resulting up/down alpha fight at the mask border halos fine
    // structures (hair). Keep true unless you deliberately want the aggressive
    // fixed-scale behavior.
    inline constexpr bool SMN_ATTENTION_PENALTY_COUPLE_TO_LOSS = true;

    // Additive floor for the coupling above (oldmode's `1e-2`). Ensures the
    // penalty never fully vanishes when the photometric loss is tiny.
    inline constexpr float SMN_ATTENTION_PENALTY_LOSS_FLOOR = 1.0e-2f;

    // -------------------------------------------------------------------------
    // 3. Penalty schedule (fractions of total training iterations)
    // -------------------------------------------------------------------------
    //
    // schedule weight w(iter), with T = total iterations:
    //
    //   iter < WARMUP*T                     -> 0            (let coarse geometry form first)
    //   WARMUP*T <= iter <= FULL*T          -> 1            (full strength)
    //   FULL*T   <  iter <  DECAY_END*T     -> linear 1->0  (release the constraint)
    //   iter >= DECAY_END*T                 -> 0            (final refinement is unconstrained)

    inline constexpr float SMN_ATTENTION_PENALTY_WARMUP_FRACTION = 0.1f;
    inline constexpr float SMN_ATTENTION_PENALTY_FULL_FRACTION = 0.5f;
    inline constexpr float SMN_ATTENTION_PENALTY_DECAY_END_FRACTION = 0.8f;

    // -------------------------------------------------------------------------
    // 4. Post-training prune (projection vote)
    // -------------------------------------------------------------------------
    //
    // After the last optimization step, every Gaussian center is projected into
    // each masked view. A Gaussian is kept only when it lands inside the mask in
    // a large enough fraction of the views that actually see it. This removes
    // background floaters the opacity penalty did not fully suppress.

    // Master switch: set to false to skip the prune stage entirely while keeping
    // the attention loss/penalty behavior intact.
    inline constexpr bool SMN_ATTENTION_PRUNE_ENABLED = true;

    // When true (and the prune runs), save an extra PLY of the model BEFORE
    // pruning, so a single MaskMode::Attention run yields both the pre-prune and
    // pruned results — no need to also train with attention_no_prune to compare.
    // The copy is named "<output>_preprune.ply" alongside the final PLY.
    inline constexpr bool SMN_ATTENTION_SAVE_PREPRUNE_COPY = true;

    // Filename suffix (before ".ply") used for the pre-prune copy above.
    inline constexpr std::string_view SMN_ATTENTION_PREPRUNE_SUFFIX = "_preprune";

    // Minimum fraction of visible views in which a Gaussian must project inside
    // the mask to survive the prune. 0.8 => "inside in at least 80% of views".
    inline constexpr float SMN_ATTENTION_PRUNE_KEEP_THRESHOLD = 0.8f;

    // A Gaussian must be visible in at least this many views before it is
    // eligible for pruning. Guards against removing splats seen only once or
    // twice, where the inside/outside vote is statistically unreliable.
    inline constexpr int SMN_ATTENTION_PRUNE_MIN_VISIBILITY = 3;

    // Near-plane distance (camera space, +Z forward). A projected center with
    // z <= this value is treated as not visible in that view.
    inline constexpr float SMN_ATTENTION_PRUNE_NEAR_PLANE = 0.01f;

    // Mask sampling threshold: mask pixels strictly greater than this count as
    // "inside". Masks reach the prune already binarized to {0,1}, so 0.5 is a
    // safe midpoint.
    inline constexpr float SMN_ATTENTION_PRUNE_MASK_THRESHOLD = 0.5f;

} // namespace lfs::training::smn
