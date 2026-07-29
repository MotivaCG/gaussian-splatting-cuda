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

    // Dilation radius (pixels) of the mask used ONLY for "where to give priority":
    // the photometric weight and the densification error map. It grows the full-
    // priority region a few pixels beyond the (tight) mask so the border band —
    // hair above all — is well reconstructed AND densified. The opacity penalty
    // (alpha) and the post-training prune keep the TIGHT mask, so this does not
    // push the background opaque (no halo) nor keep floaters. 0 disables it.
    inline constexpr int SMN_ATTENTION_PRIORITY_DILATION_PX = 3;



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

    // Darkness boost (after oldmode make_darkness_weight): the PHOTOMETRIC loss is
    // weighted per pixel by
    //   w = 1 + (1 - brightness(GT)) * SMN_ATTENTION_DARKNESS_BOOST
    // so dark target regions (dark hair, dark clothing) get more reconstruction
    // attention. brightness is Rec.601 luma (0.299 R + 0.587 G + 0.114 B) of the
    // GROUND-TRUTH image (never the render); darkness = 1 - brightness. (Oldmode's
    // extra mean-normalization is omitted here: it cancels in the fused loss, which
    // normalizes by the total weight sum.)
    //
    // This is a LOSS weight, not an opacity term — it never pushes alpha, so it
    // cannot halo edges (the halo came from wrongly placing it in the penalty).
    // It coexists with the opacity penalty (oldmode made them mutually exclusive;
    // here they are independent). A value of 0 disables it; 2.0 is the oldmode
    // default. GT-based, so it is static per image.
    inline constexpr float SMN_ATTENTION_DARKNESS_BOOST = 2.0f;

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
    // 4. One-shot opacity "kick"
    // -------------------------------------------------------------------------
    //
    // Once during training, adjust every splat's LINEAR opacity, then let the
    // optimizer + strategy decay relax it — a discrete "kick, then settle". Fires
    // only in the attention modes; strategy-agnostic (MRNF and MCMC). Two
    // compatible components are available; when both are enabled they apply in
    // sequence (POW first, then MUL).

    // Master switch for the kick.
    inline constexpr bool SMN_OPACITY_KICK_ENABLED = true;

    // Fraction of total training at which the kick fires (once).
    inline constexpr float SMN_OPACITY_KICK_AT_FRACTION = 0.75f;

    // Power (gamma) component: opacity_linear <- opacity_linear ^ POW_VALUE.
    // POW_VALUE < 1 raises opacity (toward 1, strongest on low opacities), > 1
    // lowers it, == 1 is a no-op.
    inline constexpr bool SMN_OPACITY_KICK_POW_ENABLED = true;
    inline constexpr float SMN_OPACITY_KICK_POW_VALUE = 0.45f;

    // Multiplier component (SuperSplat): opacity_linear <- opacity_linear * e^MUL_VALUE.
    // MUL_VALUE > 0 raises opacity, < 0 lowers it, == 0 is a no-op.
    inline constexpr bool SMN_OPACITY_KICK_MUL_ENABLED = false;
    inline constexpr float SMN_OPACITY_KICK_MUL_VALUE = 0.5f;

    // Reset the opacity Adam moments right after the kick so stale momentum does
    // not immediately undo it.
    inline constexpr bool SMN_OPACITY_KICK_RESET_OPTIMIZER = true;



    // -------------------------------------------------------------------------
    // 5. Post-training prune (multi-pass; see mask_pruning.hpp)
    // -------------------------------------------------------------------------
    //
    // A sequence of independent post-training passes, each toggled below. They run
    // once, right before the final save, and compound (MCMC soft-deletes are
    // compacted between passes). All values below are SMNV2's validated tuning for
    // inward-facing person captures — deliberately conservative so real geometry
    // (a pointing fingertip, hair tips) survives.

    // Master switch: false skips the whole prune stage (loss/penalty untouched).
    inline constexpr bool SMN_ATTENTION_PRUNE_ENABLED = true;

    // When true (and the prune runs), save an extra PLY of the model BEFORE
    // pruning, so a single MaskMode::Attention run yields both the pre-prune and
    // pruned results — no need to also train with attention_no_prune to compare.
    inline constexpr bool SMN_ATTENTION_SAVE_PREPRUNE_COPY = true;

    // Filename suffix (before ".ply") used for the pre-prune copy above.
    inline constexpr std::string_view SMN_ATTENTION_PREPRUNE_SUFFIX = "_preprune";

    // Per-pass enables (run in this order; each is a no-op when disabled).
    // On an inward-facing DOME the mask passes average over views, so a surface
    // hair splat (at the silhouette in only a few views, central in the rest)
    // survives. Alpha CONSENSUS is the gentlest (mass-weighted) so it is ON; the
    // harsher/redundant LEAKAGE and ELLIPSE (binary boundary, silhouette-strict)
    // stay OFF. Caveat: consensus can still trim genuinely wispy flyaway strands
    // that fall outside the tight mask in MANY views — disable it if that hurts.
    // Geometric dome (position-based) and isolation/SOR (3D outliers) never touch
    // the mask border.
    inline constexpr bool SMN_PRUNE_GEOMETRIC_ENABLED = true;
    inline constexpr bool SMN_PRUNE_CENTER_VOTE_ENABLED = true;
    inline constexpr bool SMN_PRUNE_LEAKAGE_ENABLED = false;
    inline constexpr bool SMN_PRUNE_CONSENSUS_ENABLED = true;
    inline constexpr bool SMN_PRUNE_ELLIPSE_ENABLED = false;
    inline constexpr bool SMN_PRUNE_ISOLATION_ENABLED = true;
    inline constexpr bool SMN_PRUNE_SOR_ENABLED = true;

    // Geometric dome (behind-camera + floor/ceil-Y + max-scale). WORLD-space: verify
    // the COLMAP Y axis / scale match a standing person with feet near y=0.
    inline constexpr float SMN_PRUNE_BEHIND_TOLERANCE = -0.1f;
    inline constexpr float SMN_PRUNE_FLOOR_Y = -0.30f;
    inline constexpr float SMN_PRUNE_CEIL_Y = 2.6f;
    inline constexpr float SMN_PRUNE_MAX_SCALE_M = 0.7f;

    // Center vote (projected center in mask). Reverted to the old v4 single-pass
    // behavior — strict by center, few views to be eligible, no frame margin, no
    // depth — so hair-fringe streaks whose center lands off-mask are removed (the
    // lenient SMNV2 values 0.70/0.20/0.33 relied on the now-disabled footprint
    // passes and left those streaks behind the head).
    inline constexpr float SMN_PRUNE_CENTER_VOTE_RATIO = 0.85f;
    // Low visibility threshold ~= the old classic's absolute "min 3 views": a
    // floater seen from few cameras must still be EVALUATED (and voted out for
    // projecting off-mask), not kept as "not enough evidence". ~0.02 ≈ 3 views at
    // 150 cameras. Only affects low-visibility splats (floaters); real geometry is
    // seen in many views and unaffected.
    inline constexpr float SMN_PRUNE_CENTER_MIN_VIS_RATIO = 0.02f;
    inline constexpr float SMN_PRUNE_CENTER_BORDER_MARGIN = 0.0f;
    inline constexpr bool SMN_PRUNE_CENTER_DEPTH_FILTER = false;

    // Mask leakage (footprint boundary samples outside the mask).
    inline constexpr float SMN_PRUNE_LEAK_KEEP_THRESHOLD = 0.60f;
    inline constexpr float SMN_PRUNE_LEAK_PER_VIEW_FRACTION = 0.20f;
    inline constexpr float SMN_PRUNE_LEAK_MIN_VIS_RATIO = 0.10f;
    inline constexpr float SMN_PRUNE_LEAK_MIN_PIXEL_RADIUS = 1.0f;
    inline constexpr int SMN_PRUNE_LEAK_SAMPLE_POINTS = 8;
    inline constexpr int SMN_PRUNE_LEAK_DILATE_PX = 2;

    // Alpha consensus (fraction of 2D Gaussian mass inside the mask).
    inline constexpr float SMN_PRUNE_CONSENSUS_THRESHOLD = 0.60f;
    inline constexpr int SMN_PRUNE_CONSENSUS_MIN_VIS = 5;
    inline constexpr int SMN_PRUNE_CONSENSUS_GRID = 5;

    // Ellipse boundary (boundary points vs expanded mask).
    inline constexpr float SMN_PRUNE_ELLIPSE_EXPANSION_FRACTION = 0.01f;
    inline constexpr float SMN_PRUNE_ELLIPSE_NEG_VOTE_THRESHOLD = 0.10f;
    inline constexpr int SMN_PRUNE_ELLIPSE_MIN_CAMERAS = 3;

    // Isolation (kth-neighbor outlier vs global median).
    inline constexpr int SMN_PRUNE_ISOLATION_K = 8;
    inline constexpr int SMN_PRUNE_ISOLATION_KTH = 4;
    inline constexpr float SMN_PRUNE_ISOLATION_MULTIPLIER = 32.0f;

    // SOR (mean-kNN-distance outlier + connectivity guard). std_ratio 3.5 is
    // conservative on purpose — protects extremities (pointing hand, hair tips).
    inline constexpr int SMN_PRUNE_SOR_NEIGHBORS = 30;
    inline constexpr float SMN_PRUNE_SOR_STD_RATIO = 3.5f;
    inline constexpr int SMN_PRUNE_SOR_GUARD_KTH = 4;
    inline constexpr float SMN_PRUNE_SOR_GUARD_MULTIPLIER = 5.0f;

} // namespace lfs::training::smn
