/* SPDX-FileCopyrightText: 2026 SMN | Víctor M. Feliz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "smn_attention_penalty.hpp"

#include "smn_attention_constants.h"
#include "smn_attention_penalty_kernel.hpp"
#include "smn_opacity.hpp"

#include "core/logger.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "optimizer/adam_optimizer.hpp"
#include "strategies/istrategy.hpp"

#include <algorithm>
#include <cmath>

namespace lfs::training::smn {

    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::Tensor;

    namespace {

        // Collapse an optional [1,H,W] tensor to [H,W]; pass [H,W] through.
        [[nodiscard]] Tensor to_2d(const Tensor& t) {
            return (t.is_valid() && t.ndim() == 3) ? t.squeeze(0) : t;
        }

        // Convert a mask (bool/uint8 => {0,1}, float => as-is) to Float32.
        [[nodiscard]] Tensor mask_as_float(const Tensor& mask_2d) {
            return (mask_2d.dtype() == DataType::UInt8 || mask_2d.dtype() == DataType::Bool)
                       ? mask_2d.gt(0).to(DataType::Float32)
                       : mask_2d;
        }

        // Per-pixel darkness weight from the GT image (NOT the render), after
        // oldmode's make_darkness_weight. Emphasizes dark target regions in the
        // PHOTOMETRIC loss (dark hair / clothing reconstruct more accurately):
        //   w = 1 + (1 - brightness) * boost
        // brightness is Rec.601 luma (0.299 R + 0.587 G + 0.114 B) in [0,1].
        //
        // Oldmode also divided w by its mean to keep the global loss scale stable,
        // because it normalized the L1 term by the MASK sum. Here the darkness folds
        // into the fused-loss weight, which is normalized by the TOTAL weight sum, so
        // any global per-image scalar cancels in both loss and gradient — the mean
        // normalization would be a redundant reduction and is omitted. The per-pixel
        // dark emphasis (gradient ∝ darkness·mask) is identical either way.
        //
        // This is a photometric weight, never an opacity-penalty term, so it cannot
        // halo fine edges.
        [[nodiscard]] Tensor compute_darkness_weight(const Tensor& gt_image, const float boost) {
            constexpr float kR = 0.299f, kG = 0.587f, kB = 0.114f; // Rec.601 luma
            Tensor gt = gt_image.ndim() == 4 ? gt_image.squeeze(0) : gt_image;
            Tensor gt_f = gt.dtype() == DataType::UInt8
                              ? gt.to(DataType::Float32).mul(1.0f / 255.0f)
                              : gt;
            Tensor brightness;
            if (gt_f.ndim() == 3 && gt_f.shape()[0] == 3) {
                // channel-first [3,H,W]
                brightness = gt_f.slice(0, 0, 1).squeeze(0).mul(kR)
                                 .add(gt_f.slice(0, 1, 2).squeeze(0).mul(kG))
                                 .add(gt_f.slice(0, 2, 3).squeeze(0).mul(kB));
            } else if (gt_f.ndim() == 3 && gt_f.shape()[2] == 3) {
                // channel-last [H,W,3]
                brightness = gt_f.slice(2, 0, 1).squeeze(2).mul(kR)
                                 .add(gt_f.slice(2, 1, 2).squeeze(2).mul(kG))
                                 .add(gt_f.slice(2, 2, 3).squeeze(2).mul(kB));
            } else {
                brightness = gt_f.ndim() == 3 ? gt_f.squeeze(0) : gt_f;
            }
            // w = 1 + (1 - brightness) * boost = (1 + boost) - brightness * boost.
            // No mean-normalization: it cancels in the fused loss (see note above).
            return brightness.mul(-boost).add(1.0f + boost).contiguous();
        }

        // Soft photometric weight map: 1.0 inside the mask, SMN_ATTENTION_OUT_MASK_WEIGHT
        // outside. Optionally modulated by the crop-box ROI weight so pixels the crop
        // box excludes still drop out of the loss.
        //
        //   weight = OUT_W + mask_f * (1 - OUT_W)          [then * roi_weight]
        [[nodiscard]] Tensor build_soft_photometric_weight(
            const Tensor& mask_f,
            const Tensor& roi_weight_2d) {
            Tensor weight =
                mask_f.mul(1.0f - SMN_ATTENTION_OUT_MASK_WEIGHT).add(SMN_ATTENTION_OUT_MASK_WEIGHT);
            if (roi_weight_2d.is_valid() && roi_weight_2d.numel() > 0) {
                weight = weight.mul(roi_weight_2d);
            }
            return weight.contiguous();
        }

    } // namespace

    Tensor dilate_mask(const Tensor& mask, const int radius) {
        if (radius <= 0 || !mask.is_valid() || mask.numel() == 0) {
            return mask;
        }
        const Tensor mf = mask_as_float(to_2d(mask)).contiguous(); // [H,W] Float32
        const int H = static_cast<int>(mf.shape()[0]);
        const int W = static_cast<int>(mf.shape()[1]);
        // Dilation = max filter over a (2r+1)^2 window (stride 1, padding r).
        const Tensor pooled = mf.reshape({1, 1, H, W}).max_pool2d(2 * radius + 1, 1, radius);
        return pooled.reshape({H, W}).contiguous();
    }

    float attention_penalty_schedule_weight(const int iteration, const int total_iterations) {
        if (total_iterations <= 0) {
            return 0.0f;
        }
        const float t = static_cast<float>(iteration) / static_cast<float>(total_iterations);

        // Warmup: no opacity constraint while coarse geometry is still forming.
        if (t < SMN_ATTENTION_PENALTY_WARMUP_FRACTION) {
            return 0.0f;
        }
        // Full strength band.
        if (t <= SMN_ATTENTION_PENALTY_FULL_FRACTION) {
            return 1.0f;
        }
        // Linear release of the constraint towards the end of training.
        if (t < SMN_ATTENTION_PENALTY_DECAY_END_FRACTION) {
            const float span =
                SMN_ATTENTION_PENALTY_DECAY_END_FRACTION - SMN_ATTENTION_PENALTY_FULL_FRACTION;
            const float progress = (t - SMN_ATTENTION_PENALTY_FULL_FRACTION) / span;
            return std::clamp(1.0f - progress, 0.0f, 1.0f);
        }
        // Final refinement is unconstrained.
        return 0.0f;
    }

    std::expected<AttentionLossResult, std::string> compute_attention_photometric_loss(
        const Tensor& corrected,
        const Tensor& gt_image,
        const Tensor& mask,
        const Tensor& roi_weight,
        const Tensor& alpha,
        const lfs::core::param::OptimizationParameters& opt_params,
        const Tensor& raw_rendered,
        const int iteration,
        lfs::training::kernels::LossWorkspaceArena& workspace_arena) {

        const bool has_mask = mask.is_valid() && mask.numel() > 0;
        const Tensor roi_weight_2d = to_2d(roi_weight);
        const Tensor mask_f = has_mask ? mask_as_float(to_2d(mask)) : Tensor{};

        // Warmup phase: like oldmode, the first SMN_ATTENTION_PENALTY_WARMUP_FRACTION
        // of training uses a FLAT photometric weight (no out-of-mask down-weighting)
        // and no opacity penalty, so the whole scene — including the background right
        // behind hair — forms solidly before we start emphasizing the masked object.
        // Starting the soft weighting from iteration 0 under-trains that background
        // and halos fine edges (hair).
        const int total_iters = static_cast<int>(opt_params.iterations);
        const float progress = total_iters > 0
                                   ? static_cast<float>(iteration) / static_cast<float>(total_iters)
                                   : 1.0f;
        const bool in_warmup = progress < SMN_ATTENTION_PENALTY_WARMUP_FRACTION;

        // ------------------------------------------------------------------
        // 1) Soft-weighted photometric loss via the engine's fused kernels.
        // ------------------------------------------------------------------
        // After warmup with a mask: 1.0 inside, SMN_ATTENTION_OUT_MASK_WEIGHT outside.
        // During warmup, or without a usable mask, degrade to the crop ROI weight, or
        // to a flat weight so the fused kernel reproduces the ordinary photometric loss.
        //
        // The PRIORITY mask (full-weight region) is dilated a few pixels beyond the
        // tight mask so the border band (hair) gets full reconstruction priority.
        // The opacity penalty below keeps the TIGHT mask_f (no alpha push in the band).
        const Tensor mask_priority =
            (has_mask && SMN_ATTENTION_PRIORITY_DILATION_PX > 0)
                ? dilate_mask(mask_f, SMN_ATTENTION_PRIORITY_DILATION_PX)
                : mask_f;

        Tensor photometric_weight;
        if (has_mask && !in_warmup) {
            photometric_weight = build_soft_photometric_weight(mask_priority, roi_weight_2d);
        } else if (roi_weight_2d.is_valid() && roi_weight_2d.numel() > 0) {
            photometric_weight = roi_weight_2d.contiguous();
        } else {
            const auto& sh = gt_image.shape();
            const size_t rank = gt_image.ndim();
            const lfs::core::TensorShape spatial_shape({sh[rank - 2], sh[rank - 1]});
            photometric_weight = Tensor::full(spatial_shape, 1.0f, gt_image.device());
        }

        // Darkness boost (oldmode): emphasize dark GT regions in the photometric
        // loss so dark hair / clothing reconstruct more accurately. Applied to the
        // whole image, every iteration (no schedule), and independent of the opacity
        // penalty (the two are NOT mutually exclusive here, unlike oldmode). It is a
        // photometric weight, never an opacity term, so it cannot halo edges.
        //
        // NOTE: oldmode weights ONLY the L1 term with darkness; this engine's fused
        // L1+SSIM kernel takes a single per-pixel weight, so folding it here also
        // weights SSIM. Because the darkness weight is mean-normalized (mean == 1),
        // this is a minor, scale-neutral deviation (it only re-emphasizes the ~20%
        // SSIM term spatially toward dark pixels). A bit-exact L1-only version would
        // need two separate fused passes (≈2x loss cost); folded here for speed.
        if (SMN_ATTENTION_DARKNESS_BOOST > 0.0f) {
            const Tensor darkness = compute_darkness_weight(gt_image, SMN_ATTENTION_DARKNESS_BOOST);
            photometric_weight = photometric_weight.mul(darkness).contiguous();
        }

        const bool use_decoupled =
            raw_rendered.is_valid() && raw_rendered.numel() > 0 && opt_params.lambda_dssim > 0.0f;

        Tensor loss, grad_corrected, grad_raw;
        if (use_decoupled) {
            auto& decoupled_ws = workspace_arena.masked_decoupled();
            auto [loss_tensor, ctx] = lfs::training::kernels::masked_decoupled_fused_l1_ssim_forward(
                corrected, raw_rendered, gt_image, photometric_weight, opt_params.lambda_dssim, decoupled_ws);
            auto grads = lfs::training::kernels::masked_decoupled_fused_l1_ssim_backward(ctx, decoupled_ws);
            loss = loss_tensor;
            grad_corrected = grads.grad_corrected;
            grad_raw = grads.grad_raw;
            if (grad_corrected.ndim() == 4 && corrected.ndim() == 3) {
                grad_corrected = grad_corrected.squeeze(0);
            }
            if (grad_raw.ndim() == 4 && corrected.ndim() == 3) {
                grad_raw = grad_raw.squeeze(0);
            }
        } else {
            auto& fused_ws = workspace_arena.masked_fused();
            auto [loss_tensor, ctx] = lfs::training::kernels::masked_fused_l1_ssim_forward(
                corrected, gt_image, photometric_weight, opt_params.lambda_dssim, fused_ws);
            loss = loss_tensor;
            grad_corrected = lfs::training::kernels::masked_fused_l1_ssim_backward(ctx, fused_ws);
            if (grad_corrected.ndim() == 4 && corrected.ndim() == 3) {
                grad_corrected = grad_corrected.squeeze(0);
            }
        }

        // ------------------------------------------------------------------
        // 2) Scheduled bidirectional opacity penalty (additive + grad_alpha).
        // ------------------------------------------------------------------
        //
        // penalty = w * SCALE * ( IN  * mean( (1-alpha) * inside )
        //                       + OUT * mean(  alpha     * outside ) )
        //
        // grad wrt alpha:
        //   d/dalpha = w * SCALE / N * ( OUT * outside - IN * inside )
        //
        // where inside = mask (optionally * roi), outside = (1-mask) (optionally * roi).
        // Evaluated by a single fused CUDA kernel (grad_alpha + reduced loss in one pass).
        Tensor grad_alpha;
        const float schedule_w =
            attention_penalty_schedule_weight(iteration, static_cast<int>(opt_params.iterations));
        if (schedule_w > 0.0f && has_mask && alpha.is_valid() && alpha.numel() > 0) {
            const Tensor alpha_2d = to_2d(alpha).contiguous();      // [H,W] inside indicator source
            const Tensor mask_in = mask_f.contiguous();             // [H,W] mask in {0,1}
            const bool has_roi = roi_weight_2d.is_valid() && roi_weight_2d.numel() > 0;
            const Tensor roi_c = has_roi ? roi_weight_2d.contiguous() : Tensor{};

            const float w = schedule_w * SMN_ATTENTION_PENALTY_SCALE;
            const float cin = w * SMN_ATTENTION_PENALTY_INSIDE_WEIGHT;
            const float cout = w * SMN_ATTENTION_PENALTY_OUTSIDE_WEIGHT;

            // Coupling factor (photometric_loss + floor) as a device scalar, captured
            // BEFORE folding the penalty into `loss` to avoid self-reference.
            const Tensor couple = SMN_ATTENTION_PENALTY_COUPLE_TO_LOSS
                                      ? loss.add(SMN_ATTENTION_PENALTY_LOSS_FLOOR).contiguous()
                                      : Tensor{};

            const int H = static_cast<int>(alpha_2d.shape()[0]);
            const int W = static_cast<int>(alpha_2d.shape()[1]);
            grad_alpha = Tensor::empty(
                lfs::core::TensorShape({static_cast<size_t>(H), static_cast<size_t>(W)}),
                alpha_2d.device());
            Tensor penalty_loss = Tensor::empty(lfs::core::TensorShape({1}), loss.device());

            const cudaStream_t stream = lfs::core::getCurrentCUDAStream();
            alpha_2d.sync_to_stream(stream);
            mask_in.sync_to_stream(stream);
            if (has_roi) {
                roi_c.sync_to_stream(stream);
            }
            if (couple.is_valid()) {
                couple.sync_to_stream(stream);
            }
            grad_alpha.set_stream(stream);
            penalty_loss.set_stream(stream);

            launch_attention_opacity_penalty(
                alpha_2d.ptr<float>(),
                mask_in.ptr<float>(),
                has_roi ? roi_c.ptr<float>() : nullptr,
                H, W, cin, cout,
                couple.is_valid() ? couple.ptr<float>() : nullptr,
                grad_alpha.ptr<float>(),
                penalty_loss.ptr<float>(),
                stream);

            loss = loss.add(penalty_loss);
        }

        return AttentionLossResult{
            .loss = loss,
            .grad_corrected = grad_corrected,
            .grad_raw = grad_raw,
            .grad_alpha = grad_alpha};
    }

    void attention_on_optimizer_step(
        lfs::training::IStrategy& strategy,
        const int iteration,
        const int total_iterations,
        const lfs::core::param::MaskMode mode) {

        if (!is_attention_mask_mode(mode) || total_iterations <= 0) {
            return; // attention-only behavior
        }

        // Scheduled discrete opacity kicks: at each listed fraction, apply the entry's
        // opacity op (pow or mul) once, then reset the opacity optimizer so stale
        // momentum does not immediately undo it. See smn_attention_constants.h section 4.
        if (SMN_OPACITY_KICK_ENABLED) {
            for (const OpacityKick& kick : SMN_OPACITY_KICKS) {
                const int kick_iter = static_cast<int>(std::lround(
                    kick.fraction * static_cast<float>(total_iterations)));
                if (iteration != kick_iter) {
                    continue;
                }
                const bool is_mul = kick.op == OpacityKickOp::Mul;
                if (is_mul) {
                    modify_opacity_mul(strategy.get_model(), kick.value);
                } else {
                    modify_opacity_pow(strategy.get_model(), kick.value);
                }
                if (SMN_OPACITY_KICK_RESET_OPTIMIZER) {
                    strategy.get_optimizer().reset_state(ParamType::Opacity);
                }
                LOG_INFO("[SMN opacity kick] {}({:.3f}) at {:.0f}% on {} splats{}",
                         is_mul ? "mul" : "pow",
                         kick.value,
                         kick.fraction * 100.0f,
                         strategy.get_model().size(),
                         SMN_OPACITY_KICK_RESET_OPTIMIZER ? " (opacity optimizer reset)" : "");
            }
        }
    }

} // namespace lfs::training::smn
