/* SPDX-FileCopyrightText: 2026 SMN | Víctor M. Feliz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// =============================================================================
// SMN — Attention opacity penalty: fused single-pass kernel
// =============================================================================
//
// Computes the bidirectional opacity penalty in one CUDA pass instead of a
// chain of elementwise tensor ops. For each pixel it evaluates:
//
//   m_in  = mask * roi                 (inside indicator, roi optional)
//   m_out = (1 - mask) * roi           (outside indicator)
//   grad_alpha = (cout*m_out - cin*m_in) * (1/N) * couple
//   contrib    =  cin*(1-alpha)*m_in + cout*alpha*m_out
//
// and reduces `contrib` across all pixels into the scalar penalty loss:
//
//   penalty_loss = (Σ contrib) * (1/N) * couple
//
// where cin = w*INSIDE_WEIGHT, cout = w*OUTSIDE_WEIGHT (w = schedule*scale) and
// `couple` is the optional per-image coupling factor (photometric_loss + floor),
// passed as a device scalar so the whole thing stays sync-free.
//
// Raw-pointer API (no lfs::core::Tensor) so it lives in the LibTorch-free
// kernels library, like the other SMN/roi kernels.
// =============================================================================

#include <cuda_runtime.h>

namespace lfs::training::smn {

    void launch_attention_opacity_penalty(
        const float* alpha,        // [H*W] rendered alpha, CUDA
        const float* mask,         // [H*W] inside indicator in {0,1}, CUDA
        const float* roi,          // [H*W] crop ROI weight, CUDA, or nullptr
        int height, int width,
        float cin,                 // w * SMN_ATTENTION_PENALTY_INSIDE_WEIGHT
        float cout,                // w * SMN_ATTENTION_PENALTY_OUTSIDE_WEIGHT
        const float* couple,       // device [1] coupling factor, or nullptr (=> 1)
        float* grad_alpha,         // [H*W] output d(loss)/d(alpha), CUDA
        float* penalty_loss,       // device [1] output scalar penalty, CUDA
        cudaStream_t stream = nullptr);

} // namespace lfs::training::smn
