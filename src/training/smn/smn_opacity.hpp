/* SPDX-FileCopyrightText: 2026 SMN | Víctor M. Feliz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/splat_data.hpp"

namespace lfs::training::smn {

    // Both operate on the LINEAR opacity (sigmoid of opacity_raw) and map back to
    // raw (logit) space, in place, for ALL splats. Generic primitives: they rewrite
    // opacity_raw only and do NOT touch the optimizer moments. They are compatible
    // (a caller may apply both in sequence).

    // Gamma-adjust: opacity_raw <- logit( sigmoid(opacity_raw) ^ value ).
    // Since linear opacity is in (0,1) and value > 0 the result stays in (0,1) with
    // no clamp; value < 1 raises opacity toward 1 (strongest on low opacities),
    // value > 1 lowers it, value == 1 is a no-op.
    void modify_opacity_pow(lfs::core::SplatData& model, float value);

    // Scale (SuperSplat's transparency): opacity_raw <- logit( sigmoid(opacity_raw)
    // * exp(value) ). value == 0 is a no-op, value > 0 raises opacity, value < 0
    // lowers it. logit() eps-clamps so factor > 1 saturates toward opaque.
    void modify_opacity_mul(lfs::core::SplatData& model, float value);

} // namespace lfs::training::smn
