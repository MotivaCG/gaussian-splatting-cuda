/* SPDX-FileCopyrightText: 2026 SMN | Víctor M. Feliz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "smn_opacity.hpp"

#include <cmath>

namespace lfs::training::smn {

    void modify_opacity_pow(lfs::core::SplatData& model, const float value) {
        if (value == 1.0f) {
            return; // pow(x, 1) == x: no change
        }
        lfs::core::Tensor& raw = model.opacity_raw();
        if (!raw.is_valid() || raw.numel() == 0) {
            return;
        }
        // linear = sigmoid(raw); linear = linear^value; raw = logit(linear).
        raw.copy_(raw.sigmoid().pow(value).logit().contiguous());
    }

    void modify_opacity_mul(lfs::core::SplatData& model, const float value) {
        if (value == 0.0f) {
            return; // exp(0) == 1: no change
        }
        lfs::core::Tensor& raw = model.opacity_raw();
        if (!raw.is_valid() || raw.numel() == 0) {
            return;
        }
        // linear = sigmoid(raw); linear *= exp(value); raw = logit(linear).
        raw.copy_(raw.sigmoid().mul(std::exp(value)).logit().contiguous());
    }

} // namespace lfs::training::smn
