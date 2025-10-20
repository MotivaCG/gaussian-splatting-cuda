// frequency_map.hpp
#pragma once
#include <algorithm>
#include <torch/torch.h>
#include <vector>

namespace gs::img {

    namespace F = torch::nn::functional;

    /* Build a depthwise kernel [C,1,kH,kW] from a single 2D filter [kH,kW].
     * It will be used with groups=C so that each input channel is filtered independently. */
    inline torch::Tensor make_depthwise_kernel(const torch::Tensor& k2d, int64_t C, torch::Device dev) {
        TORCH_CHECK(k2d.dim() == 2, "k2d must be [kH,kW]");
        auto w = k2d.to(torch::kFloat32).to(dev).unsqueeze(0).unsqueeze(0); // [1,1,kH,kW]
        w = w.expand({C, 1, k2d.size(0), k2d.size(1)}).contiguous();        // [C,1,kH,kW]
        return w;
    }

    /* Create a normalized 1D Gaussian kernel (odd ksize). */
    inline torch::Tensor gaussian1d(float sigma, int ksize, torch::Device dev) {
        TORCH_CHECK(sigma > 0.0f, "sigma must be > 0");
        TORCH_CHECK(ksize % 2 == 1 && ksize >= 3, "ksize must be odd and >= 3");
        const int half = ksize / 2;
        auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(dev);
        auto x = torch::arange(-half, half + 1, opts);          // [-h..h]
        auto g = torch::exp(-0.5f * (x * x) / (sigma * sigma)); // unnormalized
        g = g / g.sum();                                        // normalize
        return g.contiguous();                                  // [ksize]
    }

    /* Separable Gaussian blur for [B,1,H,W], using reflect padding to mimic OpenCV-ish borders. */
    inline torch::Tensor gaussian_blur_single(torch::Tensor x, float sigma, int ksize) {
        TORCH_CHECK(x.dim() == 4 && x.size(1) == 1, "x must be [B,1,H,W]");
        auto dev = x.device();
        const int half = ksize / 2;

        // Build 1D kernels
        auto g = gaussian1d(sigma, ksize, dev); // [ksize]
        auto gy = g.view({1, 1, ksize, 1});     // vertical
        auto gx = g.view({1, 1, 1, ksize});     // horizontal

        // Pad reflect (top,bottom,left,right)
        F::PadFuncOptions pad_y({0, 0, half, half});
        pad_y.mode(torch::kReflect);
        F::PadFuncOptions pad_x({half, half, 0, 0});
        pad_x.mode(torch::kReflect);

        // Vertical pass
        x = F::pad(x, pad_y);
        x = F::conv2d(x, gy);

        // Horizontal pass
        x = F::pad(x, pad_x);
        x = F::conv2d(x, gx);

        return x;
    }

    /* Combined frequency map:
     *  - Per-channel Sobel magnitude and/or |Laplacian|
     *  - Sum across RGB channels, average if both were used
     *  - Optional invert, gamma 2.2, Gaussian blur with sigma=0.025*min(H,W)
     *  - Two normalizations to [0,1] (pre- and post-blur), then gamma 2.2
     * Input accepted: [3,H,W], [1,3,H,W], or [H,W,3] (float32 in [0,1] or uint8).
     * Output: [H,W] float32 in [0,1] on same device. */
    inline torch::Tensor compute_frequency_map_combined(
        torch::Tensor input,
        bool useSobel,
        bool useLaplacian,
        bool invertOutput) {
        torch::NoGradGuard ng;
        TORCH_CHECK(input.defined() && input.numel() > 0, "input is empty");
        TORCH_CHECK(useSobel || useLaplacian,
                    "both useSobel and useLaplacian are false — at least one must be true.");

        // Convert dtype to float32 in [0,1]
        if (input.dtype() == torch::kByte) {
            input = input.to(torch::kFloat32).div_(255.0f);
        } else if (input.dtype() != torch::kFloat32) {
            input = input.to(torch::kFloat32);
        }

        // Accept [H,W,3] → [3,H,W]
        if (input.dim() == 3 && input.size(0) != 3 && input.size(-1) == 3) {
            input = input.permute({2, 0, 1});
        }
        // Normalize to [1,3,H,W]
        if (input.dim() == 3) {
            TORCH_CHECK(input.size(0) == 3, "expected [3,H,W] after permute");
            input = input.unsqueeze(0);
        } else {
            TORCH_CHECK(input.dim() == 4, "expected [1,3,H,W] or [3,H,W]");
        }
        TORCH_CHECK(input.size(0) == 1 && input.size(1) == 3, "expect [1,3,H,W]");

        auto dev = input.device();
        const int H = static_cast<int>(input.size(2));
        const int W = static_cast<int>(input.size(3));

        // Accumulator [B=1,1,H,W]
        auto accum = torch::zeros({1, 1, H, W}, input.options());

        // Prepare kernels (depthwise per channel)
        torch::Tensor sobelX, sobelY, lap;
        if (useSobel) {
            // OpenCV-like Sobel 3x3
            auto kx = torch::tensor({{-1.f, 0.f, 1.f},
                                     {-2.f, 0.f, 2.f},
                                     {-1.f, 0.f, 1.f}},
                                    torch::kFloat32);
            auto ky = torch::tensor({{-1.f, -2.f, -1.f},
                                     {0.f, 0.f, 0.f},
                                     {1.f, 2.f, 1.f}},
                                    torch::kFloat32);
            sobelX = make_depthwise_kernel(kx, /*C=*/3, dev);
            sobelY = make_depthwise_kernel(ky, /*C=*/3, dev);
        }
        if (useLaplacian) {
            // 4-neighborhood Laplacian 3x3
            auto lk = torch::tensor({{0.f, 1.f, 0.f},
                                     {1.f, -4.f, 1.f},
                                     {0.f, 1.f, 0.f}},
                                    torch::kFloat32);
            lap = make_depthwise_kernel(lk, /*C=*/3, dev);
        }

        // Reflect pad by 1 px for 3x3 filters
        F::PadFuncOptions pad3({1, 1, 1, 1});
        pad3.mode(torch::kReflect);

        // Sobel branch
        if (useSobel) {
            auto x_pad = F::pad(input, pad3);
            auto gx = F::conv2d(x_pad, sobelX, F::Conv2dFuncOptions().groups(3)); // [1,3,H,W]
            auto gy = F::conv2d(x_pad, sobelY, F::Conv2dFuncOptions().groups(3)); // [1,3,H,W]
            auto mag = torch::sqrt(gx.mul(gx) + gy.mul(gy));                      // [1,3,H,W]
            auto mag_sum = mag.sum(1, /*keepdim=*/true);                          // [1,1,H,W]
            accum = accum + mag_sum;
        }

        // Laplacian branch
        if (useLaplacian) {
            auto x_pad = F::pad(input, pad3);
            auto l = F::conv2d(x_pad, lap, F::Conv2dFuncOptions().groups(3)); // [1,3,H,W]
            auto l_abs = l.abs();                                             // [1,3,H,W]
            auto l_sum = l_abs.sum(1, /*keepdim=*/true);                      // [1,1,H,W]
            accum = accum + l_sum;
        }

        // Average if both were used
        const int contribPerChannel = (useSobel ? 1 : 0) + (useLaplacian ? 1 : 0);
        if (contribPerChannel > 1)
            accum = accum * (1.0f / static_cast<float>(contribPerChannel));

        // Optional invert
        if (invertOutput)
            accum = 1.0f - accum;

        // Gamma 2.2
        accum = torch::pow(accum.clamp(0.0f, 1.0f), 2.2f);

        // Sigma/ksize proportional to min dim
        const int minDim = std::min(H, W);
        const float sigma = std::max(0.001f, minDim * 0.025f); // ~2.5% of min dim
        int ksize = static_cast<int>(std::ceil(sigma * 6.0f));
        if ((ksize & 1) == 0)
            ++ksize;
        ksize = std::max(3, ksize);

        // Normalize to [0,1] before blur
        auto a_min = std::get<0>(accum.aminmax());
        auto a_max = std::get<1>(accum.aminmax());
        auto denom = (a_max - a_min).clamp_min(1e-12f);
        accum = (accum - a_min) / denom;
        accum = accum.clamp_(0.0f, 1.0f);

        // Separable Gaussian blur on [1,1,H,W]
        accum = gaussian_blur_single(accum, sigma, ksize);

        // Normalize again, then gamma 2.2
        a_min = std::get<0>(accum.aminmax());
        a_max = std::get<1>(accum.aminmax());
        denom = (a_max - a_min).clamp_min(1e-12f);
        accum = (accum - a_min) / denom;
        accum = torch::pow(accum.clamp(0.0f, 1.0f), 2.2f);

        // To [H,W]
        auto out = accum.squeeze(0).squeeze(0);

        return out;
    }

    // Convenience wrapper for images already in CHW (as returned by cam->load_and_get_image).
    inline torch::Tensor compute_frequency_map_from_cam_image(
        const torch::Tensor& chwImage,
        bool useSobel,
        bool useLaplacian,
        bool invertOutput) {
        // cam->load_and_get_image returns [3,H,W] float32 in [0,1] on CUDA
        return compute_frequency_map_combined(chwImage, useSobel, useLaplacian, invertOutput);
    }

} // namespace gs::img
