#include "core/trainer.hpp"
#include "core/fast_rasterizer.hpp"
#include "core/rasterizer.hpp"
#include "kernels/fused_ssim.cuh"
#include <ATen/cuda/CUDAEvent.h>
#include <atomic>
#include <chrono>
#include <cuda_runtime.h>
#include <expected>
#include <memory>
#include <numeric>
#include <print>
#include "core/rasterizer_autograd.hpp"

namespace gs {

    std::expected<void, std::string> Trainer::initialize_bilateral_grid() {
        if (!params_.optimization.use_bilateral_grid) {
            return {};
        }

        try {
            bilateral_grid_ = std::make_unique<gs::BilateralGrid>(
                train_dataset_size_,
                params_.optimization.bilateral_grid_X,
                params_.optimization.bilateral_grid_Y,
                params_.optimization.bilateral_grid_W);

            bilateral_grid_optimizer_ = std::make_unique<torch::optim::Adam>(
                std::vector<torch::Tensor>{bilateral_grid_->parameters()},
                torch::optim::AdamOptions(params_.optimization.bilateral_grid_lr)
                    .eps(1e-15));

            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to initialize bilateral grid: {}", e.what()));
        }
    }

    std::expected<torch::Tensor, std::string> Trainer::compute_photometric_loss(
        const RenderOutput& render_output,
        const torch::Tensor& gt_image,
        const SplatData& splatData,
        const param::OptimizationParameters& opt_params) {

        try {
            // Ensure images have same dimensions
            torch::Tensor rendered = render_output.image;
            torch::Tensor gt = gt_image;

            // Ensure both tensors are 4D (batch, height, width, channels)
            rendered = rendered.dim() == 3 ? rendered.unsqueeze(0) : rendered;
            gt = gt.dim() == 3 ? gt.unsqueeze(0) : gt;

            TORCH_CHECK(rendered.sizes() == gt.sizes(),
                        "ERROR: size mismatch – rendered ", rendered.sizes(),
                        " vs. ground truth ", gt.sizes());

            // Base loss: L1 + SSIM
            auto l1_loss = torch::l1_loss(rendered, gt);
            auto ssim_loss = 1.f - fused_ssim(rendered, gt, "valid", /*train=*/true);
            torch::Tensor loss = (1.f - opt_params.lambda_dssim) * l1_loss +
                                 opt_params.lambda_dssim * ssim_loss;
            return loss;
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error computing photometric loss: {}", e.what()));
        }
    }
    // Place this new static function inside trainer.cpp

    /**
     * @brief Applies a penalty based on the final rendered alpha channel of the image.
     *
     * This method penalizes pixels inside the mask that are semi-transparent and
     * pixels outside the mask that have any opacity. It's a more direct approach
     * than the splat-based penalty.
     *
     * @param base_loss The pre-computed photometric loss.
     * @param rendered_alpha The final accumulated alpha channel from the rasterizer [H, W].
     * @param weights The float weight map, where values > 0.5 define the ROI.
     * @param w The global weight multiplier for this penalty.
     * @param kIn The weight for penalizing transparency INSIDE the mask.
     * @param kOut The weight for penalizing opacity OUTSIDE the mask.
     * @return torch::Tensor The loss with the pixel-based opacity penalty added.
     */
    static torch::Tensor pixelBasedOpacityPenalty(const torch::Tensor& base_loss,
                                                  const torch::Tensor& rendered_alpha,
                                                  const torch::Tensor& weights,
                                                  const float w) {
        
        const float kOut = 0.333f;
        const float kIn =  0.667f;

        // Convert the float weight map to a clean boolean mask
        auto bool_mask = (weights > 0.5f);

        // 1. Penalty for being transparent INSIDE the mask
        // (1.0 - alpha) is high where pixels are transparent.
        // We multiply by the boolean mask to only consider pixels inside the ROI.
        auto inside_penalty_map = (1.0f - rendered_alpha) * bool_mask;

        // 2. Penalty for being opaque OUTSIDE the mask
        // `alpha` is high where pixels are opaque.
        // We multiply by the inverted mask to only consider pixels outside the ROI.
        auto outside_penalty_map = rendered_alpha * (~bool_mask);

        // 3. Combine and normalize the penalties
        // We sum the penalty over all pixels and normalize by the total number of pixels.
        float num_pixels = static_cast<float>(rendered_alpha.numel());
        auto total_penalty = w * (kIn * inside_penalty_map.sum() + kOut * outside_penalty_map.sum()) / num_pixels;

        // 4. Add the penalty to the base loss
        // For pixel-based losses, adding is often more stable than multiplying.
        //return base_loss + 0.01 * total_penalty;
        
        return (base_loss * (1.0f + total_penalty)) + (1e-2f * total_penalty);
    }

    /**
     * @brief Applies a penalty to guide splat opacity based on an attention mask.
     * @param base_loss The pre-computed photometric loss.
     * @param out The RenderOutput from the rasterizer.
     * @param weights The float weight map, where values > 0.5 define the ROI.
     * @param opacities_alpha The opacity alpha values (0-1) from get_opacity().
     * @param w The global weight multiplier for this penalty.
     */
    static torch::Tensor outsideMaskOpacityPenalty(const torch::Tensor& base_loss,
                                                   const RenderOutput& out,
                                                   const torch::Tensor& weights,
                                                   const torch::Tensor& opacities_alpha,
                                                   float w) {
        if (w == 0.0f) {
            return base_loss;
        }
        if (!weights.defined() || !out.means2d.defined() || !out.radii.defined()) {
            return base_loss;
        }

        torch::Tensor visible_mask = (out.radii > 0.0f);
        if (!visible_mask.any().item<bool>()) {
            return base_loss;
        }
        auto visible_indices = visible_mask.nonzero().squeeze(-1);
        auto xy = out.means2d.index({visible_indices});

        auto alpha = opacities_alpha.index({visible_indices});

        const int W = out.width, H = out.height;
        auto x = torch::round(xy.select(1, 0)).to(torch::kLong).clamp(0, W - 1);
        auto y = torch::round(xy.select(1, 1)).to(torch::kLong).clamp(0, H - 1);
        auto linear_indices = y * W + x;

        auto bool_mask = (weights > 0.5f);
        auto is_in_mask = bool_mask.to(torch::kFloat32).flatten();
        auto is_in = is_in_mask.index({linear_indices});
        auto is_out = 1.0f - is_in;

        const float kOut = 2.0f;
        const float kIn = 0.02f;
        auto outside_penalty = alpha * is_out;
        auto inside_penalty = (1.0f - alpha) * is_in;

        auto combined_penalty = w * (kOut * outside_penalty + kIn * inside_penalty);
        auto mean_penalty = combined_penalty.mean();
        
        return (base_loss * (1.0f + mean_penalty)) + (1e-5f*mean_penalty);
    }

    std::expected<torch::Tensor, std::string> Trainer::compute_photometric_loss(const RenderOutput& render_output,
                                                    const torch::Tensor& gt_image,
                                                    const torch::Tensor& weights,
                                                    const float outOfMaskAlphaPenalty,
                                                    const SplatData& splatData,
                                                    const param::OptimizationParameters& opt_params) {

        if (!weights.defined() || weights.numel() == 0) {
            // fallback to the previous mode
            return compute_photometric_loss(render_output, gt_image, splatData, opt_params);
        }

        try {
            // Ensure images have same dimensions
            torch::Tensor rendered = render_output.image;
            torch::Tensor gt = gt_image;

            // Ensure both tensors are 4D (batch, height, width, channels)
            rendered = rendered.dim() == 3 ? rendered.unsqueeze(0) : rendered;
            gt = gt.dim() == 3 ? gt.unsqueeze(0) : gt;

            const int Height = rendered.size(2);
            const int Width = rendered.size(3);
            TORCH_CHECK(rendered.sizes() == gt.sizes(), "ERROR: size mismatch - rendered ", rendered.sizes(), " vs. ground truth ", gt.sizes());
            TORCH_CHECK((Height == weights.size(1) && Width == weights.size(2)),
                        "ERROR: size mismatch - rendered ", rendered.sizes(), " vs. mask ", weights.sizes());

            torch::Tensor W = weights;

            // Pixel-wise L1 map and weighted L1 loss
            auto l1_map = torch::abs(rendered - gt).mean(/*dim=*/1); // Resultado: [B, H, W]
            auto wSum = W.sum().clamp_min(1e-6f);
            auto l1_loss = (l1_map * W).sum() / wSum;

            // 2) pixel-wise SSIM map and weighted SSIM loss
            
            // Compute the SSIM map. Using "valid" padding results in a smaller map.
            auto ssim_map = fused_ssim_map(rendered, gt, "valid", /*train=*/true);

            // Manually crop the weight map `W` to match the `ssim_map` dimensions.
            namespace I = torch::indexing;

            // Get original and target dimensions.
            const int orig_h = W.size(-2);
            const int orig_w = W.size(-1);
            const int target_h = ssim_map.size(-2);
            const int target_w = ssim_map.size(-1);

            // Calculate offsets for a centered crop.
            const int crop_h_start = (orig_h - target_h) / 2;
            const int crop_w_start = (orig_w - target_w) / 2;

            // Apply the crop using tensor slicing.
            torch::Tensor W_cropped = W.index({I::Slice(),
                                                I::Slice(crop_h_start, crop_h_start + target_h),
                                                I::Slice(crop_w_start, crop_w_start + target_w)});
            // Compute the weighted SSIM loss.
            auto ssim_loss_map = 1.0f - ssim_map;

            // Normalize by the sum of the cropped weights for correct loss scaling.
            auto W_cropped_sum = W_cropped.sum().clamp_min(1e-6f);
            auto ssim_loss = (ssim_loss_map * W_cropped).sum() / W_cropped_sum;

            // 3) combined loss
            auto loss = (1.0f - opt_params.lambda_dssim) * l1_loss + opt_params.lambda_dssim * ssim_loss;
            
        
            if (outOfMaskAlphaPenalty > 0) {
                if (!render_output.image.defined() || render_output.image.numel() == 0) {
                    printf("Image failed!\n");
                } else if (!render_output.alpha.defined() || render_output.alpha.numel() == 0) {
                    //printf("Alpha failed!\n");
                }
                else {
                    /* auto opacity = splatData.get_opacity();
                    loss = outsideMaskOpacityPenalty(loss,
                                                    render_output,
                                                    weights,
                                                    opacity,
                                                    outOfMaskAlphaPenalty);*/

                    loss = pixelBasedOpacityPenalty(loss,
                                                    render_output.alpha.squeeze(0), // Squeeze to [H, W] if needed
                                                    weights,
                                                    outOfMaskAlphaPenalty);
                }
            }
        

            return loss;
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error computing photometric loss: {}", e.what()));
        }
    }
    

    

    // Gently boost opacity for Gaussians that are inside the mask in at least
    // 'min_visible_views' visible views.
    // Policy: alpha_new = clamp( 2 * alpha_old, alpha_old, alpha_min ).
    //
    // Notes:
    // - Counts ONLY views where the Gaussian is visible (positive projected radii).
    // - Uses explicit index arrays (no boolean mask indexing) to avoid broadcasting traps.
    // - Flattens raw opacity to 1D to ensure shape consistency.
    // - Never decreases opacity; caps at 'alpha_min' (< 1.0 to keep finite logits).
    void Trainer::maybe_alpha_boost(const float alpha_min, const int min_visible_views) {
        return;
        const float min_support_ratio = 0.0f; // >= 0.0 => at least one inside vote across visible views
        const float boost_value = 3.0f;

        const float alpha_threshold = 0.0f; //ignore splats behind this level

        SplatData& model = strategy_->get_model();
        const int64_t N = model.get_means().size(0);
        if (N <= 0)
            return;

        // Accumulators: visible-only votes
        auto pos = torch::zeros({N}, torch::kInt32).to(torch::kCUDA);
        auto tot = torch::zeros({N}, torch::kInt32).to(torch::kCUDA);

        // Data loader (sequential, batch=1)
        auto loader = torch::data::make_data_loader(
            *train_dataset_,
            torch::data::samplers::SequentialSampler(train_dataset_->size().value()),
            torch::data::DataLoaderOptions().batch_size(1).workers(4));

        // Model tensors
        auto means3D = model.get_means();      // [N,3] CUDA
        auto scales = model.get_scaling();     // [N,3] CUDA
        auto rotations = model.get_rotation(); // [N,4] CUDA
        auto opacities = model.get_opacity();  // [N] or [N,1] (unused here)
        if (opacities.defined() && opacities.dim() == 2 && opacities.size(-1) == 1)
            opacities = opacities.squeeze(-1);

        // Projection constants
        const float eps2d = 0.3f, near_p = 0.01f, far_p = 10000.0f, radius_clip = 0.0f, scaling_mod = 1.0f;

        // Tally visible-only inside votes
        for (auto& batch : *loader) {
            auto cam_data = batch[0].data;
            Camera* cam = cam_data.camera;
            torch::Tensor float_weight_map = cam_data.attentionMask;
            if (!cam || !float_weight_map.defined())
                continue;

            // CPU bool mask [H,W]
            auto m3 = (float_weight_map > 0.5f);
            auto mask = (m3.dim() == 3 && m3.size(0) == 1) ? m3.squeeze(0) : m3;
            if (mask.dim() != 2)
                continue;
            mask = mask.contiguous();

            const int H = (int)mask.size(0);
            const int W = (int)mask.size(1);

            // Camera tensors (CUDA)
            auto view = cam->world_view_transform().to(torch::kCUDA);
            auto K = cam->K().to(torch::kCUDA);
            const int Wimg = (int)cam->image_width();
            const int Himg = (int)cam->image_height();

            // Projection-only (fast)
            auto settings = torch::tensor({(float)Wimg, (float)Himg, eps2d, near_p, far_p, radius_clip, scaling_mod},
                                          torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
            auto proj = ProjectionFunction::apply(means3D, rotations, scales, opacities, view, K, settings);

            torch::Tensor radii2 = proj[0];  // [1,N,2] or [N,2]
            torch::Tensor means2d = proj[1]; // [1,N,2] or [N,2]
            if (!radii2.defined() || !means2d.defined())
                continue;
            if (radii2.dim() == 3 && radii2.size(0) == 1)
                radii2 = radii2.squeeze(0);
            if (means2d.dim() == 3 && means2d.size(0) == 1)
                means2d = means2d.squeeze(0);

            // Visible-only splats (positive projected radii)
            auto visible = (radii2 > 0.0f).all(-1); // [N] bool CUDA
            if (!visible.any().item<bool>())
                continue;
            auto vidx = visible.nonzero().squeeze(-1); // [M] long CUDA

            // Pixel coords (on CPU) for mask sampling
            auto xy_cuda = means2d.index({vidx}); // [M,2] CUDA
            auto xy = xy_cuda.detach().to(torch::kCPU);
            auto x = torch::round(xy.select(1, 0)).to(torch::kLong).clamp(0, W - 1);
            auto y = torch::round(xy.select(1, 1)).to(torch::kLong).clamp(0, H - 1);
            auto lin = y * W + x; // [M] CPU long

            // Inside votes
            auto inside_cpu = mask.flatten().index({lin});                   // [M] bool CPU
            auto inside_i32 = inside_cpu.to(torch::kInt32).to(torch::kCUDA); // [M] int32 CUDA

            pos.index_add_(0, vidx, inside_i32);
            tot.index_add_(0, vidx, torch::ones_like(inside_i32, torch::kInt32));
        }

        // Decide eligibility (visible-only ratios) 
        
        // Read/prepare opacity logits as a flat 1D tensor
        torch::NoGradGuard no_grad;
        auto raw = model.opacity_raw(); // logits; can be [N] or [N,1] depending on impl
        if (raw.dim() == 2 && raw.size(-1) == 1)
            raw = raw.squeeze(-1);
        if (raw.dim() != 1)
            raw = raw.view({-1}); // flatten to [N]
        auto cur_alpha = torch::sigmoid(raw);

        auto tot_f = tot.to(torch::kFloat32).clamp_min(1.0f);
        auto ratio = pos.to(torch::kFloat32) / tot_f;                                   // [N] float
        auto eligible_mask = (tot >= min_visible_views) & (ratio >= min_support_ratio) & (cur_alpha > alpha_threshold); // [N] bool


        const float eps = 1e-6f;
        const float cap = std::min(alpha_min, 1.0f - eps);

        // Further restrict to those below the cap (avoid needless writes)
        auto below_cap = (cur_alpha < cap);          // [N] bool
        auto final_mask = eligible_mask & below_cap; // [N] bool

        auto idx = final_mask.nonzero().squeeze(-1); // [K] long
        if (idx.numel() == 0) {
            std::cout << "[Trainer] Alpha boost: no eligible splats.\n";
            return;
        }

        // Compute target alpha = clamp(2*a, a, cap) on the selected indices
        auto a = cur_alpha.index({idx}); // [K]
        auto doubled = a * boost_value;
        auto target = torch::min(doubled, torch::tensor(cap, a.options()));
        target = torch::max(target, a).clamp(eps, cap); // never decrease; keep finite logits

        // Write back logits (use max to be extra safe numerically)
        auto new_logit = torch::logit(target); // [K]
        auto old_logit = raw.index({idx});
        auto write = torch::max(old_logit, new_logit); // monotonicity guard
        raw.index_put_({idx}, write);

        std::cout << "[Trainer] Alpha boost (x" << boost_value << " capped at " << alpha_min
                  << "): updated splats\n";
    }

    void Trainer::contextual_alpha_boost_once(float target_alpha, int min_views) {
        namespace F = torch::nn::functional;
        torch::NoGradGuard no_grad;

        // ---------------- Tunables (safe defaults) ----------------
        const int core_erode_px = 3;             // core erosion to stay away from boundary
        const int edge_band_px = 2;              // width of inner edge ring (mask \ erode(mask,edge_band_px))
        const float support_thr = 0.60f;         // fraction of visible views with center in core
        const int sample_points = 8;             // ellipse sampling (NESW + diagonals)
        const float per_view_leak_frac = 0.125f; // view leaks if >12.5% sampled points outside (1 of 8)
        const float no_leak_thr = 0.90f;         // fraction of center-in views that must be no-leak
        const float edge_max_ratio = 0.20f;      // max fraction of views near boundary
        const float depth_zscore_max = 1.5f;     // mean|zscore| across views (if depth available)
        const float eps = 1e-6f;
        const float cap_alpha = std::min(target_alpha, 1.0f - 1e-6f); // keep logits finite
        const float boost_soft_cap = 0.95f;                           // safety cap after confidence scaling
        const float rmax_px = 6.0f;                                   // reject very large projected footprints
        const float rmin_px = 1.0f;                                   // reject needle-like footprints
        const float kappa_max = 3.0f;                                 // reject highly elongated footprints (rmax/rmin)

        // Ellipse sampling directions
        std::vector<std::array<float, 2>> dirs = {{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}};
        if (sample_points >= 8) {
            const float s = 0.70710678f; // 1/sqrt(2)
            dirs.push_back({s, s});
            dirs.push_back({-s, s});
            dirs.push_back({s, -s});
            dirs.push_back({-s, -s});
        }
        const int P = (int)dirs.size();

        // ---------------- Model state ----------------
        SplatData& model = strategy_->get_model();
        const int64_t N = model.get_means().size(0);
        if (N <= 0)
            return;

        auto means3D = model.get_means();      // [N,3] CUDA
        auto scales = model.get_scaling();     // [N,3] CUDA
        auto rotations = model.get_rotation(); // [N,4] CUDA
        auto opacities = model.get_opacity();  // [N] or [N,1] CUDA (unused in projection)
        if (opacities.defined() && opacities.dim() == 2 && opacities.size(-1) == 1)
            opacities = opacities.squeeze(-1);

        // Accumulators (CUDA)
        auto views_vis = torch::zeros({N}, torch::kInt32).cuda();       // visible views
        auto center_in_pos = torch::zeros({N}, torch::kInt32).cuda();   // center in core
        auto center_in_tot = torch::zeros({N}, torch::kInt32).cuda();   // == visible views considered
        auto noleak_pos = torch::zeros({N}, torch::kInt32).cuda();      // no-leak among center-in
        auto noleak_tot = torch::zeros({N}, torch::kInt32).cuda();      // center-in views
        auto edge_hits = torch::zeros({N}, torch::kInt32).cuda();       // center near boundary
        auto depth_abs_sum = torch::zeros({N}, torch::kFloat32).cuda(); // sum|zscore|
        auto depth_cnt = torch::zeros({N}, torch::kInt32).cuda();       // views with depth info
        auto bad_fp_flag = torch::zeros({N}, torch::kInt32).cuda();     // large/needle/anisotropic seen

        // Projection constants (must match your rasterizer)
        const float eps2d = 0.3f, near_p = 0.01f, far_p = 10000.0f, radius_clip = 0.0f, scaling_mod = 1.0f;

        // ---------------- Dataset sweep ----------------
        auto loader = torch::data::make_data_loader(
            *train_dataset_,
            torch::data::samplers::SequentialSampler(train_dataset_->size().value()),
            torch::data::DataLoaderOptions().batch_size(1).workers(4));

        int v = 0;
        for (auto& batch : *loader) {
            std::printf("\r[Contextual Alpha Boost] View %d/%zu", ++v, train_dataset_->size().value());
            std::fflush(stdout);

            auto cd = batch[0].data;
            Camera* cam = cd.camera;
            torch::Tensor wmap = cd.attentionMask; // CPU [1,H,W] or [H,W]
            if (!cam || !wmap.defined())
                continue;

            // --- Mask (CPU) -> core & edge ring ---
            auto m = (wmap > 0.5f).to(torch::kFloat32);                      // [1,H,W] or [H,W]
            auto mask = (m.dim() == 3 && m.size(0) == 1) ? m.squeeze(0) : m; // [H,W] float on CPU
            if (mask.dim() != 2)
                continue;

            const int H = (int)mask.size(0), W = (int)mask.size(1);

            auto erode_fn = [&](const torch::Tensor& bin01, int r) -> torch::Tensor {
                if (r <= 0)
                    return bin01.clamp(0, 1);
                const int k = 2 * r + 1;
                auto inv = 1.0f - bin01;
                auto inv_dil = F::max_pool2d(inv.unsqueeze(0).unsqueeze(0),
                                             F::MaxPool2dFuncOptions(k).stride(1).padding(r))
                                   .squeeze(0)
                                   .squeeze(0);
                return (1.0f - inv_dil).clamp(0, 1);
            };
            auto core = erode_fn(mask, core_erode_px);       // [H,W] float {0,1}
            auto edge_core = erode_fn(mask, edge_band_px);   // [H,W]
            auto edge_ring = (mask - edge_core).clamp(0, 1); // inside ring near boundary

            // Also keep a CUDA version for fast sampling along ellipse
            auto mask01_cuda = mask.to(torch::kCUDA); // [H,W] float {0,1} CUDA
            auto core01_cpu = (core > 0.5f);          // CPU bool
            auto ring01_cpu = (edge_ring > 0.5f);     // CPU bool

            // --- Optional render depth (for coherence); safe handling of shapes/devices ---
            torch::Tensor depth_cpu;
            {
                RenderOutput out = fast_rasterize(*cam, model, background_);
                // depth can be undefined or of shape [1,H,W] or [H,W]
                if (out.depth.defined()) {
                    auto d = out.depth;
                    if (d.dim() == 3 && d.size(0) == 1)
                        d = d.squeeze(0);
                    depth_cpu = d.contiguous().to(torch::kCPU); // sample on CPU
                }
            }

            // --- Projection-only (CUDA) ---
            auto view = cam->world_view_transform().to(torch::kCUDA); // [1,4,4]
            auto K = cam->K().to(torch::kCUDA);                       // [1,3,3] or [3,3]
            auto settings = torch::tensor({(float)W, (float)H, eps2d, near_p, far_p, radius_clip, scaling_mod},
                                          torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
            auto proj = ProjectionFunction::apply(means3D, rotations, scales, opacities, view, K, settings);

            auto radii = proj[0];   // [1,N,2] or [N,2]
            auto means2d = proj[1]; // [1,N,2] or [N,2]
            if (!radii.defined() || !means2d.defined())
                continue;
            if (radii.dim() == 3 && radii.size(0) == 1)
                radii = radii.squeeze(0);
            if (means2d.dim() == 3 && means2d.size(0) == 1)
                means2d = means2d.squeeze(0);

            auto visible = (radii > 0.0f).all(-1); // [N] bool CUDA
            if (!visible.any().item<bool>())
                continue;
            auto vidx = visible.nonzero().squeeze(-1); // [M] long CUDA

            // Footprint guards (CUDA)
            auto rx = radii.index({vidx, 0}).abs();
            auto ry = radii.index({vidx, 1}).abs();
            auto rmax = torch::max(rx, ry);
            auto rmin = torch::min(rx, ry);
            auto kappa = rmax / (rmin + 1e-6f);
            auto bad_fp = (rmax > rmax_px) | (rmin < rmin_px) | (kappa > kappa_max); // [M] bool CUDA

            // Center pixels for CPU mask sampling
            auto xy_cuda = means2d.index({vidx});           // [M,2] CUDA
            auto xy_cpu = xy_cuda.detach().to(torch::kCPU); // CPU
            auto cx = torch::round(xy_cpu.select(1, 0)).to(torch::kLong).clamp(0, W - 1);
            auto cy = torch::round(xy_cpu.select(1, 1)).to(torch::kLong).clamp(0, H - 1);
            auto lin_cpu = cy * W + cx; // [M] long CPU

            // Center in core? Near edge?
            auto center_in_core = core01_cpu.flatten().index({lin_cpu}); // [M] bool CPU
            auto near_edge = ring01_cpu.flatten().index({lin_cpu});      // [M] bool CPU

            // Push to CUDA for accumulation
            auto center_i32 = center_in_core.to(torch::kInt32).to(torch::kCUDA); // [M]
            auto edge_i32 = near_edge.to(torch::kInt32).to(torch::kCUDA);        // [M]
            views_vis.index_add_(0, vidx, torch::ones_like(center_i32));
            center_in_pos.index_add_(0, vidx, center_i32);
            center_in_tot.index_add_(0, vidx, torch::ones_like(center_i32));
            edge_hits.index_add_(0, vidx, edge_i32);
            bad_fp_flag.index_add_(0, vidx, bad_fp.to(torch::kInt32));

            // ---- Per-view leak test only for center-in pixels ----
            if (center_in_core.any().item<bool>()) {
                auto kidx = center_in_core.to(torch::kCUDA).nonzero().squeeze(-1); // [K] indices inside core among vidx
                if (kidx.numel() > 0) {
                    auto rxk = rx.index({kidx});
                    auto ryk = ry.index({kidx});
                    // center coords also on CUDA for fast linear sampling
                    auto cx_cuda = cx.to(torch::kCUDA).index({kidx});
                    auto cy_cuda = cy.to(torch::kCUDA).index({kidx});

                    // Sample ellipse contour on CUDA
                    auto sx = torch::empty({kidx.size(0), P}, torch::kLong).to(torch::kCUDA);
                    auto sy = torch::empty_like(sx);
                    for (int p = 0; p < P; ++p) {
                        auto dx = torch::round(rxk * dirs[p][0]).to(torch::kLong);
                        auto dy = torch::round(ryk * dirs[p][1]).to(torch::kLong);
                        sx.index_put_({torch::indexing::Slice(), p}, (cx_cuda + dx).clamp(0, W - 1));
                        sy.index_put_({torch::indexing::Slice(), p}, (cy_cuda + dy).clamp(0, H - 1));
                    }
                    auto lin_cuda = (sy * W + sx).reshape({-1});                                 // [K*P]
                    auto mvals = mask01_cuda.view({-1}).index({lin_cuda}).view({-1, P});         // [K,P] in {0,1}
                    auto outside_ratio = (1.0f - mvals).mean(1);                                 // [K] float
                    auto no_leak_here = (outside_ratio <= per_view_leak_frac).to(torch::kInt32); // [K]

                    auto ids = vidx.index({kidx}); // original Gaussian ids
                    noleak_pos.index_add_(0, ids, no_leak_here);
                    noleak_tot.index_add_(0, ids, torch::ones_like(no_leak_here));
                }
            }

            // ---- Depth coherence (optional) ----
            if (depth_cpu.defined()) {
                // Sample only visible positions
                auto dvals = depth_cpu.flatten().index({lin_cpu}); // [M] CPU float
                // z-score within this view (only visible set)
                auto mean = dvals.mean();
                auto stdv = dvals.std().clamp_min(1e-6);
                auto zabs = (dvals - mean).abs() / stdv; // [M] CPU
                auto z_cuda = zabs.to(torch::kCUDA);
                depth_abs_sum.index_add_(0, vidx, z_cuda);
                depth_cnt.index_add_(0, vidx, torch::ones_like(center_i32));
            }
        }
        std::printf("\n");

        // ---------------- Final decision ----------------
        auto vis_ok = (views_vis >= min_views);
        auto support = center_in_pos.to(torch::kFloat32) / center_in_tot.to(torch::kFloat32).clamp_min(1.0f);
        auto noleak_r = noleak_pos.to(torch::kFloat32) / noleak_tot.to(torch::kFloat32).clamp_min(1.0f);
        auto edge_r = edge_hits.to(torch::kFloat32) / views_vis.to(torch::kFloat32).clamp_min(1.0f);
        auto bad_fp = (bad_fp_flag > 0);

        // Depth
        auto depth_mean_abs = depth_abs_sum / depth_cnt.to(torch::kFloat32).clamp_min(1.0f);
        auto depth_ok = torch::ones_like(depth_mean_abs, depth_mean_abs.options().dtype(torch::kBool));
        if (depth_cnt.sum().item<int>() > 0) {
            depth_ok = (depth_mean_abs <= depth_zscore_max);
        }

        auto cand = vis_ok & (support >= support_thr) & (noleak_r >= no_leak_thr) & (edge_r <= edge_max_ratio) & (~bad_fp) & depth_ok; // [N] bool CUDA

        // ---------------- Apply boost (monotonic, capped) ----------------
        auto raw = model.opacity_raw(); // logits
        if (raw.dim() == 2 && raw.size(-1) == 1)
            raw = raw.squeeze(-1);
        if (raw.dim() != 1)
            raw = raw.view({-1}); // [N]
        raw = raw.contiguous();

        auto cur_a = torch::sigmoid(raw); // [N]
        auto need = cand & (cur_a < cap_alpha);
        auto idx = need.nonzero().squeeze(-1); // [K]
        if (idx.numel() == 0) {
            std::cout << "[Contextual Alpha Boost] No eligible candidates.\n";
            return;
        }

        // Confidence: combine support & no-leak and de-emphasize boundary
        auto conf = (0.5f * support + 0.5f * noleak_r) * (1.0f - edge_r);
        conf = conf.clamp(0.0f, 1.0f);
        auto conf_k = conf.index({idx}); // [K]

        auto a = cur_a.index({idx});                                            // [K]
        auto targ = a + (cap_alpha - a) * conf_k;                               // interpolate towards cap by confidence
        targ = torch::min(targ, torch::tensor(boost_soft_cap, targ.options())); // extra cap
        targ = torch::max(targ, a).clamp(eps, cap_alpha);                       // never decrease

        auto new_logit = torch::logit(targ);
        auto old_logit = raw.index({idx});
        raw.index_put_({idx}, torch::max(old_logit, new_logit));

        std::cout << "[Contextual Alpha Boost] Boosted " << idx.size(0)
                  << " splats (min_views=" << min_views
                  << ", support>=" << (int)(support_thr * 100)
                  << "%, no-leak>=" << (int)(no_leak_thr * 100)
                  << "%, edge<=" << (int)(edge_max_ratio * 100)
                  << "%, cap=" << target_alpha << ")\n";
    }

    std::expected<torch::Tensor, std::string> Trainer::compute_scale_reg_loss(
        const SplatData& splatData,
        const param::OptimizationParameters& opt_params) {

        try {
            if (opt_params.scale_reg > 0.0f) {
                auto scale_l1 = splatData.get_scaling().mean();
                return opt_params.scale_reg * scale_l1;
            }
            return torch::zeros({1}, torch::kFloat32).requires_grad_();
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error computing scale regularization loss: {}", e.what()));
        }
    }

    std::expected<torch::Tensor, std::string> Trainer::compute_opacity_reg_loss(
        const SplatData& splatData,
        const param::OptimizationParameters& opt_params) {

        try {
            if (opt_params.opacity_reg > 0.0f) {
                auto opacity_l1 = splatData.get_opacity().mean();
                return opt_params.opacity_reg * opacity_l1;
            }
            return torch::zeros({1}, torch::kFloat32).requires_grad_();
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error computing opacity regularization loss: {}", e.what()));
        }
    }

    std::expected<torch::Tensor, std::string> Trainer::compute_bilateral_grid_tv_loss(
        const std::unique_ptr<gs::BilateralGrid>& bilateral_grid,
        const param::OptimizationParameters& opt_params) {

        try {
            if (opt_params.use_bilateral_grid) {
                return opt_params.tv_loss_weight * bilateral_grid->tv_loss();
            }
            return torch::zeros({1}, torch::kFloat32).requires_grad_();
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error computing bilateral grid TV loss: {}", e.what()));
        }
    }

    Trainer::Trainer(std::shared_ptr<CameraDataset> dataset,
                     std::unique_ptr<IStrategy> strategy,
                     const param::TrainingParameters& params)
        : strategy_(std::move(strategy)),
          params_(params) {

        if (!torch::cuda::is_available()) {
            throw std::runtime_error("CUDA is not available – aborting.");
        }

        // Handle dataset split based on evaluation flag
        if (params.optimization.enable_eval) {
            // Create train/val split
            train_dataset_ = std::make_shared<CameraDataset>(
                dataset->get_cameras(), params.dataset, CameraDataset::Split::TRAIN);
            val_dataset_ = std::make_shared<CameraDataset>(
                dataset->get_cameras(), params.dataset, CameraDataset::Split::VAL);

            std::println("Created train/val split: {} train, {} val images",
                         train_dataset_->size().value(),
                         val_dataset_->size().value());
        } else {
            // Use all images for training
            train_dataset_ = dataset;
            val_dataset_ = nullptr;

            std::println("Using all {} images for training (no evaluation)",
                         train_dataset_->size().value());
        }

        train_dataset_size_ = train_dataset_->size().value();

        strategy_->initialize(params.optimization);

        // Initialize bilateral grid if enabled
        if (auto result = initialize_bilateral_grid(); !result) {
            throw std::runtime_error(result.error());
        }

        background_ = torch::tensor({0.f, 0.f, 0.f}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        // Create progress bar based on headless flag
        if (params.optimization.headless) {
            progress_ = std::make_unique<TrainingProgress>(
                params.optimization.iterations,
                /*update_frequency=*/100);
        }

        // Initialize the evaluator - it handles all metrics internally
        evaluator_ = std::make_unique<metrics::MetricsEvaluator>(params);

        // setup camera cache
        for (const auto& cam : dataset->get_cameras()) {
            m_cam_id_to_cam[cam->uid()] = cam;
        }

        // Print render mode configuration
        std::println("Render mode: {}", params.optimization.render_mode);
        std::println("Visualization: {}", params.optimization.headless ? "disabled" : "enabled");
        std::println("Strategy: {}", params.optimization.strategy);
    }

    Trainer::~Trainer() {
        // Ensure training is stopped
        stop_requested_ = true;

        // Wait for callback to finish if busy
        if (callback_busy_.load()) {
            callback_stream_.synchronize();
        }
    }

    void Trainer::handle_control_requests(int iter, std::stop_token stop_token) {
        // Check stop token first
        if (stop_token.stop_requested()) {
            stop_requested_ = true;
            return;
        }

        // Handle pause/resume
        if (pause_requested_.load() && !is_paused_.load()) {
            is_paused_ = true;
            if (progress_) {
                progress_->pause();
            }
            std::println("\nTraining paused at iteration {}", iter);
            std::println("Click 'Resume Training' to continue.");
        } else if (!pause_requested_.load() && is_paused_.load()) {
            is_paused_ = false;
            if (progress_) {
                progress_->resume(iter, current_loss_.load(), static_cast<int>(strategy_->get_model().size()));
            }
            std::println("\nTraining resumed at iteration {}", iter);
        }

        // Handle save request
        if (save_requested_.exchange(false)) {
            std::println("\nSaving checkpoint at iteration {}...", iter);
            auto checkpoint_path = params_.dataset.output_path / "checkpoints";
            save_ply(checkpoint_path, iter, /*join=*/true);

            std::println("Checkpoint saved to {}", checkpoint_path.string());

            // Emit checkpoint saved event
            events::state::CheckpointSaved{
                .iteration = iter,
                .path = checkpoint_path}
                .emit();
        }

        // Handle stop request - this permanently stops training
        if (stop_requested_.load()) {
            std::println("\nStopping training permanently at iteration {}...", iter);
            std::println("Saving final model...");
            save_ply(params_.dataset.output_path, iter, /*join=*/true);
            is_running_ = false;
        }
    }

    std::expected<Trainer::StepResult, std::string> Trainer::train_step(
        int iter,
        Camera* cam,
        torch::Tensor gt_image,
        torch::Tensor weights,
        RenderMode render_mode,
        bool out_of_mask_penalty,
        std::stop_token stop_token) {

        try {
            if (cam->radial_distortion().numel() != 0 ||
                cam->tangential_distortion().numel() != 0) {
                return std::unexpected("Training on cameras with distortion is not supported yet.");
            }
            if (cam->camera_model_type() != gsplat::CameraModelType::PINHOLE) {
                return std::unexpected("Training on cameras with non-pinhole model is not supported yet.");
            }

            current_iteration_ = iter;

            // Check control requests at the beginning
            handle_control_requests(iter, stop_token);

            // If stop requested, return Stop
            if (stop_requested_.load() || stop_token.stop_requested()) {
                return StepResult::Stop;
            }

            // If paused, wait
            while (is_paused_.load() && !stop_requested_.load() && !stop_token.stop_requested()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                handle_control_requests(iter, stop_token);
            }

            // Check stop again after potential pause
            if (stop_requested_.load() || stop_token.stop_requested()) {
                return StepResult::Stop;
            }

            // Use the render mode from parameters
            RenderOutput r_output = fast_rasterize(*cam, strategy_->get_model(), background_);

            // Apply bilateral grid if enabled
            if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
                r_output.image = bilateral_grid_->apply(r_output.image, cam->uid());
            }

            // Compute loss using the factored-out function
            std::expected<torch::Tensor, std::string> loss_result;
            
            const int total_iters = params_.optimization.iterations;
            const int warmup_end_iter = total_iters * 0.1f;

            const int alpha_boost_iter = warmup_end_iter + 0.0;
           if (weights.defined() && iter == alpha_boost_iter)
                maybe_alpha_boost(0.90f /*alpha_min*/, 3 /*min_visible_views */);
                //contextual_alpha_boost_once();

            if (!weights.defined() || iter < warmup_end_iter) {
                loss_result = compute_photometric_loss(r_output,
                                                        gt_image,
                                                        strategy_->get_model(),
                                                        params_.optimization);
            } 
            else {
                const int full_penalty_end_iter = total_iters * 0.5f;
                const int decay_end_iter = total_iters * 0.8f;
                float current_penalty_w = 0.0f;
                if (out_of_mask_penalty) {
                    if (iter <= full_penalty_end_iter) {
                        current_penalty_w = 1;
                    } else if (iter < decay_end_iter) {
                        const int decay_start_iter = full_penalty_end_iter;
                        const int decay_duration = decay_end_iter - decay_start_iter;
                        const float decay_progress = static_cast<float>(iter - decay_start_iter) / decay_duration;
                        current_penalty_w = 1.0f - decay_progress;
                    }

                    // Floor at 0.2 so it never vanishes
                    current_penalty_w = std::max(0.2f, current_penalty_w);
                }

                loss_result = compute_photometric_loss( r_output,
                                                        gt_image,
                                                        weights,
                                                        current_penalty_w,
                                                        strategy_->get_model(),
                                                        params_.optimization);
            }
            
            if (!loss_result) {
                return std::unexpected(loss_result.error());
            }

            torch::Tensor loss = *loss_result;
            loss.backward();
            float loss_value = loss.item<float>();

            // Scale regularization loss
            auto scale_loss_result = compute_scale_reg_loss(strategy_->get_model(), params_.optimization);
            if (!scale_loss_result) {
                return std::unexpected(scale_loss_result.error());
            }
            loss = *scale_loss_result;
            loss.backward();
            loss_value += loss.item<float>();

            // Opacity regularization loss
            auto opacity_loss_result = compute_opacity_reg_loss(strategy_->get_model(), params_.optimization);
            if (!opacity_loss_result) {
                return std::unexpected(opacity_loss_result.error());
            }
            loss = *opacity_loss_result;
            loss.backward();
            loss_value += loss.item<float>();

            // Bilateral grid TV loss
            auto tv_loss_result = compute_bilateral_grid_tv_loss(bilateral_grid_, params_.optimization);
            if (!tv_loss_result) {
                return std::unexpected(tv_loss_result.error());
            }
            loss = *tv_loss_result;
            loss.backward();
            loss_value += loss.item<float>();

            // Store the loss value immediately
            current_loss_ = loss_value;

            // Update progress synchronously if needed
            if (progress_) {
                progress_->update(iter, loss_value,
                                  static_cast<int>(strategy_->get_model().size()),
                                  strategy_->is_refining(iter));
            }

            // Emit training progress event (throttled to reduce GUI updates)
            if (iter % 10 == 0 || iter == 1) { // Only update every 10 iterations
                events::state::TrainingProgress{
                    .iteration = iter,
                    .loss = loss_value,
                    .num_gaussians = static_cast<int>(strategy_->get_model().size()),
                    .is_refining = strategy_->is_refining(iter)}
                    .emit();
            }

            {
                torch::NoGradGuard no_grad;

                DeferredEvents deferred;

                {
                    std::unique_lock<std::shared_mutex> lock(render_mutex_);

                    // Execute strategy post-backward and step
                    strategy_->post_backward(iter, r_output);
                    strategy_->step(iter);

                    if (params_.optimization.use_bilateral_grid) {
                        bilateral_grid_optimizer_->step();
                        bilateral_grid_optimizer_->zero_grad(true);
                    }

                    // Queue event for emission after lock release
                    deferred.add(events::state::ModelUpdated{
                        .iteration = iter,
                        .num_gaussians = static_cast<size_t>(strategy_->get_model().size())});
                } // Lock released here

                // Events automatically emitted here when deferred destructs

                // Clean evaluation - let the evaluator handle everything
                if (evaluator_->is_enabled() && evaluator_->should_evaluate(iter)) {
                    evaluator_->print_evaluation_header(iter);
                    auto metrics = evaluator_->evaluate(iter,
                                                        strategy_->get_model(),
                                                        val_dataset_,
                                                        background_);
                    std::println("{}", metrics.to_string());
                }

                // Save model at specified steps
                if (!params_.optimization.skip_intermediate_saving) {
                    for (size_t save_step : params_.optimization.save_steps) {
                        if (iter == static_cast<int>(save_step) && iter != params_.optimization.iterations) {
                            const bool join_threads = (iter == params_.optimization.save_steps.back());
                            auto save_path = params_.dataset.output_path;
                            save_ply(save_path, iter, /*join=*/join_threads);
                            // Emit checkpoint saved event
                            events::state::CheckpointSaved{
                                .iteration = iter,
                                .path = save_path}
                                .emit();
                        }
                    }
                }
            }

            // Return Continue if we should continue training
            if (iter < params_.optimization.iterations && !stop_requested_.load() && !stop_token.stop_requested()) {
                return StepResult::Continue;
            } else {
                return StepResult::Stop;
            }

        } catch (const std::exception& e) {
            return std::unexpected(std::format("Training step failed: {}", e.what()));
        }
    }

    std::expected<void, std::string> Trainer::train(std::stop_token stop_token) {
        is_running_ = false;
        training_complete_ = false;
        ready_to_start_ = false; // Reset the flag

        // Event-based ready signaling
        if (!params_.optimization.headless) {
            // Subscribe to start signal (no need to store handle)
            events::internal::TrainingReadyToStart::when([this](const auto&) {
                ready_to_start_ = true;
            });

            // Signal we're ready
            events::internal::TrainerReady{}.emit();

            // Wait for start signal
            while (!ready_to_start_.load() && !stop_token.stop_requested()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }

        is_running_ = true; // Now we can start

        try {
            int iter = 1;
            const int num_workers = 16;
            const RenderMode render_mode = stringToRenderMode(params_.optimization.render_mode);

            if (progress_) {
                progress_->update(iter, current_loss_.load(),
                                  static_cast<int>(strategy_->get_model().size()),
                                  strategy_->is_refining(iter));
            }

            // Use infinite dataloader to avoid epoch restarts
            auto train_dataloader = create_infinite_dataloader_from_dataset(train_dataset_, num_workers);
            auto loader = train_dataloader->begin();

            // Single loop without epochs
            while (iter <= params_.optimization.iterations) {
                if (stop_token.stop_requested() || stop_requested_.load()) {
                    break;
                }

                // Wait for previous callback if still running
                if (callback_busy_.load()) {
                    callback_stream_.synchronize();
                }

                auto& batch = *loader;
                auto camera_with_image = batch[0].data;
                Camera* cam = camera_with_image.camera;
                torch::Tensor gt_image = std::move(camera_with_image.image).to(torch::kCUDA, /*non_blocking=*/true);


                std::expected<Trainer::StepResult, std::string> step_result;
                if (!params_.optimization.use_attention_mask || !camera_with_image.attentionMask.defined()) {
                    step_result = train_step(iter, cam, gt_image, torch::Tensor(), render_mode, false, stop_token);
                } else {
                    torch::Tensor attention_image = std::move(camera_with_image.attentionMask);
                    bool out_of_mask_penalty = true;
                    step_result = train_step(iter, cam, gt_image, attention_image, render_mode, out_of_mask_penalty, stop_token);
                }

                
                if (!step_result) {
                    return std::unexpected(step_result.error());
                }

                if (*step_result == StepResult::Stop) {
                    break;
                }

                // Launch callback for async progress update (except first iteration)
                if (iter > 1 && callback_) {
                    callback_busy_ = true;
                    auto err = cudaLaunchHostFunc(
                        callback_stream_.stream(),
                        [](void* self) {
                            auto* trainer = static_cast<Trainer*>(self);
                            if (trainer->callback_) {
                                trainer->callback_();
                            }
                            trainer->callback_busy_ = false;
                        },
                        this);
                    if (err != cudaSuccess) {
                        std::cerr << "Warning: Failed to launch callback: " << cudaGetErrorString(err) << std::endl;
                        callback_busy_ = false;
                    }
                }

                ++iter;
                ++loader;
            }

            // Ensure callback is finished before final save
            if (callback_busy_.load()) {
                callback_stream_.synchronize();
            }

            training_complete:
                if (params_.optimization.use_attention_mask)
                    prune_after_training(0.8, 0.80);
            
            // Final save if not already saved by stop request
            if (!stop_requested_.load() && !stop_token.stop_requested()) {
                auto final_path = params_.dataset.output_path;
                save_ply(final_path, iter - 1, /*join=*/true);
                // Emit final checkpoint saved event
                events::state::CheckpointSaved{
                    .iteration = iter,
                    .path = final_path}
                    .emit();

                events::notify::Log{
                    .level = events::notify::Log::Level::Info,
                    .message = std::format("Training completed. Final model saved at iteration {}", iter),
                    .source = "Trainer"}
                    .emit();
            }

            if (progress_) {
                progress_->complete();
            }
            evaluator_->save_report();
            if (progress_) {
                progress_->print_final_summary(static_cast<int>(strategy_->get_model().size()));
            }
            
            is_running_ = false;
            training_complete_ = true;

            return {};

        } catch (const std::exception& e) {
            is_running_ = false;
            return std::unexpected(std::format("Training failed: {}", e.what()));
        }
    }

    std::shared_ptr<const Camera> Trainer::getCamById(int camId) const {
        const auto it = m_cam_id_to_cam.find(camId);
        if (it == m_cam_id_to_cam.end()) {
            std::cerr << "error: getCamById - could not find cam with cam id " << camId << std::endl;
            return nullptr;
        }
        return it->second;
    }

    std::vector<std::shared_ptr<const Camera>> Trainer::getCamList() const {

        std::vector<std::shared_ptr<const Camera>> cams;
        cams.reserve(m_cam_id_to_cam.size());
        for (auto& [key, value] : m_cam_id_to_cam) {
            cams.push_back(value);
        }

        return cams;
    }

    /* void Trainer::prune_after_training(float threshold) {
        torch::NoGradGuard no_grad;

        // 0) Access current Gaussian model
        SplatData& model = strategy_->get_model();
        const int64_t N = model.get_means().size(0);
        if (N == 0)
            return;

        // 1) Allocate vote buffers on GPU
        torch::Tensor pos = torch::zeros({N}, torch::kInt32).cuda(); // positive votes
        torch::Tensor tot = torch::zeros({N}, torch::kInt32).cuda(); // total votes

        // 2) Build the same DataLoader you already use (mask comes from batch.data.attentionMask)
        std::cout << "Optimized pruning (projection-only): Using DataLoader..." << std::endl;
        auto pruning_dataloader = torch::data::make_data_loader(
            *train_dataset_,
            torch::data::samplers::SequentialSampler(train_dataset_->size().value()),
            torch::data::DataLoaderOptions().batch_size(1).workers(4) // keep your current setting
        );

        // 3) Prepare model tensors once (CUDA)
        auto means3D = model.get_means();      // [N,3], CUDA
        auto scales = model.get_scaling();     // [N,3], CUDA
        auto rotations = model.get_rotation(); // [N,4] or [N,3x3], CUDA
        auto opacities = model.get_opacity();  // [N] or [N,1], CUDA / may be undefined but Tensor

        if (opacities.defined() && opacities.dim() == 2 && opacities.size(-1) == 1) {
            opacities = opacities.squeeze(-1);
        }

        // Projection numeric constants (keep in sync with rasterizer)
        const float eps2d = 0.3f;
        const float near_plane = 0.01f;
        const float far_plane = 10000.0f;
        const float radius_clip = 0.0f;
        const float scaling_mod = 1.0f;

        std::cout << "Optimized pruning: Fetched..." << std::endl;
        int index = 1;

        for (auto& batch : *pruning_dataloader) {
            // Progress heartbeat
            printf("\rPrunning image %i", index++);
            fflush(stdout);

            // 3.a) Unpack camera and attention mask from your batch
            auto camera_with_data = batch[0].data;
            Camera* cam = camera_with_data.camera;
            torch::Tensor float_weight_map = camera_with_data.attentionMask;
            if (!cam || !float_weight_map.defined()) {
                continue;
            }

            // Make a [H,W] boolean mask on CPU
            auto bool_mask_3d = (float_weight_map > 0.5f); // [1,H,W] or [H,W]
            auto bool_mask = (bool_mask_3d.dim() == 3 && bool_mask_3d.size(0) == 1)
                                 ? bool_mask_3d.squeeze(0)
                                 : bool_mask_3d;
            TORCH_CHECK(bool_mask.dim() == 2, "Attention mask must be [H,W] or [1,H,W]");
            bool_mask = bool_mask.contiguous(); // CPU bool [H,W]

            const int H = static_cast<int>(bool_mask.size(0));
            const int W = static_cast<int>(bool_mask.size(1));

            // 3.b) Camera tensors (CUDA)
            auto viewmat = cam->world_view_transform().to(torch::kCUDA); // [1,4,4]
            auto K = cam->K().to(torch::kCUDA);                          // [1,3,3] or [3,3]

            // Prefer camera's declared image size for projection
            const int image_width = static_cast<int>(cam->image_width());
            const int image_height = static_cast<int>(cam->image_height());

            // 3.c) Projection-only (no rendering)
            auto proj_settings = torch::tensor(
                {static_cast<float>(image_width),
                 static_cast<float>(image_height),
                 eps2d, near_plane, far_plane, radius_clip, scaling_mod},
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

            // NOTE: This lives in the same namespace (gs) – call without "gs::" prefix,
            // exactly like in rasterizer.cpp.
            auto proj_out = ProjectionFunction::apply(
                means3D, rotations, scales, opacities, viewmat, K, proj_settings);

            torch::Tensor radii2 = proj_out[0];  // [1,N,2] or [N,2]
            torch::Tensor means2d = proj_out[1]; // [1,N,2] or [N,2]

            if (!radii2.defined() || !means2d.defined()) {
                printf("radii2 or means2d failed\n");
                continue; // Projection failed, skip gracefully
            }
            if (radii2.dim() == 3 && radii2.size(0) == 1)
                radii2 = radii2.squeeze(0);
            if (means2d.dim() == 3 && means2d.size(0) == 1)
                means2d = means2d.squeeze(0);

            // 3.d) Visibility: positive projected radius
            torch::Tensor visible;
            if (radii2.dim() == 2 && radii2.size(1) >= 1) {
                visible = (radii2 > 0.0f).all(-1); // [N]
            } else if (radii2.dim() == 1) {
                visible = (radii2 > 0.0f);
            } else {
                continue;
            }
            if (!visible.any().item<bool>())
                continue;

            auto idx = visible.nonzero().squeeze(); // [M], CUDA

            // 3.e) Gather 2D positions (CPU) and vote on the CPU mask
            auto xy_cuda = means2d.index({idx}); // [M,2], CUDA
            auto xy = xy_cuda.detach().to(torch::kCPU);

            // Round and clamp to mask bounds
            auto x = torch::round(xy.select(1, 0)).to(torch::kLong).clamp(0, W - 1);
            auto y = torch::round(xy.select(1, 1)).to(torch::kLong).clamp(0, H - 1);
            auto lin = y * W + x; // [M], CPU long

            // Sample mask and accumulate votes
            auto white_cpu = bool_mask.flatten().index({lin});                  // CPU bool
            auto white_i32_cuda = white_cpu.to(torch::kInt32).to(torch::kCUDA); // CUDA int32

            pos.index_add_(0, idx, white_i32_cuda);
            tot.index_add_(0, idx, torch::ones_like(white_i32_cuda, torch::kInt32));
        }

        // Newline after progress
        printf("\n");

        // 4) Final pruning by vote ratio
        const int min_visibility_count = 3;
        auto tot_safe = tot.to(torch::kFloat32).clamp_min(1.0f);
        auto ratio = pos.to(torch::kFloat32) / tot_safe;

        auto keep_mask = (tot >= min_visibility_count) & (ratio >= threshold);

        const int removed = (keep_mask == 0).sum().item<int>();
        model.filterByMask(keep_mask);

        std::cout << "[Trainer] prune_after_training (projection-only): removed "
                  << removed << " / " << N << " splats (thr=" << threshold
                  << ", min_vis=" << min_visibility_count << ")\n";
    }*/
    
    void Trainer::prune_by_center_vote(float center_keep_threshold, int min_visibility_count) {
        torch::NoGradGuard no_grad;

        // 0) Access model
        SplatData& model = strategy_->get_model();
        const int64_t N = model.get_means().size(0);
        if (N == 0) {

            std::cout << "[Trainer] Prune center-vote: has no means!" << std::endl;
            return;
        }

        // 1) Vote buffers (CUDA int32)
        torch::Tensor pos = torch::zeros({N}, torch::kInt32).cuda(); // inside votes
        torch::Tensor tot = torch::zeros({N}, torch::kInt32).cuda(); // visibility count

        // 2) Rebuild the same DataLoader (batch=1, sequential)
        auto pruning_dataloader = torch::data::make_data_loader(
            *train_dataset_,
            torch::data::samplers::SequentialSampler(train_dataset_->size().value()),
            torch::data::DataLoaderOptions().batch_size(1).workers(4));

        // 3) Prepare model tensors (CUDA)
        auto means3D = model.get_means();      // [N,3]
        auto scales = model.get_scaling();     // [N,3]
        auto rotations = model.get_rotation(); // [N,4]
        auto opacities = model.get_opacity();  // [N] or [N,1]
        if (opacities.defined() && opacities.dim() == 2 && opacities.size(-1) == 1)
            opacities = opacities.squeeze(-1);

        // Projection constants
        const float eps2d = 0.3f, near_plane = 0.01f, far_plane = 10000.0f, radius_clip = 0.0f, scaling_mod = 1.0f;

        int idx_img = 1;
        for (auto& batch : *pruning_dataloader) {
            // Progress heartbeat
            std::printf("\r[Prune Center] image %d", idx_img++);
            std::fflush(stdout);

            auto camera_with_data = batch[0].data;
            Camera* cam = camera_with_data.camera;
            torch::Tensor float_weight_map = camera_with_data.attentionMask;
            if (!cam || !float_weight_map.defined())
                continue;

            // CPU mask [H,W] bool
            auto bool_mask_3d = (float_weight_map > 0.5f);
            auto bool_mask = (bool_mask_3d.dim() == 3 && bool_mask_3d.size(0) == 1) ? bool_mask_3d.squeeze(0) : bool_mask_3d;
            TORCH_CHECK(bool_mask.dim() == 2, "Attention mask must be [H,W] or [1,H,W]");
            bool_mask = bool_mask.contiguous();
            const int H = (int)bool_mask.size(0);
            const int W = (int)bool_mask.size(1);

            // Camera tensors
            auto viewmat = cam->world_view_transform().to(torch::kCUDA); // [1,4,4]
            auto K = cam->K().to(torch::kCUDA);                          // [1,3,3] or [3,3]

            const int image_w = (int)cam->image_width();
            const int image_h = (int)cam->image_height();

            // Projection-only
            auto proj_settings = torch::tensor({(float)image_w, (float)image_h, eps2d, near_plane, far_plane, radius_clip, scaling_mod},
                                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
            auto proj_out = ProjectionFunction::apply(means3D, rotations, scales, opacities, viewmat, K, proj_settings);

            torch::Tensor radii2 = proj_out[0];  // [1,N,2] or [N,2]
            torch::Tensor means2d = proj_out[1]; // [1,N,2] or [N,2]
            if (!radii2.defined() || !means2d.defined())
                continue;
            if (radii2.dim() == 3 && radii2.size(0) == 1)
                radii2 = radii2.squeeze(0);
            if (means2d.dim() == 3 && means2d.size(0) == 1)
                means2d = means2d.squeeze(0);

            // Visibility: positive projected radius
            torch::Tensor visible = (radii2.dim() == 2 && radii2.size(1) >= 1) ? (radii2 > 0.0f).all(-1) : (radii2 > 0.0f);
            if (!visible.any().item<bool>())
                continue;
            auto idx = visible.nonzero().squeeze(); // [M], CUDA

            // Gather 2D & vote on CPU mask
            auto xy_cuda = means2d.index({idx}); // [M,2], CUDA
            auto xy = xy_cuda.detach().to(torch::kCPU);
            auto x = torch::round(xy.select(1, 0)).to(torch::kLong).clamp(0, W - 1);
            auto y = torch::round(xy.select(1, 1)).to(torch::kLong).clamp(0, H - 1);
            auto lin = y * W + x; // [M]

            auto white_cpu = bool_mask.flatten().index({lin});                  // CPU bool
            auto white_i32_cuda = white_cpu.to(torch::kInt32).to(torch::kCUDA); // CUDA int32

            pos.index_add_(0, idx, white_i32_cuda);
            tot.index_add_(0, idx, torch::ones_like(white_i32_cuda, torch::kInt32));
        }
        std::printf("\n");

        // 4) Final keep & prune
        auto tot_safe = tot.to(torch::kFloat32).clamp_min(1.0f);
        auto ratio = pos.to(torch::kFloat32) / tot_safe;
        auto keep_mask = (tot >= min_visibility_count) & (ratio >= center_keep_threshold);

        const int removed = (keep_mask == 0).sum().item<int>();
        model.filterByMask(keep_mask);

        std::cout << "[Trainer] Prune center-vote: removed " << removed
                  << " / " << N << " splats (thr=" << center_keep_threshold
                  << ", min_vis=" << min_visibility_count << ")\n";
    }

    // Prune Gaussians whose projected footprint "leaks" outside the mask too often.
    // Success-rate semantics: keep only if (views_without_leak / candidate_views) >= leak_keep_threshold.
    // - Only views where the Gaussian is visible are considered.
    // - Optional mask dilation (dilate_px) provides pixel tolerance at the boundary.
    // - per_view_leak_frac is the fraction of sampled contour points that must be outside
    //   in a given view to count that view as a "leak". E.g., 0.25 means at least 25% of
    //   sampled points outside the (dilated) mask to mark the view as leak.
    void Trainer::prune_by_mask_leakage(float leak_keep_threshold,
                                        float min_pixel_radius,
                                        float min_center_mask,
                                        int sample_points,
                                        int dilate_px,
                                        float per_view_leak_frac) {
        torch::NoGradGuard no_grad;

        // 0) Access model
        SplatData& model = strategy_->get_model();
        const int64_t N = model.get_means().size(0);
        if (N == 0)
            return;

        // 1) Leak vote buffers (CUDA int32)
        torch::Tensor leak_votes = torch::zeros({N}, torch::kInt32).to(torch::kCUDA); // count of leak-views
        torch::Tensor vis_counts = torch::zeros({N}, torch::kInt32).to(torch::kCUDA); // candidate views counted

        // 2) DataLoader (sequential, batch=1)
        auto pruning_dataloader = torch::data::make_data_loader(
            *train_dataset_,
            torch::data::samplers::SequentialSampler(train_dataset_->size().value()),
            torch::data::DataLoaderOptions().batch_size(1).workers(4));

        // 3) Model tensors (CUDA)
        auto means3D = model.get_means();      // [N,3]
        auto scales = model.get_scaling();     // [N,3]
        auto rotations = model.get_rotation(); // [N,4]
        auto opacities = model.get_opacity();  // [N] or [N,1]
        if (opacities.defined() && opacities.dim() == 2 && opacities.size(-1) == 1)
            opacities = opacities.squeeze(-1);

        // Projection constants (keep in sync with rasterizer)
        const float eps2d = 0.3f;
        const float near_plane = 0.01f;
        const float far_plane = 10000.0f;
        const float radius_clip = 0.0f;
        const float scaling_mod = 1.0f;

        // Directions on the ellipse: NESW (+ diagonals if requested).
        // Note: we only support 4 or 8 samples in this fast path; sample_points>=8 -> 8 else 4.
        std::vector<std::array<float, 2>> dirs = {{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}};
        if (sample_points >= 8) {
            const float s = 0.70710678f;
            dirs.push_back({s, s});
            dirs.push_back({-s, s});
            dirs.push_back({s, -s});
            dirs.push_back({-s, -s});
        }

        int idx_img = 1;
        for (auto& batch : *pruning_dataloader) {
            std::printf("\r[Prune Leakage] image %d", idx_img++);
            std::fflush(stdout);

            auto camera_with_data = batch[0].data;
            Camera* cam = camera_with_data.camera;
            torch::Tensor float_weight_map = camera_with_data.attentionMask;
            if (!cam || !float_weight_map.defined())
                continue;

            // Mask {0,1} on CUDA, shape [H,W]
            auto mask01_3d = (float_weight_map > 0.5f).to(torch::kFloat32); // [1,H,W] or [H,W]
            auto mask01 = (mask01_3d.dim() == 3 && mask01_3d.size(0) == 1) ? mask01_3d.squeeze(0) : mask01_3d;
            TORCH_CHECK(mask01.dim() == 2, "Attention mask must be [H,W] or [1,H,W]");
            mask01 = mask01.contiguous().to(torch::kCUDA);

            // Optional dilation (pixel tolerance at boundary): k = 2*dilate_px+1, stride=1, same padding.
            if (dilate_px > 0) {
                const int k = 2 * dilate_px + 1;
                auto m4 = mask01.unsqueeze(0).unsqueeze(0); // [1,1,H,W]
                auto opts = torch::nn::functional::MaxPool2dFuncOptions(k).stride(1).padding(dilate_px);
                mask01 = torch::nn::functional::max_pool2d(m4, opts).squeeze(0).squeeze(0); // [H,W]
            }

            const int H = static_cast<int>(mask01.size(0));
            const int W = static_cast<int>(mask01.size(1));

            // Camera tensors (CUDA)
            auto viewmat = cam->world_view_transform().to(torch::kCUDA); // [1,4,4]
            auto K = cam->K().to(torch::kCUDA);                          // [1,3,3] or [3,3]
            const int image_w = static_cast<int>(cam->image_width());
            const int image_h = static_cast<int>(cam->image_height());

            // Projection-only: get means2d & radii
            auto settings = torch::tensor(
                {static_cast<float>(image_w), static_cast<float>(image_h),
                 eps2d, near_plane, far_plane, radius_clip, scaling_mod},
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

            auto proj_out = ProjectionFunction::apply(means3D, rotations, scales, opacities, viewmat, K, settings);
            torch::Tensor radii = proj_out[0];   // [1,N,2] or [N,2]
            torch::Tensor means2d = proj_out[1]; // [1,N,2] or [N,2]
            if (!radii.defined() || !means2d.defined())
                continue;
            if (radii.dim() == 3 && radii.size(0) == 1)
                radii = radii.squeeze(0);
            if (means2d.dim() == 3 && means2d.size(0) == 1)
                means2d = means2d.squeeze(0);

            // Visibility: positive projected radius -> only visible splats count.
            auto visible = (radii > 0.0f).all(-1); // [N]
            if (!visible.any().item<bool>())
                continue;
            auto vidx = visible.nonzero().squeeze(-1); // [M]

            // Centers & radii for visible splats
            auto xy = means2d.index({vidx});        // [M,2]
            auto rx = radii.index({vidx, 0}).abs(); // [M]
            auto ry = radii.index({vidx, 1}).abs(); // [M]

            // Pixel centers
            auto cx = torch::round(xy.select(1, 0)).to(torch::kLong).clamp(0, W - 1);
            auto cy = torch::round(xy.select(1, 1)).to(torch::kLong).clamp(0, H - 1);

            // Center must be inside the (dilated) mask
            auto center_mask = mask01.index({cy, cx});       // [M] in {0,1}
            auto center_in = center_mask >= min_center_mask; // bool

            // Footprint must be large enough (filters micro-splats)
            auto large_fp = (torch::max(rx, ry) >= min_pixel_radius);

            auto keep = center_in & large_fp;
            if (!keep.any().item<bool>())
                continue;

            auto kidx = keep.nonzero().squeeze(-1); // [K]
            auto rxk = rx.index({kidx});
            auto ryk = ry.index({kidx});
            auto cxk = cx.index({kidx});
            auto cyk = cy.index({kidx});

            // Sample points around the ellipse (axis-aligned approx; fast and robust).
            const int P = static_cast<int>(dirs.size());
            auto sx = torch::empty({kidx.size(0), P}, torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));
            auto sy = torch::empty_like(sx);
            for (int p = 0; p < P; ++p) {
                auto dx = torch::round(rxk * dirs[p][0]).to(torch::kLong);
                auto dy = torch::round(ryk * dirs[p][1]).to(torch::kLong);
                sx.index_put_({torch::indexing::Slice(), p}, (cxk + dx).clamp(0, W - 1));
                sy.index_put_({torch::indexing::Slice(), p}, (cyk + dy).clamp(0, H - 1));
            }

            // Gather mask values at sampled points and compute per-splat outside fraction.
            auto lin = (sy * W + sx).reshape({-1});                    // [K*P]
            auto mvals = mask01.view({-1}).index({lin}).view({-1, P}); // [K,P] in {0,1}
            auto outside_ratio = (1.0f - mvals).mean(1);               // [K] in [0,1]

            // Per-view leak decision:
            // Count a view as "leak" only if the fraction of sampled contour points
            // outside the (dilated) mask exceeds per_view_leak_frac.
            auto leak_here = (outside_ratio > per_view_leak_frac).to(torch::kInt32); // [K]
            auto ids = vidx.index({kidx});                                           // [K] original Gaussian ids

            vis_counts.index_add_(0, ids, torch::ones_like(leak_here));
            leak_votes.index_add_(0, ids, leak_here);
        }
        std::printf("\n");

        // 4) Final decision (success-rate semantics)
        auto vis_ok = vis_counts >= 1;
        if (!vis_ok.any().item<bool>())
            return;

        auto leak_fail = leak_votes.to(torch::kFloat32) / vis_counts.to(torch::kFloat32).clamp_min(1.0f); // failures in [0,1]
        auto leak_ok = 1.0f - leak_fail;                                                                  // successes in [0,1]
        auto is_prune = vis_ok & (leak_ok < leak_keep_threshold);
        auto keep_mask = (~is_prune).to(torch::kBool);

        const int removed = is_prune.sum().item<int>();
        model.filterByMask(keep_mask);

        std::cout << "[Trainer] Prune mask-leakage: removed " << removed
                  << " / " << N << " splats (keep_thr=" << leak_keep_threshold
                  << ", min_radius=" << min_pixel_radius
                  << ", samples=" << (sample_points >= 8 ? 8 : 4)
                  << ", dilate_px=" << dilate_px
                  << ", per_view_leak_frac=" << per_view_leak_frac
                  << ")\n";
    }



    void Trainer::prune_after_training(float vote_ratio_threshold, float leak_keep_threshold) {
        // 1) First, center-vote pruning (your original logic)
        prune_by_center_vote(vote_ratio_threshold);

        // 2) Re-fetch model and then leakage pruning on remaining splats
        prune_by_mask_leakage(leak_keep_threshold);
    }




    void Trainer::save_ply(const std::filesystem::path& save_path, int iter_num, bool join_threads) {
        strategy_->get_model().save_ply(save_path, iter_num + 1, /*join=*/join_threads);
        if (lf_project_) {
            const std::string ply_name = "splat_" + std::to_string(iter_num + 1);
            const std::filesystem::path ply_path = save_path / (ply_name + ".ply");
            lf_project_->addPly(gs::management::PlyData(false, ply_path, iter_num, ply_name));
        }
    }

} // namespace gs
