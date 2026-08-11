/* SPDX-FileCopyrightText: 2026 SMN | Víctor M. Feliz
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/parameters.hpp"
#include "training/smn/smn_hybrid.hpp"
#include "training/smn/smn_step_scaling.hpp"

#include <array>
#include <cmath>
#include <gtest/gtest.h>

using lfs::core::param::OptimizationParameters;

TEST(SMNHybridTest, SwitchFractionMatchesOriginalEffectiveRuntimeTotal) {
    struct Case {
        float scaler;
        size_t iterations;
        int switch_iteration;
    };
    constexpr std::array cases{
        Case{0.5f, 15'000, 7'500},
        Case{1.0f, 30'000, 11'250},
        Case{2.0f, 60'000, 18'750},
        Case{3.25f, 97'500, 28'125},
    };

    for (const auto& test : cases) {
        auto params = OptimizationParameters::mcmc_defaults();
        params.steps_scaler = test.scaler;
        params.apply_step_scaling();
        params.enable_sparsity = true;
        params.sparsify_steps = 15'000;
        params.smn_switch_strategy_to = "mrnf";
        params.smn_switch_at_fraction = 0.25f;

        EXPECT_EQ(params.iterations, test.iterations);
        EXPECT_EQ(
            lfs::training::smn::resolve_strategy_switch_iteration(params),
            test.switch_iteration);
    }
}

TEST(SMNHybridTest, McmcToMrnfPreservesProfileAndPhasesFirstRefineFromHandoff) {
    auto source = OptimizationParameters::mcmc_defaults();
    source.steps_scaler = 2.0f;
    source.apply_step_scaling();
    source.mask_mode = lfs::core::param::MaskMode::Attention;
    source.max_cap = 1'234'567;
    source.enable_sparsity = true;
    source.smn_switch_strategy_to = "mrnf";
    source.smn_switch_at_fraction = 0.25f;
    const int switch_iteration =
        lfs::training::smn::resolve_strategy_switch_iteration(source);

    const auto target = lfs::training::smn::make_in_process_target_params(
        source, "mrnf", switch_iteration);

    EXPECT_EQ(target.strategy, "mrnf");
    EXPECT_EQ(target.iterations, 60'000u);
    EXPECT_EQ(target.refine_every, 200u);
    EXPECT_EQ(target.start_refine, static_cast<size_t>(switch_iteration));
    EXPECT_EQ(target.stop_refine, 50'000u);
    EXPECT_EQ(target.grow_until_iter, 30'000u);
    EXPECT_EQ(target.max_cap, source.max_cap);
    EXPECT_EQ(target.mask_mode, lfs::core::param::MaskMode::Attention);
    EXPECT_FLOAT_EQ(target.opacity_reg, source.opacity_reg);
    EXPECT_FLOAT_EQ(target.opacity_decay, source.opacity_decay);
    EXPECT_FLOAT_EQ(target.scale_reg, source.scale_reg);
    EXPECT_FLOAT_EQ(target.scaling_lr, source.scaling_lr);
    EXPECT_FLOAT_EQ(target.rotation_lr, source.rotation_lr);
    EXPECT_TRUE(target.smn_hybrid_mcmc_densification);
    EXPECT_TRUE(lfs::training::smn::uses_mcmc_densification_signal(target));
    EXPECT_FALSE(lfs::training::smn::uses_mcmc_densification_signal(source));
    EXPECT_TRUE(target.smn_switch_strategy_to.empty());
    EXPECT_FLOAT_EQ(target.smn_switch_at_fraction, 0.0f);

    const auto restored = OptimizationParameters::from_json(target.to_json());
    EXPECT_TRUE(restored.smn_hybrid_mcmc_densification);
    EXPECT_TRUE(lfs::training::smn::uses_mcmc_densification_signal(restored));

    const int refine_every = static_cast<int>(restored.refine_every);
    EXPECT_FALSE(lfs::training::smn::is_mrnf_refine_due(restored, switch_iteration));
    EXPECT_FALSE(lfs::training::smn::is_mrnf_refine_due(restored, switch_iteration + refine_every - 1));
    EXPECT_TRUE(lfs::training::smn::is_mrnf_refine_due(restored, switch_iteration + refine_every));
    EXPECT_FALSE(lfs::training::smn::is_mrnf_refine_due(restored, switch_iteration + refine_every + 1));
}

TEST(SMNHybridTest, PlainMrnfRetainsGlobalRefineCadence) {
    auto params = OptimizationParameters::mrnf_defaults();

    EXPECT_FALSE(lfs::training::smn::is_mrnf_refine_due(params, 199));
    EXPECT_TRUE(lfs::training::smn::is_mrnf_refine_due(params, 200));
    EXPECT_TRUE(lfs::training::smn::is_mrnf_refine_due(params, 400));
}

TEST(SMNStepScalingTest, ScalesWarmupsAndExplicitActivationButKeepsSentinels) {
    auto params = OptimizationParameters::mrnf_defaults();
    params.ppisp_warmup_steps = 500;
    params.ppisp_controller_activation_step = 12'000;
    params.scale_steps(0.5f);

    EXPECT_EQ(params.ppisp_warmup_steps, 250);
    EXPECT_EQ(params.ppisp_controller_activation_step, 6'000);
    EXPECT_EQ(lfs::training::smn::scaled_component_steps(1'000, 0.5f), 500);
    EXPECT_EQ(lfs::training::smn::scaled_component_steps(100, 2.0f), 200);

    params.ppisp_warmup_steps = 0;
    params.ppisp_controller_activation_step = -1;
    params.scale_steps(2.0f);
    EXPECT_EQ(params.ppisp_warmup_steps, 0);
    EXPECT_EQ(params.ppisp_controller_activation_step, -1);
}

TEST(SMNStepScalingTest, DefaultControllerTailAlsoShrinksBelowOne) {
    auto params = OptimizationParameters::mrnf_defaults();
    params.steps_scaler = 0.5f;
    params.apply_step_scaling();

    EXPECT_EQ(params.iterations, 15'000u);
    EXPECT_EQ(
        params.resolved_ppisp_controller_activation_step(
            params.resolved_total_iterations()),
        12'500);
}

TEST(SMNStepScalingTest, RequiredIntervalsNeverRoundToZero) {
    auto params = OptimizationParameters::mcmc_defaults();
    params.scale_steps(0.0001f);

    EXPECT_EQ(params.iterations, 3u);
    EXPECT_EQ(params.refine_every, 1u);
    EXPECT_EQ(params.reset_every, 1u);
    EXPECT_EQ(params.sh_degree_interval, 1u);
}
