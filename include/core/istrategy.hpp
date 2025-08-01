#pragma once

#include "core/parameters.hpp"
#include "core/scheduler.hpp"
#include "core/splat_data.hpp"

namespace gs {
    struct RenderOutput;
}

class IStrategy {
public:
    virtual ~IStrategy() = default;

    virtual void initialize(const gs::param::OptimizationParameters& optimParams) = 0;
    virtual void pre_backward(gs::RenderOutput& render_output) = 0;
    virtual void post_backward(int iter, gs::RenderOutput& render_output, bool packed = false) = 0;
    virtual void step(int iter) = 0;
    virtual bool is_refining(int iter) const = 0;
    // Get the underlying Gaussian model for rendering
    virtual gs::SplatData& get_model() = 0;
    virtual const gs::SplatData& get_model() const = 0;
};
