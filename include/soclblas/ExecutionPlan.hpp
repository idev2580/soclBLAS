#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <socl/Buffer.hpp>
#include <socl/Context.hpp>
#include <socl/DescriptorSet.hpp>
#include <socl/ShaderPipeline.hpp>

namespace soclblas{
    struct BufferBinding{
        std::uint32_t binding;
        socl::Buffer buffer;
        socl::BufferAccess access;
    };

    struct DispatchPlanResource{
        socl::DescriptorSet descriptorSet;
        socl::Context* context = nullptr;
    };

    struct DispatchPlan{
        socl::ShaderPipeline pipeline;
        std::vector<BufferBinding> bindings;
        std::vector<std::byte> pushConstants;
        std::uint32_t dispatchX;
        std::uint32_t dispatchY;
        std::uint32_t dispatchZ;
        mutable std::optional<DispatchPlanResource> resource;

        void allocate(socl::Context& ctx) const;
        void record(socl::Context& ctx) const;
    };

    class ExecutionPlan{
        private:
        std::vector<DispatchPlan> plans;

        public:
        void append(DispatchPlan plan);
        void record(socl::Context& ctx) const;
        [[nodiscard]] socl::DispatchToken execute(socl::Context& ctx, bool recordGpuTimestamp = false) const;
    };
}
