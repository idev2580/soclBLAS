#pragma once
#include "socl/Buffer.hpp"
#include "socl/DescriptorSet.hpp"
#include <cstdint>
#include <socl/Context.hpp>
#include <soclblas/ops/Operator.hpp>

namespace soclblas{
    // TODO: For broadcasted operations, elementwise operators should also need to support strided operation.
    struct AxpyArguments{
        uint32_t n;
        float alpha;
    };
    class Axpy: public Operator{
        private:
        socl::Context& ctx;
        socl::ShaderPipeline pipeline;
        socl::DescriptorSet descSet;
        uint32_t thread_num;

        public:
        Axpy(
            socl::Context& ctx,
            uint32_t thread_num = 64
        );
        virtual void execute(
            std::span<socl::Buffer> inputs,
            std::span<socl::Buffer> inouts,
            std::span<socl::Buffer> outputs,
            const void* args,
            std::size_t argsSize
        ) override;

        virtual void operator()(
            socl::Buffer A,
            socl::Buffer B,
            socl::Buffer C,
            const AxpyArguments& args
        );
    };
}