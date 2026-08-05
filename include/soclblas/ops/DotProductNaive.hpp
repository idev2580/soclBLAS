#pragma once
#include "socl/Buffer.hpp"
#include "socl/Context.hpp"
#include "socl/DescriptorSet.hpp"
#include <soclblas/ops/Reduction.hpp>

namespace soclblas{
    class DotProductNaiveFP32:public BinaryReduction{
        private:
        socl::Context& ctx;
        socl::ShaderPipeline pipeline;
        socl::DescriptorSet descSet;
        uint32_t thread_num;
        uint32_t values_per_thread;

        public:
        DotProductNaiveFP32(
            socl::Context& ctx,
            uint32_t thread_num = 32,
            uint32_t values_per_thread = 64
        );

        virtual socl::DispatchToken execute(
            std::span<socl::Buffer> inputs,
            std::span<socl::Buffer> inouts,
            std::span<socl::Buffer> outputs,
            const void* args,
            std::size_t argsSize
        ) override;

        virtual socl::DispatchToken operator()(
            socl::Buffer a,
            socl::Buffer b,
            socl::Buffer out,
            const BinaryReductionArguments& args
        ) override;
    };
}
