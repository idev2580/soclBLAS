#pragma once
#include "socl/Buffer.hpp"
#include "socl/Context.hpp"
#include <soclblas/ops/Reduction.hpp>

namespace soclblas{
    class DotProductNaiveFP32:public BinaryReduction{
        private:
        socl::ShaderPipeline pipeline;
        uint32_t thread_num;
        uint32_t values_per_thread;

        public:
        DotProductNaiveFP32(
            socl::Context& ctx,
            uint32_t thread_num = 32,
            uint32_t values_per_thread = 64
        );

        virtual DispatchPlan execute(
            std::span<const BufferView> inputs,
            std::span<const BufferView> inouts,
            std::span<const BufferView> outputs,
            const void* args,
            std::size_t argsSize
        ) override;

        virtual DispatchPlan operator()(
            BufferView a,
            BufferView b,
            BufferView out,
            const BinaryReductionArguments& args
        ) override;
    };
}
