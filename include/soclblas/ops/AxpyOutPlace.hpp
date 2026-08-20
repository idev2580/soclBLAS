#pragma once
#include "socl/Buffer.hpp"
#include <cstdint>
#include <socl/Context.hpp>
#include <soclblas/ops/Axpy.hpp>
#include <soclblas/ops/Operator.hpp>

namespace soclblas{
    class AxpyOutPlaceFP32: public Operator{
        private:
        socl::ShaderPipeline pipeline;
        uint32_t thread_num;

        public:
        AxpyOutPlaceFP32(
            socl::Context& ctx,
            uint32_t thread_num = 64
        );
        virtual DispatchPlan execute(
            std::span<socl::Buffer> inputs,
            std::span<socl::Buffer> inouts,
            std::span<socl::Buffer> outputs,
            const void* args,
            std::size_t argsSize
        ) override;

        virtual DispatchPlan operator()(
            socl::Buffer A,
            socl::Buffer B,
            socl::Buffer outB,
            const AxpyOutPlaceArguments& args
        );
        virtual DispatchPlan operator()(
            socl::Buffer A,
            socl::Buffer B,
            socl::Buffer outB,
            const AxpyArguments& args
        );
    };
}
