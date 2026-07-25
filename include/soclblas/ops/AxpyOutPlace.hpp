#pragma once
#include "socl/Buffer.hpp"
#include "socl/DescriptorSet.hpp"
#include <cstdint>
#include <socl/Context.hpp>
#include <soclblas/ops/Axpy.hpp>
#include <soclblas/ops/Operator.hpp>

namespace soclblas{
    class AxpyOutPlace: public Operator{
        private:
        socl::Context& ctx;
        socl::ShaderPipeline pipeline;
        socl::DescriptorSet descSet;
        uint32_t thread_num;

        public:
        AxpyOutPlace(
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
            socl::Buffer outB,
            const AxpyOutPlaceArguments& args
        );
        virtual void operator()(
            socl::Buffer A,
            socl::Buffer B,
            socl::Buffer outB,
            const AxpyArguments& args
        );
    };
}
