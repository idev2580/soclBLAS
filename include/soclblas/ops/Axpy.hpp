#pragma once
#include "socl/Buffer.hpp"
#include "socl/DescriptorSet.hpp"
#include <cstdint>
#include <socl/Context.hpp>
#include <soclblas/ops/Operator.hpp>

namespace soclblas{
    // TODO: For broadcasted operations, elementwise operators should also need to support strided operation.
    struct AxpyArguments{
        uint32_t b;
        uint32_t n;
        float alpha;
        uint32_t a_b_stride;
        uint32_t a_n_stride;
        uint32_t b_b_stride;
        uint32_t b_n_stride;
    };
    struct AxpyOutPlaceArguments{
        uint32_t b;
        uint32_t n;
        float alpha;
        uint32_t a_b_stride;
        uint32_t a_n_stride;
        uint32_t b_b_stride;
        uint32_t b_n_stride;
        uint32_t out_b_b_stride;
        uint32_t out_b_n_stride;

        static AxpyOutPlaceArguments sameOutputLayout(const AxpyArguments& args);
        void fromInPlace(const AxpyArguments& args);
    };
    class AxpyFP32: public Operator{
        private:
        socl::Context& ctx;
        socl::ShaderPipeline pipeline;
        socl::DescriptorSet descSet;
        uint32_t thread_num;

        public:
        AxpyFP32(
            socl::Context& ctx,
            uint32_t thread_num = 64
        );
        virtual socl::DispatchToken execute(
            std::span<socl::Buffer> inputs,
            std::span<socl::Buffer> inouts,
            std::span<socl::Buffer> outputs,
            const void* args,
            std::size_t argsSize
        ) override;

        virtual socl::DispatchToken operator()(
            socl::Buffer A,
            socl::Buffer B,
            const AxpyArguments& args
        );
    };
}
