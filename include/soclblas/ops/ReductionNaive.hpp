#pragma once
#include "socl/Buffer.hpp"
#include "socl/Context.hpp"
#include "socl/DescriptorSet.hpp"
#include <soclblas/ops/Reduction.hpp>

namespace soclblas{
    class MaxNaiveFP32:public IndexedUnaryReduction{
        private:
        socl::Context& ctx;
        socl::ShaderPipeline pipeline;
        socl::DescriptorSet descSet;
        uint32_t thread_num;

        public:
        MaxNaiveFP32(
            socl::Context& ctx,
            uint32_t thread_num = 32
        );

        virtual void execute(
            std::span<socl::Buffer> inputs,
            std::span<socl::Buffer> inouts,
            std::span<socl::Buffer> outputs,
            const void* args,
            std::size_t argsSize
        ) override;

        virtual void operator()(
            socl::Buffer a,
            socl::Buffer outValue,
            socl::Buffer outIndex,
            const IndexedUnaryReductionArguments& args
        ) override;
    };
    class MinNaiveFP32:public IndexedUnaryReduction{
        private:
        socl::Context& ctx;
        socl::ShaderPipeline pipeline;
        socl::DescriptorSet descSet;
        uint32_t thread_num;

        public:
        MinNaiveFP32(
            socl::Context& ctx,
            uint32_t thread_num = 32
        );

        virtual void execute(
            std::span<socl::Buffer> inputs,
            std::span<socl::Buffer> inouts,
            std::span<socl::Buffer> outputs,
            const void* args,
            std::size_t argsSize
        ) override;

        virtual void operator()(
            socl::Buffer a,
            socl::Buffer outValue,
            socl::Buffer outIndex,
            const IndexedUnaryReductionArguments& args
        ) override;
    };
    class AvgNaiveFP32:public UnaryReduction{
        private:
        socl::Context& ctx;
        socl::ShaderPipeline pipeline;
        socl::DescriptorSet descSet;
        uint32_t thread_num;

        public:
        AvgNaiveFP32(
            socl::Context& ctx,
            uint32_t thread_num = 32
        );

        virtual void execute(
            std::span<socl::Buffer> inputs,
            std::span<socl::Buffer> inouts,
            std::span<socl::Buffer> outputs,
            const void* args,
            std::size_t argsSize
        ) override;

        virtual void operator()(
            socl::Buffer a,
            socl::Buffer out,
            const UnaryReductionArguments& args
        ) override;
    };
    class SumNaiveFP32:public UnaryReduction{
        private:
        socl::Context& ctx;
        socl::ShaderPipeline pipeline;
        socl::DescriptorSet descSet;
        uint32_t thread_num;

        public:
        SumNaiveFP32(
            socl::Context& ctx,
            uint32_t thread_num = 32
        );

        virtual void execute(
            std::span<socl::Buffer> inputs,
            std::span<socl::Buffer> inouts,
            std::span<socl::Buffer> outputs,
            const void* args,
            std::size_t argsSize
        ) override;

        virtual void operator()(
            socl::Buffer a,
            socl::Buffer out,
            const UnaryReductionArguments& args
        ) override;
    };
}
