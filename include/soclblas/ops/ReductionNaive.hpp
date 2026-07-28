#pragma once
#include "socl/Buffer.hpp"
#include "socl/Context.hpp"
#include "socl/DescriptorSet.hpp"
#include <soclblas/ops/Reduction.hpp>

namespace soclblas{
    class MaxNaive:public IndexedUnaryReduction{
        private:
        socl::Context& ctx;
        socl::ShaderPipeline pipeline;
        socl::DescriptorSet descSet;
        uint32_t thread_num;

        public:
        MaxNaive(
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
    class MinNaive:public IndexedUnaryReduction{
        private:
        socl::Context& ctx;
        socl::ShaderPipeline pipeline;
        socl::DescriptorSet descSet;
        uint32_t thread_num;

        public:
        MinNaive(
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
    class AvgNaive:public UnaryReduction{
        private:
        socl::Context& ctx;
        socl::ShaderPipeline pipeline;
        socl::DescriptorSet descSet;
        uint32_t thread_num;

        public:
        AvgNaive(
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
    class SumNaive:public UnaryReduction{
        private:
        socl::Context& ctx;
        socl::ShaderPipeline pipeline;
        socl::DescriptorSet descSet;
        uint32_t thread_num;

        public:
        SumNaive(
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
