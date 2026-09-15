#pragma once
#include "socl/Buffer.hpp"
#include "socl/Context.hpp"
#include <soclblas/ops/Reduction.hpp>

namespace soclblas{
    class MaxNaiveFP32:public IndexedUnaryReduction{
        private:
        socl::ShaderPipeline pipeline;
        uint32_t thread_num;
        uint32_t values_per_thread;

        public:
        MaxNaiveFP32(
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
            BufferView outValue,
            BufferView outIndex,
            const IndexedUnaryReductionArguments& args
        ) override;
    };
    class MinNaiveFP32:public IndexedUnaryReduction{
        private:
        socl::ShaderPipeline pipeline;
        uint32_t thread_num;
        uint32_t values_per_thread;

        public:
        MinNaiveFP32(
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
            BufferView outValue,
            BufferView outIndex,
            const IndexedUnaryReductionArguments& args
        ) override;
    };
    class AvgNaiveFP32:public UnaryReduction{
        private:
        socl::ShaderPipeline pipeline;
        uint32_t thread_num;
        uint32_t values_per_thread;

        public:
        AvgNaiveFP32(
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
            BufferView out,
            const UnaryReductionArguments& args
        ) override;
    };
    class SumNaiveFP32:public UnaryReduction{
        private:
        socl::ShaderPipeline pipeline;
        uint32_t thread_num;
        uint32_t values_per_thread;

        public:
        SumNaiveFP32(
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
            BufferView out,
            const UnaryReductionArguments& args
        ) override;
    };
}
