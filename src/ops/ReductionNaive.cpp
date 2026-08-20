#include "socl/ShaderPipeline.hpp"
#include <soclblas/ops/ReductionNaive.hpp>
#include <AvgNaiveFP32_SPIRV>
#include <MaxNaiveFP32_SPIRV>
#include <MinNaiveFP32_SPIRV>
#include <SumNaiveFP32_SPIRV>

namespace soclblas{
    MaxNaiveFP32::MaxNaiveFP32(
        socl::Context& ctx,
        uint32_t thread_num,
        uint32_t values_per_thread
    ):thread_num(thread_num), values_per_thread(values_per_thread){
        this->pipeline = ctx.createShaderPipeline({
            .spirv = MaxNaiveFP32_SPIRV,
            .bindings = {
                {0, socl::DescriptorType::StorageBuffer},
                {1, socl::DescriptorType::StorageBuffer},
                {2, socl::DescriptorType::StorageBuffer},
            },
            .pushConstantSize = sizeof(IndexedUnaryReductionArguments),
            .specConstants = {
                {0, socl::specConstant(std::uint32_t{thread_num})},
                {1, socl::specConstant(std::uint32_t{values_per_thread})},
            }
        });
    }

    DispatchPlan MaxNaiveFP32::execute(
        std::span<socl::Buffer> inputs,
        std::span<socl::Buffer> inouts,
        std::span<socl::Buffer> outputs,
        const void* args,
        std::size_t argsSize
    ){
        const auto* reductionArgs =
            static_cast<const IndexedUnaryReductionArguments*>(args);
        return {
            .pipeline = pipeline,
            .bindings = {
                {0, inputs[0], socl::BufferAccess::Read},
                {1, outputs[0], socl::BufferAccess::Write},
                {2, outputs[1], socl::BufferAccess::Write},
            },
            .pushConstants = copyPushConstants(args, argsSize),
            .dispatchX = reductionArgs->b,
            .dispatchY = 1,
            .dispatchZ = 1,
        };
    }

    DispatchPlan MaxNaiveFP32::operator()(
        socl::Buffer a,
        socl::Buffer outValue,
        socl::Buffer outIndex,
        const IndexedUnaryReductionArguments& args
    ){
        std::vector<socl::Buffer> inputs = {a};
        std::vector<socl::Buffer> inouts = {};
        std::vector<socl::Buffer> outputs = {outValue, outIndex};
        return this->execute(
            inputs,
            inouts,
            outputs,
            &args,
            sizeof(IndexedUnaryReductionArguments)
        );
    }

    MinNaiveFP32::MinNaiveFP32(
        socl::Context& ctx,
        uint32_t thread_num,
        uint32_t values_per_thread
    ):thread_num(thread_num), values_per_thread(values_per_thread){
        this->pipeline = ctx.createShaderPipeline({
            .spirv = MinNaiveFP32_SPIRV,
            .bindings = {
                {0, socl::DescriptorType::StorageBuffer},
                {1, socl::DescriptorType::StorageBuffer},
                {2, socl::DescriptorType::StorageBuffer},
            },
            .pushConstantSize = sizeof(IndexedUnaryReductionArguments),
            .specConstants = {
                {0, socl::specConstant(std::uint32_t{thread_num})},
                {1, socl::specConstant(std::uint32_t{values_per_thread})},
            }
        });
    }

    DispatchPlan MinNaiveFP32::execute(
        std::span<socl::Buffer> inputs,
        std::span<socl::Buffer> inouts,
        std::span<socl::Buffer> outputs,
        const void* args,
        std::size_t argsSize
    ){
        const auto* reductionArgs =
            static_cast<const IndexedUnaryReductionArguments*>(args);
        return {
            .pipeline = pipeline,
            .bindings = {
                {0, inputs[0], socl::BufferAccess::Read},
                {1, outputs[0], socl::BufferAccess::Write},
                {2, outputs[1], socl::BufferAccess::Write},
            },
            .pushConstants = copyPushConstants(args, argsSize),
            .dispatchX = reductionArgs->b,
            .dispatchY = 1,
            .dispatchZ = 1,
        };
    }

    DispatchPlan MinNaiveFP32::operator()(
        socl::Buffer a,
        socl::Buffer outValue,
        socl::Buffer outIndex,
        const IndexedUnaryReductionArguments& args
    ){
        std::vector<socl::Buffer> inputs = {a};
        std::vector<socl::Buffer> inouts = {};
        std::vector<socl::Buffer> outputs = {outValue, outIndex};
        return this->execute(
            inputs,
            inouts,
            outputs,
            &args,
            sizeof(IndexedUnaryReductionArguments)
        );
    }

    AvgNaiveFP32::AvgNaiveFP32(
        socl::Context& ctx,
        uint32_t thread_num,
        uint32_t values_per_thread
    ):thread_num(thread_num), values_per_thread(values_per_thread){
        this->pipeline = ctx.createShaderPipeline({
            .spirv = AvgNaiveFP32_SPIRV,
            .bindings = {
                {0, socl::DescriptorType::StorageBuffer},
                {1, socl::DescriptorType::StorageBuffer},
            },
            .pushConstantSize = sizeof(UnaryReductionArguments),
            .specConstants = {
                {0, socl::specConstant(std::uint32_t{thread_num})},
                {1, socl::specConstant(std::uint32_t{values_per_thread})},
            }
        });
    }

    DispatchPlan AvgNaiveFP32::execute(
        std::span<socl::Buffer> inputs,
        std::span<socl::Buffer> inouts,
        std::span<socl::Buffer> outputs,
        const void* args,
        std::size_t argsSize
    ){
        const auto* reductionArgs =
            static_cast<const UnaryReductionArguments*>(args);
        return {
            .pipeline = pipeline,
            .bindings = {
                {0, inputs[0], socl::BufferAccess::Read},
                {1, outputs[0], socl::BufferAccess::Write},
            },
            .pushConstants = copyPushConstants(args, argsSize),
            .dispatchX = reductionArgs->b,
            .dispatchY = 1,
            .dispatchZ = 1,
        };
    }

    DispatchPlan AvgNaiveFP32::operator()(
        socl::Buffer a,
        socl::Buffer out,
        const UnaryReductionArguments& args
    ){
        std::vector<socl::Buffer> inputs = {a};
        std::vector<socl::Buffer> inouts = {};
        std::vector<socl::Buffer> outputs = {out};
        return this->execute(inputs, inouts, outputs, &args, sizeof(UnaryReductionArguments));
    }

    SumNaiveFP32::SumNaiveFP32(
        socl::Context& ctx,
        uint32_t thread_num,
        uint32_t values_per_thread
    ):thread_num(thread_num), values_per_thread(values_per_thread){
        this->pipeline = ctx.createShaderPipeline({
            .spirv = SumNaiveFP32_SPIRV,
            .bindings = {
                {0, socl::DescriptorType::StorageBuffer},
                {1, socl::DescriptorType::StorageBuffer},
            },
            .pushConstantSize = sizeof(UnaryReductionArguments),
            .specConstants = {
                {0, socl::specConstant(std::uint32_t{thread_num})},
                {1, socl::specConstant(std::uint32_t{values_per_thread})},
            }
        });
    }

    DispatchPlan SumNaiveFP32::execute(
        std::span<socl::Buffer> inputs,
        std::span<socl::Buffer> inouts,
        std::span<socl::Buffer> outputs,
        const void* args,
        std::size_t argsSize
    ){
        const auto* reductionArgs =
            static_cast<const UnaryReductionArguments*>(args);
        return {
            .pipeline = pipeline,
            .bindings = {
                {0, inputs[0], socl::BufferAccess::Read},
                {1, outputs[0], socl::BufferAccess::Write},
            },
            .pushConstants = copyPushConstants(args, argsSize),
            .dispatchX = reductionArgs->b,
            .dispatchY = 1,
            .dispatchZ = 1,
        };
    }

    DispatchPlan SumNaiveFP32::operator()(
        socl::Buffer a,
        socl::Buffer out,
        const UnaryReductionArguments& args
    ){
        std::vector<socl::Buffer> inputs = {a};
        std::vector<socl::Buffer> inouts = {};
        std::vector<socl::Buffer> outputs = {out};
        return this->execute(inputs, inouts, outputs, &args, sizeof(UnaryReductionArguments));
    }
}
