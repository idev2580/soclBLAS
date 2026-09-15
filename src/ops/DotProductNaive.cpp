#include "socl/ShaderPipeline.hpp"
#include <soclblas/ops/DotProductNaive.hpp>
#include <DotProductNaiveFP32_SPIRV>

namespace soclblas{
    DotProductNaiveFP32::DotProductNaiveFP32(
        socl::Context& ctx,
        uint32_t thread_num,
        uint32_t values_per_thread
    ):thread_num(thread_num), values_per_thread(values_per_thread){
        this->pipeline = ctx.createShaderPipeline({
            .spirv = DotProductNaiveFP32_SPIRV,
            .bindings = {
                {0, socl::DescriptorType::StorageBuffer},
                {1, socl::DescriptorType::StorageBuffer},
                {2, socl::DescriptorType::StorageBuffer},
            },
            .pushConstantSize = sizeof(BinaryReductionArguments),
            .specConstants = {
                {0, socl::specConstant(std::uint32_t{thread_num})},
                {1, socl::specConstant(std::uint32_t{values_per_thread})},
            }
        });
    }

    DispatchPlan DotProductNaiveFP32::execute(
        std::span<const BufferView> inputs,
        std::span<const BufferView> inouts,
        std::span<const BufferView> outputs,
        const void* args,
        std::size_t argsSize
    ){
        const auto* reductionArgs =
            static_cast<const BinaryReductionArguments*>(args);
        return {
            .pipeline = pipeline,
            .bindings = {
                {0, inputs[0], socl::BufferAccess::Read},
                {1, inputs[1], socl::BufferAccess::Read},
                {2, outputs[0], socl::BufferAccess::Write},
            },
            .pushConstants = copyPushConstants(args, argsSize),
            .dispatchX = reductionArgs->b,
            .dispatchY = 1,
            .dispatchZ = 1,
        };
    }

    DispatchPlan DotProductNaiveFP32::operator()(
        BufferView a,
        BufferView b,
        BufferView out,
        const BinaryReductionArguments& args
    ){
        std::vector<BufferView> inputs = {a, b};
        std::vector<BufferView> inouts = {};
        std::vector<BufferView> outputs = {out};
        return this->execute(inputs, inouts, outputs, &args, sizeof(BinaryReductionArguments));
    }
}
