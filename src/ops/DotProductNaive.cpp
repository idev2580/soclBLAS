#include "socl/ShaderPipeline.hpp"
#include <soclblas/ops/DotProductNaive.hpp>
#include <DotProductNaiveFP32_SPIRV>

namespace soclblas{
    DotProductNaiveFP32::DotProductNaiveFP32(
        socl::Context& ctx,
        uint32_t thread_num
    ):ctx(ctx), thread_num(thread_num){
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
            }
        });
        this->descSet = ctx.createDescriptorSet(pipeline);
    }

    socl::DispatchToken DotProductNaiveFP32::execute(
        std::span<socl::Buffer> inputs,
        std::span<socl::Buffer> inouts,
        std::span<socl::Buffer> outputs,
        const void* args,
        std::size_t argsSize
    ){
        this->descSet.bindBuffer(0, inputs[0]);
        this->descSet.bindBuffer(1, inputs[1]);
        this->descSet.bindBuffer(2, outputs[0]);
        this->descSet.update();

        ctx.begin();
        ctx.use(pipeline);
        ctx.bind(descSet);
        ctx.push(args, argsSize);

        const BinaryReductionArguments* reductionArgs =
            (const BinaryReductionArguments*)args;
        ctx.dispatch(reductionArgs->b, 1, 1);
        return ctx.submitAsync();
    }

    socl::DispatchToken DotProductNaiveFP32::operator()(
        socl::Buffer a,
        socl::Buffer b,
        socl::Buffer out,
        const BinaryReductionArguments& args
    ){
        std::vector<socl::Buffer> inputs = {a, b};
        std::vector<socl::Buffer> inouts = {};
        std::vector<socl::Buffer> outputs = {out};
        return this->execute(inputs, inouts, outputs, &args, sizeof(BinaryReductionArguments));
    }
}
