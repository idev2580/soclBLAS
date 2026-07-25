#include "socl/ShaderPipeline.hpp"
#include <soclblas/ops/Axpy.hpp>
#include <AxpyFP32_SPIRV>

namespace soclblas{
    Axpy::Axpy(
        socl::Context& ctx,
        uint32_t thread_num
    ):ctx(ctx), thread_num(thread_num){
        this->pipeline = ctx.createShaderPipeline({
            .spirv = AxpyFP32_SPIRV,
            .bindings = {
                {0, socl::DescriptorType::StorageBuffer},
                {1, socl::DescriptorType::StorageBuffer},
            },
            .pushConstantSize = sizeof(AxpyArguments),
            .specConstants = {
                {0, socl::specConstant(std::uint32_t{thread_num})},
            }
        });
        this->descSet = ctx.createDescriptorSet(pipeline);
    }
    void Axpy::execute(
        std::span<socl::Buffer> inputs,
        std::span<socl::Buffer> inouts,
        std::span<socl::Buffer> outputs,
        const void* args,
        std::size_t argsSize
    ){
        this->descSet.bindBuffer(0, inputs[0]);
        this->descSet.bindBuffer(1, inouts[0]);
        this->descSet.update();

        ctx.begin();
        ctx.use(pipeline);
        ctx.bind(descSet);
        ctx.push(args, argsSize);

        AxpyArguments* axpyArgs = (AxpyArguments*)args;
        const uint32_t group_cnt = axpyArgs->n / thread_num + (axpyArgs->n % thread_num == 0 ? 0 : 1);
        ctx.dispatch(axpyArgs->b, group_cnt, 1);
        ctx.submitAndWait();
    }
    void Axpy::operator()(
        socl::Buffer A,
        socl::Buffer B,
        const AxpyArguments& args
    ){
        std::vector<socl::Buffer> inputs = {A};
        std::vector<socl::Buffer> inouts = {B};
        std::vector<socl::Buffer> outputs = {};
        this->execute(inputs, inouts, outputs, &args, sizeof(AxpyArguments));
    }
}
