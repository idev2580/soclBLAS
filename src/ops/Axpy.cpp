#include "socl/ShaderPipeline.hpp"
#include <soclblas/ops/Axpy.hpp>
#include <AxpyFP32_SPIRV>

namespace soclblas{
    AxpyFP32::AxpyFP32(
        socl::Context& ctx,
        uint32_t thread_num
    ):thread_num(thread_num){
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
    }
    DispatchPlan AxpyFP32::execute(
        std::span<socl::Buffer> inputs,
        std::span<socl::Buffer> inouts,
        std::span<socl::Buffer> outputs,
        const void* args,
        std::size_t argsSize
    ){
        const auto* axpyArgs = static_cast<const AxpyArguments*>(args);
        const uint32_t group_cnt = axpyArgs->n / thread_num + (axpyArgs->n % thread_num == 0 ? 0 : 1);
        return {
            .pipeline = pipeline,
            .bindings = {
                {0, inputs[0], socl::BufferAccess::Read},
                {1, inouts[0], socl::BufferAccess::ReadWrite},
            },
            .pushConstants = copyPushConstants(args, argsSize),
            .dispatchX = axpyArgs->b,
            .dispatchY = group_cnt,
            .dispatchZ = 1,
        };
    }
    DispatchPlan AxpyFP32::operator()(
        socl::Buffer A,
        socl::Buffer B,
        const AxpyArguments& args
    ){
        std::vector<socl::Buffer> inputs = {A};
        std::vector<socl::Buffer> inouts = {B};
        std::vector<socl::Buffer> outputs = {};
        return this->execute(inputs, inouts, outputs, &args, sizeof(AxpyArguments));
    }
}
