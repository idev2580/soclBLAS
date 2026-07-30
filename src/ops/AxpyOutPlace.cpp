#include "socl/ShaderPipeline.hpp"
#include <soclblas/ops/AxpyOutPlace.hpp>
#include <AxpyOutPlaceFP32_SPIRV>

namespace soclblas{
    AxpyOutPlaceArguments AxpyOutPlaceArguments::sameOutputLayout(const AxpyArguments& args){
        return {
            .b = args.b,
            .n = args.n,
            .alpha = args.alpha,
            .a_b_stride = args.a_b_stride,
            .a_n_stride = args.a_n_stride,
            .b_b_stride = args.b_b_stride,
            .b_n_stride = args.b_n_stride,
            .out_b_b_stride = args.b_b_stride,
            .out_b_n_stride = args.b_n_stride
        };
    }

    void AxpyOutPlaceArguments::fromInPlace(const AxpyArguments& args){
        *this = sameOutputLayout(args);
    }

    AxpyOutPlaceFP32::AxpyOutPlaceFP32(
        socl::Context& ctx,
        uint32_t thread_num
    ):ctx(ctx), thread_num(thread_num){
        this->pipeline = ctx.createShaderPipeline({
            .spirv = AxpyOutPlaceFP32_SPIRV,
            .bindings = {
                {0, socl::DescriptorType::StorageBuffer},
                {1, socl::DescriptorType::StorageBuffer},
                {2, socl::DescriptorType::StorageBuffer},
            },
            .pushConstantSize = sizeof(AxpyOutPlaceArguments),
            .specConstants = {
                {0, socl::specConstant(std::uint32_t{thread_num})},
            }
        });
        this->descSet = ctx.createDescriptorSet(pipeline);
    }

    socl::DispatchToken AxpyOutPlaceFP32::execute(
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

        const AxpyOutPlaceArguments* axpyArgs = (const AxpyOutPlaceArguments*)args;
        const uint32_t group_cnt =
            axpyArgs->n / thread_num + (axpyArgs->n % thread_num == 0 ? 0 : 1);
        ctx.dispatch(axpyArgs->b, group_cnt, 1);
        return ctx.submitAsync();
    }

    socl::DispatchToken AxpyOutPlaceFP32::operator()(
        socl::Buffer A,
        socl::Buffer B,
        socl::Buffer outB,
        const AxpyOutPlaceArguments& args
    ){
        std::vector<socl::Buffer> inputs = {A, B};
        std::vector<socl::Buffer> inouts = {};
        std::vector<socl::Buffer> outputs = {outB};
        return this->execute(inputs, inouts, outputs, &args, sizeof(AxpyOutPlaceArguments));
    }

    socl::DispatchToken AxpyOutPlaceFP32::operator()(
        socl::Buffer A,
        socl::Buffer B,
        socl::Buffer outB,
        const AxpyArguments& args
    ){
        const AxpyOutPlaceArguments outPlaceArgs =
            AxpyOutPlaceArguments::sameOutputLayout(args);
        return (*this)(A, B, outB, outPlaceArgs);
    }
}
