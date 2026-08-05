#include "ShaderTemplate.hpp"
#include "ShaderTemplates.hpp"
#include <soclblas/ops/ElementWiseTemplate.hpp>

namespace soclblas{
    namespace{
        constexpr std::string_view operationMarker =
            "/*__SOCLBLAS_OPERATION__*/";
    }

    UnaryElementwiseTemplateFP32::UnaryElementwiseTemplateFP32(
        socl::Context& ctx,
        std::string_view operationSource,
        uint32_t thread_num
    ):UnaryElementwiseTemplateFP32(
        ctx,
        detail::compileShaderTemplate(
            detail::unaryElementwiseShaderTemplate(),
            operationSource,
            "UnaryElementwiseTemplateFP32.comp",
            operationMarker
        ),
        thread_num
    ){}

    UnaryElementwiseTemplateFP32::UnaryElementwiseTemplateFP32(
        socl::Context& ctx,
        std::vector<uint32_t> spirv,
        uint32_t thread_num
    ):ctx(ctx), thread_num(thread_num){
        this->pipeline = ctx.createShaderPipeline({
            .spirv = spirv,
            .bindings = {
                {0, socl::DescriptorType::StorageBuffer},
                {1, socl::DescriptorType::StorageBuffer},
            },
            .pushConstantSize = sizeof(UnaryElementwiseArguments),
            .specConstants = {
                {0, socl::specConstant(std::uint32_t{thread_num})},
            }
        });
        this->descSet = ctx.createDescriptorSet(pipeline);
    }

    socl::DispatchToken UnaryElementwiseTemplateFP32::execute(
        std::span<socl::Buffer> inputs,
        std::span<socl::Buffer> inouts,
        std::span<socl::Buffer> outputs,
        const void* args,
        std::size_t argsSize
    ){
        this->descSet.bindBuffer(0, inputs[0]);
        this->descSet.bindBuffer(1, outputs[0]);
        this->descSet.update();

        ctx.begin();
        ctx.use(pipeline);
        ctx.bind(descSet);
        ctx.push(args, argsSize);

        const UnaryElementwiseArguments* elementwiseArgs =
            (const UnaryElementwiseArguments*)args;
        const uint32_t groupCount =
            elementwiseArgs->size / thread_num +
            (elementwiseArgs->size % thread_num == 0 ? 0 : 1);
        ctx.dispatch(groupCount, 1, 1);
        return ctx.submitAsync();
    }

    socl::DispatchToken UnaryElementwiseTemplateFP32::operator()(
        socl::Buffer a,
        socl::Buffer out,
        const UnaryElementwiseArguments& args
    ){
        std::vector<socl::Buffer> inputs = {a};
        std::vector<socl::Buffer> inouts = {};
        std::vector<socl::Buffer> outputs = {out};
        return this->execute(
            inputs,
            inouts,
            outputs,
            &args,
            sizeof(UnaryElementwiseArguments)
        );
    }

    BinaryElementwiseTemplateFP32::BinaryElementwiseTemplateFP32(
        socl::Context& ctx,
        std::string_view operationSource,
        uint32_t thread_num
    ):BinaryElementwiseTemplateFP32(
        ctx,
        detail::compileShaderTemplate(
            detail::binaryElementwiseShaderTemplate(),
            operationSource,
            "BinaryElementwiseTemplateFP32.comp",
            operationMarker
        ),
        thread_num
    ){}

    BinaryElementwiseTemplateFP32::BinaryElementwiseTemplateFP32(
        socl::Context& ctx,
        std::vector<uint32_t> spirv,
        uint32_t thread_num
    ):ctx(ctx), thread_num(thread_num){
        this->pipeline = ctx.createShaderPipeline({
            .spirv = spirv,
            .bindings = {
                {0, socl::DescriptorType::StorageBuffer},
                {1, socl::DescriptorType::StorageBuffer},
                {2, socl::DescriptorType::StorageBuffer},
            },
            .pushConstantSize = sizeof(BinaryElementwiseArguments),
            .specConstants = {
                {0, socl::specConstant(std::uint32_t{thread_num})},
            }
        });
        this->descSet = ctx.createDescriptorSet(pipeline);
    }

    socl::DispatchToken BinaryElementwiseTemplateFP32::execute(
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

        const BinaryElementwiseArguments* elementwiseArgs =
            (const BinaryElementwiseArguments*)args;
        const uint32_t groupCount =
            elementwiseArgs->size / thread_num +
            (elementwiseArgs->size % thread_num == 0 ? 0 : 1);
        ctx.dispatch(groupCount, 1, 1);
        return ctx.submitAsync();
    }

    socl::DispatchToken BinaryElementwiseTemplateFP32::operator()(
        socl::Buffer a,
        socl::Buffer b,
        socl::Buffer out,
        const BinaryElementwiseArguments& args
    ){
        std::vector<socl::Buffer> inputs = {a, b};
        std::vector<socl::Buffer> inouts = {};
        std::vector<socl::Buffer> outputs = {out};
        return this->execute(
            inputs,
            inouts,
            outputs,
            &args,
            sizeof(BinaryElementwiseArguments)
        );
    }
}
