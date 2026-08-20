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
    ):thread_num(thread_num){
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
    }

    DispatchPlan UnaryElementwiseTemplateFP32::execute(
        std::span<socl::Buffer> inputs,
        std::span<socl::Buffer> inouts,
        std::span<socl::Buffer> outputs,
        const void* args,
        std::size_t argsSize
    ){
        const auto* elementwiseArgs =
            static_cast<const UnaryElementwiseArguments*>(args);
        const uint32_t groupCount =
            elementwiseArgs->size / thread_num +
            (elementwiseArgs->size % thread_num == 0 ? 0 : 1);
        return {
            .pipeline = pipeline,
            .bindings = {
                {0, inputs[0], socl::BufferAccess::Read},
                {1, outputs[0], socl::BufferAccess::Write},
            },
            .pushConstants = copyPushConstants(args, argsSize),
            .dispatchX = groupCount,
            .dispatchY = 1,
            .dispatchZ = 1,
        };
    }

    DispatchPlan UnaryElementwiseTemplateFP32::operator()(
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
    ):thread_num(thread_num){
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
    }

    DispatchPlan BinaryElementwiseTemplateFP32::execute(
        std::span<socl::Buffer> inputs,
        std::span<socl::Buffer> inouts,
        std::span<socl::Buffer> outputs,
        const void* args,
        std::size_t argsSize
    ){
        const auto* elementwiseArgs =
            static_cast<const BinaryElementwiseArguments*>(args);
        const uint32_t groupCount =
            elementwiseArgs->size / thread_num +
            (elementwiseArgs->size % thread_num == 0 ? 0 : 1);
        return {
            .pipeline = pipeline,
            .bindings = {
                {0, inputs[0], socl::BufferAccess::Read},
                {1, inputs[1], socl::BufferAccess::Read},
                {2, outputs[0], socl::BufferAccess::Write},
            },
            .pushConstants = copyPushConstants(args, argsSize),
            .dispatchX = groupCount,
            .dispatchY = 1,
            .dispatchZ = 1,
        };
    }

    DispatchPlan BinaryElementwiseTemplateFP32::operator()(
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
