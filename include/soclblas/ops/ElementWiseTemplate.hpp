#pragma once
#include <soclblas/ops/ElementWise.hpp>

#include <cstdint>
#include <string_view>
#include <vector>

namespace soclblas{
    class UnaryElementwiseTemplateFP32: public UnaryElementwise{
        private:
        socl::Context& ctx;
        socl::ShaderPipeline pipeline;
        socl::DescriptorSet descSet;
        uint32_t thread_num;

        UnaryElementwiseTemplateFP32(
            socl::Context& ctx,
            std::vector<uint32_t> spirv,
            uint32_t thread_num
        );

        public:
        UnaryElementwiseTemplateFP32(
            socl::Context& ctx,
            std::string_view operationSource,
            uint32_t thread_num = 64
        );

        virtual socl::DispatchToken execute(
            std::span<socl::Buffer> inputs,
            std::span<socl::Buffer> inouts,
            std::span<socl::Buffer> outputs,
            const void* args,
            std::size_t argsSize
        ) override;

        virtual socl::DispatchToken operator()(
            socl::Buffer a,
            socl::Buffer out,
            const UnaryElementwiseArguments& args
        ) override;
    };

    class BinaryElementwiseTemplateFP32: public BinaryElementwise{
        private:
        socl::Context& ctx;
        socl::ShaderPipeline pipeline;
        socl::DescriptorSet descSet;
        uint32_t thread_num;

        BinaryElementwiseTemplateFP32(
            socl::Context& ctx,
            std::vector<uint32_t> spirv,
            uint32_t thread_num
        );

        public:
        BinaryElementwiseTemplateFP32(
            socl::Context& ctx,
            std::string_view operationSource,
            uint32_t thread_num = 64
        );

        virtual socl::DispatchToken execute(
            std::span<socl::Buffer> inputs,
            std::span<socl::Buffer> inouts,
            std::span<socl::Buffer> outputs,
            const void* args,
            std::size_t argsSize
        ) override;

        virtual socl::DispatchToken operator()(
            socl::Buffer a,
            socl::Buffer b,
            socl::Buffer out,
            const BinaryElementwiseArguments& args
        ) override;
    };
}
