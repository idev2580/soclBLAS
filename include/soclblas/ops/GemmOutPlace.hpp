#pragma once
#include "socl/Buffer.hpp"
#include "socl/DescriptorSet.hpp"
#include <cstdint>
#include <socl/Context.hpp>
#include <soclblas/ops/Operator.hpp>
#include <soclblas/ops/GemmArguments.hpp>

namespace soclblas{
    class GemmOutPlace: public Operator{
        private:
        socl::Context& ctx;
        socl::ShaderPipeline pipeline;
        socl::DescriptorSet descSet;

        uint32_t tile_m;
        uint32_t tile_n;
        uint32_t tile_p;

        public:
        GemmOutPlace(
            socl::Context& ctx,
            std::span<const uint32_t> shaderBytecodes,
            uint32_t tile_m = 8,
            uint32_t tile_n = 4,
            uint32_t tile_p = 4
        );
        virtual socl::DispatchToken execute(
            std::span<socl::Buffer> inputs,
            std::span<socl::Buffer> inouts,
            std::span<socl::Buffer> outputs,
            const void* args,
            std::size_t argsSize
        ) override;

        virtual socl::DispatchToken operator()(
            socl::Buffer A,
            socl::Buffer B,
            socl::Buffer C,
            socl::Buffer outC,
            const GemmOutPlaceArguments& args
        );
        virtual socl::DispatchToken operator()(
            socl::Buffer A,
            socl::Buffer B,
            socl::Buffer C,
            socl::Buffer outC,
            const GemmArguments& args
        );
    };
}
