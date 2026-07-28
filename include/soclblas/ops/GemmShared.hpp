#pragma once
#include "socl/Buffer.hpp"
#include "socl/Context.hpp"
#include "socl/DescriptorSet.hpp"
#include <soclblas/ops/Gemm.hpp>

namespace soclblas{
    class GemmSharedFP32: public Gemm{
        private:
        socl::Context& ctx;
        socl::ShaderPipeline pipeline;
        socl::DescriptorSet descSet;
        uint32_t block_m;
        uint32_t block_n;
        uint32_t block_p;
        uint32_t thread_tile_m;
        uint32_t thread_tile_p;

        public:
        GemmSharedFP32(
            socl::Context& ctx,
            uint32_t block_m = 64,
            uint32_t block_n = 16,
            uint32_t block_p = 64,
            uint32_t thread_tile_m = 4,
            uint32_t thread_tile_p = 4
        );

        virtual void execute(
            std::span<socl::Buffer> inputs,
            std::span<socl::Buffer> inouts,
            std::span<socl::Buffer> outputs,
            const void* args,
            std::size_t argsSize
        ) override;

        virtual void operator()(
            socl::Buffer A,
            socl::Buffer B,
            socl::Buffer C,
            const GemmArguments& args
        ) override;
    };

}
