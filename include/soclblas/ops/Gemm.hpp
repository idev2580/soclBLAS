#pragma once
#include "socl/Buffer.hpp"
#include "socl/DescriptorSet.hpp"
#include <cstdint>
#include <socl/Context.hpp>
#include <soclblas/ops/Operator.hpp>

namespace soclblas{
    struct GemmArguments{
        uint32_t b;
        uint32_t m;
        uint32_t n;
        uint32_t p;
        float alpha;
        float beta;
        uint32_t a_stride;
        uint32_t b_stride;
        uint32_t c_stride;
        uint32_t a_m_stride;
        uint32_t a_n_stride;
        uint32_t b_n_stride;
        uint32_t b_p_stride;
        uint32_t c_m_stride;
        uint32_t c_p_stride;
    };

    class Gemm: public Operator{
        private:
        socl::Context& ctx;
        socl::ShaderPipeline pipeline;
        socl::DescriptorSet descSet;

        uint32_t tile_m;
        uint32_t tile_n;
        uint32_t tile_p;

        public:
        Gemm(
            socl::Context& ctx,
            std::span<const uint32_t> shaderBytecodes,
            uint32_t tile_m = 8,
            uint32_t tile_n = 4,
            uint32_t tile_p = 4
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
        );
    };
}