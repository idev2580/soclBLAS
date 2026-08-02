#pragma once
#include "socl/Buffer.hpp"
#include "socl/DescriptorSet.hpp"
#include <cstdint>
#include <socl/Context.hpp>
#include <soclblas/ops/Operator.hpp>

namespace soclblas{
    struct MatMulArguments{
        uint32_t b;
        uint32_t m;
        uint32_t n;
        uint32_t p;
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

    class MatMul: public Operator{
        private:
        socl::Context& ctx;
        socl::ShaderPipeline pipeline;
        socl::DescriptorSet descSet;

        uint32_t tile_m;
        uint32_t tile_n;
        uint32_t tile_p;

        public:
        MatMul(
            socl::Context& ctx,
            std::span<const uint32_t> shaderBytecodes,
            uint32_t subgroup_tile_m,
            uint32_t subgroup_tile_n,
            uint32_t subgroup_tile_p,
            uint32_t subgroup_tile_cnt_m,
            uint32_t subgroup_tile_cnt_p,
            uint32_t shared_tile_n_multiplier,
            uint32_t reg_tile_m,
            uint32_t inner_tile_n,
            uint32_t reg_tile_p
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
            const MatMulArguments& args
        );
    };
}
