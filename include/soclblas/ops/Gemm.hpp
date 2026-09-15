#pragma once
#include "socl/Buffer.hpp"
#include <cstdint>
#include <socl/Context.hpp>
#include <soclblas/ops/Operator.hpp>
#include <soclblas/ops/GemmArguments.hpp>

namespace soclblas{
    class Gemm: public Operator{
        private:
        socl::ShaderPipeline pipeline;

        uint32_t tile_m;
        uint32_t tile_n;
        uint32_t tile_p;

        protected:
        Gemm(
            socl::Context& ctx
        );

        public:
        Gemm(
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
        virtual DispatchPlan execute(
            std::span<const BufferView> inputs,
            std::span<const BufferView> inouts,
            std::span<const BufferView> outputs,
            const void* args,
            std::size_t argsSize
        ) override;

        virtual DispatchPlan operator()(
            BufferView A,
            BufferView B,
            BufferView C,
            const GemmArguments& args
        );
    };
}
