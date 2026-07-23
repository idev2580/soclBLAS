#pragma once
#include "soclblas/ops/Operator.hpp"
#include "soclblas/ops/Gemm.hpp"
#include <cstdint>

namespace soclblas{
    struct GemvArguments{
        uint32_t b;
        uint32_t m;
        uint32_t n;
        float alpha;
        float beta;
        uint32_t a_m_stride;
        uint32_t a_n_stride;
        uint32_t x_n_stride;
        uint32_t x_b_stride;
        uint32_t y_m_stride;
        uint32_t y_b_stride;
    };

    class Gemv: public Operator{
        private:
        Gemm gemm;
        public:
        Gemv(
            socl::Context& ctx,
            std::span<const uint32_t> gemmShaderBytecodes,
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
            socl::Buffer X,
            socl::Buffer Y,
            const GemvArguments& args
        );
    };
}