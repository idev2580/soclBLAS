#pragma once
#include "soclblas/ops/Operator.hpp"
#include "soclblas/ops/Gemm.hpp"
#include <soclblas/ops/GemvArguments.hpp>
#include <cstdint>

namespace soclblas{
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
        virtual socl::DispatchToken execute(
            std::span<socl::Buffer> inputs,
            std::span<socl::Buffer> inouts,
            std::span<socl::Buffer> outputs,
            const void* args,
            std::size_t argsSize
        ) override;

        virtual socl::DispatchToken operator()(
            socl::Buffer A,
            socl::Buffer X,
            socl::Buffer Y,
            const GemvArguments& args
        );
    };
}
