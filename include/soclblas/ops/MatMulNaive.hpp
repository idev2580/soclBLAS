#pragma once
#include "socl/Buffer.hpp"
#include "socl/Context.hpp"
#include <soclblas/ops/MatMul.hpp>
#include <MatMulNaiveFP32_SPIRV>

namespace soclblas{
    class MatMulNaiveFP32: public MatMul{
        public:
        MatMulNaiveFP32(
            socl::Context& ctx,
            uint32_t tile_m,
            uint32_t tile_n,
            uint32_t tile_k
        );
    };
}
