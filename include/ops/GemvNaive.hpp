#pragma once
#include "ops/GemmNaive.hpp"
#include <kompute/operations/OpAlgoDispatch.hpp>
#include <kompute/Kompute.hpp>

namespace kpblas{
    class GemvNaiveFP32: public GemmNaiveFP32{
        public:
        GemvNaiveFP32(
            std::vector<std::shared_ptr<kp::Tensor>> tensors, 
            std::shared_ptr<kp::Algorithm> algorithm,
            uint32_t b,
            uint32_t m,
            uint32_t n,
            float alpha,
            float beta,
            bool a_transposed = false
        );
    };
}