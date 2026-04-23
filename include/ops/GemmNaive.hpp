#pragma once
#include <kompute/operations/OpAlgoDispatch.hpp>
#include <kompute/Kompute.hpp>

#include <GemmNaiveFP32_SPIRV>

namespace kpblas{
    class GemmNaiveFP32: public kp::OpAlgoDispatch{
        public:
        GemmNaiveFP32(
            std::vector<std::shared_ptr<kp::Tensor>> tensors, 
            std::shared_ptr<kp::Algorithm> algorithm,
            uint32_t b,
            uint32_t m,
            uint32_t n,
            uint32_t p,
            float alpha,
            float beta
        );
    };
}