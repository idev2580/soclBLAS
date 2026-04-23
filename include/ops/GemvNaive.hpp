#pragma once
#include <kompute/operations/OpAlgoDispatch.hpp>
#include <kompute/Kompute.hpp>

#include <GemvNaiveFP32_SPIRV>

namespace kpblas{
    class GemvNaiveFP32: public kp::OpAlgoDispatch{
        public:
        GemvNaiveFP32(
            std::vector<std::shared_ptr<kp::Tensor>> tensors, 
            std::shared_ptr<kp::Algorithm> algorithm,
            uint32_t b,
            uint32_t m,
            uint32_t n,
            float alpha,
            float beta
        );
    };
}