#pragma once
#include <ops/Gemm.hpp>
#include <GemmNaiveFP32_SPIRV>

namespace kpblas{
    class GemmNaiveFP32: public Gemm{
        public:
        GemmNaiveFP32(
            std::vector<std::shared_ptr<kp::Tensor>> tensors, 
            std::shared_ptr<kp::Algorithm> algorithm,
            const GemmArguments& args
        );
    };
}