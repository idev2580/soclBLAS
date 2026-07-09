#pragma once
#include "ops/GemmNaive.hpp"
#include <kompute/operations/OpAlgoDispatch.hpp>
#include <kompute/Kompute.hpp>

#include <ops/Gemv.hpp>

namespace kpblas{
    class GemvNaiveFP32: public GemmNaiveFP32{
        public:
        GemvNaiveFP32(
            std::vector<std::shared_ptr<kp::Tensor>> tensors, 
            std::shared_ptr<kp::Algorithm> algorithm,
            const GemvArguments& args
        );
    };
}