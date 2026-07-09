#pragma once
#include <kompute/operations/OpAlgoDispatch.hpp>
#include <kompute/Kompute.hpp>

#include <ops/Gemv.hpp>
#include <GemvNaiveFP32_SPIRV>

namespace kpblas{
    class GemvNaiveFP32: public kp::OpAlgoDispatch{
        public:
        GemvNaiveFP32(
            std::vector<std::shared_ptr<kp::Tensor>> tensors, 
            std::shared_ptr<kp::Algorithm> algorithm,
            const GemvArguments& args
        );
    };
}