#pragma once
#include <cstdint>
#include <kompute/operations/OpAlgoDispatch.hpp>
#include <kompute/Kompute.hpp>

namespace kpblas{
    struct GemmArguments{
        uint32_t b;
        uint32_t m;
        uint32_t n;
        uint32_t p;
        float alpha;
        float beta;
    };

    class Gemm: public kp::OpAlgoDispatch{
        protected:
        std::vector<float> specConstant;
        public:
        Gemm(
            std::vector<std::shared_ptr<kp::Tensor>> tensors, 
            std::shared_ptr<kp::Algorithm> algorithm,
            const GemmArguments& args
        );
    };
}