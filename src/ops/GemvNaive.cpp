#include <ops/GemvNaive.hpp>

namespace kpblas{
    GemvNaiveFP32::GemvNaiveFP32(
        std::vector<std::shared_ptr<kp::Tensor>> tensors, 
        std::shared_ptr<kp::Algorithm> algorithm,
        uint32_t b,
        uint32_t m,
        uint32_t n,
        float alpha,
        float beta
    ):kp::OpAlgoDispatch(algorithm){
        // Initialize specialized constants
        std::vector<float> specConstant;
        specConstant.resize(5);
        uint8_t* target = (uint8_t*)specConstant.data();
        memcpy(target, &b, 4);
        memcpy(target+4, &m, 4);
        memcpy(target+8, &n, 4);
        memcpy(target+12, &alpha, 4);
        memcpy(target+16, &beta, 4);

        kp::Workgroup wgrp({b, m});

        algorithm->rebuild(tensors, std::vector<uint32_t>(
            std::begin(GemvNaiveFP32_SPIRV), 
            std::end(GemvNaiveFP32_SPIRV)
            ), 
            wgrp,
            specConstant
        );
    }
}