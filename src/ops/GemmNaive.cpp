#include <ops/GemmNaive.hpp>

//Should have same size as the shader code.
#define TILE_SIZE 4 

namespace kpblas{
    GemmNaiveFP32::GemmNaiveFP32(
        std::vector<std::shared_ptr<kp::Tensor>> tensors, 
        std::shared_ptr<kp::Algorithm> algorithm,
        uint32_t b,
        uint32_t m,
        uint32_t n,
        uint32_t p,
        float alpha,
        float beta
    ):kp::OpAlgoDispatch(algorithm){
        // Initialize specialized constants
        std::vector<float> specConstant;
        specConstant.resize(6);
        uint8_t* target = (uint8_t*)specConstant.data();
        memcpy(target, &b, 4);
        memcpy(target+4, &m, 4);
        memcpy(target+8, &n, 4);
        memcpy(target+12, &p, 4);
        memcpy(target+16, &alpha, 4);
        memcpy(target+20, &beta, 4);


        const uint32_t tiled_m = (m / TILE_SIZE) + (m % TILE_SIZE != 0);
        const uint32_t tiled_p = (p / TILE_SIZE) + (p % TILE_SIZE != 0);
        auto wgrp = kp::Workgroup({b, tiled_m, tiled_p});
        
        algorithm->rebuild(tensors, std::vector<uint32_t>(
            std::begin(GemmNaiveFP32_SPIRV), 
            std::end(GemmNaiveFP32_SPIRV)
            ), 
            wgrp,
            specConstant
        );
    }
}