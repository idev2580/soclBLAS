#include <ops/GemmNaive.hpp>

//Should have same size as the shader code.
#define TILE_R_SIZE 8
#define TILE_C_SIZE 4 

namespace kpblas{
    GemmNaiveFP32::GemmNaiveFP32(
        std::vector<std::shared_ptr<kp::Tensor>> tensors, 
        std::shared_ptr<kp::Algorithm> algorithm,
        const GemmArguments& args
    ):Gemm(
        tensors, algorithm, args
    ){
        const uint32_t tiled_m = (args.m / TILE_R_SIZE) + (args.m % TILE_R_SIZE != 0);
        const uint32_t tiled_p = (args.p / TILE_C_SIZE) + (args.p % TILE_C_SIZE != 0);
        auto wgrp = kp::Workgroup({args.b, tiled_m, tiled_p});
        
        algorithm->rebuild(tensors, std::vector<uint32_t>(
            std::begin(GemmNaiveFP32_SPIRV), 
            std::end(GemmNaiveFP32_SPIRV)
            ), 
            wgrp,
            specConstant
        );
    }
}