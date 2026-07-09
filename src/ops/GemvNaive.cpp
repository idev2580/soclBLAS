#include <ops/GemvNaive.hpp>

namespace kpblas{
    GemvNaiveFP32::GemvNaiveFP32(
        std::vector<std::shared_ptr<kp::Tensor>> tensors, 
        std::shared_ptr<kp::Algorithm> algorithm,
        const GemvArguments& args
    ):kp::OpAlgoDispatch(algorithm){
        // Initialize specialized constants
        std::vector<float> specConstant;
        specConstant.resize(sizeof(GemvArguments) / 4);
        uint8_t* target = (uint8_t*)specConstant.data();
        memcpy(target, &args, sizeof(GemvArguments));
        
        kp::Workgroup wgrp({args.b, args.m});
        algorithm->rebuild(tensors, std::vector<uint32_t>(
            std::begin(GemvNaiveFP32_SPIRV), 
            std::end(GemvNaiveFP32_SPIRV)
            ), 
            wgrp,
            specConstant
        );
    }
}