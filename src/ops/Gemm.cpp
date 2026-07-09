#include <ops/Gemm.hpp>

namespace kpblas{
    Gemm::Gemm(
        std::vector<std::shared_ptr<kp::Tensor>> tensors, 
        std::shared_ptr<kp::Algorithm> algorithm,
        const GemmArguments& args
    ):kp::OpAlgoDispatch(algorithm){
        // Write into specConstant buffer.
        specConstant.resize(sizeof(GemmArguments) / 4);
        uint8_t* target = (uint8_t*)specConstant.data();
        memcpy(target, &args, sizeof(GemmArguments));
        
        // Tiling and other things should be done in real implementation
    }
}
