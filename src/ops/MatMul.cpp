#include "socl/ShaderPipeline.hpp"
#include <soclblas/ops/MatMul.hpp>

namespace soclblas{
    MatMul::MatMul(
        socl::Context& ctx,
        std::span<const uint32_t> shaderBytecodes,
        uint32_t tile_m,
        uint32_t tile_n,
        uint32_t tile_k
    ):ctx(ctx), tile_m(tile_m), tile_n(tile_n), tile_k(tile_k){
        this->pipeline = ctx.createShaderPipeline({
            .spirv = shaderBytecodes,
            .bindings = {
                {0, socl::DescriptorType::StorageBuffer},
                {1, socl::DescriptorType::StorageBuffer},
                {2, socl::DescriptorType::StorageBuffer},
            },
            .pushConstantSize = sizeof(MatMulArguments),
            .specConstants = {
                {0, socl::specConstant(std::uint32_t{tile_m})},
                {1, socl::specConstant(std::uint32_t{tile_n})},
                {2, socl::specConstant(std::uint32_t{tile_k})}
            }
        });
        this->descSet = ctx.createDescriptorSet(pipeline);
    }
    void MatMul::execute(
        std::span<socl::Buffer> inputs,
        std::span<socl::Buffer> inouts,
        std::span<socl::Buffer> outputs,
        const void* args,
        std::size_t argsSize
    ){
        // Implement the forward pass of GEMM operation
        this->descSet.bindBuffer(0, inputs[0]);
        this->descSet.bindBuffer(1, inputs[1]);
        this->descSet.bindBuffer(2, outputs[0]);
        this->descSet.update();

        ctx.begin();
        ctx.use(pipeline);
        ctx.bind(descSet);
        ctx.push(args, argsSize);

        MatMulArguments* matmulArgs = (MatMulArguments*)args;
        const uint32_t tile_r_size = tile_m;
        const uint32_t tile_c_size = tile_n;
        const uint32_t tiled_m = (matmulArgs->m / tile_r_size) + (matmulArgs->m % tile_r_size != 0);
        const uint32_t tiled_p = (matmulArgs->p / tile_c_size) + (matmulArgs->p % tile_c_size != 0);
        ctx.dispatch(matmulArgs->b, tiled_m, tiled_p);
        ctx.submitAndWait();
    }
    void MatMul::operator()(
        socl::Buffer A,
        socl::Buffer B,
        socl::Buffer C,
        const MatMulArguments& args
    ){
        std::vector<socl::Buffer> inputs = {A, B};
        std::vector<socl::Buffer> inouts = {};
        std::vector<socl::Buffer> outputs = {C};
        this->execute(inputs, inouts, outputs, &args, sizeof(MatMulArguments));
    }
}
