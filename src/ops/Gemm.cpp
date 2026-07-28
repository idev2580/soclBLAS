#include "socl/ShaderPipeline.hpp"
#include <soclblas/ops/Gemm.hpp>

namespace soclblas{
    Gemm::Gemm(
        socl::Context& ctx
    ):ctx(ctx), tile_m(1), tile_n(1), tile_p(1){}

    Gemm::Gemm(
        socl::Context& ctx,
        std::span<const uint32_t> shaderBytecodes,
        uint32_t tile_m,
        uint32_t tile_n,
        uint32_t tile_p
    ):ctx(ctx), tile_m(tile_m), tile_n(tile_n), tile_p(tile_p){
        this->pipeline = ctx.createShaderPipeline({
            .spirv = shaderBytecodes,
            .bindings = {
                {0, socl::DescriptorType::StorageBuffer},
                {1, socl::DescriptorType::StorageBuffer},
                {2, socl::DescriptorType::StorageBuffer},
            },
            .pushConstantSize = sizeof(GemmArguments),
            .specConstants = {
                {0, socl::specConstant(std::uint32_t{tile_m})},
                {1, socl::specConstant(std::uint32_t{tile_n})},
                {2, socl::specConstant(std::uint32_t{tile_p})}
            }
        });
        this->descSet = ctx.createDescriptorSet(pipeline);
    }
    void Gemm::execute(
        std::span<socl::Buffer> inputs,
        std::span<socl::Buffer> inouts,
        std::span<socl::Buffer> outputs,
        const void* args,
        std::size_t argsSize
    ){
        // Implement the forward pass of GEMM operation
        this->descSet.bindBuffer(0, inputs[0]);
        this->descSet.bindBuffer(1, inputs[1]);
        this->descSet.bindBuffer(2, inouts[0]);
        this->descSet.update();

        ctx.begin();
        ctx.use(pipeline);
        ctx.bind(descSet);
        ctx.push(args, argsSize);

        GemmArguments* gemmArgs = (GemmArguments*)args;
        const uint32_t tile_r_size = tile_m;
        const uint32_t tile_c_size = tile_p;
        const uint32_t tiled_m = (gemmArgs->m / tile_r_size) + (gemmArgs->m % tile_r_size != 0);
        const uint32_t tiled_p = (gemmArgs->p / tile_c_size) + (gemmArgs->p % tile_c_size != 0);
        ctx.dispatch(gemmArgs->b, tiled_m, tiled_p);
        ctx.submitAndWait();
    }
    void Gemm::operator()(
        socl::Buffer A,
        socl::Buffer B,
        socl::Buffer C,
        const GemmArguments& args
    ){
        std::vector<socl::Buffer> inputs = {A, B};
        std::vector<socl::Buffer> inouts = {C};
        std::vector<socl::Buffer> outputs = {};
        this->execute(inputs, inouts, outputs, &args, sizeof(GemmArguments));
    }
}
