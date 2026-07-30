#include "socl/ShaderPipeline.hpp"
#include <soclblas/ops/GemmOutPlace.hpp>

namespace soclblas{
    GemmOutPlace::GemmOutPlace(
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
                {3, socl::DescriptorType::StorageBuffer}
            },
            .pushConstantSize = sizeof(GemmOutPlaceArguments),
            .specConstants = {
                {0, socl::specConstant(std::uint32_t{tile_m})},
                {1, socl::specConstant(std::uint32_t{tile_n})},
                {2, socl::specConstant(std::uint32_t{tile_p})}
            }
        });
        this->descSet = ctx.createDescriptorSet(pipeline);
    }
    socl::DispatchToken GemmOutPlace::execute(
        std::span<socl::Buffer> inputs,
        std::span<socl::Buffer> inouts,
        std::span<socl::Buffer> outputs,
        const void* args,
        std::size_t argsSize
    ){
        // Implement the forward pass of GEMM operation
        this->descSet.bindBuffer(0, inputs[0]);
        this->descSet.bindBuffer(1, inputs[1]);
        this->descSet.bindBuffer(2, inputs[2]);
        this->descSet.bindBuffer(3, outputs[0]);
        this->descSet.update();

        ctx.begin();
        ctx.use(pipeline);
        ctx.bind(descSet);
        ctx.push(args, argsSize);

        GemmOutPlaceArguments* gemmArgs = (GemmOutPlaceArguments*)args;
        const uint32_t tile_r_size = tile_m;
        const uint32_t tile_c_size = tile_p;
        const uint32_t tiled_m = (gemmArgs->m / tile_r_size) + (gemmArgs->m % tile_r_size != 0);
        const uint32_t tiled_p = (gemmArgs->p / tile_c_size) + (gemmArgs->p % tile_c_size != 0);
        ctx.dispatch(gemmArgs->b, tiled_m, tiled_p);
        return ctx.submitAsync();
    }
    socl::DispatchToken GemmOutPlace::operator()(
        socl::Buffer A,
        socl::Buffer B,
        socl::Buffer C,
        socl::Buffer outC,
        const GemmOutPlaceArguments& args
    ){
        std::vector<socl::Buffer> inputs = {A, B, C};
        std::vector<socl::Buffer> inouts = {};
        std::vector<socl::Buffer> outputs = {outC};
        return this->execute(inputs, inouts, outputs, &args, sizeof(GemmOutPlaceArguments));
    }

    socl::DispatchToken GemmOutPlace::operator()(
        socl::Buffer A,
        socl::Buffer B,
        socl::Buffer C,
        socl::Buffer outC,
        const GemmArguments& args
    ){
        const GemmOutPlaceArguments outPlaceArgs = GemmOutPlaceArguments::sameOutputLayout(args);
        return (*this)(A, B, C, outC, outPlaceArgs);
    }
}
