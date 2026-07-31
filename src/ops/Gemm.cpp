#include "socl/ShaderPipeline.hpp"
#include <soclblas/ops/Gemm.hpp>

namespace soclblas{
    Gemm::Gemm(
        socl::Context& ctx
    ):ctx(ctx), tile_m(1), tile_n(1), tile_p(1){}

    Gemm::Gemm(
        socl::Context& ctx,
        std::span<const uint32_t> shaderBytecodes,
        uint32_t subgroup_tile_m,
        uint32_t subgroup_tile_n,
        uint32_t subgroup_tile_p,
        uint32_t subgroup_tile_cnt_m,
        uint32_t subgroup_tile_cnt_p,
        uint32_t shared_tile_n_multiplier,
        uint32_t reg_tile_m,
        uint32_t inner_tile_n,
        uint32_t reg_tile_p
    ):
        ctx(ctx),
        tile_m(subgroup_tile_cnt_m * subgroup_tile_m * reg_tile_m),
        tile_n(shared_tile_n_multiplier * subgroup_tile_n * inner_tile_n),
        tile_p(subgroup_tile_cnt_p * subgroup_tile_p * reg_tile_p){
        const uint32_t threadgroup_tile_m =
            subgroup_tile_cnt_m * subgroup_tile_m;
        const uint32_t threadgroup_tile_p =
            subgroup_tile_cnt_p * subgroup_tile_p;

        this->pipeline = ctx.createShaderPipeline({
            .spirv = shaderBytecodes,
            .bindings = {
                {0, socl::DescriptorType::StorageBuffer},
                {1, socl::DescriptorType::StorageBuffer},
                {2, socl::DescriptorType::StorageBuffer},
            },
            .pushConstantSize = sizeof(GemmArguments),
            .specConstants = {
                {0, socl::specConstant(shared_tile_n_multiplier)},
                {1, socl::specConstant(subgroup_tile_cnt_m)},
                {2, socl::specConstant(subgroup_tile_cnt_p)},
                {3, socl::specConstant(subgroup_tile_m)},
                {4, socl::specConstant(subgroup_tile_n)},
                {5, socl::specConstant(subgroup_tile_p)},
                {6, socl::specConstant(threadgroup_tile_m)},
                {7, socl::specConstant(threadgroup_tile_p)},
                {8, socl::specConstant(reg_tile_m)},
                {9, socl::specConstant(inner_tile_n)},
                {10, socl::specConstant(reg_tile_p)}
            }
        });
        this->descSet = ctx.createDescriptorSet(pipeline);
    }
    socl::DispatchToken Gemm::execute(
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
        return ctx.submitAsync();
    }
    socl::DispatchToken Gemm::operator()(
        socl::Buffer A,
        socl::Buffer B,
        socl::Buffer C,
        const GemmArguments& args
    ){
        std::vector<socl::Buffer> inputs = {A, B};
        std::vector<socl::Buffer> inouts = {C};
        std::vector<socl::Buffer> outputs = {};
        return this->execute(inputs, inouts, outputs, &args, sizeof(GemmArguments));
    }
}
