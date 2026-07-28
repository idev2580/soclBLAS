#include <soclblas/ops/GemmShared.hpp>
#include <GemmSharedFP32_SPIRV>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace soclblas{
    namespace {
        constexpr uint32_t max_thread_tile_m = 8;
        constexpr uint32_t max_thread_tile_p = 8;

        void validate_gemm_shared_config(
            uint32_t block_m,
            uint32_t block_n,
            uint32_t block_p,
            uint32_t thread_tile_m,
            uint32_t thread_tile_p
        ) {
            if(
                block_m == 0 ||
                block_n == 0 ||
                block_p == 0 ||
                thread_tile_m == 0 ||
                thread_tile_p == 0
            ) {
                throw std::invalid_argument("GemmSharedFP32 block and thread tile sizes must be greater than zero");
            }
            if(block_m % thread_tile_m != 0 || block_p % thread_tile_p != 0) {
                throw std::invalid_argument("GemmSharedFP32 block_m/block_p must be divisible by thread tile sizes");
            }
            if(thread_tile_m > max_thread_tile_m || thread_tile_p > max_thread_tile_p) {
                throw std::invalid_argument("GemmSharedFP32 thread tile sizes must be less than or equal to 8");
            }

            const uint32_t local_m = block_m / thread_tile_m;
            const uint32_t local_p = block_p / thread_tile_p;
            if(uint64_t(local_m) * uint64_t(local_p) > 256) {
                throw std::invalid_argument("GemmSharedFP32 workgroup size must be less than or equal to 256");
            }

            const uint64_t shared_elements =
                uint64_t(block_m) * uint64_t(block_n) +
                uint64_t(block_n) * uint64_t(block_p);
            if(shared_elements * sizeof(float) > 32ull * 1024ull) {
                throw std::invalid_argument("GemmSharedFP32 shared memory usage must be less than or equal to 32 KiB");
            }
        }
    }

    GemmSharedFP32::GemmSharedFP32(
        socl::Context& ctx,
        uint32_t block_m,
        uint32_t block_n,
        uint32_t block_p,
        uint32_t thread_tile_m,
        uint32_t thread_tile_p
    ):Gemm(ctx),
        ctx(ctx),
        block_m(block_m),
        block_n(block_n),
        block_p(block_p),
        thread_tile_m(thread_tile_m),
        thread_tile_p(thread_tile_p)
    {
        validate_gemm_shared_config(
            block_m,
            block_n,
            block_p,
            thread_tile_m,
            thread_tile_p
        );

        const uint32_t local_m = block_m / thread_tile_m;
        const uint32_t local_p = block_p / thread_tile_p;
        this->pipeline = ctx.createShaderPipeline({
            .spirv = std::span<const uint32_t>(GemmSharedFP32_SPIRV),
            .bindings = {
                {0, socl::DescriptorType::StorageBuffer},
                {1, socl::DescriptorType::StorageBuffer},
                {2, socl::DescriptorType::StorageBuffer},
            },
            .pushConstantSize = sizeof(GemmArguments),
            .specConstants = {
                {0, socl::specConstant(std::uint32_t{block_m})},
                {1, socl::specConstant(std::uint32_t{block_n})},
                {2, socl::specConstant(std::uint32_t{block_p})},
                {3, socl::specConstant(std::uint32_t{local_m})},
                {4, socl::specConstant(std::uint32_t{local_p})},
                {5, socl::specConstant(std::uint32_t{thread_tile_m})},
                {6, socl::specConstant(std::uint32_t{thread_tile_p})},
            }
        });
        this->descSet = ctx.createDescriptorSet(pipeline);
    }

    void GemmSharedFP32::execute(
        std::span<socl::Buffer> inputs,
        std::span<socl::Buffer> inouts,
        std::span<socl::Buffer> outputs,
        const void* args,
        std::size_t argsSize
    ){
        this->descSet.bindBuffer(0, inputs[0]);
        this->descSet.bindBuffer(1, inputs[1]);
        this->descSet.bindBuffer(2, inouts[0]);
        this->descSet.update();

        ctx.begin();
        ctx.use(pipeline);
        ctx.bind(descSet);
        ctx.push(args, argsSize);

        const GemmArguments* gemmArgs = (const GemmArguments*)args;
        const uint32_t tiled_m =
            (gemmArgs->m / block_m) + (gemmArgs->m % block_m != 0);
        const uint32_t tiled_p =
            (gemmArgs->p / block_p) + (gemmArgs->p % block_p != 0);
        ctx.dispatch(gemmArgs->b, tiled_m, tiled_p);
        ctx.submitAndWait();
    }

    void GemmSharedFP32::operator()(
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
