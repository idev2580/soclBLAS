#include <soclblas/ops/GemmContiguousNaive.hpp>
#include <GemmContiguousNaiveFP32_SPIRV>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace soclblas{
    namespace {
        constexpr uint64_t max_thread_accum_count = 128;
        constexpr uint64_t max_shared_memory_bytes = 64ull * 1024ull;

        void validate_gemm_contiguous_config(
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
                throw std::invalid_argument("GemmContiguousNaiveFP32 block and thread tile sizes must be greater than zero");
            }
            if(block_m % thread_tile_m != 0 || block_p % thread_tile_p != 0) {
                throw std::invalid_argument("GemmContiguousNaiveFP32 block_m/block_p must be divisible by thread tile sizes");
            }
            if(
                block_m % 4 != 0 ||
                block_n % 4 != 0 ||
                block_p % 4 != 0
            ) {
                throw std::invalid_argument("GemmContiguousNaiveFP32 block sizes must be divisible by 4");
            }
            if(uint64_t(thread_tile_m) * uint64_t(thread_tile_p) > max_thread_accum_count) {
                throw std::invalid_argument("GemmContiguousNaiveFP32 thread tile must contain at most 128 output elements");
            }

            const uint32_t local_m = block_m / thread_tile_m;
            const uint32_t local_p = block_p / thread_tile_p;
            if(uint64_t(local_m) * uint64_t(local_p) > 256) {
                throw std::invalid_argument("GemmContiguousNaiveFP32 workgroup size must be less than or equal to 256");
            }

            const uint64_t shared_elements =
                uint64_t(block_m) * uint64_t(block_n) +
                uint64_t(block_n) * uint64_t(block_p);
            if(shared_elements * sizeof(float) > max_shared_memory_bytes) {
                throw std::invalid_argument("GemmContiguousNaiveFP32 shared memory usage must be less than or equal to 64 KiB");
            }
        }

        void validate_gemm_contiguous_arguments(const GemmContiguousArguments& args) {
            if((args.flags & ~GemmContiguousFlagsMask) != 0) {
                throw std::invalid_argument("GemmContiguousNaiveFP32 received unknown contiguous GEMM flags");
            }
            if(
                args.a_stride % 4 != 0 ||
                args.b_stride % 4 != 0 ||
                args.c_stride % 4 != 0 ||
                args.a_major_stride % 4 != 0 ||
                args.b_major_stride % 4 != 0 ||
                args.c_major_stride % 4 != 0
            ) {
                throw std::invalid_argument("GemmContiguousNaiveFP32 matrix and major strides must be divisible by 4");
            }
        }
    }

    GemmContiguousNaiveFP32::GemmContiguousNaiveFP32(
        socl::Context& ctx,
        uint32_t block_m,
        uint32_t block_n,
        uint32_t block_p,
        uint32_t thread_tile_m,
        uint32_t thread_tile_p
    ):ctx(ctx),
        block_m(block_m),
        block_n(block_n),
        block_p(block_p),
        thread_tile_m(thread_tile_m),
        thread_tile_p(thread_tile_p)
    {
        validate_gemm_contiguous_config(
            block_m,
            block_n,
            block_p,
            thread_tile_m,
            thread_tile_p
        );

        const uint32_t local_m = block_m / thread_tile_m;
        const uint32_t local_p = block_p / thread_tile_p;
        this->pipeline = ctx.createShaderPipeline({
            .spirv = std::span<const uint32_t>(GemmContiguousNaiveFP32_SPIRV),
            .bindings = {
                {0, socl::DescriptorType::StorageBuffer},
                {1, socl::DescriptorType::StorageBuffer},
                {2, socl::DescriptorType::StorageBuffer},
            },
            .pushConstantSize = sizeof(GemmContiguousArguments),
            .specConstants = {
                {0, socl::specConstant(std::uint32_t{block_m})},
                {1, socl::specConstant(std::uint32_t{block_n})},
                {2, socl::specConstant(std::uint32_t{block_p})},
                {3, socl::specConstant(std::uint32_t{block_m / 4})},
                {4, socl::specConstant(std::uint32_t{block_n / 4})},
                {5, socl::specConstant(std::uint32_t{block_p / 4})},
                {6, socl::specConstant(std::uint32_t{local_m})},
                {7, socl::specConstant(std::uint32_t{local_p})},
                {8, socl::specConstant(std::uint32_t{thread_tile_m})},
                {9, socl::specConstant(std::uint32_t{thread_tile_p})},
            }
        });
        this->descSet = ctx.createDescriptorSet(pipeline);
    }

    socl::DispatchToken GemmContiguousNaiveFP32::execute(
        std::span<socl::Buffer> inputs,
        std::span<socl::Buffer> inouts,
        std::span<socl::Buffer> outputs,
        const void* args,
        std::size_t argsSize
    ){
        if(argsSize != sizeof(GemmContiguousArguments)) {
            throw std::invalid_argument("GemmContiguousNaiveFP32 argsSize must match GemmContiguousArguments");
        }

        const GemmContiguousArguments* gemmArgs = (const GemmContiguousArguments*)args;
        validate_gemm_contiguous_arguments(*gemmArgs);
        if(
            gemmArgs->m % block_m != 0 ||
            gemmArgs->n % block_n != 0 ||
            gemmArgs->p % block_p != 0
        ) {
            throw std::invalid_argument("GemmContiguousNaiveFP32 dimensions must be divisible by block_m/block_n/block_p");
        }

        this->descSet.bindBuffer(0, inputs[0]);
        this->descSet.bindBuffer(1, inputs[1]);
        this->descSet.bindBuffer(2, inouts[0]);
        this->descSet.update();

        ctx.begin();
        ctx.use(pipeline);
        ctx.bind(descSet);
        ctx.push(args, argsSize);

        const uint32_t tiled_m = gemmArgs->m / block_m;
        const uint32_t tiled_p = gemmArgs->p / block_p;
        // Map P to X to favor A-tile cache reuse on implementations that
        // schedule neighboring X workgroups close together.
        ctx.dispatch(tiled_p, tiled_m, gemmArgs->b);
        return ctx.submitAsync();
    }

    socl::DispatchToken GemmContiguousNaiveFP32::operator()(
        socl::Buffer A,
        socl::Buffer B,
        socl::Buffer C,
        const GemmContiguousArguments& args
    ){
        std::vector<socl::Buffer> inputs = {A, B};
        std::vector<socl::Buffer> inouts = {C};
        std::vector<socl::Buffer> outputs = {};
        return this->execute(inputs, inouts, outputs, &args, sizeof(GemmContiguousArguments));
    }
}
