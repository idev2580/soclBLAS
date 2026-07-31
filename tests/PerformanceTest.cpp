#include <chrono>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <socl/Context.hpp>
#include <soclblas/ops/GemmContiguousNaive.hpp>
#include <soclblas/ops/GemmNaive.hpp>
#include <soclblas/ops/GemmOutPlaceNaive.hpp>
#include <soclblas/ops/GemmShared.hpp>
/* Argument sweep targets
 *
 * build/socl_performance_test --cont batch M N P tile_m tile_n tile_p thread_tile_m thread_tile_p iterations
 * Current best (Naive case): ./build/soclblas_performance_tests --naive 4 4096 4096 4096 4 4 8 2 4 1 8 8 8 20 (5 TFLOPs on Radeon 760M, same as theoretical performance)
 * Current best (Shared Memory 32KiB limit) : 
 * ./build/soclblas_performance_tests --cont 8 4096 1024 1024 64 16 256 8 8 2
 * ./build/soclblas_performance_tests --cont 8 4096 1024 1024 64 16 256 16 4 20
 * Current best (Shared Memory 64KiB limit) : ./build/soclblas_performance_tests --cont 8 4096 1024 1024 64 16 256 8 8 20
 *
 * 1. Shared Memory
 * ./build/soclblas_performance_tests --cont 8 4096 1024 1024 128 32 128 8 8 20
 * ./build/soclblas_performance_tests --cont 8 4096 1024 1024 128 16 128 8 8 20
 * ./build/soclblas_performance_tests --cont 8 4096 1024 1024 128 8 128 8 8 20
 *
 * 2. Tile reusage
 * ./build/soclblas_performance_tests --cont 8 4096 1024 1024 64 32 64 8 8 20
 * ./build/soclblas_performance_tests --cont 8 4096 1024 1024 64 32 128 8 8 20
 * ./build/soclblas_performance_tests --cont 8 4096 1024 1024 128 32 64 8 8 20
 * ./build/soclblas_performance_tests --cont 8 4096 1024 1024 128 32 128 8 8 20
 *
 * 3. Register occupancy
 * ./build/soclblas_performance_tests --cont 8 4096 1024 1024 64 32 64 4 4 20
 * ./build/soclblas_performance_tests --cont 8 4096 1024 1024 64 32 64 8 4 20
 * ./build/soclblas_performance_tests --cont 8 4096 1024 1024 64 32 64 4 8 20
 * ./build/soclblas_performance_tests --cont 8 4096 1024 1024 64 32 64 8 8 20
 *
 * 4. Subgroup/shared-memory naive kernels
 * ./build/soclblas_performance_tests --naive 8 4096 1024 1024 8 4 4 2 2 2 8 8 8 20
 * ./build/soclblas_performance_tests --naive-oop 8 4096 1024 1024 8 4 4 2 2 2 8 8 8 20
 */
namespace {
    constexpr int gpu_idx = 1;
    enum class GemmPerfMode {
        Shared,
        Contiguous,
        Naive,
        NaiveOutPlace
    };

    struct GemmPerfConfig {
        uint32_t batch = 64;
        uint32_t m = 4096;
        uint32_t n = 1024;
        uint32_t p = 1024;
        uint32_t tile_m = 128;
        uint32_t tile_n = 32;
        uint32_t tile_p = 128;
        uint32_t thread_tile_m = 8;
        uint32_t thread_tile_p = 8;
        uint32_t subgroup_tile_cnt_m = 2;
        uint32_t subgroup_tile_cnt_p = 2;
        uint32_t shared_tile_n_multiplier = 2;
        uint32_t reg_tile_n = 8;
        uint32_t iterations = 50;
        uint32_t seed = 2580;
    };

    struct ParsedArgs {
        GemmPerfMode mode = GemmPerfMode::Shared;
        GemmPerfConfig config;
    };

    uint32_t parse_u32_arg(
        int argc,
        char** argv,
        int index,
        uint32_t default_value,
        const char* name
    ) {
        if(index >= argc) {
            return default_value;
        }

        const unsigned long value = std::stoul(argv[index]);
        if(value == 0 || value > std::numeric_limits<uint32_t>::max()) {
            throw std::out_of_range(std::string(name) + " must fit in uint32_t and be greater than zero");
        }
        return static_cast<uint32_t>(value);
    }

    GemmPerfConfig parse_config(int argc, char** argv, int first_config_arg) {
        GemmPerfConfig config;
        config.batch = parse_u32_arg(argc, argv, first_config_arg, config.batch, "batch");
        config.m = parse_u32_arg(argc, argv, first_config_arg + 1, config.m, "m");
        config.n = parse_u32_arg(argc, argv, first_config_arg + 2, config.n, "n");
        config.p = parse_u32_arg(argc, argv, first_config_arg + 3, config.p, "p");
        config.tile_m = parse_u32_arg(argc, argv, first_config_arg + 4, config.tile_m, "tile_m");
        config.tile_n = parse_u32_arg(argc, argv, first_config_arg + 5, config.tile_n, "tile_n");
        config.tile_p = parse_u32_arg(argc, argv, first_config_arg + 6, config.tile_p, "tile_p");
        config.thread_tile_m = parse_u32_arg(argc, argv, first_config_arg + 7, config.thread_tile_m, "thread_tile_m");
        config.thread_tile_p = parse_u32_arg(argc, argv, first_config_arg + 8, config.thread_tile_p, "thread_tile_p");
        config.iterations = parse_u32_arg(argc, argv, first_config_arg + 9, config.iterations, "iterations");
        return config;
    }

    GemmPerfConfig parse_naive_config(
        int argc,
        char** argv,
        int first_config_arg
    ) {
        GemmPerfConfig config;
        config.tile_m = 8;
        config.tile_n = 4;
        config.tile_p = 4;

        config.batch = parse_u32_arg(argc, argv, first_config_arg, config.batch, "batch");
        config.m = parse_u32_arg(argc, argv, first_config_arg + 1, config.m, "m");
        config.n = parse_u32_arg(argc, argv, first_config_arg + 2, config.n, "n");
        config.p = parse_u32_arg(argc, argv, first_config_arg + 3, config.p, "p");
        config.tile_m = parse_u32_arg(argc, argv, first_config_arg + 4, config.tile_m, "subgroup_tile_m");
        config.tile_n = parse_u32_arg(argc, argv, first_config_arg + 5, config.tile_n, "subgroup_tile_n");
        config.tile_p = parse_u32_arg(argc, argv, first_config_arg + 6, config.tile_p, "subgroup_tile_p");
        config.subgroup_tile_cnt_m = parse_u32_arg(
            argc,
            argv,
            first_config_arg + 7,
            config.subgroup_tile_cnt_m,
            "subgroup_tile_cnt_m"
        );
        config.subgroup_tile_cnt_p = parse_u32_arg(
            argc,
            argv,
            first_config_arg + 8,
            config.subgroup_tile_cnt_p,
            "subgroup_tile_cnt_p"
        );
        config.shared_tile_n_multiplier = parse_u32_arg(
            argc,
            argv,
            first_config_arg + 9,
            config.shared_tile_n_multiplier,
            "shared_tile_n_multiplier"
        );
        config.thread_tile_m = parse_u32_arg(
            argc,
            argv,
            first_config_arg + 10,
            config.thread_tile_m,
            "reg_tile_m"
        );
        config.reg_tile_n = parse_u32_arg(
            argc,
            argv,
            first_config_arg + 11,
            config.reg_tile_n,
            "reg_tile_n"
        );
        config.thread_tile_p = parse_u32_arg(
            argc,
            argv,
            first_config_arg + 12,
            config.thread_tile_p,
            "reg_tile_p"
        );
        config.iterations = parse_u32_arg(
            argc,
            argv,
            first_config_arg + 13,
            config.iterations,
            "iterations"
        );
        return config;
    }

    bool is_naive_mode(GemmPerfMode mode) {
        return
            mode == GemmPerfMode::Naive ||
            mode == GemmPerfMode::NaiveOutPlace;
    }

    ParsedArgs parse_args(int argc, char** argv) {
        ParsedArgs parsed;
        int first_config_arg = 1;

        if(argc > 1) {
            const std::string mode_arg = argv[1];
            if(mode_arg == "--cont" || mode_arg == "--contiguous") {
                parsed.mode = GemmPerfMode::Contiguous;
                first_config_arg = 2;
            } else if(mode_arg == "--shared") {
                parsed.mode = GemmPerfMode::Shared;
                first_config_arg = 2;
            } else if(mode_arg == "--naive") {
                parsed.mode = GemmPerfMode::Naive;
                first_config_arg = 2;
            } else if(mode_arg == "--naive-oop") {
                parsed.mode = GemmPerfMode::NaiveOutPlace;
                first_config_arg = 2;
            }
        }

        parsed.config =
            is_naive_mode(parsed.mode)
                ? parse_naive_config(argc, argv, first_config_arg)
                : parse_config(argc, argv, first_config_arg);
        return parsed;
    }

    void validate_tile_config(GemmPerfMode mode, const GemmPerfConfig& config) {
        const uint64_t thread_accum_count =
            uint64_t(config.thread_tile_m) * uint64_t(config.thread_tile_p);

        if(is_naive_mode(mode)) {
            const uint64_t subgroup_size =
                uint64_t(config.tile_m) * uint64_t(config.tile_p);
            if(subgroup_size != 32) {
                throw std::out_of_range(
                    "naive GEMM requires subgroup_tile_m * subgroup_tile_p == 32"
                );
            }
            if(
                config.reg_tile_n % config.tile_m != 0 ||
                config.reg_tile_n % config.tile_p != 0
            ) {
                throw std::out_of_range(
                    "naive GEMM reg_tile_n must be divisible by subgroup_tile_m and subgroup_tile_p"
                );
            }
            if(thread_accum_count > 128) {
                throw std::out_of_range(
                    "naive GEMM register tile must contain at most 128 output elements"
                );
            }

            const uint64_t workgroup_threads =
                uint64_t(config.subgroup_tile_cnt_m) *
                uint64_t(config.subgroup_tile_cnt_p) *
                subgroup_size;
            if(workgroup_threads > 256) {
                throw std::out_of_range(
                    "naive GEMM workgroup must contain at most 256 invocations"
                );
            }

            const uint64_t shared_m =
                uint64_t(config.subgroup_tile_cnt_m) *
                uint64_t(config.tile_m) *
                uint64_t(config.thread_tile_m);
            const uint64_t shared_n =
                uint64_t(config.shared_tile_n_multiplier) *
                uint64_t(config.tile_n) *
                uint64_t(config.reg_tile_n);
            const uint64_t shared_p =
                uint64_t(config.subgroup_tile_cnt_p) *
                uint64_t(config.tile_p) *
                uint64_t(config.thread_tile_p);
            const uint64_t shared_bytes =
                shared_n * (shared_m + shared_p) * sizeof(float);
            if(shared_bytes > 64ull * 1024ull) {
                throw std::out_of_range(
                    "naive GEMM shared-memory tiles must use at most 64 KiB"
                );
            }
            return;
        }

        if(mode == GemmPerfMode::Shared) {
            if(config.thread_tile_m > 8 || config.thread_tile_p > 8) {
                throw std::out_of_range("shared GEMM thread tile dimensions must be less than or equal to 8");
            }
        } else if(thread_accum_count > 128) {
            throw std::out_of_range("contiguous GEMM thread tile must contain at most 128 output elements");
        }

        if(config.tile_m % config.thread_tile_m != 0 || config.tile_p % config.thread_tile_p != 0) {
            throw std::out_of_range("tile_m and tile_p must be divisible by thread tile sizes");
        }

        const uint64_t workgroup_threads =
            uint64_t(config.tile_m / config.thread_tile_m) *
            uint64_t(config.tile_p / config.thread_tile_p);
        if(workgroup_threads > 256) {
            throw std::out_of_range("(tile_m / thread_tile_m) * (tile_p / thread_tile_p) must be less than or equal to 256");
        }

        const uint64_t shared_elements =
            uint64_t(config.tile_m) * uint64_t(config.tile_n) +
            uint64_t(config.tile_n) * uint64_t(config.tile_p);
        const uint64_t shared_bytes = shared_elements * sizeof(float);
        const uint64_t max_shared_bytes =
            mode == GemmPerfMode::Contiguous ? 64ull * 1024ull : 32ull * 1024ull;
        if(shared_bytes > max_shared_bytes) {
            throw std::out_of_range(
                mode == GemmPerfMode::Contiguous
                    ? "contiguous GEMM shared memory tile must be less than or equal to 64 KiB"
                    : "shared GEMM shared memory tile must be less than or equal to 32 KiB"
            );
        }
    }

    uint64_t matrix_elements(uint32_t batch, uint32_t rows, uint32_t cols) {
        return uint64_t(batch) * uint64_t(rows) * uint64_t(cols);
    }

    size_t checked_bytes(uint64_t elements) {
        if(elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
            throw std::overflow_error("matrix allocation size exceeds size_t");
        }
        return static_cast<size_t>(elements) * sizeof(float);
    }

    size_t checked_total_bytes(size_t a_bytes, size_t b_bytes, size_t c_bytes) {
        if(a_bytes > std::numeric_limits<size_t>::max() - b_bytes) {
            throw std::overflow_error("total matrix allocation size exceeds size_t");
        }
        const size_t ab_bytes = a_bytes + b_bytes;
        if(ab_bytes > std::numeric_limits<size_t>::max() - c_bytes) {
            throw std::overflow_error("total matrix allocation size exceeds size_t");
        }
        return ab_bytes + c_bytes;
    }

    size_t checked_add_bytes(size_t lhs, size_t rhs) {
        if(lhs > std::numeric_limits<size_t>::max() - rhs) {
            throw std::overflow_error("total matrix allocation size exceeds size_t");
        }
        return lhs + rhs;
    }

    uint32_t checked_stride(uint32_t lhs, uint32_t rhs, const char* name) {
        const uint64_t value = uint64_t(lhs) * uint64_t(rhs);
        if(value > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error(std::string(name) + " exceeds uint32_t");
        }
        return static_cast<uint32_t>(value);
    }

    double bytes_to_mib(size_t bytes) {
        return static_cast<double>(bytes) / (1024.0 * 1024.0);
    }

    void fill_random(std::vector<float>& values, std::mt19937& gen) {
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for(float& value : values) {
            value = dist(gen);
        }
    }

    void print_usage(const char* program_name) {
        std::cout
            << "Usage:\n"
            << "  " << program_name
            << " [--shared|--cont]"
            << " [batch] [m] [n] [p]"
            << " [tile_m] [tile_n] [tile_p]"
            << " [thread_tile_m] [thread_tile_p] [iterations]\n"
            << "  " << program_name
            << " [--naive|--naive-oop]"
            << " [batch] [m] [n] [p]"
            << " [subgroup_tile_m] [subgroup_tile_n] [subgroup_tile_p]"
            << " [subgroup_tile_cnt_m] [subgroup_tile_cnt_p]"
            << " [shared_tile_n_multiplier]"
            << " [reg_tile_m] [reg_tile_n] [reg_tile_p] [iterations]\n";
    }

    const char* mode_name(GemmPerfMode mode) {
        switch(mode) {
            case GemmPerfMode::Shared:
                return "shared";
            case GemmPerfMode::Contiguous:
                return "contiguous";
            case GemmPerfMode::Naive:
                return "naive";
            case GemmPerfMode::NaiveOutPlace:
                return "naive-oop";
        }
        return "unknown";
    }

    double total_flops(const GemmPerfConfig& config) {
        return
            2.0 *
            static_cast<double>(config.batch) *
            static_cast<double>(config.m) *
            static_cast<double>(config.n) *
            static_cast<double>(config.p);
    }

    void print_config(
        GemmPerfMode mode,
        const GemmPerfConfig& config,
        size_t total_bytes
    ) {
        std::cout << std::fixed << std::setprecision(3);
        std::cout
            << "GEMM performance test\n"
            << "  mode=" << mode_name(mode) << "\n"
            << "  batch=" << config.batch
            << ", m=" << config.m
            << ", n=" << config.n
            << ", p=" << config.p << "\n";

        if(is_naive_mode(mode)) {
            std::cout
                << "  subgroup_tile_m=" << config.tile_m
                << ", subgroup_tile_n=" << config.tile_n
                << ", subgroup_tile_p=" << config.tile_p << "\n"
                << "  subgroup_tile_cnt_m=" << config.subgroup_tile_cnt_m
                << ", subgroup_tile_cnt_p=" << config.subgroup_tile_cnt_p
                << ", shared_tile_n_multiplier="
                << config.shared_tile_n_multiplier << "\n"
                << "  reg_tile_m=" << config.thread_tile_m
                << ", reg_tile_n=" << config.reg_tile_n
                << ", reg_tile_p=" << config.thread_tile_p << "\n";
        } else {
            std::cout
                << "  tile_m=" << config.tile_m
                << ", tile_n=" << config.tile_n
                << ", tile_p=" << config.tile_p << "\n"
                << "  thread_tile_m=" << config.thread_tile_m
                << ", thread_tile_p=" << config.thread_tile_p << "\n";
        }

        std::cout
            << "  iterations=" << config.iterations
            << " (async submit per iteration, wait after all submissions)\n"
            << "  host/device buffer bytes=" << total_bytes
            << " (" << bytes_to_mib(total_bytes) << " MiB)\n";
    }

    void print_result(
        const GemmPerfConfig& config,
        double seconds
    ) {
        const double flops = total_flops(config) * static_cast<double>(config.iterations);
        const double gflops = flops / seconds / 1.0e9;
        const double seconds_per_iteration = seconds / static_cast<double>(config.iterations);

        std::cout
            << "  elapsed_ms_total=" << seconds * 1000.0 << "\n"
            << "  elapsed_ms_per_iteration=" << seconds_per_iteration * 1000.0 << "\n"
            << "  total_flops=" << flops << "\n"
            << "  gpu_gflops=" << gflops << "\n";
    }

    template<typename DispatchFunction>
    double measure_dispatches(
        const GemmPerfConfig& config,
        DispatchFunction dispatch
    ) {
        std::vector<socl::DispatchToken> tokens;
        tokens.reserve(config.iterations);

        const auto start = std::chrono::steady_clock::now();
        for(uint32_t i = 0; i < config.iterations; i++) {
            tokens.emplace_back(dispatch());
        }
        for(socl::DispatchToken& token : tokens) {
            token.wait();
        }
        const auto end = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(end - start).count();
    }

    void test_gemm(const GemmPerfConfig& config) {
        const uint64_t a_elements = matrix_elements(config.batch, config.m, config.n);
        const uint64_t b_elements = matrix_elements(config.batch, config.n, config.p);
        const uint64_t c_elements = matrix_elements(config.batch, config.m, config.p);

        const size_t a_bytes = checked_bytes(a_elements);
        const size_t b_bytes = checked_bytes(b_elements);
        const size_t c_bytes = checked_bytes(c_elements);
        const size_t total_bytes = checked_total_bytes(a_bytes, b_bytes, c_bytes);
        const uint32_t a_stride = checked_stride(config.m, config.n, "a_stride");
        const uint32_t b_stride = checked_stride(config.n, config.p, "b_stride");
        const uint32_t c_stride = checked_stride(config.m, config.p, "c_stride");

        print_config(GemmPerfMode::Shared, config, total_bytes);

        std::vector<float> a(static_cast<size_t>(a_elements));
        std::vector<float> b(static_cast<size_t>(b_elements));
        std::vector<float> c(static_cast<size_t>(c_elements), 0.0f);

        socl::Context ctx({gpu_idx});
        // ctx.printGpuInfo(std::cout);
        soclblas::GemmSharedFP32 gemm(
            ctx,
            config.tile_m,
            config.tile_n,
            config.tile_p,
            config.thread_tile_m,
            config.thread_tile_p
        );

        auto bufferA = ctx.createBuffer(a_bytes, socl::BufferType::Auto);
        auto bufferB = ctx.createBuffer(b_bytes, socl::BufferType::Auto);
        auto bufferC = ctx.createBuffer(c_bytes, socl::BufferType::Auto);

        bufferA.write(a.data(), a_bytes);
        bufferB.write(b.data(), b_bytes);
        bufferC.write(c.data(), c_bytes);

        soclblas::GemmArguments args = {
            .b = config.batch,
            .m = config.m,
            .n = config.n,
            .p = config.p,
            .alpha = 1.0f,
            .beta = 0.0f,
            .a_stride = a_stride,
            .b_stride = b_stride,
            .c_stride = c_stride,
            .a_m_stride = config.n,
            .a_n_stride = 1,
            .b_n_stride = config.p,
            .b_p_stride = 1,
            .c_m_stride = config.p,
            .c_p_stride = 1
        };

        std::vector<socl::DispatchToken> tokens;
        tokens.reserve(config.iterations);

        const auto start = std::chrono::steady_clock::now();
        for(uint32_t i = 0; i < config.iterations; i++) {
            tokens.emplace_back(gemm(bufferA, bufferB, bufferC, args));
        }
        for(socl::DispatchToken& token : tokens) {
            token.wait();
        }
        const auto end = std::chrono::steady_clock::now();

        print_result(config, std::chrono::duration<double>(end - start).count());
    }

    void test_cont_gemm(const GemmPerfConfig& config) {
        const uint64_t a_elements = matrix_elements(config.batch, config.m, config.n);
        const uint64_t b_elements = matrix_elements(config.batch, config.n, config.p);
        const uint64_t c_elements = matrix_elements(config.batch, config.m, config.p);

        const size_t a_bytes = checked_bytes(a_elements);
        const size_t b_bytes = checked_bytes(b_elements);
        const size_t c_bytes = checked_bytes(c_elements);
        const size_t total_bytes = checked_total_bytes(a_bytes, b_bytes, c_bytes);
        const uint32_t a_stride = checked_stride(config.m, config.n, "a_stride");
        const uint32_t b_stride = checked_stride(config.n, config.p, "b_stride");
        const uint32_t c_stride = checked_stride(config.m, config.p, "c_stride");

        print_config(GemmPerfMode::Contiguous, config, total_bytes);

        std::vector<float> a(static_cast<size_t>(a_elements));
        std::vector<float> b(static_cast<size_t>(b_elements));
        std::vector<float> c(static_cast<size_t>(c_elements), 0.0f);

        socl::Context ctx({gpu_idx});
        // ctx.printGpuInfo(std::cout);
        soclblas::GemmContiguousNaiveFP32 gemm(
            ctx,
            config.tile_m,
            config.tile_n,
            config.tile_p,
            config.thread_tile_m,
            config.thread_tile_p
        );

        auto bufferA = ctx.createBuffer(a_bytes, socl::BufferType::Auto);
        auto bufferB = ctx.createBuffer(b_bytes, socl::BufferType::Auto);
        auto bufferC = ctx.createBuffer(c_bytes, socl::BufferType::Auto);

        bufferA.write(a.data(), a_bytes);
        bufferB.write(b.data(), b_bytes);
        bufferC.write(c.data(), c_bytes);

        soclblas::GemmContiguousArguments args = {
            .b = config.batch,
            .m = config.m,
            .n = config.n,
            .p = config.p,
            .alpha = 1.0f,
            .beta = 0.0f,
            .a_stride = a_stride,
            .b_stride = b_stride,
            .c_stride = c_stride,
            .a_major_stride = config.n,
            .b_major_stride = config.p,
            .c_major_stride = config.p,
            .flags = 0
        };

        std::vector<socl::DispatchToken> tokens;
        tokens.reserve(config.iterations);

        const auto start = std::chrono::steady_clock::now();
        for(uint32_t i = 0; i < config.iterations; i++) {
            tokens.emplace_back(gemm(bufferA, bufferB, bufferC, args));
        }
        for(socl::DispatchToken& token : tokens) {
            token.wait();
        }
        const auto end = std::chrono::steady_clock::now();

        print_result(config, std::chrono::duration<double>(end - start).count());
    }

    void test_naive_gemm(
        GemmPerfMode mode,
        const GemmPerfConfig& config
    ) {
        const bool out_of_place = mode == GemmPerfMode::NaiveOutPlace;
        const uint64_t a_elements =
            matrix_elements(config.batch, config.m, config.n);
        const uint64_t b_elements =
            matrix_elements(config.batch, config.n, config.p);
        const uint64_t c_elements =
            matrix_elements(config.batch, config.m, config.p);

        const size_t a_bytes = checked_bytes(a_elements);
        const size_t b_bytes = checked_bytes(b_elements);
        const size_t c_bytes = checked_bytes(c_elements);
        size_t total_bytes = checked_total_bytes(a_bytes, b_bytes, c_bytes);
        if(out_of_place) {
            total_bytes = checked_add_bytes(total_bytes, c_bytes);
        }

        print_config(mode, config, total_bytes);

        std::vector<float> a(static_cast<size_t>(a_elements));
        std::vector<float> b(static_cast<size_t>(b_elements));
        std::vector<float> c(static_cast<size_t>(c_elements), 0.0f);
        std::vector<float> out_c;
        if(out_of_place) {
            out_c.resize(static_cast<size_t>(c_elements));
        }

        const uint32_t a_stride =
            checked_stride(config.m, config.n, "a_stride");
        const uint32_t b_stride =
            checked_stride(config.n, config.p, "b_stride");
        const uint32_t c_stride =
            checked_stride(config.m, config.p, "c_stride");

        const soclblas::GemmArguments args = {
            .b = config.batch,
            .m = config.m,
            .n = config.n,
            .p = config.p,
            .alpha = 1.0f,
            .beta = 0.0f,
            .a_stride = a_stride,
            .b_stride = b_stride,
            .c_stride = c_stride,
            .a_m_stride = config.n,
            .a_n_stride = 1,
            .b_n_stride = config.p,
            .b_p_stride = 1,
            .c_m_stride = config.p,
            .c_p_stride = 1
        };

        socl::Context ctx({gpu_idx});
        auto bufferA = ctx.createBuffer(a_bytes, socl::BufferType::Auto);
        auto bufferB = ctx.createBuffer(b_bytes, socl::BufferType::Auto);
        auto bufferC = ctx.createBuffer(c_bytes, socl::BufferType::Auto);

        bufferA.write(a.data(), a_bytes);
        bufferB.write(b.data(), b_bytes);
        bufferC.write(c.data(), c_bytes);

        if(out_of_place) {
            soclblas::GemmOutPlaceNaiveFP32 gemm(
                ctx,
                config.tile_m,
                config.tile_n,
                config.tile_p,
                config.subgroup_tile_cnt_m,
                config.subgroup_tile_cnt_p,
                config.shared_tile_n_multiplier,
                config.thread_tile_m,
                config.reg_tile_n,
                config.thread_tile_p
            );
            auto bufferOutC =
                ctx.createBuffer(c_bytes, socl::BufferType::Auto);
            bufferOutC.write(out_c.data(), c_bytes);

            const soclblas::GemmOutPlaceArguments out_args =
                soclblas::GemmOutPlaceArguments::sameOutputLayout(args);
            const double seconds = measure_dispatches(config, [&]() {
                return gemm(
                    bufferA,
                    bufferB,
                    bufferC,
                    bufferOutC,
                    out_args
                );
            });
            print_result(config, seconds);
            return;
        }

        soclblas::GemmNaiveFP32 gemm(
            ctx,
            config.tile_m,
            config.tile_n,
            config.tile_p,
            config.subgroup_tile_cnt_m,
            config.subgroup_tile_cnt_p,
            config.shared_tile_n_multiplier,
            config.thread_tile_m,
            config.reg_tile_n,
            config.thread_tile_p
        );
        const double seconds = measure_dispatches(config, [&]() {
            return gemm(bufferA, bufferB, bufferC, args);
        });
        print_result(config, seconds);
    }
}

int main(int argc, char** argv) {
    try {
        if(argc > 1 && std::string(argv[1]) == "--help") {
            print_usage(argv[0]);
            return 0;
        }

        const ParsedArgs parsed = parse_args(argc, argv);
        const GemmPerfConfig& config = parsed.config;
        validate_tile_config(parsed.mode, config);
        if(is_naive_mode(parsed.mode)) {
            test_naive_gemm(parsed.mode, config);
        } else if(parsed.mode == GemmPerfMode::Contiguous) {
            test_cont_gemm(config);
        } else {
            test_gemm(config);
        }
    } catch(const std::exception& e) {
        std::cerr << "Performance test failed: " << e.what() << "\n";
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
