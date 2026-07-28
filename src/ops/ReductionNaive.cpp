#include "socl/ShaderPipeline.hpp"
#include <soclblas/ops/ReductionNaive.hpp>
#include <AvgNaiveFP32_SPIRV>
#include <MaxNaiveFP32_SPIRV>
#include <MinNaiveFP32_SPIRV>
#include <SumNaiveFP32_SPIRV>

namespace soclblas{
    MaxNaive::MaxNaive(
        socl::Context& ctx,
        uint32_t thread_num
    ):ctx(ctx), thread_num(thread_num){
        this->pipeline = ctx.createShaderPipeline({
            .spirv = MaxNaiveFP32_SPIRV,
            .bindings = {
                {0, socl::DescriptorType::StorageBuffer},
                {1, socl::DescriptorType::StorageBuffer},
                {2, socl::DescriptorType::StorageBuffer},
            },
            .pushConstantSize = sizeof(IndexedUnaryReductionArguments),
            .specConstants = {
                {0, socl::specConstant(std::uint32_t{thread_num})},
            }
        });
        this->descSet = ctx.createDescriptorSet(pipeline);
    }

    void MaxNaive::execute(
        std::span<socl::Buffer> inputs,
        std::span<socl::Buffer> inouts,
        std::span<socl::Buffer> outputs,
        const void* args,
        std::size_t argsSize
    ){
        this->descSet.bindBuffer(0, inputs[0]);
        this->descSet.bindBuffer(1, outputs[0]);
        this->descSet.bindBuffer(2, outputs[1]);
        this->descSet.update();

        ctx.begin();
        ctx.use(pipeline);
        ctx.bind(descSet);
        ctx.push(args, argsSize);

        const IndexedUnaryReductionArguments* reductionArgs =
            (const IndexedUnaryReductionArguments*)args;
        ctx.dispatch(reductionArgs->b, 1, 1);
        ctx.submitAndWait();
    }

    void MaxNaive::operator()(
        socl::Buffer a,
        socl::Buffer outValue,
        socl::Buffer outIndex,
        const IndexedUnaryReductionArguments& args
    ){
        std::vector<socl::Buffer> inputs = {a};
        std::vector<socl::Buffer> inouts = {};
        std::vector<socl::Buffer> outputs = {outValue, outIndex};
        this->execute(
            inputs,
            inouts,
            outputs,
            &args,
            sizeof(IndexedUnaryReductionArguments)
        );
    }

    MinNaive::MinNaive(
        socl::Context& ctx,
        uint32_t thread_num
    ):ctx(ctx), thread_num(thread_num){
        this->pipeline = ctx.createShaderPipeline({
            .spirv = MinNaiveFP32_SPIRV,
            .bindings = {
                {0, socl::DescriptorType::StorageBuffer},
                {1, socl::DescriptorType::StorageBuffer},
                {2, socl::DescriptorType::StorageBuffer},
            },
            .pushConstantSize = sizeof(IndexedUnaryReductionArguments),
            .specConstants = {
                {0, socl::specConstant(std::uint32_t{thread_num})},
            }
        });
        this->descSet = ctx.createDescriptorSet(pipeline);
    }

    void MinNaive::execute(
        std::span<socl::Buffer> inputs,
        std::span<socl::Buffer> inouts,
        std::span<socl::Buffer> outputs,
        const void* args,
        std::size_t argsSize
    ){
        this->descSet.bindBuffer(0, inputs[0]);
        this->descSet.bindBuffer(1, outputs[0]);
        this->descSet.bindBuffer(2, outputs[1]);
        this->descSet.update();

        ctx.begin();
        ctx.use(pipeline);
        ctx.bind(descSet);
        ctx.push(args, argsSize);

        const IndexedUnaryReductionArguments* reductionArgs =
            (const IndexedUnaryReductionArguments*)args;
        ctx.dispatch(reductionArgs->b, 1, 1);
        ctx.submitAndWait();
    }

    void MinNaive::operator()(
        socl::Buffer a,
        socl::Buffer outValue,
        socl::Buffer outIndex,
        const IndexedUnaryReductionArguments& args
    ){
        std::vector<socl::Buffer> inputs = {a};
        std::vector<socl::Buffer> inouts = {};
        std::vector<socl::Buffer> outputs = {outValue, outIndex};
        this->execute(
            inputs,
            inouts,
            outputs,
            &args,
            sizeof(IndexedUnaryReductionArguments)
        );
    }

    AvgNaive::AvgNaive(
        socl::Context& ctx,
        uint32_t thread_num
    ):ctx(ctx), thread_num(thread_num){
        this->pipeline = ctx.createShaderPipeline({
            .spirv = AvgNaiveFP32_SPIRV,
            .bindings = {
                {0, socl::DescriptorType::StorageBuffer},
                {1, socl::DescriptorType::StorageBuffer},
            },
            .pushConstantSize = sizeof(UnaryReductionArguments),
            .specConstants = {
                {0, socl::specConstant(std::uint32_t{thread_num})},
            }
        });
        this->descSet = ctx.createDescriptorSet(pipeline);
    }

    void AvgNaive::execute(
        std::span<socl::Buffer> inputs,
        std::span<socl::Buffer> inouts,
        std::span<socl::Buffer> outputs,
        const void* args,
        std::size_t argsSize
    ){
        this->descSet.bindBuffer(0, inputs[0]);
        this->descSet.bindBuffer(1, outputs[0]);
        this->descSet.update();

        ctx.begin();
        ctx.use(pipeline);
        ctx.bind(descSet);
        ctx.push(args, argsSize);

        const UnaryReductionArguments* reductionArgs =
            (const UnaryReductionArguments*)args;
        ctx.dispatch(reductionArgs->b, 1, 1);
        ctx.submitAndWait();
    }

    void AvgNaive::operator()(
        socl::Buffer a,
        socl::Buffer out,
        const UnaryReductionArguments& args
    ){
        std::vector<socl::Buffer> inputs = {a};
        std::vector<socl::Buffer> inouts = {};
        std::vector<socl::Buffer> outputs = {out};
        this->execute(inputs, inouts, outputs, &args, sizeof(UnaryReductionArguments));
    }

    SumNaive::SumNaive(
        socl::Context& ctx,
        uint32_t thread_num
    ):ctx(ctx), thread_num(thread_num){
        this->pipeline = ctx.createShaderPipeline({
            .spirv = SumNaiveFP32_SPIRV,
            .bindings = {
                {0, socl::DescriptorType::StorageBuffer},
                {1, socl::DescriptorType::StorageBuffer},
            },
            .pushConstantSize = sizeof(UnaryReductionArguments),
            .specConstants = {
                {0, socl::specConstant(std::uint32_t{thread_num})},
            }
        });
        this->descSet = ctx.createDescriptorSet(pipeline);
    }

    void SumNaive::execute(
        std::span<socl::Buffer> inputs,
        std::span<socl::Buffer> inouts,
        std::span<socl::Buffer> outputs,
        const void* args,
        std::size_t argsSize
    ){
        this->descSet.bindBuffer(0, inputs[0]);
        this->descSet.bindBuffer(1, outputs[0]);
        this->descSet.update();

        ctx.begin();
        ctx.use(pipeline);
        ctx.bind(descSet);
        ctx.push(args, argsSize);

        const UnaryReductionArguments* reductionArgs =
            (const UnaryReductionArguments*)args;
        ctx.dispatch(reductionArgs->b, 1, 1);
        ctx.submitAndWait();
    }

    void SumNaive::operator()(
        socl::Buffer a,
        socl::Buffer out,
        const UnaryReductionArguments& args
    ){
        std::vector<socl::Buffer> inputs = {a};
        std::vector<socl::Buffer> inouts = {};
        std::vector<socl::Buffer> outputs = {out};
        this->execute(inputs, inouts, outputs, &args, sizeof(UnaryReductionArguments));
    }
}
