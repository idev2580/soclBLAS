#include <soclblas/ops/Operator.hpp>

#include <cstring>

namespace soclblas{
    std::vector<std::byte> Operator::copyPushConstants(
        const void* args,
        std::size_t argsSize
    ){
        std::vector<std::byte> bytes(argsSize);
        if(argsSize != 0){
            std::memcpy(bytes.data(), args, argsSize);
        }
        return bytes;
    }
}
