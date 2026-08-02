#pragma once
#include <cstdint>
#include <string_view>
#include <vector>

namespace soclblas::detail{
    std::vector<uint32_t> compileShaderTemplate(
        std::string_view shaderTemplate,
        std::string_view epilogueSource,
        std::string_view sourceName
    );
}
