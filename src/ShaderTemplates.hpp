#pragma once
#include <string_view>

namespace soclblas::detail{
    std::string_view gemmNaiveShaderTemplate();
    std::string_view matmulNaiveShaderTemplate();
    std::string_view gemmOutPlaceNaiveShaderTemplate();
}
