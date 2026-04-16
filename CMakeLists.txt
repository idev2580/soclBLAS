cmake_minimum_required(VERSION 3.14)
project(kpBLAS)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include(FetchContent)

add_subdirectory(shaders)
add_subdirectory(src)
# add_subdirectory(tests)

# Copy-paste compile_commands.json automatically after build
add_custom_command(TARGET kpblas POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_BINARY_DIR}/compile_commands.json
        ${CMAKE_SOURCE_DIR}/compile_commands.json
)