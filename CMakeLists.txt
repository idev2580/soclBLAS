cmake_minimum_required(VERSION 3.14)
project(kpBLAS)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include(FetchContent)

add_definitions(-DKOMPUTE_DISABLE_VK_DEBUG_LAYERS=1) #Due to buggy Kompute implementation(Compile not done!)
find_package(kompute REQUIRED)

add_subdirectory(shaders)
add_subdirectory(src)
# add_subdirectory(tests)

include_directories("${CMAKE_SOURCE_DIR}/include" "${CMAKE_BINARY_DIR}/shaders")
add_library(kpblas SHARED ${SOURCES})
target_link_libraries(kpblas PUBLIC kompute)