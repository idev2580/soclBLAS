cmake_minimum_required(VERSION 3.14)
project(kpBLAS)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)
FetchContent_MakeAvailable(googletest)
FetchContent_Declare(
  kompute
  GIT_REPOSITORY https://github.com/KomputeProject/kompute.git
  GIT_TAG        v0.9.0
)
set(KOMPUTE_OPT_LOG_LEVEL "Error" CACHE STRING "" FORCE)
FetchContent_MakeAvailable(kompute)
find_package(fmt REQUIRED)

add_subdirectory(shaders)
add_subdirectory(src)
add_subdirectory(tests)

include_directories("${CMAKE_SOURCE_DIR}/include" "${CMAKE_BINARY_DIR}/shaders")
add_library(kpblas SHARED ${SOURCES})
target_link_libraries(kpblas PUBLIC kompute fmt)

if(TEST_SOURCES)
    add_executable(kpblas_tests ${TEST_SOURCES})
    target_link_libraries(kpblas_tests PRIVATE kpblas GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(kpblas_tests)
endif()