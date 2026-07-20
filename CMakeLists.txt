cmake_minimum_required(VERSION 3.14)
project(soclblas)

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
  socl
  GIT_REPOSITORY https://github.com/idev2580/socl.git
  GIT_TAG        v0.0.3a
)
FetchContent_MakeAvailable(socl)

add_subdirectory(shaders)
add_subdirectory(src)
add_subdirectory(tests)

add_library(soclblas SHARED ${SOURCES})
add_dependencies(soclblas compile_shaders)
target_include_directories(soclblas
    PUBLIC
        "${CMAKE_SOURCE_DIR}/include"
        "${CMAKE_BINARY_DIR}/shaders"
)
target_link_libraries(soclblas PUBLIC socl)

if(TEST_SOURCES)
    add_executable(soclblas_tests ${TEST_SOURCES})
    target_link_libraries(soclblas_tests PRIVATE soclblas GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(soclblas_tests)
endif()
