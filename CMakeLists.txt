cmake_minimum_required(VERSION 3.14)
project(soclblas)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include(FetchContent)

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(SOCLBLAS_BUILD_TESTS_DEFAULT ON)
else()
    set(SOCLBLAS_BUILD_TESTS_DEFAULT OFF)
endif()
option(
    SOCLBLAS_BUILD_TESTS
    "Build soclBLAS tests and performance executables"
    ${SOCLBLAS_BUILD_TESTS_DEFAULT}
)

FetchContent_Declare(
  socl
  GIT_REPOSITORY https://github.com/idev2580/socl.git
  GIT_TAG        v0.0.10
)
FetchContent_MakeAvailable(socl)

set(GEMM_NAIVE_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/templates/GemmNaiveFP32.comp"
)
set(MATMUL_NAIVE_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/templates/MatMulNaiveFP32.comp"
)
set(GEMM_OUT_PLACE_NAIVE_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/templates/GemmOutPlaceNaiveFP32.comp"
)
set(UNARY_ELEMENTWISE_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/templates/UnaryElementwiseFP32.comp"
)
set(BINARY_ELEMENTWISE_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/templates/BinaryElementwiseFP32.comp"
)
set(EMBEDDED_SHADER_TEMPLATES_CPP
    "${CMAKE_CURRENT_BINARY_DIR}/shaders/EmbeddedShaderTemplates.cpp"
)
add_custom_command(
    OUTPUT "${EMBEDDED_SHADER_TEMPLATES_CPP}"
    COMMAND ${CMAKE_COMMAND}
        -DGEMM_TEMPLATE=${GEMM_NAIVE_TEMPLATE}
        -DMATMUL_TEMPLATE=${MATMUL_NAIVE_TEMPLATE}
        -DGEMM_OUT_PLACE_TEMPLATE=${GEMM_OUT_PLACE_NAIVE_TEMPLATE}
        -DUNARY_ELEMENTWISE_TEMPLATE=${UNARY_ELEMENTWISE_TEMPLATE}
        -DBINARY_ELEMENTWISE_TEMPLATE=${BINARY_ELEMENTWISE_TEMPLATE}
        -DOUTPUT_FILE=${EMBEDDED_SHADER_TEMPLATES_CPP}
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedShaderTemplates.cmake"
    DEPENDS
        "${GEMM_NAIVE_TEMPLATE}"
        "${MATMUL_NAIVE_TEMPLATE}"
        "${GEMM_OUT_PLACE_NAIVE_TEMPLATE}"
        "${UNARY_ELEMENTWISE_TEMPLATE}"
        "${BINARY_ELEMENTWISE_TEMPLATE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedShaderTemplates.cmake"
    COMMENT "Embedding soclBLAS shader templates"
    VERBATIM
)

add_subdirectory(shaders)
add_subdirectory(src)
if(SOCLBLAS_BUILD_TESTS)
    add_subdirectory(tests)
endif()

add_library(
    soclblas SHARED
    ${SOURCES}
    "${EMBEDDED_SHADER_TEMPLATES_CPP}"
)
set_target_properties(soclblas PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
add_dependencies(soclblas soclblas_compile_shaders)
target_include_directories(soclblas
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_BINARY_DIR}/shaders"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(soclblas PUBLIC socl)

function(copy_target_runtime_dlls target_name)
    if(WIN32)
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_RUNTIME_DLLS:${target_name}>
                $<TARGET_FILE_DIR:${target_name}>
            COMMAND_EXPAND_LISTS
        )
    endif()
endfunction()

if(TEST_SOURCES)
    if(MSVC)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    endif()
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
    )
    FetchContent_MakeAvailable(googletest)
    add_executable(soclblas_tests ${TEST_SOURCES})
    target_link_libraries(soclblas_tests PRIVATE soclblas GTest::gtest_main)
    copy_target_runtime_dlls(soclblas_tests)

    include(GoogleTest)
    gtest_discover_tests(soclblas_tests)
endif()

if(PERFORMANCE_TEST_SOURCES)
    add_executable(soclblas_performance_tests ${PERFORMANCE_TEST_SOURCES})
    target_link_libraries(soclblas_performance_tests PRIVATE soclblas)
    copy_target_runtime_dlls(soclblas_performance_tests)
endif()

if(PERFORMANCE_SWEEP_TEST_SOURCES)
    add_executable(
        soclblas_performance_sweep_tests
        ${PERFORMANCE_SWEEP_TEST_SOURCES}
    )
    target_link_libraries(
        soclblas_performance_sweep_tests
        PRIVATE soclblas
    )
    copy_target_runtime_dlls(soclblas_performance_sweep_tests)
endif()
