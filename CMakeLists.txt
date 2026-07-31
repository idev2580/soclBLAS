cmake_minimum_required(VERSION 3.14)
project(soclblas)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include(FetchContent)

FetchContent_Declare(
  socl
  GIT_REPOSITORY https://github.com/idev2580/socl.git
  GIT_TAG        v0.0.8
)
FetchContent_MakeAvailable(socl)

add_subdirectory(shaders)
add_subdirectory(src)
add_subdirectory(tests)

add_library(soclblas SHARED ${SOURCES})
set_target_properties(soclblas PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
add_dependencies(soclblas compile_shaders)
target_include_directories(soclblas
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_BINARY_DIR}/shaders"
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
