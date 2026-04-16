# Include
include_directories("${CMAKE_SOURCE_DIR}/include" "${CMAKE_BINARY_DIR}/shaders")

# Scan CPP files
file(GLOB SOURCES "src/*.cpp" "src/ops/*.cpp")
add_library(kpblas SHARED ${SOURCES})

# Link libraries
target_link_libraries(kpblas PUBLIC kompute)
