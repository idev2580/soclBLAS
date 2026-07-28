# Scan generated shader files
file(GLOB GEN_SHADERS "${CMAKE_CURRENT_SOURCE_DIR}/generated/*.comp")
# Add manually coded shader files
set(
    MANUAL_SHADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/AxpyFP32.comp"
    "${CMAKE_CURRENT_SOURCE_DIR}/AxpyOutPlaceFP32.comp"
    "${CMAKE_CURRENT_SOURCE_DIR}/AvgNaiveFP32.comp"
    "${CMAKE_CURRENT_SOURCE_DIR}/DotProductNaiveFP32.comp"
    # "${CMAKE_SOURCE_DIR}/shaders/GemvNaiveFP32.comp"
    "${CMAKE_CURRENT_SOURCE_DIR}/GemmNaiveFP32.comp"
    "${CMAKE_CURRENT_SOURCE_DIR}/GemmOutPlaceNaiveFP32.comp"
    "${CMAKE_CURRENT_SOURCE_DIR}/MaxNaiveFP32.comp"
    "${CMAKE_CURRENT_SOURCE_DIR}/MinNaiveFP32.comp"
    "${CMAKE_CURRENT_SOURCE_DIR}/MatMulNaiveFP32.comp"
    "${CMAKE_CURRENT_SOURCE_DIR}/SumNaiveFP32.comp"
)
set(
    ALL_SHADER_SOURCES
    ${GEN_SHADERS}
    ${MANUAL_SHADERS}
)

set(SHADER_HEADERS "" )

# Compile shader codes at build time
foreach(SHADER_PATH IN LISTS ALL_SHADER_SOURCES)
  get_filename_component(FILE_NAME ${SHADER_PATH} NAME)

  string(REPLACE ".comp" "_SPIRV" VAR_NAME ${FILE_NAME})

  set(SHADER_HEADER "${CMAKE_BINARY_DIR}/shaders/${VAR_NAME}")

  add_custom_command(
    OUTPUT ${SHADER_HEADER}
    COMMAND glslangValidator --target-env vulkan1.4 ${SHADER_PATH} -x -o ${SHADER_HEADER} --vn ${VAR_NAME}
    DEPENDS ${SHADER_PATH}
    COMMENT "Generating header for: ${FILE_NAME}"
  )
  list(APPEND SHADER_HEADERS ${SHADER_HEADER})
endforeach()

add_custom_target(compile_shaders ALL DEPENDS ${SHADER_HEADERS})
set(SHADER_HEADERS ${SHADER_HEADERS} PARENT_SCOPE)
