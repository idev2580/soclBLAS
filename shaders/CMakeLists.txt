# Scan generated shader files
file(GLOB GEN_SHADERS "${CMAKE_SOURCE_DIR}/shaders/generated/*.comp")
# Add manually coded shader files
set(
    MANUAL_SHADERS
    "${CMAKE_SOURCE_DIR}/shaders/AxpyFP32.comp"
    "${CMAKE_SOURCE_DIR}/shaders/GemvNaiveFP32.comp"
    "${CMAKE_SOURCE_DIR}/shaders/GemmNaiveFP32.comp"
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