# Scan CPP files
file(GLOB SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp" "${CMAKE_SOURCE_DIR}/src/ops/*.cpp")

# Export to parent
set(SOURCES ${SOURCES} PARENT_SCOPE)

