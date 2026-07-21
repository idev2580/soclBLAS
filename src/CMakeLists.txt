# Scan CPP files
file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/ops/*.cpp")

# Export to parent
set(SOURCES ${SOURCES} PARENT_SCOPE)

