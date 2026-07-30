# Scan CPP files
set(SOURCES 
	"${CMAKE_CURRENT_SOURCE_DIR}/ops/Axpy.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/ops/AxpyOutPlace.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/ops/DotProductNaive.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/ops/Gemm.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/ops/GemmArguments.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/ops/GemmContiguousNaive.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/ops/GemmNaive.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/ops/GemmOutPlace.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/ops/GemmOutPlaceNaive.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/ops/GemmShared.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/ops/Gemv.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/ops/GemvArguments.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/ops/GemvNaive.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/ops/GemvOutPlace.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/ops/GemvOutPlaceNaive.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/ops/MatMul.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/ops/MatMulNaive.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/ops/Operator.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/ops/ReductionNaive.cpp"
)

# Export to parent
set(SOURCES ${SOURCES} PARENT_SCOPE)
