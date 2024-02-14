
project(lpa_lib)

set (SOURCES
        compressor.cpp
        file.cpp)

set (CUDA_SOURCES decompressor.cu)

add_library(${PROJECT_NAME} ${SOURCES} )


