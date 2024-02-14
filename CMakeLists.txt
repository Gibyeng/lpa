cmake_minimum_required(VERSION 3.10)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(cuda_lpa CXX C CUDA)

set(CMAKE_CUDA_FLAGS "--expt-extended-lambda -lineinfo -gencode arch=compute_70,code=sm_70 -O3")
set(CMAKE_CUDA_STANDARD 14)

include_directories(
        ${PROJECT_SOURCE_DIR}/include/
)
include_directories(
        ${PROJECT_SOURCE_DIR}/3rd_party/moderngpu/src/moderngpu
)
add_subdirectory(src)
add_subdirectory(baselines)
add_subdirectory(test)
add_subdirectory(3rd_party/google-test)

