project(novaplus)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_MWAITXINTRIN_H_INCLUDED")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

set(LINKER
        empty.cpp)
 add_executable(novaplus novaplus.cu ${LINKER} )

target_link_libraries(${PROJECT_NAME}
        PUBLIC
        lpa_lib)
set_target_properties(
        ${PROJECT_NAME}
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
