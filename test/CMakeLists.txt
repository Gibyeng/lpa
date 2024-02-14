project(test_run)

enable_testing()

add_executable(${PROJECT_NAME} decompress_test.cu)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_MWAITXINTRIN_H_INCLUDED")
target_link_libraries(${PROJECT_NAME}
        PUBLIC
        lpa_lib
        GTest::GTest
        GTest::Main)
add_test(test_all ${PROJECT_NAME})
set_target_properties(
        ${PROJECT_NAME}
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)