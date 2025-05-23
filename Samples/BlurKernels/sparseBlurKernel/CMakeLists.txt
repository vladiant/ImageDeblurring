cmake_minimum_required(VERSION 3.10)

project(sparseBlurKernel)

add_executable(
    ${PROJECT_NAME}
    include/SparseBlurkernel.hpp
    src/SparseBlurkernel.cpp
    src/SparseBlurkernel_test.cpp
)

target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
        include
)

set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

