cmake_minimum_required(VERSION 3.20)

project(BlurKernel LANGUAGES CXX)

add_library(
    ${PROJECT_NAME}
    BlurKernelUtils.h
    GyroBlurKernelBuilder.h
    GyroBlurParams.h
    SparseBlurKernel.h
    BlurKernelUtils.cpp
    GyroBlurKernelBuilder.cpp
    GyroBlurParams.cpp
    SparseBlurKernel.cpp
)

set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        LINKER_LANGUAGE CXX
)

target_link_libraries(
    ${PROJECT_NAME}
    PUBLIC
        Utils
)

target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
        .
)

target_compile_options(
    ${PROJECT_NAME}
    PRIVATE
         $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
         $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Werror>
)

