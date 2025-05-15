cmake_minimum_required(VERSION 3.28)

project(BlurKernelExample LANGUAGES CXX C)

# Commons
add_library(
    Commons
    Commons/include/export.h
    Commons/include/local_debug.h
    Commons/src/local_debug.c
)

set_target_properties(
    Commons
    PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        C_STANDARD 23
        C_STANDARD_REQUIRED YES
        C_EXTENSIONS NO
)

target_compile_options(
    Commons
    PRIVATE
         $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
         $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)

target_include_directories(
    Commons
    PUBLIC
        Commons/include
)

# BlurKernel
add_library(
    BlurKernel
    BlurKernel/include/blur_kernel.h
    BlurKernel/include/BlurKernel.hpp
    BlurKernel/include/PSF.hpp
    BlurKernel/src/blur_kernel.c
    BlurKernel/src/BlurKernel.cpp
)

set_target_properties(
    BlurKernel
    PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        C_STANDARD 23
        C_STANDARD_REQUIRED YES
        C_EXTENSIONS NO
)

target_compile_options(
    BlurKernel
    PRIVATE
         $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
         $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)

target_include_directories(
    BlurKernel
    PUBLIC
        BlurKernel/include
)

target_link_libraries(
    BlurKernel
    PRIVATE
        Commons
)

# testBlurKernel
add_executable(
    testBlurKernel
    testBlurKernel/main.cpp
)

set_target_properties(
    testBlurKernel
    PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

target_compile_options(
    testBlurKernel
    PRIVATE
         $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
         $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Werror>
)

target_link_libraries(
    testBlurKernel
    PRIVATE
        BlurKernel
)

# testBlurKernelC
add_executable(
    testBlurKernelC
    testBlurKernelC/main.c
)

set_target_properties(
    testBlurKernelC
    PROPERTIES
        C_STANDARD 23
        C_STANDARD_REQUIRED YES
        C_EXTENSIONS NO
)

target_compile_options(
    testBlurKernelC
    PRIVATE
         $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
         $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Werror>
)

target_link_libraries(
    testBlurKernelC
    PRIVATE
        BlurKernel
)



