cmake_minimum_required(VERSION 3.20)

project(FftDeblurrer LANGUAGES CXX)

add_library(
    ${PROJECT_NAME}
    FftDeblurrer.h
    FftDeblurrer.cpp
    GradientFftRegularizer.h
    GradientFftRegularizer.cpp
    HLTwoThirdsFftRegularizer.h
    HLTwoThirdsFftRegularizer.cpp
    LaplaceFftRegularizer.h
    LaplaceFftRegularizer.cpp
    TmFftRegularizer.h
    TmFftRegularizer.cpp
    CutOffFftDeblurrer.h
    CutOffFftDeblurrer.cpp
    FftRegularizer.h
    FftRegularizer.cpp
    HLFftRegularizer.h
    HLFftRegularizer.cpp
    InverseFftDeblurrer.h
    InverseFftDeblurrer.cpp
    LsFftDeblurrer.h
    LsFftDeblurrer.cpp
    TvFftRegularizer.h
    TvFftRegularizer.cpp
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
        ImageDeblurrer
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

