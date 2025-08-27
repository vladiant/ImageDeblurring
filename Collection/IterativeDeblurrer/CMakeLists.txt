cmake_minimum_required(VERSION 3.20)

project(IterativeDeblurrer LANGUAGES CXX)

add_library(
    ${PROJECT_NAME}
    vanCittertDeblurrer.h
    vanCittertDeblurrer.cpp
    TmIterationRegularizer.h
    TmIterationRegularizer.cpp
    LrDeblurrer.h
    LrDeblurrer.cpp
    IterativeDeblurrer.h
    IterativeDeblurrer.cpp
    HLIterationRegularizer.h 
    HLIterationRegularizer.cpp
    CgDeblurrer.h
    CgDeblurrer.cpp
    TvIterationRegularizer.h
    TvIterationRegularizer.cpp
    MaskLaplaceIterationRegularizer.h
    MaskLaplaceIterationRegularizer.cpp
    LaplaceIterationRegularizer.h
    LaplaceIterationRegularizer.cpp
    IterationRegularizer.h
    IterationRegularizer.cpp
    GradientsIterationRegularizer.h
    GradientsIterationRegularizer.cpp  
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
        Collection
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

