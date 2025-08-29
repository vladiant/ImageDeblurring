cmake_minimum_required(VERSION 3.20)

project(experiment LANGUAGES CXX)

find_package(OpenCV REQUIRED)

add_library(
    ${PROJECT_NAME}
    CgTmDeblurrer.h
    CgTmDeblurrer.cpp
    CgTvDeblurrer.h
    CgTvDeblurrer.cpp
    LrTmDeblurrer.h
    LrTmDeblurrer.cpp
    LrTvDeblurrer.h
    LrTvDeblurrer.cpp
    vanCittertTmDeblurrer.h
    vanCittertTmDeblurrer.cpp
    vanCittertTvDeblurrer.h
    vanCittertTvDeblurrer.cpp
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
        ${OpenCV_LIBS}
        ImageDeblurrer
        IterativeDeblurrer
)

target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
        ${OpenCV_INCLUDE_DIRS}
        .
)

target_compile_options(
    ${PROJECT_NAME}
    PRIVATE
         $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
         $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Werror>
)

