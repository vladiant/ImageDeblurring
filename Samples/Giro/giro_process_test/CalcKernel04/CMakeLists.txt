cmake_minimum_required(VERSION 3.20)

project(CalcKernel04)

find_package(OpenCV REQUIRED)

add_executable(
    ${PROJECT_NAME}
    BlockDeblur.h
    GyroBlurKernel.h
    main.cpp
)

set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
        GiroProcessor
        ${OpenCV_LIBS}
)

target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
        ${OpenCV_INCLUDE_DIRS}
)
