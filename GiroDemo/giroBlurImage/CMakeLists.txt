cmake_minimum_required(VERSION 3.20)

project(giroBlurImage)

find_package(OpenCV REQUIRED)

add_executable(
    ${PROJECT_NAME}
    include/BlurKernelUtils.h
    include/GyroBlurKernelBuilder.h
    include/GyroBlurParams.h
    include/GyroDataReader.h
    include/SparseBlurKernel.h
    src/BlurKernelUtils.cpp
    src/GyroBlurKernelBuilder.cpp
    src/GyroBlurParams.cpp
    src/GyroDataReader.cpp
    src/SparseBlurKernel.cpp
    src/main.cpp
)

set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
        include
)

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
        ${OpenCV_LIBS}
        GiroProcessor
)

target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
        ${OpenCV_INCLUDE_DIRS}
)
