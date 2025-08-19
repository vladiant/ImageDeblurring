cmake_minimum_required(VERSION 3.20)

project(giroDeblurImage)

find_package(OpenCV REQUIRED)

add_executable(
    ${PROJECT_NAME}
    include/BlurKernelUtils.h
    include/GyroBlurKernelBuilder.h
    include/HLIterationRegularizer.h
    include/IterativeDeblurrer.h
    include/MaskLaplaceIterationRegularizer.h
    include/TvIterationRegularizer.h
    include/CgDeblurrer.h
    include/GyroBlurParams.h
    include/ImageDeblurrer.h
    include/LaplaceIterationRegularizer.h
    include/SparseBlurKernel.h
    include/vanCittertDeblurrer.h
    include/GradientsIterationRegularizer.h
    include/GyroDataReader.h
    include/IterationRegularizer.h
    include/LrDeblurrer.h
    include/TmIterationRegularizer.h
    src/BlurKernelUtils.cpp
    src/GyroBlurParams.cpp
    src/IterationRegularizer.cpp
    src/main.cpp
    src/TvIterationRegularizer.cpp
    src/CgDeblurrer.cpp
    src/GyroDataReader.cpp
    src/IterativeDeblurrer.cpp
    src/MaskLaplaceIterationRegularizer.cpp
    src/vanCittertDeblurrer.cpp
    src/GradientsIterationRegularizer.cpp
    src/HLIterationRegularizer.cpp
    src/LaplaceIterationRegularizer.cpp
    src/SparseBlurKernel.cpp
    src/GyroBlurKernelBuilder.cpp
    src/ImageDeblurrer.cpp
    src/LrDeblurrer.cpp
    src/TmIterationRegularizer.cpp
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
