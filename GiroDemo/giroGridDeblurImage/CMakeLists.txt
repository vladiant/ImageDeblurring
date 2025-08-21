cmake_minimum_required(VERSION 3.20)

project(giroGridDeblurImage)

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
    include/LsFftDeblurrer.h
    include/CutOffFftDeblurrer.h
    include/InverseFftDeblurrer.h
    include/HLFftRegularizer.h
    include/HLTwoThirdsFftRegularizer.h
    include/TvFftRegularizer.h
    include/LaplaceFftRegularizer.h
    include/GradientFftRegularizer.h
    include/TmFftRegularizer.h
    include/FftDeblurrer.h
    include/FftRegularizer.h
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
    src/LsFftDeblurrer.cpp
    src/CutOffFftDeblurrer.cpp
    src/InverseFftDeblurrer.cpp
    src/HLFftRegularizer.cpp
    src/HLTwoThirdsFftRegularizer.cpp
    src/TvFftRegularizer.cpp
    src/LaplaceFftRegularizer.cpp
    src/GradientFftRegularizer.cpp
    src/TmFftRegularizer.cpp
    src/FftDeblurrer.cpp
    src/FftRegularizer.cpp
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
