cmake_minimum_required(VERSION 3.20)

project(Collection LANGUAGES CXX)

add_subdirectory(Utils)
add_subdirectory(BlurKernel)
add_subdirectory(ImageDeblurrer)
add_subdirectory(IterativeDeblurrer)
add_subdirectory(experiment)
add_subdirectory(FftDeblurrer)
add_subdirectory(OcvFftDeblurrer)


find_package(OpenCV REQUIRED)

add_executable(
    ${PROJECT_NAME}
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
        ${OpenCV_LIBS}
        experiment
        FftDeblurrer
        OcvFftDeblurrer
)

target_include_directories(
    ${PROJECT_NAME}
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
)
