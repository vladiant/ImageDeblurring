cmake_minimum_required(VERSION 3.20)

project(regularization03)

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
)

target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
        ${OpenCV_INCLUDE_DIRS}
)
