cmake_minimum_required(VERSION 3.20)

project(implementations)

find_package(OpenCV REQUIRED)

add_executable(
    ${PROJECT_NAME}
    include/fft.hpp
    src/fft.cpp
    include/impl1.hpp
    src/impl1.cpp
    include/impl2.hpp
    src/impl2.cpp
    include/impl3.hpp
    src/impl3.cpp
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
        include
        ${OpenCV_INCLUDE_DIRS}
)
