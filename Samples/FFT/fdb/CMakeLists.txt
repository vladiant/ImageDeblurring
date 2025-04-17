cmake_minimum_required(VERSION 3.20)

project(fdb_demo LANGUAGES CXX)

find_package(OpenCV REQUIRED)

add_library(
    fdb
    fdb.hpp
    fdb.cpp
)

set_target_properties(
    fdb
    PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        LINKER_LANGUAGE CXX
)

target_link_libraries(
    fdb
    PUBLIC
        ${OpenCV_LIBS}
)

target_include_directories(
    fdb
    PUBLIC
        ${OpenCV_INCLUDE_DIRS}
)

add_executable(
    fdb_demo
    main.cpp
)

target_include_directories(
    fdb_demo
    PUBLIC
        ${OpenCV_INCLUDE_DIRS}
)

set_target_properties(
    fdb_demo
    PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

target_compile_options(
    fdb_demo
    PRIVATE
         $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
         $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Werror>
)

target_link_libraries(
    fdb_demo
    PRIVATE
        fdb
        ${OpenCV_LIBS}
)
