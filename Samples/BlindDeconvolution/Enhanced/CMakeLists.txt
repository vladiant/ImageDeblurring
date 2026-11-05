cmake_minimum_required(VERSION 3.16)
project(enhanced_deconvblind LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED)

add_library(deconvblind STATIC deconvblind.cpp)
target_include_directories(deconvblind PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(deconvblind PUBLIC ${OpenCV_LIBS})

add_executable(deconvblind_app main.cpp)
target_link_libraries(deconvblind_app PRIVATE deconvblind)
