cmake_minimum_required(VERSION 3.20)

project(1d_blind_deconvolution_cga)

find_package(PkgConfig REQUIRED)
pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)


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

if(FFTW3_THREAD_LIBRARY)
    add_compile_definitions(FFTW_HAS_THREADS)
endif()

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
        PkgConfig::FFTW
)

target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
        PkgConfig::FFTW
)
