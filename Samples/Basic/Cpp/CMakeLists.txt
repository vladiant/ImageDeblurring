cmake_minimum_required(VERSION 3.10)

project(BasicDeblur)

find_package(OpenCV REQUIRED)

configure_file(licenseplate_motion.jpg . COPYONLY)
configure_file(text_defocus.jpg . COPYONLY)
configure_file(text_motion.jpg . COPYONLY)

add_executable(
    ${PROJECT_NAME}
    main.cpp
)

set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
        ${OpenCV_LIBS}
)
