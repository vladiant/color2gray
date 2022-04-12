cmake_minimum_required(VERSION 3.10)

project(color2gray_opencv)

find_package(OpenCV REQUIRED)

add_executable(
    ${PROJECT_NAME}
    utils.hpp
    utils.cpp
    rgb.hpp
    amy_xyz.cpp
    amy_xyz.cpp
    amy_lab.hpp
    amy_lab.cpp
    gray_image.hpp
    gray_image.cpp
    color_image.hpp
    color_image.cpp
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
