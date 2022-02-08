cmake_minimum_required(VERSION 3.10)

project(color2gray_simple)

add_executable(
    ${PROJECT_NAME}
    amy_colors.h
    image_io.cpp
    images.h
    main.cpp
)

set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

