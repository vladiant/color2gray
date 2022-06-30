cmake_minimum_required(VERSION 3.10)

project(color2gray_opencv)

find_package(OpenCV REQUIRED)

add_library(
    ColorToGrayLib
    lib/include/utils.hpp
    lib/src/utils.cpp
    lib/include/rgb.hpp
    lib/src/amy_xyz.cpp
    lib/src/amy_xyz.cpp
    lib/include/amy_lab.hpp
    lib/src/amy_lab.cpp
    lib/include/gray_image.hpp
    lib/src/gray_image.cpp
    lib/include/color_image.hpp
    lib/src/color_image.cpp
)

set_target_properties(
    ColorToGrayLib
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        LINKER_LANGUAGE CXX
        SOVERSION 1
        VERSION 1.0.0
)

target_compile_options(
    ColorToGrayLib
    PRIVATE
         $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
         $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)

target_link_libraries(
    ColorToGrayLib
    PUBLIC
        ${OpenCV_LIBS}
)

target_include_directories(
    ColorToGrayLib
    PUBLIC
        lib/include/
)

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

target_compile_options(
    ${PROJECT_NAME}
    PRIVATE
         $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
         $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
        ${OpenCV_LIBS}
        ColorToGrayLib
)
