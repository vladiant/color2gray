cmake_minimum_required(VERSION 3.20)

project(color2gray_bmp)

add_library(
    ColorToGrayLib
    lib/include/time_bench.hpp
    lib/src/time_bench.cpp
    lib/include/bitmap.hpp
    lib/src/bitmap.cpp
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
        CXX_STANDARD 20
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

target_include_directories(
    ColorToGrayLib
    PUBLIC
        lib/include/
        argspp
)

add_executable(
    ${PROJECT_NAME}
    argspp/args.h
    argspp/args.cpp
    main.cpp
)

set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
        CXX_STANDARD 20
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
        ColorToGrayLib
)

# Tests
include(CTest)

add_executable(
    TestColor2Gray
    test/test_data.hpp
    test/test_data.cpp
    test/test_utils.hpp
    test/test_utils.cpp
    test/test_color2gray.cpp
    test/main.cpp
)

set_target_properties(
    TestColor2Gray
    PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

target_include_directories(
    TestColor2Gray
    PRIVATE
        test
)

target_link_libraries(
    TestColor2Gray
    PRIVATE
        ColorToGrayLib
)

add_test(NAME TestColor2Gray COMMAND TestColor2Gray)
