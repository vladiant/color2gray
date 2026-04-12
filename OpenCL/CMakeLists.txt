cmake_minimum_required(VERSION 3.20)

project(color2gray_bmp)

# --- Optional OpenCL support ---
option(USE_OPENCL "Enable OpenCL acceleration" ON)
if(USE_OPENCL)
    find_package(OpenCL)
endif()

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
    lib/include/opencl_context.hpp
    lib/src/opencl_context.cpp
)

if(OpenCL_FOUND)
    target_compile_definitions(ColorToGrayLib PUBLIC HAS_OPENCL)
    target_link_libraries(ColorToGrayLib PUBLIC OpenCL::OpenCL)
    message(STATUS "OpenCL found — GPU acceleration enabled")

    # Copy kernel files to build directory
    file(GLOB KERNEL_FILES "${CMAKE_SOURCE_DIR}/lib/kernels/*.cl")
    foreach(KERNEL ${KERNEL_FILES})
        get_filename_component(KERNEL_NAME ${KERNEL} NAME)
        configure_file(${KERNEL} "${CMAKE_BINARY_DIR}/kernels/${KERNEL_NAME}" COPYONLY)
    endforeach()
else()
    message(STATUS "OpenCL not found — building CPU-only")
endif()

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

# Benchmark
add_executable(
    Benchmark
    benchmark.cpp
)

set_target_properties(
    Benchmark
    PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

target_compile_options(
    Benchmark
    PRIVATE
         $<$<CXX_COMPILER_ID:MSVC>:/W4>
         $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -O3>
)

target_link_libraries(
    Benchmark
    PRIVATE
        ColorToGrayLib
)
