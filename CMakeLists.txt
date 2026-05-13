cmake_minimum_required(VERSION 3.20)
project(ellph LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type if not set from command line
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(ALGLIB_ROOT "" CACHE PATH "Path to ALGLIB C++ source tree")
if(NOT ALGLIB_ROOT)
    if(DEFINED ENV{ALGLIB_ROOT})
        set(ALGLIB_ROOT "$ENV{ALGLIB_ROOT}")
    else()
        set(ALGLIB_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/alglib-cpp")
    endif()
endif()

set(ALGLIB_SRC_DIR "${ALGLIB_ROOT}/src")
set(ALGLIB_REQUIRED_FILES
    "${ALGLIB_SRC_DIR}/ap.cpp"
    "${ALGLIB_SRC_DIR}/ap.h"
    "${ALGLIB_SRC_DIR}/alglibinternal.cpp"
    "${ALGLIB_SRC_DIR}/alglibinternal.h"
    "${ALGLIB_SRC_DIR}/alglibmisc.cpp"
    "${ALGLIB_SRC_DIR}/alglibmisc.h"
    "${ALGLIB_SRC_DIR}/linalg.cpp"
    "${ALGLIB_SRC_DIR}/linalg.h"
    "${ALGLIB_SRC_DIR}/solvers.cpp"
    "${ALGLIB_SRC_DIR}/solvers.h"
    "${ALGLIB_SRC_DIR}/optimization.cpp"
    "${ALGLIB_SRC_DIR}/optimization.h"
    "${ALGLIB_SRC_DIR}/kernels_avx2.h"
    "${ALGLIB_SRC_DIR}/kernels_fma.h"
    "${ALGLIB_SRC_DIR}/kernels_sse2.h"
    "${ALGLIB_SRC_DIR}/stdafx.h"
)
foreach(ALGLIB_FILE IN LISTS ALGLIB_REQUIRED_FILES)
    if(NOT EXISTS "${ALGLIB_FILE}")
        message(FATAL_ERROR
            "Missing ALGLIB file: ${ALGLIB_FILE}\n"
            "Download ALGLIB C++ and either place it at third_party/alglib-cpp "
            "or configure with -DALGLIB_ROOT=/path/to/alglib-cpp.")
    endif()
endforeach()

add_library(alglib STATIC
    "${ALGLIB_SRC_DIR}/ap.cpp"
    "${ALGLIB_SRC_DIR}/alglibinternal.cpp"
    "${ALGLIB_SRC_DIR}/alglibmisc.cpp"
    "${ALGLIB_SRC_DIR}/linalg.cpp"
    "${ALGLIB_SRC_DIR}/solvers.cpp"
    "${ALGLIB_SRC_DIR}/optimization.cpp"
)
target_include_directories(alglib PUBLIC "${ALGLIB_SRC_DIR}")

# all src/*.cpp plus the benchmark main
file(GLOB ELLPH_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/geometry/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/solvers/*.cpp"
)

add_executable(main
    ${ELLPH_SOURCES}
    main.cpp
)



if(APPLE)
    # Include directories
    target_include_directories(main PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/geometry"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/solvers"
        "/opt/homebrew/include"
        "/opt/homebrew/opt/eigen/include/eigen3"
        "/opt/homebrew/opt/boost/include"
        "/opt/homebrew/opt/nlopt/include"
    )

    # Library search paths
    target_link_directories(main PRIVATE
        "/opt/homebrew/opt/boost/lib"
        "/opt/homebrew/opt/nlopt/lib"
    )

    # Link against NLopt (Boost is header-only for what you are doing)
    target_link_libraries(main PRIVATE
        nlopt
        alglib
    )
else()
    # Include directories
    target_include_directories(main PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/geometry"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/solvers"
    )
    
    # Find and link NLopt
    find_package(nlopt CONFIG REQUIRED)
    if(TARGET nlopt::nlopt)
        target_link_libraries(main PRIVATE nlopt::nlopt alglib)
    elseif(TARGET NLopt::nlopt)
        target_link_libraries(main PRIVATE NLopt::nlopt alglib)
    else()
        message(FATAL_ERROR "nlopt found but no imported target")
    endif()
    # Find Eigen3
    find_package(Eigen3 CONFIG REQUIRED)
    target_link_libraries(main PRIVATE Eigen3::Eigen)
    
    # Boost (header-only)
    find_package(Boost CONFIG REQUIRED)
    target_include_directories(main PRIVATE ${Boost_INCLUDE_DIRS})
endif()

# Put the binary in build/output/main
set_target_properties(main PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output"
)
