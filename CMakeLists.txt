cmake_minimum_required(VERSION 3.20)
project(ellph LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type if not set from command line
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Your sources: all src/*.cpp plus the benchmark main
file(GLOB ELLPH_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

add_executable(benchmark_stats2
    ${ELLPH_SOURCES}
    benchmark_stats2.cpp
)



if(APPLE)
    # Include directories
    target_include_directories(benchmark_stats2 PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "/opt/homebrew/include"
        "/opt/homebrew/opt/eigen/include/eigen3"
        "/opt/homebrew/opt/boost/include"
        "/opt/homebrew/opt/nlopt/include"
    )

    # Library search paths
    target_link_directories(benchmark_stats2 PRIVATE
        "/opt/homebrew/opt/boost/lib"
        "/opt/homebrew/opt/nlopt/lib"
    )

    # Link against NLopt (Boost is header-only for what you are doing)
    target_link_libraries(benchmark_stats2 PRIVATE
        nlopt
    )
else()
    # Include directories
    target_include_directories(benchmark_stats2 PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
    
    # Find and link NLopt
    find_package(NLopt REQUIRED)
    target_link_libraries(benchmark_stats2 PRIVATE NLopt::nlopt)
    
    # Find Eigen3
    find_package(Eigen3 REQUIRED)
    target_link_libraries(benchmark_stats2 PRIVATE Eigen3::Eigen)
    
    # Boost (header-only)
    find_package(Boost REQUIRED)
    target_include_directories(benchmark_stats2 PRIVATE ${Boost_INCLUDE_DIRS})
endif()

# Put the binary in build/output/benchmark_stats2
set_target_properties(benchmark_stats2 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output"
)