cmake_minimum_required(VERSION 3.18)
project(stochastic_market_maker_research LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SMM_ENABLE_EIGEN "Use Eigen3 for the Hawkes spectral-radius calculation when available" ON)
option(SMM_NATIVE_OPT "Use -march=native on GCC/Clang builds" ON)

find_package(Eigen3 QUIET NO_MODULE)

add_library(smm_research_core
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/hjb_qvi.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/hawkes.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sim.cpp
)

target_include_directories(smm_research_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}
)

if(MSVC)
    target_compile_options(smm_research_core PRIVATE /O2 /W4 /permissive-)
else()
    target_compile_options(smm_research_core PRIVATE -O3 -Wall -Wextra -Wpedantic)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(smm_research_core PRIVATE -fopenmp-simd)
        if(SMM_NATIVE_OPT)
            target_compile_options(smm_research_core PRIVATE -march=native)
        endif()
    endif()
endif()

if(SMM_ENABLE_EIGEN AND Eigen3_FOUND)
    target_link_libraries(smm_research_core PUBLIC Eigen3::Eigen)
    target_compile_definitions(smm_research_core PRIVATE SMM_HAS_EIGEN=1)
endif()

add_executable(smm_research ${CMAKE_CURRENT_SOURCE_DIR}/../src/research_runner.cpp)
target_link_libraries(smm_research PRIVATE smm_research_core)

if(MSVC)
    target_compile_options(smm_research PRIVATE /O2 /W4 /permissive-)
else()
    target_compile_options(smm_research PRIVATE -O3 -Wall -Wextra -Wpedantic)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(smm_research PRIVATE -fopenmp-simd)
        if(SMM_NATIVE_OPT)
            target_compile_options(smm_research PRIVATE -march=native)
        endif()
    endif()
endif()

add_executable(smm_unit_tests ${CMAKE_CURRENT_SOURCE_DIR}/../tests/unit.cpp)
target_link_libraries(smm_unit_tests PRIVATE smm_research_core)

if(MSVC)
    target_compile_options(smm_unit_tests PRIVATE /O2 /W4 /permissive-)
else()
    target_compile_options(smm_unit_tests PRIVATE -O2 -Wall -Wextra -Wpedantic)
endif()

enable_testing()
add_test(NAME smm_unit_tests COMMAND smm_unit_tests)
