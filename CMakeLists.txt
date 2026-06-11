cmake_minimum_required(VERSION 3.21)

# ----------------------------------------------------------------------------
# Compiler requirement: Clang
#
# This project depends on Enzyme (https://enzyme.mit.edu) for automatic
# differentiation. Enzyme ships as an LLVM/Clang plugin, so the project can
# only be built with a Clang compiler that matches the LLVM version Enzyme was
# built against.
#
# If the user has not explicitly selected a compiler, default to clang++ so a
# fresh `cmake --preset ...` works out of the box. An explicit choice (via
# -DCMAKE_CXX_COMPILER=... or the CXX environment variable) is still honoured,
# but is validated to be Clang right after project().
# ----------------------------------------------------------------------------
if(NOT DEFINED CMAKE_CXX_COMPILER AND NOT DEFINED ENV{CXX})
    find_program(EOSLAB_CLANGXX NAMES clang++ clang++-22 clang++-21 clang++-20)
    if(EOSLAB_CLANGXX)
        set(CMAKE_CXX_COMPILER "${EOSLAB_CLANGXX}")
    endif()
endif()

project(MyProject
    VERSION 0.1.0
    DESCRIPTION "A C++ library"
    LANGUAGES CXX
)

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR
        "EoSLab requires the Clang compiler because it depends on Enzyme, "
        "which is an LLVM/Clang plugin.\n"
        "Detected C++ compiler: '${CMAKE_CXX_COMPILER_ID}' "
        "(${CMAKE_CXX_COMPILER}).\n"
        "Configure with Clang, e.g.:\n"
        "  cmake --preset debug -DCMAKE_CXX_COMPILER=clang++\n"
        "or set the CXX environment variable to a clang++ executable."
    )
endif()

# ----------------------------------------------------------------------------
# Enzyme
#
# Locate Enzyme's CMake package.
#
# find_package(Enzyme) creates the imported INTERFACE target `ClangEnzymeFlags`
# which adds `-fplugin=ClangEnzyme-<ver>.so` to the compile line. Linking it to
# our interface library propagates the plugin flag to every consumer (tests).
# ----------------------------------------------------------------------------
find_package(Enzyme REQUIRED CONFIG)

# Prevent in-source builds
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR
        "In-source builds are not allowed.\n"
        "Run: cmake --preset debug  (or release / release-max)"
    )
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Symlink compile_commands.json into the source root for clangd
file(REMOVE "${CMAKE_SOURCE_DIR}/compile_commands.json")
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E create_symlink
        "${CMAKE_BINARY_DIR}/compile_commands.json"
        "${CMAKE_SOURCE_DIR}/compile_commands.json"
)

# Warnings — linked only to our targets so FetchContent deps are unaffected
add_library(${PROJECT_NAME}_warnings INTERFACE)
target_compile_options(${PROJECT_NAME}_warnings INTERFACE
    -Wall -Wextra -Wpedantic
)

# Sanitizers — empty unless ENABLE_SANITIZERS=ON (set by the debug preset)
#
# NOTE: only AddressSanitizer is enabled, NOT UndefinedBehaviorSanitizer.
# UBSan rewrites arithmetic/pointer operations to call its runtime handlers
# (__ubsan_handle_*); Enzyme cannot differentiate through those opaque calls and
# aborts with "No create nofree of empty function (__ubsan_handle_*)". ASan adds
# no such instrumentation inside the differentiated code and works fine.
add_library(${PROJECT_NAME}_sanitizers INTERFACE)
option(ENABLE_SANITIZERS "Enable AddressSanitizer (UBSan is incompatible with Enzyme)" OFF)
if(ENABLE_SANITIZERS)
    target_compile_options(${PROJECT_NAME}_sanitizers INTERFACE
        -fsanitize=address
        -fno-omit-frame-pointer
    )
    target_link_options(${PROJECT_NAME}_sanitizers INTERFACE
        -fsanitize=address
    )
endif()

# clang-tidy — enable with ENABLE_CLANG_TIDY=ON or add to a preset
option(ENABLE_CLANG_TIDY "Run clang-tidy during compilation" OFF)
if(ENABLE_CLANG_TIDY)
    find_program(CLANG_TIDY_EXE NAMES clang-tidy REQUIRED)
    set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
endif()

# Header-only interface library
add_library(${PROJECT_NAME} INTERFACE)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_23)

# Enzyme autodiff plugin flags. The Enzyme documentation recommends linking the
# imported `ClangEnzymeFlags` target, which propagates `-fplugin=...` to every
# consumer of this interface library.
target_link_libraries(${PROJECT_NAME} INTERFACE ClangEnzymeFlags)

option(BUILD_TESTING "Build tests" ON)
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

# Benchmarks for the bundled ideal models (off by default; needs Google
# Benchmark). Build with a release preset for meaningful numbers.
option(BUILD_BENCHMARKS "Build the ideal-model benchmarks (requires Google Benchmark)" OFF)
if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
