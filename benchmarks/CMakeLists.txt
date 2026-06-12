# ----------------------------------------------------------------------------
# Benchmarks for the bundled ideal-gas models.
#
# Built only when BUILD_BENCHMARKS=ON. Uses Google Benchmark and links the main
# header-only library target (which propagates the include path and the Enzyme
# `-fplugin=` flag). Configure with the `release` / `release-max` preset for
# meaningful numbers; the autodiff is correct at every level but only fast under
# optimization.
# ----------------------------------------------------------------------------

# Google Benchmark (installed). Allow the user to override the location.
if(NOT benchmark_DIR)
    set(benchmark_DIR "/home/dannylong/programming/install/benchmark/lib/cmake/benchmark")
endif()
find_package(benchmark REQUIRED CONFIG)

add_executable(bench_ideal_models src/bench_ideal_models.cpp)
target_link_libraries(bench_ideal_models PRIVATE
    ${PROJECT_NAME}::${PROJECT_NAME}
    benchmark::benchmark
)
target_compile_options(bench_ideal_models PRIVATE -Wall -Wextra)

# ----------------------------------------------------------------------------
# bench_cubic_vs_teqp: head-to-head comparison of this library's cubic models
# against NIST's teqp (https://github.com/usnistgov/teqp).
#
# teqp is pulled in via FetchContent with its full CMake build system; the
# python bindings, tests, and C++ wrapper are switched off, and the benchmark
# links teqp's `teqpinterface` header-only target.
# ----------------------------------------------------------------------------
include(FetchContent)
set(TEQP_NO_PYTHON ON CACHE BOOL "" FORCE)
set(TEQP_NO_TESTS ON CACHE BOOL "" FORCE)
set(TEQP_NO_TEQPCPP ON CACHE BOOL "" FORCE)
FetchContent_Declare(teqp
    GIT_REPOSITORY https://github.com/usnistgov/teqp.git
    GIT_TAG v0.23.1
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(teqp)

add_executable(bench_cubic_vs_teqp src/bench_cubic_vs_teqp.cpp)
target_link_libraries(bench_cubic_vs_teqp PRIVATE
    ${PROJECT_NAME}::${PROJECT_NAME}
    benchmark::benchmark
    teqpinterface
    # teqpinterface does not propagate the include paths of these two teqp
    # dependencies; its compiled targets link them separately.
    autodiff
    nlohmann_json_schema_validator
)
target_compile_options(bench_cubic_vs_teqp PRIVATE -Wall -Wextra)
