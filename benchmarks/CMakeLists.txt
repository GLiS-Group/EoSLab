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
